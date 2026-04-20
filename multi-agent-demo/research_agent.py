"""
Research Agent - LangGraph-based agent for enriching partner context
Goal: Enrich context with partner + sales history by:
1. Retrieving internal sales data from Excel files
2. Retrieving external context via Tavily web search
3. Returning enriched partner profile with insights
"""

from typing import Annotated, Literal, NotRequired
from typing_extensions import TypedDict
import pandas as pd
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ibm import ChatWatsonx
from langchain_tavily import TavilySearch
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()


# ============================================================================
# State Definition
# ============================================================================

class ResearchState(TypedDict):
    """State for the Research Agent workflow."""
    partner_name: str
    messages: list
    internal_data: NotRequired[dict]
    external_data: NotRequired[dict]
    enriched_profile: NotRequired[dict]
    next_step: NotRequired[str]


# ============================================================================
# Internal Sales Data Tools
# ============================================================================

def _load_sales_df() -> pd.DataFrame:
    excel_path = Path("docs/Confluent Sales Cloud Infor.xlsx")
    if not excel_path.exists():
        raise FileNotFoundError("Sales data file not found")

    df = pd.read_excel(excel_path, sheet_name="Sheet1", header=0)
    if all('Unnamed' in str(col) for col in df.columns):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

    if pd.isna(df.columns[0]) or str(df.columns[0]).strip() == '':
        df = df.iloc[:, 1:]

    df.columns = df.columns.str.strip()
    return df


def _parse_crm_date(date_value) -> dict:
    """
    Parse CRM date and return dict with original, parsed, and ISO format.
    
    Handles missing dates (NaN, NaT, "nan" strings) gracefully.
    Missing expiration dates indicate the contract was not signed/won yet.
    
    Args:
        date_value: Date value from CRM (can be string, datetime, NaN, or other)
        
    Returns:
        Dict with 'original', 'parsed' (datetime object), and 'iso' (ISO string) keys
        Returns None values for missing/invalid dates (expected for unsigned contracts)
    """
    from dateutil import parser as date_parser
    
    result = {
        'original': None,
        'parsed': None,
        'iso': None
    }
    
    # Check for various forms of missing data BEFORE attempting to parse
    # 1. pandas NA/NaN/NaT
    if pd.isna(date_value):
        return result
    
    # 2. pandas NaT (Not a Time) - explicit check
    if pd.api.types.is_scalar(date_value) and hasattr(pd, 'NaT') and date_value is pd.NaT:
        return result
    
    # 3. String "nan", "NaN", "NaT", or empty strings
    if isinstance(date_value, str):
        date_str_lower = date_value.strip().lower()
        if date_str_lower in ['nan', 'nat', '', 'none', 'n/a']:
            return result
    
    # 4. None value
    if date_value is None:
        return result
    
    # If we get here, we have a potentially valid date - store original
    result['original'] = str(date_value)
    
    try:
        # Try to parse the date
        if isinstance(date_value, str):
            parsed_date = date_parser.parse(date_value)
        elif hasattr(date_value, 'to_pydatetime'):
            # Handle pandas Timestamp
            parsed_date = date_value.to_pydatetime()
        else:
            parsed_date = date_parser.parse(str(date_value))
        
        result['parsed'] = parsed_date
        result['iso'] = parsed_date.strftime('%Y-%m-%d')
    except Exception as e:
        # If parsing fails, keep None values (this is expected for invalid data)
        pass
    
    return result


@tool
def retrieve_sales_history(partner_name: Annotated[str, "Name of the partner company"]) -> dict:
    """
    Retrieve internal sales data from Confluent Sales Cloud Excel file.
    Returns prior opportunities, stage history, previous follow-ups, and known stakeholders.
    """
    try:
        df = _load_sales_df()

        # Special handling for Confluent: since this is the Confluent sales file,
        # ALL rows are Confluent opportunities, even if "Confluent" isn't in the name
        if partner_name.lower() == 'confluent':
            partner_data = df
        else:
            # For other partners, search across key columns
            mask = (
                df['Opportunity Name'].astype(str).str.contains(partner_name, case=False, na=False) |
                df['Next Steps'].astype(str).str.contains(partner_name, case=False, na=False) |
                df['Products'].astype(str).str.contains(partner_name, case=False, na=False)
            )
            partner_data = df[mask]

            # If still empty, search all columns
            if partner_data.empty:
                mask = df.apply(lambda row: row.astype(str).str.contains(partner_name, case=False, na=False).any(), axis=1)
                partner_data = df[mask]

        if partner_data.empty:
            return {
                "partner_name": partner_name,
                "found": False,
                "message": f"No sales history found for {partner_name}"
            }

        opportunities = []
        for row_num, (idx, row) in enumerate(partner_data.iterrows(), start=1):
            # Extract numeric amount for comparison
            amount_value = row['Amount']
            amount_numeric = None
            if isinstance(amount_value, (int, float)):
                amount_numeric = float(amount_value)
            elif isinstance(amount_value, str):
                import re
                amount_match = re.search(r'[\$]?\s*([\d,]+\.?\d*)', str(amount_value))
                if amount_match:
                    try:
                        amount_numeric = float(amount_match.group(1).replace(',', ''))
                    except ValueError:
                        pass
            
            # Parse dates for better comparison
            close_date_parsed = _parse_crm_date(row['Close Date'])
            contract_exp_parsed = _parse_crm_date(row.get('Contract Expiration Date', 'N/A'))
            
            # Calculate opportunity number (1-based row number from Excel)
            # row_num starts at 1 (first data row), which corresponds to Excel row 2
            # So row_num directly equals the opportunity number (1, 2, 3, ...)
            opportunity_number = row_num
            
            opp = {
                "opportunity_number": opportunity_number,
                "opportunity_name": row['Opportunity Name'],
                "owner": row['Owner Full Name'],
                "stage": row['Stage'],
                "amount": row['Amount'],
                "amount_numeric": amount_numeric,
                "close_date": str(row['Close Date']),
                "close_date_parsed": close_date_parsed['parsed'],
                "close_date_iso": close_date_parsed['iso'],
                "contract_expiration_date": str(row.get('Contract Expiration Date', 'N/A')),
                "contract_expiration_parsed": contract_exp_parsed['parsed'],
                "contract_expiration_iso": contract_exp_parsed['iso'],
                "products": row['Products'],
                "next_steps": row['Next Steps']
            }
            opportunities.append(opp)

        total_opportunities = len(opportunities)
        won_deals = [o for o in opportunities if o['stage'] == 'Won']
        lost_deals = [o for o in opportunities if o['stage'] == 'Lost']
        active_deals = [o for o in opportunities if o['stage'] not in ['Won', 'Lost']]

        total_won_amount = sum(o['amount'] for o in won_deals if isinstance(o['amount'], (int, float)))
        total_active_amount = sum(o['amount'] for o in active_deals if isinstance(o['amount'], (int, float)))

        stakeholders = list(set(o['owner'] for o in opportunities))

        products_used = set()
        for opp in opportunities:
            if pd.notna(opp['products']):
                products = [p.strip() for p in str(opp['products']).split(',')]
                products_used.update(products)

        deal_blockers = []
        for opp in lost_deals:
            if pd.notna(opp['next_steps']):
                deal_blockers.append({
                    "opportunity": opp['opportunity_name'],
                    "reason": opp['next_steps']
                })

        return {
            "partner_name": partner_name,
            "found": True,
            "summary": {
                "total_opportunities": total_opportunities,
                "won_deals": len(won_deals),
                "lost_deals": len(lost_deals),
                "active_deals": len(active_deals),
                "total_won_amount": total_won_amount,
                "total_active_amount": total_active_amount,
                "win_rate": f"{(len(won_deals) / total_opportunities * 100):.1f}%" if total_opportunities > 0 else "0%"
            },
            "opportunities": opportunities,
            "stakeholders": stakeholders,
            "products_used": list(products_used),
            "deal_blockers": deal_blockers,
            "sales_velocity": "High" if len(won_deals) >= 3 else "Medium" if len(won_deals) >= 1 else "Low"
        }

    except Exception as e:
        return {"error": f"Error retrieving sales history: {str(e)}"}


@tool
def get_recent_and_upcoming_contract_actions(partner_name: Annotated[str, "Name of the partner company"]) -> dict:
    """
    Summarize CRM opportunities that suggest renewals, expirations, follow-ups, and product-specific actions.
    """
    try:
        df = _load_sales_df()
        
        # Special handling for Confluent: since this is the Confluent sales file,
        # ALL rows are Confluent opportunities
        if partner_name.lower() == 'confluent':
            partner_data = df
        else:
            mask = df['Opportunity Name'].astype(str).str.contains(partner_name, case=False, na=False)
            partner_data = df[mask]

        action_flags = []
        for _, row in partner_data.iterrows():
            opp_name = str(row.get('Opportunity Name', ''))
            next_steps = str(row.get('Next Steps', ''))
            products = str(row.get('Products', ''))
            close_date = str(row.get('Close Date', ''))

            if "renew" in next_steps.lower() or "renew" in opp_name.lower():
                action_flags.append({
                    "opportunity_name": opp_name,
                    "close_date": close_date,
                    "action_type": "renewal",
                    "recommended_action": f"Prepare renewal outreach for {opp_name}"
                })

            if "cognos" in products.lower() or "cognos" in opp_name.lower():
                action_flags.append({
                    "opportunity_name": opp_name,
                    "close_date": close_date,
                    "action_type": "cognos_review",
                    "recommended_action": f"Meet with the account before renewal because Cognos is in the renewal scope for {opp_name}"
                })

        return {
            "partner_name": partner_name,
            "action_flags": action_flags
        }
    except Exception as e:
        return {"error": f"Error retrieving renewal actions: {str(e)}"}


@tool
def analyze_partner_maturity(sales_data: Annotated[dict, "Sales history data for the partner"]) -> dict:
    """
    Analyze partner maturity level based on sales history.
    Returns maturity assessment and recommendations.
    """
    if not sales_data.get("found"):
        return {
            "maturity_level": "New Partner",
            "assessment": "No prior engagement history",
            "recommendation": "Focus on discovery and relationship building"
        }
    
    summary = sales_data.get("summary", {})
    won_deals = summary.get("won_deals", 0)
    total_won_amount = summary.get("total_won_amount", 0)
    active_deals = summary.get("active_deals", 0)
    
    # Determine maturity level
    if won_deals >= 3 and total_won_amount >= 1000000:
        maturity_level = "Strategic Partner"
        assessment = "Established relationship with multiple successful deals"
        recommendation = "Focus on expansion and upsell opportunities"
    elif won_deals >= 1 and total_won_amount >= 250000:
        maturity_level = "Growing Partner"
        assessment = "Proven success with initial deals"
        recommendation = "Nurture relationship and explore additional use cases"
    elif active_deals > 0:
        maturity_level = "Engaged Prospect"
        assessment = "Active opportunities in pipeline"
        recommendation = "Focus on closing active deals and addressing concerns"
    else:
        maturity_level = "Early Stage"
        assessment = "Limited or no successful engagement"
        recommendation = "Rebuild relationship and understand past challenges"
    
    return {
        "maturity_level": maturity_level,
        "assessment": assessment,
        "recommendation": recommendation,
        "engagement_score": min(100, (won_deals * 20) + (active_deals * 10))
    }


# ============================================================================
# External Context Tools
# ============================================================================

@tool
def search_partner_background(partner_name: Annotated[str, "Name of the partner company"]) -> str:
    """
    Search for partner company background, announcements, and market signals using Tavily.
    IMPORTANT: Focuses on information BEFORE IBM acquisition.
    """
    tavily = TavilySearch(max_results=5)
    query = f"{partner_name} company background news technology partnerships BEFORE IBM acquisition pre-acquisition independent"
    results = tavily.invoke(query)
    return results


@tool
def search_technology_alignment(
    partner_name: Annotated[str, "Name of the partner company"],
    products: Annotated[list[str], "List of products to check alignment for"]
) -> str:
    """
    Search for technology alignment between partner and specific products.
    """
    tavily = TavilySearch(max_results=3)
    products_str = ", ".join(products[:3])  # Limit to top 3 products
    query = f"{partner_name} {products_str} integration technology stack compatibility"
    results = tavily.invoke(query)
    return results


@tool
def search_pre_acquisition_executives(partner_name: Annotated[str, "Name of the partner company"]) -> str:
    """
    Search for key decision-makers and executives at the partner company prior to IBM acquisition.
    Finds people a seller would want to contact: CPO, CTO, CEO, VP Procurement, VP Engineering, etc.
    This is critical for understanding who to contact at the partner company.
    IMPORTANT: Searches for executives AT the partner company, not IBM executives.
    """
    tavily = TavilySearch(max_results=8)
    
    # Search for multiple key roles that sellers would contact
    query = (
        f"{partner_name} company leadership team executives "
        f"CPO \"Chief Procurement Officer\" "
        f"CTO \"Chief Technology Officer\" "
        f"CEO \"Chief Executive Officer\" "
        f"CFO \"Chief Financial Officer\" "
        f"\"VP Procurement\" \"VP Engineering\" \"VP Technology\" "
        f"\"Head of Procurement\" \"Head of Technology\" "
        f"prior to IBM acquisition 2024 2025 "
        f"-IBM -\"Arvind Krishna\" -\"IBM executives\""
    )
    
    results = tavily.invoke(query)
    return results


@tool
def search_revenue_and_growth(partner_name: Annotated[str, "Name of the partner company"]) -> str:
    """
    Search for revenue data and growth trends prior to IBM acquisition.
    Focuses on financial performance before the acquisition.
    """
    tavily = TavilySearch(max_results=5)
    query = f"{partner_name} revenue financial performance growth 2024 2025 prior to the IBM acquisition"
    results = tavily.invoke(query)
    return results


@tool
def search_key_announcements(partner_name: Annotated[str, "Name of the partner company"]) -> str:
    """
    Search for key public releases, product announcements, and strategic initiatives prior to IBM acquisition.
    """
    tavily = TavilySearch(max_results=5)
    query = f"{partner_name} announcements product releases strategic initiatives 2024 2025 prior to the IBM acquisition"
    results = tavily.invoke(query)
    return results


# ============================================================================
# Agent Nodes
# ============================================================================

def research_internal_node(state: ResearchState) -> dict:
    """Node to retrieve internal sales data."""
    partner_name = state["partner_name"]
    
    # Retrieve sales history
    sales_data = retrieve_sales_history.invoke({"partner_name": partner_name})

    # Analyze maturity
    maturity_data = analyze_partner_maturity.invoke({"sales_data": sales_data})
    renewal_actions = get_recent_and_upcoming_contract_actions.invoke({"partner_name": partner_name})

    internal_data = {
        "sales_history": sales_data,
        "maturity_analysis": maturity_data,
        "renewal_actions": renewal_actions
    }
    
    return {"internal_data": internal_data}


def research_external_node(state: ResearchState) -> dict:
    """Node to retrieve external context via web search with pre-acquisition executive data."""
    partner_name = state["partner_name"]
    internal_data = state.get("internal_data", {})
    
    # Get partner background
    background = search_partner_background.invoke({"partner_name": partner_name})
    
    # Get pre-acquisition executive information (CPO, CTO, CEO)
    executives = search_pre_acquisition_executives.invoke({"partner_name": partner_name})
    
    # Get revenue and growth data
    revenue_data = search_revenue_and_growth.invoke({"partner_name": partner_name})
    
    # Get key announcements and releases
    announcements = search_key_announcements.invoke({"partner_name": partner_name})
    
    # Get technology alignment if we have product data
    tech_alignment = None
    sales_history = internal_data.get("sales_history", {})
    if sales_history.get("found") and sales_history.get("products_used"):
        products = sales_history["products_used"]
        tech_alignment = search_technology_alignment.invoke({
            "partner_name": partner_name,
            "products": products
        })
    
    external_data = {
        "background": background,
        "executives": executives,
        "revenue_data": revenue_data,
        "announcements": announcements,
        "technology_alignment": tech_alignment
    }
    
    return {"external_data": external_data}


def synthesize_profile_node(state: ResearchState) -> dict:
    """Node to synthesize enriched partner profile using LLM."""
    partner_name = state["partner_name"]
    internal_data = state.get("internal_data", {})
    external_data = state.get("external_data", {})

    # ------------------------------------------------------------------
    # HARD-CODED: Confluent pre-acquisition executive roster.
    # Tavily frequently surfaces IBM executives instead of the Confluent
    # leadership team that was in place before the acquisition.  We inject
    # the known executive data here so that downstream agents always have
    # reliable names for CPO / CTO / CFO / CEO outreach.
    # ------------------------------------------------------------------
    CONFLUENT_KNOWN_EXECUTIVES = {
        "CPO": {"name": "Shaun Clowes", "title": "Chief Product Officer (CPO)", "note": "Pre-acquisition"},
        "CTO": {"name": "Stephen Deasy", "title": "Chief Technology Officer (CTO)", "note": "Pre-acquisition"},
        "CFO": {"name": "Rohan Sivaram", "title": "Chief Financial Officer (CFO)", "note": "Pre-acquisition"},
        "CEO": {"name": "Jay Kreps", "title": "Chief Executive Officer (CEO) & Co-Founder", "note": "Pre-acquisition"},
    }

    if "confluent" in partner_name.lower():
        # Build a human-readable block for the LLM prompt and for
        # the external_data dict so action_agent can also consume it.
        exec_lines = [
            f"{info['title']}: {info['name']} ({info['note']})"
            for info in CONFLUENT_KNOWN_EXECUTIVES.values()
        ]
        hardcoded_exec_text = (
            "HARD-CODED PRE-ACQUISITION CONFLUENT EXECUTIVES "
            "(use these when Tavily search is unreliable):\n" +
            "\n".join(exec_lines)
        )

        # Merge into external_data so the action_agent's extract_executive_info
        # regex can optionally find them (belt-and-suspenders - action_agent
        # also has its own hard-coded lookup as the primary fallback).
        existing_exec = external_data.get("executives", "") or ""
        if isinstance(existing_exec, str):
            external_data = dict(external_data)  # shallow copy before mutating
            external_data["executives"] = hardcoded_exec_text + "\n\n" + existing_exec
            external_data["known_confluent_executives"] = CONFLUENT_KNOWN_EXECUTIVES
        
        print("INFO: Confluent pre-acquisition executive data injected into partner profile.")

    # Prepare synthesis prompt using Watsonx
    from langchain_ibm import WatsonxLLM

    llm = WatsonxLLM(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url="https://us-south.ml.cloud.ibm.com",
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        apikey=os.getenv("WATSONX_APIKEY"),
        params={
            "max_new_tokens": 1000,
            "temperature": 0.1,
        }
    )

    prompt = f"""You are a sales intelligence analyst. Synthesize the internal sales data and external research into a comprehensive partner profile.

Focus on:
1. Partner maturity level and engagement history
2. Historical sales velocity and deal patterns
3. Prior deal blockers and how to address them
4. Relevant external signals (news, technology trends)
5. Actionable recommendations for next steps

Provide a structured, concise summary.

Partner: {partner_name}

INTERNAL SALES DATA:
{internal_data}

EXTERNAL RESEARCH:
{external_data}

Synthesize this information into an enriched partner profile:"""

    response = llm.invoke(prompt)

    # Create enriched profile
    enriched_profile = {
        "partner_name": partner_name,
        "maturity_level": internal_data.get("maturity_analysis", {}).get("maturity_level", "Unknown"),
        "sales_velocity": internal_data.get("sales_history", {}).get("sales_velocity", "Unknown"),
        "deal_blockers": internal_data.get("sales_history", {}).get("deal_blockers", []),
        "external_signals": external_data.get("background", "No external data available"),
        "synthesis": response,
        "internal_data": internal_data,
        "external_data": external_data
    }

    return {"enriched_profile": enriched_profile}


# ============================================================================
# Graph Construction
# ============================================================================

def create_research_agent():
    """Create and compile the Research Agent graph."""
    
    # Define the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("research_internal", research_internal_node)
    workflow.add_node("research_external", research_external_node)
    workflow.add_node("synthesize_profile", synthesize_profile_node)
    
    # Define edges
    workflow.add_edge(START, "research_internal")
    workflow.add_edge("research_internal", "research_external")
    workflow.add_edge("research_external", "synthesize_profile")
    workflow.add_edge("synthesize_profile", END)
    
    # Compile the graph
    return workflow.compile()


# ============================================================================
# Main Execution
# ============================================================================

def research_partner(partner_name: str) -> dict:
    """
    Main function to research a partner and return enriched profile.
    
    Args:
        partner_name: Name of the partner company to research
        
    Returns:
        Enriched partner profile with internal and external insights
    """
    agent = create_research_agent()
    
    result = agent.invoke({
        "partner_name": partner_name,
        "messages": []
    })
    
    return result.get("enriched_profile", {})


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Research Agent - Partner Intelligence System")
    print("=" * 80)
    
    # Research Confluent
    profile = research_partner("Confluent")
    
    print(f"\n{'=' * 80}")
    print(f"ENRICHED PARTNER PROFILE: {profile['partner_name']}")
    print(f"{'=' * 80}")
    print(f"\nMaturity Level: {profile['maturity_level']}")
    print(f"Sales Velocity: {profile['sales_velocity']}")
    print(f"\nDeal Blockers: {len(profile['deal_blockers'])}")
    for blocker in profile['deal_blockers']:
        print(f"  - {blocker['opportunity']}: {blocker['reason']}")
    
    print(f"\n{'=' * 80}")
    print("SYNTHESIS:")
    print(f"{'=' * 80}")
    print(profile['synthesis'])