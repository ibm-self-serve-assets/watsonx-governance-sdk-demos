"""
Action Agent - LangGraph-based agent for determining next best sales actions

This agent:
1. Analyzes historical patterns by querying sales CRM data
2. Determines what actions have been most successful in similar deals
3. Identifies risks and recommends next steps
4. Creates actionable artifacts (CRM updates, draft emails)
"""

from typing import TypedDict, Annotated, Optional, Dict, List
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import operator
import os
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()


# ============================================================================
# State Definition
# ============================================================================

class ActionState(TypedDict):
    """State for Action Agent workflow"""
    contract_summary: dict  # From Contract Agent
    partner_profile: dict   # From Research Agent
    seller_query: NotRequired[str]
    messages: Annotated[list, operator.add]
    historical_patterns: NotRequired[dict]
    risk_assessment: NotRequired[dict]
    recommended_action: NotRequired[dict]
    crm_updates: NotRequired[dict]
    draft_email: NotRequired[str]
    final_output: NotRequired[str]


# ============================================================================
# Action Agent Class
# ============================================================================

class ActionAgent:
    """
    Action Agent using LangGraph for next best action determination.
    
    This agent:
    1. Analyzes historical action patterns from CRM
    2. Evaluates contract status and partner readiness
    3. Determines recommended next step
    4. Creates actionable artifacts (CRM updates, emails)
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/llama-3-3-70b-instruct",
        url: str = "https://us-south.ml.cloud.ibm.com",
        apikey: Optional[str] = None,
        project_id: Optional[str] = None,
        crm_file_path: str = "docs/Confluent Sales Cloud Infor.xlsx",
        scenario_actions_path: str = "docs/ScenarioActions.pdf"
    ):
        """
        Initialize Action Agent.
        
        Args:
            model_id: Watsonx model ID for text generation
            url: Watsonx API URL
            apikey: IBM Cloud API key (defaults to WATSONX_APIKEY env var)
            project_id: Watsonx project ID (defaults to WATSONX_PROJECT_ID env var)
            crm_file_path: Path to CRM Excel file
            scenario_actions_path: Path to scenario actions PDF
        """
        self.model_id = model_id
        self.url = url
        # Use environment variables if not provided
        self.apikey = apikey or os.getenv("WATSONX_APIKEY")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.crm_file_path = crm_file_path
        self.scenario_actions_path = scenario_actions_path
        self.graph = None
        self.scenario_actions_text = None
        
        # Validate credentials
        if not self.apikey:
            raise ValueError("WATSONX_APIKEY must be provided or set in environment variables")
        if not self.project_id:
            raise ValueError("WATSONX_PROJECT_ID must be provided or set in environment variables")
        
        # Load scenario actions on initialization
        self._load_scenario_actions()
        
    def _load_scenario_actions(self) -> None:
        """Load scenario actions from PDF file"""
        try:
            pdf_path = Path(self.scenario_actions_path)
            if not pdf_path.exists():
                print(f"Warning: Scenario actions file not found: {pdf_path}")
                self.scenario_actions_text = ""
                return
            
            # Extract text from PDF
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_parts = []
                for page in pdf_reader.pages:
                    text_parts.append(page.extract_text())
                self.scenario_actions_text = '\n'.join(text_parts)
            
            print(f"Loaded scenario actions from {pdf_path} ({len(self.scenario_actions_text)} characters)")
        except Exception as e:
            print(f"Error loading scenario actions: {str(e)}")
            self.scenario_actions_text = ""
    
    def _load_crm_data(self) -> pd.DataFrame:
        """Load CRM data from Excel file"""
        try:
            excel_path = Path(self.crm_file_path)
            if not excel_path.exists():
                raise FileNotFoundError(f"CRM file not found: {excel_path}")
            
            # Read Excel with header in first row
            df = pd.read_excel(excel_path, sheet_name="Sheet1", header=0)
            
            # Handle unnamed columns
            if all('Unnamed' in str(col) for col in df.columns):
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
            
            # Drop empty first column if exists
            if pd.isna(df.columns[0]) or str(df.columns[0]).strip() == '':
                df = df.iloc[:, 1:]
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            return df
        except Exception as e:
            print(f"Error loading CRM data: {str(e)}")
            return pd.DataFrame()
    
    def _analyze_historical_patterns(self, contract_type: str, partner_stage: str) -> dict:
        """
        Analyze historical patterns for similar deals.
        
        Args:
            contract_type: Type of contract (e.g., 'ESA', 'MSA')
            partner_stage: Partner maturity stage
            
        Returns:
            Dictionary with historical pattern analysis
        """
        df = self._load_crm_data()
        
        if df.empty:
            return {
                "error": "No CRM data available",
                "patterns": []
            }
        
        # Analyze won deals to find successful patterns
        won_deals = df[df['Stage'] == 'Won'].copy()
        
        if won_deals.empty:
            return {
                "total_won_deals": 0,
                "patterns": [],
                "message": "No won deals found for pattern analysis"
            }
        
        # Extract common next steps from won deals
        next_steps_analysis = []
        if 'Next Steps' in won_deals.columns:
            next_steps = won_deals['Next Steps'].dropna()
            for step in next_steps:
                if isinstance(step, str) and step.strip():
                    next_steps_analysis.append(step)
        
        # Calculate average time to close for won deals
        avg_deal_size = won_deals['Amount'].mean() if 'Amount' in won_deals.columns else 0
        
        # Identify most successful products
        product_success = {}
        if 'Products' in won_deals.columns:
            for products in won_deals['Products'].dropna():
                if isinstance(products, str):
                    for product in products.split(','):
                        product = product.strip()
                        product_success[product] = product_success.get(product, 0) + 1
        
        # Find most successful owners (for best practices)
        top_performers = []
        if 'Owner Full Name' in won_deals.columns:
            owner_counts = won_deals['Owner Full Name'].value_counts()
            top_performers = owner_counts.head(3).to_dict()
        
        return {
            "total_won_deals": len(won_deals),
            "average_deal_size": float(avg_deal_size) if pd.notna(avg_deal_size) else 0,
            "successful_next_steps": next_steps_analysis[:5],  # Top 5
            "successful_products": product_success,
            "top_performers": top_performers,
            "pattern_confidence": "High" if len(won_deals) >= 10 else "Medium" if len(won_deals) >= 5 else "Low"
        }
    
    def _assess_risk(self, contract_summary: dict, partner_profile: dict) -> dict:
        """
        Assess risk level based on contract and partner data.
        """
        risk_factors = []
        risk_score = 0

        portfolio = contract_summary.get("portfolio_summary", {})
        renewal_candidates = portfolio.get("renewal_candidates", [])
        recently_expired = portfolio.get("recently_expired_contracts", [])

        maturity = partner_profile.get('maturity_level', 'Unknown')
        if maturity == 'New Partner':
            risk_factors.append("New partner with no prior engagement history")
            risk_score += 30
        elif maturity == 'Early Stage':
            risk_factors.append("Limited successful engagement history")
            risk_score += 20

        deal_blockers = partner_profile.get('deal_blockers', [])
        if deal_blockers:
            risk_factors.append(f"Historical deal blockers: {len(deal_blockers)} identified")
            risk_score += min(30, len(deal_blockers) * 10)

        velocity = partner_profile.get('sales_velocity', 'Unknown')
        if velocity == 'Low':
            risk_factors.append("Low historical sales velocity")
            risk_score += 15

        if renewal_candidates:
            risk_factors.append(f"{len(renewal_candidates)} contract(s) are within the renewal window")
            risk_score += min(20, len(renewal_candidates) * 10)

        if recently_expired:
            risk_factors.append(f"{len(recently_expired)} contract(s) expired recently")
            risk_score += min(30, len(recently_expired) * 15)

        recommended_next_steps = portfolio.get("recommended_next_steps", [])
        if any("cognos" in step.lower() for step in recommended_next_steps):
            risk_factors.append("Cognos renewal requires proactive stakeholder meeting")
            risk_score += 10

        if risk_score >= 60:
            risk_level = "High"
        elif risk_score >= 30:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "mitigation_required": risk_level in ["High", "Medium"]
        }
    
    def build_agent(self):
        """Build the LangGraph for action determination"""
        
        def analyze_patterns_node(state: ActionState) -> dict:
            """Analyze historical patterns from CRM"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            
            # Extract contract type
            structured = contract_summary.get("structured_summary", {})
            contract_type = structured.get("contract_type", "Unknown") if isinstance(structured, dict) else "Unknown"
            
            # Extract partner stage
            partner_stage = partner_profile.get("maturity_level", "Unknown")
            
            # Analyze patterns
            patterns = self._analyze_historical_patterns(contract_type, partner_stage)
            
            return {
                "historical_patterns": patterns,
                "messages": ["Historical patterns analyzed"]
            }
        
        def assess_risk_node(state: ActionState) -> dict:
            """Assess risk based on contract and partner data"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            
            risk_assessment = self._assess_risk(contract_summary, partner_profile)
            
            return {
                "risk_assessment": risk_assessment,
                "messages": [f"Risk assessed: {risk_assessment['risk_level']}"]
            }
        
        def determine_action_node(state: ActionState) -> dict:
            """Determine recommended next best action for workflows 3, 4, and 5"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            seller_query = state.get("seller_query", "")
            lowered_query = seller_query.lower()

            portfolio = contract_summary.get("portfolio_summary", {})
            active_contracts = portfolio.get("active_contracts", [])
            renewal_candidates = portfolio.get("renewal_candidates", [])
            recently_expired = portfolio.get("recently_expired_contracts", [])
            portfolio_steps = portfolio.get("recommended_next_steps", [])
            renewal_actions = partner_profile.get("internal_data", {}).get("renewal_actions", {}).get("action_flags", [])

            workflow_name = "portfolio_overview_30_day_actions"
            if "renewal" in lowered_query or "expired" in lowered_query or "expir" in lowered_query:
                workflow_name = "renewal_expiration_awareness"
            elif "draft me an email" in lowered_query or "reach out to the cpo" in lowered_query:
                workflow_name = "executive_outreach"

            concrete_steps = []

            if workflow_name == "portfolio_overview_30_day_actions":
                for contract in active_contracts[:3]:
                    filename = contract.get("file_name", "Unknown contract")
                    concrete_steps.append(
                        f"Review {filename} this week to confirm active status, enabled permissions, and any outstanding obligations."
                    )
                concrete_steps.extend([
                    "Schedule a portfolio review with the Confluent account team within the next 30 days.",
                    "Confirm whether any downstream SOW or enablement step is still required before additional co-sell activity."
                ])

            elif workflow_name == "renewal_expiration_awareness":
                for contract in renewal_candidates[:3]:
                    filename = contract.get("file_name", "Unknown contract")
                    renewal_date = contract.get("derived_end_date", "Unknown date")
                    concrete_steps.append(
                        f"Start renewal outreach for {filename} now and align legal and partner management before {renewal_date}."
                    )
                for contract in recently_expired[:3]:
                    filename = contract.get("file_name", "Unknown contract")
                    concrete_steps.append(
                        f"Review the recently expired contract {filename} and decide this week whether to renew, replace, or close it out."
                    )
                concrete_steps.append("Prepare a renewal and expiration dashboard for Confluent with notice windows and owners.")

            elif workflow_name == "executive_outreach":
                concrete_steps.extend([
                    "Draft an executive email to the Confluent CPO with strategic partnership status and a clear call to action.",
                    "Reference the signed ESA and current active agreements at a high level without legal detail.",
                    "Propose a short alignment meeting in the next 2 weeks to discuss enablement and next-phase opportunities."
                ])

            for action in renewal_actions[:3]:
                action_text = action.get("recommended_action")
                if action_text and action_text not in concrete_steps:
                    concrete_steps.append(action_text)

            for step in portfolio_steps[:5]:
                if step not in concrete_steps:
                    concrete_steps.append(step)

            if not concrete_steps:
                concrete_steps = [
                    "Review all active Confluent contracts and confirm current status.",
                    "Update the CRM with the top three actions the seller should take next.",
                    "Prepare outreach to Confluent stakeholders for the next milestone."
                ]

            top_action = concrete_steps[0]

            if workflow_name == "portfolio_overview_30_day_actions":
                action_lines = [
                    "MATCHING_SCENARIO: Contract portfolio overview + next 30 day actions",
                    f"ACTION: {top_action}",
                    "PRIORITY: High",
                    "TIMELINE: Next 30 Days"
                ]
            elif workflow_name == "renewal_expiration_awareness":
                action_lines = [
                    "MATCHING_SCENARIO: Renewal and expiration awareness",
                    f"ACTION: {top_action}",
                    "PRIORITY: High",
                    "TIMELINE: Immediate and next 90 days"
                ]
            else:
                action_lines = [
                    "MATCHING_SCENARIO: Executive outreach (CPO email draft)",
                    "ACTION: Draft and send an executive-ready email to the Confluent CPO focused on partnership status, enablement progress, and the proposed next meeting.",
                    "PRIORITY: High",
                    "TIMELINE: This Week"
                ]

            action_lines.extend([
                f"RATIONALE: {' '.join(concrete_steps[:3])}",
                "SUCCESS_CRITERIA: Seller has a concrete checklist, CRM Agent Next Steps is overwritten, and the next outreach is ready to send.",
                "OWNER: IBM Seller"
            ])

            recommended_action = {
                "workflow_name": workflow_name,
                "raw_recommendation": "\n".join(action_lines),
                "ranked_next_steps": concrete_steps,
                "timestamp": datetime.now().isoformat()
            }

            return {
                "recommended_action": recommended_action,
                "messages": ["Next best action determined"]
            }
        
        def create_artifacts_node(state: ActionState) -> dict:
            """Create actionable artifacts (CRM updates, draft email)"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            recommended_action = state.get("recommended_action", {})
            risk_assessment = state.get("risk_assessment", {})

            partner_name = partner_profile.get("partner_name", "Partner")
            portfolio = contract_summary.get("portfolio_summary", {})
            ranked_next_steps = recommended_action.get("ranked_next_steps", [])
            top_step = ranked_next_steps[0] if ranked_next_steps else "Review contract portfolio and schedule follow-up."

            crm_updates = {
                "opportunity_name": f"{partner_name} - Contract Portfolio Review",
                "stage": "Renewal / Expansion Review",
                "next_step": top_step,
                "owner": "IBM Seller",
                "due_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                "priority": risk_assessment.get("risk_level", "Medium"),
                "contracts_reviewed": len(portfolio.get("contract_paths", [])),
                "agent_next_steps": ranked_next_steps,
                "updated_timestamp": datetime.now().isoformat()
            }

            try:
                df = self._load_crm_data()
                if not df.empty:
                    if "Agent Next Steps" not in df.columns:
                        df["Agent Next Steps"] = ""

                    agent_next_steps_text = " | ".join(ranked_next_steps[:5])

                    partner_mask = (
                        df["Opportunity Name"].astype(str).str.contains(partner_name, case=False, na=False)
                        if "Opportunity Name" in df.columns
                        else pd.Series([False] * len(df))
                    )

                    if partner_mask.any():
                        df.loc[partner_mask, "Agent Next Steps"] = agent_next_steps_text

                        if "Next Steps" in df.columns:
                            df.loc[partner_mask, "Next Steps"] = top_step
                    else:
                        new_row = {col: "" for col in df.columns}
                        if "Opportunity Name" in df.columns:
                            new_row["Opportunity Name"] = crm_updates["opportunity_name"]
                        if "Owner Full Name" in df.columns:
                            new_row["Owner Full Name"] = crm_updates["owner"]
                        if "Stage" in df.columns:
                            new_row["Stage"] = crm_updates["stage"]
                        if "Close Date" in df.columns:
                            new_row["Close Date"] = crm_updates["due_date"]
                        if "Next Steps" in df.columns:
                            new_row["Next Steps"] = crm_updates["next_step"]
                        if "Products" in df.columns:
                            new_row["Products"] = "Portfolio Review"
                        if "Agent Next Steps" in df.columns:
                            new_row["Agent Next Steps"] = agent_next_steps_text

                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                    df.to_excel(self.crm_file_path, sheet_name="Sheet1", index=False)
                    crm_updates["crm_file_updated"] = True
                else:
                    crm_updates["crm_file_updated"] = False
            except Exception as e:
                crm_updates["crm_file_updated"] = False
                crm_updates["crm_update_error"] = str(e)

            email_prompt = ChatPromptTemplate.from_template(
                "Write a professional outreach email from an IBM seller to the CPO at {partner_name}.\n\n"
                "Context:\n"
                "- Risk Level: {risk_level}\n"
                "- Top Next Step: {top_step}\n"
                "- Additional Steps: {additional_steps}\n"
                "- Partner Maturity: {maturity}\n\n"
                "The email should:\n"
                "1. Use executive tone and strategic framing\n"
                "2. Reference active IBM-Confluent agreements at a high level only\n"
                "3. Avoid legal language and operational overload\n"
                "4. Include a clear call to action and suggested meeting\n"
                "Return a full email with subject line."
            )

            llm = WatsonxLLM(
                model_id=self.model_id,
                url=self.url,
                apikey=self.apikey,
                project_id=self.project_id,
                params={
                    "max_new_tokens": 500,
                    "temperature": 0.3,
                    "decoding_method": "sample"
                }
            )

            formatted_prompt = email_prompt.invoke({
                "partner_name": partner_name,
                "risk_level": risk_assessment.get("risk_level", "Unknown"),
                "top_step": top_step,
                "additional_steps": "; ".join(ranked_next_steps[1:3]) if len(ranked_next_steps) > 1 else "None",
                "maturity": partner_profile.get("maturity_level", "Partner")
            })

            result = llm.invoke(formatted_prompt)
            draft_email = result.content if hasattr(result, "content") else str(result)

            return {
                "crm_updates": crm_updates,
                "draft_email": draft_email,
                "messages": ["Artifacts created: CRM update and draft email"]
            }
        
        def generate_final_output_node(state: ActionState) -> dict:
            """Generate final comprehensive output"""
            recommended_action = state.get("recommended_action", {})
            risk_assessment = state.get("risk_assessment", {})
            crm_updates = state.get("crm_updates", {})
            draft_email = state.get("draft_email", "")
            historical_patterns = state.get("historical_patterns", {})
            
            output_parts = [
                "=" * 80,
                "ACTION AGENT - NEXT BEST STEP RECOMMENDATION",
                "=" * 80,
                "",
                "RECOMMENDED ACTION:",
                "-" * 80,
                recommended_action.get("raw_recommendation", "No action determined"),
                "",
                "RISK ASSESSMENT:",
                "-" * 80,
                f"Risk Level: {risk_assessment.get('risk_level', 'Unknown')}",
                f"Risk Score: {risk_assessment.get('risk_score', 0)}/100",
                "Risk Factors:",
            ]
            
            for factor in risk_assessment.get('risk_factors', []):
                output_parts.append(f"  - {factor}")
            
            output_parts.extend([
                "",
                "HISTORICAL CONTEXT:",
                "-" * 80,
                f"Pattern Confidence: {historical_patterns.get('pattern_confidence', 'Unknown')}",
                f"Based on {historical_patterns.get('total_won_deals', 0)} won deals",
                "",
                "CRM UPDATE:",
                "-" * 80,
                json.dumps(crm_updates, indent=2),
                "",
                "DRAFT FOLLOW-UP EMAIL:",
                "-" * 80,
                draft_email,
                "",
                "=" * 80,
                "Action recommendation complete. Ready for seller execution.",
                "=" * 80
            ])
            
            final_output = "\n".join(output_parts)
            
            return {
                "final_output": final_output,
                "messages": ["Final output generated"]
            }
        
        # Build the graph
        graph = StateGraph(ActionState)
        
        # Add nodes
        graph.add_node("analyze_patterns", analyze_patterns_node)
        graph.add_node("assess_risk", assess_risk_node)
        graph.add_node("determine_action", determine_action_node)
        graph.add_node("create_artifacts", create_artifacts_node)
        graph.add_node("generate_output", generate_final_output_node)
        
        # Add edges - linear workflow
        graph.add_edge(START, "analyze_patterns")
        graph.add_edge("analyze_patterns", "assess_risk")
        graph.add_edge("assess_risk", "determine_action")
        graph.add_edge("determine_action", "create_artifacts")
        graph.add_edge("create_artifacts", "generate_output")
        graph.add_edge("generate_output", END)
        
        self.graph = graph.compile()
        return self.graph
    
    def run(self, contract_summary: dict, partner_profile: dict, seller_query: str = "") -> dict:
        """
        Run the action agent to determine next best action.
        
        Args:
            contract_summary: Output from Contract Agent
            partner_profile: Output from Research Agent
            
        Returns:
            Final state with action recommendation and artifacts
        """
        if self.graph is None:
            self.build_agent()
        
        initial_state = {
            "contract_summary": contract_summary,
            "partner_profile": partner_profile,
            "seller_query": seller_query,
            "messages": []
        }
        
        result = self.graph.invoke(initial_state)
        return result


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Mock data for testing
    mock_contract_summary = {
        "structured_summary": {
            "contract_type": "ESA",
            "parties": ["Confluent", "IBM"],
            "effective_date": "2024-01-30",
            "risk_level": "Medium"
        }
    }
    
    mock_partner_profile = {
        "partner_name": "IBM",
        "maturity_level": "Strategic Partner",
        "sales_velocity": "High",
        "deal_blockers": []
    }
    
    agent = ActionAgent(
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )
    
    result = agent.run(mock_contract_summary, mock_partner_profile)
    
    print(result["final_output"])

# Made with Bob
