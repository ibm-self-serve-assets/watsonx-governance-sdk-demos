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
    matching_data: NotRequired[dict]  # NEW: From Matching Agent
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
        crm_file_path: str = "docs/Confluent Sales Cloud Infor.xlsx"
    ):
        """
        Initialize Action Agent.
        
        Args:
            model_id: Watsonx model ID for text generation
            url: Watsonx API URL
            apikey: IBM Cloud API key (defaults to WATSONX_APIKEY env var)
            project_id: Watsonx project ID (defaults to WATSONX_PROJECT_ID env var)
            crm_file_path: Path to CRM Excel file
        """
        self.model_id = model_id
        self.url = url
        # Use environment variables if not provided
        self.apikey = apikey or os.getenv("WATSONX_APIKEY")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.crm_file_path = crm_file_path
        self.graph = None

        # Validate credentials
        if not self.apikey:
            raise ValueError("WATSONX_APIKEY must be provided or set in environment variables")
        if not self.project_id:
            raise ValueError("WATSONX_PROJECT_ID must be provided or set in environment variables")
        
        # NOTE: ScenarioActions.pdf loading removed - was never used in the code
        # If scenario-based actions are needed in the future, implement them in the
        # determine_action_node with actual logic rather than just loading unused text
    
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
        
        # Calculate average deal size for won deals
        # Clean amount strings (remove $, commas, spaces) and convert to float
        avg_deal_size = 0
        if 'Amount' in won_deals.columns:
            try:
                # Clean the Amount column: remove $, commas, and extra spaces
                cleaned_amounts = won_deals['Amount'].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
                # Convert to numeric, coercing errors to NaN
                numeric_amounts = pd.to_numeric(cleaned_amounts, errors='coerce')
                # Calculate mean, ignoring NaN values
                avg_deal_size = numeric_amounts.mean() if not numeric_amounts.isna().all() else 0
            except Exception as e:
                print(f"Warning: Could not calculate average deal size: {e}")
                avg_deal_size = 0
        
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
    
    def _extract_first_email(self, text: str) -> str:
        """
        Extract only the first complete email from LLM output.
        Handles cases where LLM generates multiple versions, markdown fences, and preamble text.
        """
        # Remove markdown code fences if present
        text = text.replace("```", "").strip()
        
        # If text is short, return as-is
        if len(text) < 100:
            return text.strip()
        
        # Find the first "Subject:" line
        subject_idx = text.find("Subject:")
        if subject_idx == -1:
            # No subject found, return as-is
            return text.strip()
        
        # Find the signature "IBM Seller"
        signature_idx = text.find("IBM Seller", subject_idx)
        if signature_idx == -1:
            # No signature found, return everything from subject onwards
            return text[subject_idx:].strip()
        
        # Find the end of the signature line (look for newline after "IBM Seller")
        end_of_signature = signature_idx + len("IBM Seller")
        newline_after_sig = text.find("\n", end_of_signature)
        
        if newline_after_sig != -1:
            end_idx = newline_after_sig
        else:
            end_idx = end_of_signature
        
        # Extract the first email
        first_email = text[subject_idx:end_idx].strip()
        
        # Check if there's duplicate content after this (like "Here is the rewritten response:" or another "Subject:")
        remaining_text = text[end_idx:].strip()
        
        # Look for common preamble phrases that indicate duplicate/extra content
        duplicate_indicators = [
            "Here is the rewritten",
            "Here is the edited",
            "Subject:",
            "Dear ",
        ]
        
        # If we find any duplicate indicators in the remaining text within the first 200 chars,
        # we know we extracted correctly and should ignore the rest
        if remaining_text and len(remaining_text) > 10:
            for indicator in duplicate_indicators:
                if indicator in remaining_text[:200]:
                    # Duplicate detected, we're good with what we extracted
                    break
        
        return first_email
    
    def _extract_contract_value(self, contract):
        """Extract contract value from structured summary"""
        structured = contract.get("structured_summary", {})
        return structured.get("amount") if isinstance(structured, dict) else "Unknown"
    
    def _extract_products(self, contract):
        """Extract product names from contract structured data or filename"""
        # First try to get products from structured_summary (already extracted by contract agent)
        structured = contract.get("structured_summary", {})
        if isinstance(structured, dict):
            products = structured.get("products", [])
            if products and products != ["Unknown"]:
                return products
        
        # Fallback: parse from filename if structured data unavailable
        fn = contract.get("file_name", "").lower()
        products = []
        if "cognos" in fn:
            products.append("Cognos")
        if "watsonx" in fn:
            if "orchestrate" in fn:
                products.append("watsonx Orchestrate")
            if "governance" in fn:
                products.append("watsonx.governance")
            if "data" in fn:
                products.append("watsonx.data")
            if "ai" in fn or not products:
                products.append("watsonx.ai" if "ai" in fn else "watsonx ESA")
        
        # Return products or indicate they need to be extracted
        return products if products else ["Products not yet extracted - run cache_contracts.py"]
    
    def _find_matching_data_for_contract(self, contract, matched_contracts_list):
        """Find the matching data for a specific contract from Matching Agent output"""
        contract_filename = contract.get("file_name", "")
        for match in matched_contracts_list:
            match_contract = match.get("contract", {})
            if match_contract.get("file_name") == contract_filename:
                # Return the first opportunity from the match
                opportunities = match.get("opportunities", [])
                return opportunities[0] if opportunities else None
        return None
    
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
            """Determine recommended next best action with detailed reasoning"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            matching_data = state.get("matching_data", {})  # NEW: Get from Matching Agent
            seller_query = state.get("seller_query", "")
            lowered_query = seller_query.lower()

            portfolio = contract_summary.get("portfolio_summary", {})
            active_contracts = portfolio.get("active_contracts", [])
            renewal_candidates = portfolio.get("renewal_candidates", [])
            recently_expired = portfolio.get("recently_expired_contracts", [])
            portfolio_steps = portfolio.get("recommended_next_steps", [])
            
            # Get matched contracts from Matching Agent (replaces manual matching)
            matched_contracts = matching_data.get("matched_contracts", [])
            unmatched_contracts = matching_data.get("unmatched_contracts", [])
            
            # Utility functions for contract analysis
            def calculate_urgency(contract):
                """Calculate urgency from contract end date"""
                now, end_date_str = datetime.now().date(), contract.get("end_date", "Unknown")
                if end_date_str == "Unknown":
                    return None
                try:
                    days_diff = (datetime.fromisoformat(end_date_str).date() - now).days
                    if days_diff < 0:
                        return {"status": "EXPIRED", "days_expired": abs(days_diff), "days_until_expiration": None, "urgency_level": "CRITICAL"}
                    elif days_diff <= 30:
                        return {"status": "EXPIRING_SOON", "days_expired": None, "days_until_expiration": days_diff, "urgency_level": "HIGH"}
                    elif days_diff <= 90:
                        return {"status": "RENEWAL_WINDOW", "days_expired": None, "days_until_expiration": days_diff, "urgency_level": "MEDIUM"}
                    return {"status": "ACTIVE", "days_expired": None, "days_until_expiration": days_diff, "urgency_level": "LOW"}
                except:
                    return None
            
            
            def extract_executive_info(partner_profile, role):
                """
                Extract executive name from research data.

                HARD-CODED FALLBACKS for Confluent (pre-IBM acquisition):
                  - CPO : Shaun Clowes  (Chief Product Officer)
                  - CTO : Stephen Deasy (Chief Technology Officer)
                  - CFO : Rohan Sivaram (Chief Financial Officer)
                  - CEO : Jay Kreps (Chief Executive Officer / Co-Founder)

                These are used when Tavily cannot reliably surface the data because
                Confluent is now an IBM entity and public search results surface IBM
                executives instead.  If Tavily does successfully return a name for the
                requested role that name will take precedence.
                """
                import re

                # ----------------------------------------------------------------
                # HARD-CODED: Confluent pre-acquisition executive roster
                # Used as the authoritative fallback when Tavily search fails.
                # ----------------------------------------------------------------
                CONFLUENT_EXECUTIVES = {
                    "CPO": {
                        "name": "Shaun Clowes",
                        "title": "Chief Product Officer (CPO)",
                        "role": "CPO",
                        "note": "Pre-acquisition Confluent executive (prior to IBM acquisition)"
                    },
                    "CTO": {
                        "name": "Stephen Deasy",
                        "title": "Chief Technology Officer (CTO)",
                        "role": "CTO",
                        "note": "Pre-acquisition Confluent executive (prior to IBM acquisition)"
                    },
                    "CFO": {
                        "name": "Rohan Sivaram",
                        "title": "Chief Financial Officer (CFO)",
                        "role": "CFO",
                        "note": "Pre-acquisition Confluent executive (prior to IBM acquisition)"
                    },
                    "CEO": {
                        "name": "Jay Kreps",
                        "title": "Chief Executive Officer (CEO) & Co-Founder",
                        "role": "CEO",
                        "note": "Pre-acquisition Confluent executive / Co-Founder (prior to IBM acquisition)"
                    },
                }

                partner_name_lower = partner_profile.get("partner_name", "").lower()
                is_confluent = "confluent" in partner_name_lower

                # ------------------------------------------------------------------
                # Step 1: Try to extract from Tavily / research-agent external data.
                # ------------------------------------------------------------------
                exec_data = partner_profile.get("external_data", {}).get("executives", "")
                if not isinstance(exec_data, str):
                    exec_data = str(exec_data) if exec_data else ""

                patterns = {
                    "CPO": [
                        r'CPO[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        r'Chief\s+Product\s+Officer[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        r'Chief\s+Procurement\s+Officer[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        r'([A-Z][a-z]+\s+[A-Z][a-z]+)[,\s]+CPO',
                    ],
                    "CTO": [
                        r'CTO[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        r'Chief\s+Technology\s+Officer[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        r'([A-Z][a-z]+\s+[A-Z][a-z]+)[,\s]+CTO',
                    ],
                    "CFO": [
                        r'CFO[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        r'Chief\s+Financial\s+Officer[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        r'([A-Z][a-z]+\s+[A-Z][a-z]+)[,\s]+CFO',
                    ],
                    "CEO": [
                        r'CEO[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        r'Chief\s+Executive\s+Officer[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
                        r'([A-Z][a-z]+\s+[A-Z][a-z]+)[,\s]+CEO',
                    ],
                }

                # Guard: if Confluent, skip any Tavily result that contains IBM-era
                # executive names to avoid surfacing post-acquisition IBM leadership.
                IBM_EXEC_NAMES = {"arvind krishna", "jim kavanaugh", "gary cohn"}
                
                # Filter out common title words that might be mistaken for names
                TITLE_WORDS = {"chief", "financial", "technology", "product", "executive",
                               "officer", "procurement", "operating", "information", "marketing",
                               "sales", "human", "resources", "legal", "compliance"}

                for pattern in patterns.get(role, []):
                    match = re.search(pattern, exec_data)
                    if match:
                        found_name = match.group(1)
                        # Check if this is actually a title, not a name
                        name_words = found_name.lower().split()
                        if any(word in TITLE_WORDS for word in name_words):
                            # This is a title (e.g., "Chief Financial"), not a name - skip it
                            continue
                        if is_confluent and found_name.lower() in IBM_EXEC_NAMES:
                            # Skip IBM executive — fall through to hard-coded data
                            continue
                        return {"name": found_name, "title": role, "role": role}

                # ------------------------------------------------------------------
                # Step 2: Tavily did not return a usable result — use hard-coded data.
                # ------------------------------------------------------------------
                if is_confluent and role in CONFLUENT_EXECUTIVES:
                    print(f"INFO: Using hard-coded Confluent executive for role {role} "
                          f"(Tavily search did not return reliable pre-acquisition data).")
                    return CONFLUENT_EXECUTIVES[role]

                # ------------------------------------------------------------------
                # Step 3: No data available — return role-only placeholder.
                # ------------------------------------------------------------------
                return {"name": None, "title": role, "role": role}
            
            
            def extract_contract_signers(contract):
                """Extract previous contract signers and key people from contract"""
                structured = contract.get("structured_summary", {})
                parties = structured.get("parties", [])
                
                # Extract names from parties list
                signers = []
                for party in parties:
                    if isinstance(party, str):
                        # Look for person names (capitalized words)
                        import re
                        names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', party)
                        signers.extend(names)
                
                return list(set(signers)) if signers else ["Unknown"]
            
            def detect_expansion_opportunity(contract, crm_match):
                """Detect if there's an upsell/expansion opportunity"""
                expansion_signals = []
                can_expand = False
                
                # Check contract value vs CRM opportunity value
                contract_value = self._extract_contract_value(contract)
                if crm_match and isinstance(contract_value, (int, float)):
                    crm_amount = crm_match.get("amount", 0)
                    # Format amounts safely for display
                    if isinstance(crm_amount, (int, float)) and crm_amount > contract_value * 1.2:
                        crm_amt_str = f"${crm_amount:,.0f}" if isinstance(crm_amount, (int, float)) else str(crm_amount)
                        contract_val_str = f"${contract_value:,.0f}" if isinstance(contract_value, (int, float)) else str(contract_value)
                        expansion_signals.append(f"CRM opportunity ({crm_amt_str}) is 20%+ higher than contract value ({contract_val_str})")
                        can_expand = True
                
                # Check for expansion keywords in CRM next steps
                if crm_match:
                    next_steps = str(crm_match.get("next_steps", "")).lower()
                    if "expand" in next_steps or "expansion" in next_steps:
                        expansion_signals.append("CRM next steps mention expansion")
                        can_expand = True
                    if "upsell" in next_steps or "additional" in next_steps:
                        expansion_signals.append("CRM next steps mention upsell/additional products")
                        can_expand = True
                    if "sizing" in next_steps:
                        expansion_signals.append("Customer actively sizing - potential for expansion")
                        can_expand = True
                
                # Check product mix for expansion potential
                products = self._extract_products(contract)
                if len(products) == 1 and products[0] in ["Cognos", "watsonx ESA"]:
                    expansion_signals.append(f"Single product ({products[0]}) - opportunity to introduce complementary solutions")
                    can_expand = True
                
                return {
                    "can_expand": can_expand,
                    "signals": expansion_signals,
                    "recommendation": "Explore expansion opportunities during renewal discussion" if can_expand else "Focus on renewal at current scope"
                }
            
            def analyze_why_expired(contract, crm_match):
                """Analyze why a contract expired without renewal"""
                reasons = []
                
                if not crm_match:
                    reasons.append("❌ No CRM opportunity created - likely no one reached out")
                    reasons.append("⚠️ Lack of proactive engagement from sales team")
                    return {
                        "primary_reason": "No proactive outreach - no CRM tracking",
                        "details": reasons,
                        "action": "URGENT: Create CRM opportunity and reach out immediately"
                    }
                
                # If there is a CRM match, analyze the stage
                stage = crm_match.get("stage", "Unknown")
                next_steps = str(crm_match.get("next_steps", "")).lower()
                
                if stage == "Lost":
                    reasons.append(f"❌ Deal marked as Lost in CRM")
                    reasons.append(f"📝 Reason: {crm_match.get('next_steps', 'No reason provided')}")
                    return {
                        "primary_reason": "Deal was lost",
                        "details": reasons,
                        "action": "Review loss reason and develop win-back strategy"
                    }
                
                if stage == "Won":
                    reasons.append("✅ Deal was won but contract may have expired naturally")
                    return {
                        "primary_reason": "Natural expiration after successful engagement",
                        "details": reasons,
                        "action": "Reach out for renewal discussion"
                    }
                
                # Active engagement but still expired
                if "sizing" in next_steps or "working" in next_steps:
                    reasons.append("🔄 Active engagement but contract expired during negotiations")
                    reasons.append("⏰ Timing issue - discussions ongoing but contract lapsed")
                    return {
                        "primary_reason": "Timing gap - active discussions but contract expired",
                        "details": reasons,
                        "action": "Accelerate renewal process - customer is engaged"
                    }
                
                reasons.append("⚠️ Unknown reason - requires investigation")
                return {
                    "primary_reason": "Requires investigation",
                    "details": reasons,
                    "action": "Contact account owner for status update"
                }
            
            def determine_recipient(contract, crm_match, query, partner_profile):
                """
                Determine email recipient based on seller query, CRM next steps,
                product type, and contract urgency.

                Priority order:
                  1. Explicit role mention in seller query (cpo / cto / cfo / ceo)
                  2. Role mentioned in CRM next steps
                  3. Product-based heuristic (AI/tech products → CTO)
                  4. Renewal / procurement focus → CPO
                  5. Critical urgency → CEO
                  6. Default → CPO
                """
                products = self._extract_products(contract)
                query_lower = query.lower()
                crm_next_steps = str(crm_match.get("next_steps", "") if crm_match else "").lower()

                # 1. Explicit mention in seller query
                if "cfo" in query_lower:
                    role = "CFO"
                elif "cpo" in query_lower:
                    role = "CPO"
                elif "cto" in query_lower:
                    role = "CTO"
                elif "ceo" in query_lower:
                    role = "CEO"
                # 2. Role mentioned in CRM next steps
                elif "cfo" in crm_next_steps:
                    role = "CFO"
                elif "cto" in crm_next_steps or "chief technology" in crm_next_steps:
                    role = "CTO"
                elif "cpo" in crm_next_steps or "chief product" in crm_next_steps or "chief procurement" in crm_next_steps:
                    role = "CPO"
                elif "ceo" in crm_next_steps or "chief executive" in crm_next_steps:
                    role = "CEO"
                # 3. Product heuristic
                elif any(p in ["watsonx.ai", "watsonx.governance", "watsonx Orchestrate"] for p in products):
                    role = "CTO"
                # 4. Renewal / procurement focus
                elif crm_match and "renew" in crm_next_steps:
                    role = "CPO"
                # 5. Critical urgency
                elif (urgency := calculate_urgency(contract)) and urgency["urgency_level"] == "CRITICAL":
                    role = "CEO"
                # 6. Default
                else:
                    role = "CPO"

                return extract_executive_info(partner_profile, role)
            
            # Create detailed reasoning for each contract
            detailed_actions = []
            
            # Detect renewal chains - group contracts by product to find renewal series
            def detect_renewal_chains(all_contracts):
                """
                Detect renewal chains and return only the most recent contract in each series.
                Contracts with same product and sequential dates are considered a renewal chain.
                """
                from collections import defaultdict
                
                # Group contracts by product
                product_groups = defaultdict(list)
                for contract in all_contracts:
                    products = self._extract_products(contract)
                    product_key = tuple(sorted(products))  # Use sorted tuple as key
                    product_groups[product_key].append(contract)
                
                # For each product group, keep only the most recent contract
                filtered_contracts = []
                for product_key, contracts in product_groups.items():
                    if len(contracts) == 1:
                        filtered_contracts.append(contracts[0])
                    else:
                        # Sort by end date (most recent first)
                        sorted_contracts = sorted(
                            contracts,
                            key=lambda c: c.get('end_date', '1900-01-01'),
                            reverse=True
                        )
                        # Keep only the most recent
                        most_recent = sorted_contracts[0]
                        filtered_contracts.append(most_recent)
                        
                        # Log the renewal chain detection
                        print(f"\n🔄 Renewal chain detected for {', '.join(product_key)}:")
                        for i, c in enumerate(sorted_contracts):
                            marker = "→ MOST RECENT" if i == 0 else "  (older, filtered out)"
                            print(f"   {c.get('file_name', 'Unknown')} - End: {c.get('end_date', 'Unknown')} {marker}")
                
                return filtered_contracts
            
            # Filter expired contracts to remove old renewals
            all_expired = recently_expired.copy()
            filtered_expired = detect_renewal_chains(all_expired)
            
            print(f"\n📊 Contract filtering: {len(recently_expired)} expired → {len(filtered_expired)} after removing old renewals")
            
            # Process filtered expired contracts first (highest priority)
            for contract in filtered_expired:
                urgency = calculate_urgency(contract)
                crm_match = self._find_matching_data_for_contract(contract, matched_contracts)
                products = self._extract_products(contract)
                value = self._extract_contract_value(contract)
                recipient_info = determine_recipient(contract, crm_match, lowered_query, partner_profile)
                signers = extract_contract_signers(contract)
                expansion = detect_expansion_opportunity(contract, crm_match)
                expiration_analysis = analyze_why_expired(contract, crm_match)
                
                reasoning_parts = [
                    f"CONTRACT: {contract.get('file_name', 'Unknown')}",
                    f"- Product(s): {', '.join(products)}",
                    f"- Value: {value}",
                    f"- Previous Signers: {', '.join(signers)}",
                ]
                
                if urgency:
                    reasoning_parts.append(f"- Status: {urgency['status']} ({urgency['days_expired']} days ago)")
                    reasoning_parts.append(f"- End Date: {contract.get('end_date', 'Unknown')}")
                
                reasoning_parts.append("")
                reasoning_parts.append("CRM LINKAGE:")
                
                if crm_match:
                    stage = crm_match.get('stage', 'Unknown')
                    stage_emoji = "✅" if stage == "Won" else "❌" if stage == "Lost" else "🔄"
                    stage_status = "CONTRACT SIGNED" if stage == "Won" else "FAILED" if stage == "Lost" else "ACTIVE ENGAGEMENT"
                    
                    reasoning_parts.extend([
                        f"- {stage_emoji} CRM Status: {stage_status}",
                        f"- Opportunity: \"{crm_match.get('opportunity_name', 'Unknown')}\"",
                        f"- Owner: {crm_match.get('owner', 'Unknown')}",
                        f"- Stage: {stage}",
                        f"- Amount: {crm_match.get('amount', '$0')}",
                        f"- Close Date: {crm_match.get('close_date', 'Unknown')}",
                        f"- Next Steps: \"{crm_match.get('next_steps', 'None')}\"",
                    ])
                else:
                    reasoning_parts.append("- ⚠️ WARNING: No matching CRM opportunity found!")
                    reasoning_parts.append("- 🚨 ACTION REQUIRED: Create CRM opportunity to track renewal")
                    reasoning_parts.append("- 📝 This means NO ONE is actively working on this renewal")
                
                reasoning_parts.append("")
                reasoning_parts.append("WHY DID THIS CONTRACT EXPIRE?")
                reasoning_parts.append(f"- Primary Reason: {expiration_analysis['primary_reason']}")
                for detail in expiration_analysis['details']:
                    reasoning_parts.append(f"  {detail}")
                
                reasoning_parts.append("")
                reasoning_parts.append("EXPANSION OPPORTUNITY ANALYSIS:")
                reasoning_parts.append(f"- Can Expand: {'YES ✅' if expansion['can_expand'] else 'NO'}")
                if expansion['signals']:
                    reasoning_parts.append("- Signals:")
                    for signal in expansion['signals']:
                        reasoning_parts.append(f"  • {signal}")
                reasoning_parts.append(f"- Recommendation: {expansion['recommendation']}")
                
                reasoning_parts.append("")
                reasoning_parts.append("URGENCY ANALYSIS:")
                
                if urgency and crm_match:
                    urgency_reasons = []
                    urgency_reasons.append(f"Contract expired {urgency['days_expired']} days ago")
                    urgency_reasons.append(f"${value} renewal opportunity at risk")
                    
                    next_steps = str(crm_match.get('next_steps', '')).lower()
                    if "sizing" in next_steps:
                        urgency_reasons.append("Customer actively sizing - positive engagement signal")
                    elif "renew" in next_steps:
                        urgency_reasons.append("Renewal discussions in progress")
                    elif crm_match.get('stage') == 'Qualified':
                        urgency_reasons.append("Opportunity qualified but needs immediate action")
                    
                    urgency_reasons.append("Competitor could enter during gap period")
                    
                    reasoning_parts.extend([f"- {reason}" for reason in urgency_reasons])
                elif urgency:
                    reasoning_parts.append(f"- Contract expired {urgency['days_expired']} days ago with no CRM tracking")
                    reasoning_parts.append(f"- ${value} at risk with no visibility into renewal status")
                    reasoning_parts.append(f"- 🚨 CRITICAL: {expiration_analysis['action']}")
                
                reasoning_parts.append("")
                reasoning_parts.append("KEY PEOPLE TO CONTACT:")
                
                # Format recipient display - always show name if available, with role in parentheses
                if recipient_info["name"]:
                    recipient_display = f"{recipient_info['name']} ({recipient_info['role']})"
                else:
                    recipient_display = recipient_info['role']
                
                if recipient_info.get("note"):
                    reasoning_parts.append(f"- Primary: {recipient_display} - {recipient_info['note']}")
                else:
                    reasoning_parts.append(f"- Primary: {recipient_display}")
                
                if signers and signers != ["Unknown"]:
                    reasoning_parts.append(f"- Previous Signers: {', '.join(signers)} (may still be involved)")
                
                if crm_match:
                    reasoning_parts.append(f"- CRM Owner: {crm_match.get('owner', 'Unknown')} (currently managing this opportunity)")
                
                reasoning_parts.append("")
                reasoning_parts.append("RECOMMENDED ACTIONS:")
                
                if crm_match:
                    owner = crm_match.get('owner', 'the account owner')
                    stage = crm_match.get('stage', 'Unknown')
                    
                    if stage == "Lost":
                        reasoning_parts.extend([
                            f"1. Contact {owner} to understand why deal was lost",
                            f"2. Review loss reason: \"{crm_match.get('next_steps', 'None')}\"",
                            f"3. Develop win-back strategy with management",
                            f"4. Reach out to {recipient_display} with new value proposition",
                        ])
                    elif stage == "Won":
                        reasoning_parts.extend([
                            f"1. Contact {owner} to confirm renewal status",
                            f"2. Verify if new contract has been signed",
                            f"3. Update CRM with current contract status",
                            f"4. Schedule follow-up meeting with {recipient_display}",
                        ])
                    else:  # Active engagement
                        reasoning_parts.extend([
                            f"1. Contact {owner} TODAY to get status update",
                            f"2. Review CRM notes: \"{crm_match.get('next_steps', 'None')}\"",
                            f"3. Draft executive email to {recipient_display} addressing renewal urgency",
                            f"4. {'Discuss expansion opportunities' if expansion['can_expand'] else 'Focus on renewal at current scope'}",
                            f"5. Target resolution within 14 days",
                        ])
                else:
                    reasoning_parts.extend([
                        "1. 🚨 Create CRM opportunity immediately to track this expired contract",
                        "2. Research customer contact and current relationship status",
                        f"3. Reach out to previous signers: {', '.join(signers)}",
                        f"4. Draft outreach email to {recipient_display} to re-engage",
                        "5. Escalate to management if no response within 7 days",
                    ])
                
                reasoning_parts.append("")
                reasoning_parts.append("WHY THIS MATTERS:")
                reasoning_parts.extend([
                    f"- ${value} revenue at risk",
                    f"- Customer relationship continuity depends on quick action",
                    f"- Extended gap increases competitive vulnerability",
                    f"- May impact future expansion opportunities with this customer",
                ])
                
                detailed_actions.append({
                    "contract": contract.get("file_name", "Unknown"),
                    "priority": 100 if urgency and urgency["urgency_level"] == "CRITICAL" else 80,
                    "reasoning": "\n".join(reasoning_parts),
                    "specific_action": reasoning_parts[reasoning_parts.index("RECOMMENDED ACTIONS:") + 1] if "RECOMMENDED ACTIONS:" in reasoning_parts else "Review expired contract",
                    "recipient_info": recipient_info,
                    "crm_owner": crm_match.get('owner') if crm_match else None,
                    "urgency_level": urgency["urgency_level"] if urgency else "HIGH"
                })
            
            # Process renewal candidates
            for contract in renewal_candidates:
                urgency = calculate_urgency(contract)
                crm_match = self._find_matching_data_for_contract(contract, matched_contracts)
                products = self._extract_products(contract)
                value = self._extract_contract_value(contract)
                recipient_info = determine_recipient(contract, crm_match, lowered_query, partner_profile)
                
                reasoning_parts = [
                    f"CONTRACT: {contract.get('file_name', 'Unknown')}",
                    f"- Product(s): {', '.join(products)}",
                    f"- Value: {value}",
                ]
                
                if urgency:
                    reasoning_parts.append(f"- Status: {urgency['status']}")
                    reasoning_parts.append(f"- Days Until Expiration: {urgency['days_until_expiration']}")
                    reasoning_parts.append(f"- End Date: {contract.get('end_date', 'Unknown')}")
                
                reasoning_parts.append("")
                reasoning_parts.append("CRM LINKAGE:")
                
                if crm_match:
                    reasoning_parts.extend([
                        f"- Opportunity: \"{crm_match.get('opportunity_name', 'Unknown')}\"",
                        f"- Owner: {crm_match.get('owner', 'Unknown')}",
                        f"- Stage: {crm_match.get('stage', 'Unknown')}",
                        f"- Amount: {crm_match.get('amount', '$0')}",
                        f"- Next Steps: \"{crm_match.get('next_steps', 'None')}\"",
                    ])
                else:
                    reasoning_parts.append("- WARNING: No matching CRM opportunity found for upcoming renewal")
                
                reasoning_parts.append("")
                reasoning_parts.append("URGENCY ANALYSIS:")
                
                if urgency:
                    if urgency["days_until_expiration"] <= 30:
                        reasoning_parts.append(f"- URGENT: Only {urgency['days_until_expiration']} days until expiration")
                    else:
                        reasoning_parts.append(f"- {urgency['days_until_expiration']} days until expiration - renewal window open")
                    
                    if crm_match:
                        reasoning_parts.append(f"- CRM shows {crm_match.get('stage')} stage - {crm_match.get('next_steps', 'no next steps defined')}")
                
                reasoning_parts.append("")
                reasoning_parts.append("RECOMMENDED ACTIONS:")
                
                # Format recipient display
                recipient_display = recipient_info["name"] if recipient_info["name"] else recipient_info["role"]
                
                if crm_match:
                    owner = crm_match.get('owner', 'the account owner')
                    reasoning_parts.extend([
                        f"1. Coordinate with {owner} on renewal timeline",
                        f"2. Prepare renewal proposal with updated terms",
                        f"3. Schedule meeting with customer {recipient_display} ({recipient_info['role']}) within 2 weeks",
                        f"4. Align legal and procurement teams",
                    ])
                else:
                    reasoning_parts.extend([
                        "1. Create CRM renewal opportunity immediately",
                        "2. Identify customer stakeholders and decision makers",
                        f"3. Initiate renewal discussion with {recipient_display} ({recipient_info['role']})",
                        "4. Set up internal renewal planning meeting",
                    ])
                
                detailed_actions.append({
                    "contract": contract.get("file_name", "Unknown"),
                    "priority": 70 if urgency and urgency["urgency_level"] == "HIGH" else 50,
                    "reasoning": "\n".join(reasoning_parts),
                    "specific_action": reasoning_parts[reasoning_parts.index("RECOMMENDED ACTIONS:") + 1] if "RECOMMENDED ACTIONS:" in reasoning_parts else "Start renewal process",
                    "recipient_info": recipient_info,
                    "crm_owner": crm_match.get('owner') if crm_match else None,
                    "urgency_level": urgency["urgency_level"] if urgency else "MEDIUM"
                })
            
            # Sort by priority
            detailed_actions.sort(key=lambda x: x["priority"], reverse=True)
            
            # Determine workflow name
            workflow_name = "portfolio_overview_30_day_actions"
            if "renewal" in lowered_query or "expired" in lowered_query or "expir" in lowered_query:
                workflow_name = "renewal_expiration_awareness"
            elif "draft me an email" in lowered_query or "reach out" in lowered_query:
                workflow_name = "executive_outreach"
            
            # Create comprehensive recommendation
            if detailed_actions:
                top_action = detailed_actions[0]
                
                action_lines = [
                    "=" * 80,
                    "DETAILED CONTRACT ANALYSIS WITH REASONING",
                    "=" * 80,
                    "",
                    f"PRIORITY 1: {top_action['urgency_level']} URGENCY - IMMEDIATE ACTION REQUIRED",
                    "",
                    top_action["reasoning"],
                    "",
                    "=" * 80,
                ]
                
                # Add additional high-priority items
                if len(detailed_actions) > 1:
                    action_lines.extend([
                        "",
                        "ADDITIONAL HIGH-PRIORITY ITEMS:",
                        "-" * 80,
                    ])
                    for i, action in enumerate(detailed_actions[1:3], 2):
                        action_lines.extend([
                            "",
                            f"PRIORITY {i}: {action['contract']}",
                            f"Urgency: {action['urgency_level']}",
                            f"Action: {action['specific_action']}",
                            f"Recipient: {action['recipient_info']['name'] if action['recipient_info']['name'] else action['recipient_info']['role']} ({action['recipient_info']['role']})",
                        ])
                        if action.get('crm_owner'):
                            action_lines.append(f"CRM Owner: {action['crm_owner']}")
                
                recommended_action = {
                    "workflow_name": workflow_name,
                    "raw_recommendation": "\n".join(action_lines),
                    "ranked_next_steps": [action["specific_action"] for action in detailed_actions],
                    "detailed_actions": detailed_actions,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Fallback if no detailed actions
                recommended_action = {
                    "workflow_name": workflow_name,
                    "raw_recommendation": "No urgent contract actions identified. Review portfolio for upcoming renewals.",
                    "ranked_next_steps": ["Review contract portfolio", "Update CRM with current status"],
                    "detailed_actions": [],
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "recommended_action": recommended_action,
                "messages": ["Next best action determined with detailed reasoning"]
            }
        
        def create_artifacts_node(state: ActionState) -> dict:
            """Create actionable artifacts (CRM updates, draft email) with detailed context"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            recommended_action = state.get("recommended_action", {})
            risk_assessment = state.get("risk_assessment", {})
            seller_query = state.get("seller_query", "")
            matching_data = state.get("matching_data", {})

            partner_name = partner_profile.get("partner_name", "Partner")
            portfolio = contract_summary.get("portfolio_summary", {})
            ranked_next_steps = recommended_action.get("ranked_next_steps", [])
            matched_contracts = matching_data.get("matched_contracts", [])
            detailed_actions = recommended_action.get("detailed_actions", [])
            top_step = ranked_next_steps[0] if ranked_next_steps else "Review contract portfolio and schedule follow-up."
            
            # Get potential products from contract summary
            potential_products = []
            if portfolio.get("active_contracts"):
                # Extract products from active contracts
                for contract in portfolio.get("active_contracts", [])[:3]:
                    filename = contract.get("file_name", "")
                    if filename:
                        potential_products.append(filename.replace("_", " ").replace(".docx", ""))
            
            # Also check partner profile for products used
            if partner_profile.get("internal_data", {}).get("sales_history", {}).get("products_used"):
                potential_products.extend(partner_profile["internal_data"]["sales_history"]["products_used"][:3])
            
            # Remove duplicates and limit
            potential_products = list(dict.fromkeys(potential_products))[:5]
            products_str = ", ".join(potential_products) if potential_products else "Portfolio Review"

            crm_updates = {
                "opportunity_name": f"{partner_name} - Seller Inquiry {datetime.now().strftime('%Y-%m-%d')}",
                "stage": "Seller Inquiry",
                "next_step": top_step,  # Single top priority action
                "owner": "IBM Seller",
                "due_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                "priority": risk_assessment.get("risk_level", "Medium"),
                "contracts_reviewed": len(portfolio.get("contract_paths", [])),
                # Removed duplicate "next_steps" list - keeping only "next_step" (singular)
                "seller_query": seller_query,
                "potential_products": products_str,
                "updated_timestamp": datetime.now().isoformat()
            }

            # CRM UPDATE DISABLED - Uncomment below to re-enable
            # try:
            #     df = self._load_crm_data()
            #     if not df.empty:
            #         # Create full next steps text with all recommendations
            #         agent_next_steps_text = " | ".join(ranked_next_steps[:5])
            #
            #         # ALWAYS create a new row for each seller inquiry
            #         new_row = {col: "" for col in df.columns}
            #
            #         # Populate the new row with seller inquiry data
            #         if "Opportunity Name" in df.columns:
            #             new_row["Opportunity Name"] = crm_updates["opportunity_name"]
            #         if "Owner Full Name" in df.columns:
            #             new_row["Owner Full Name"] = crm_updates["owner"]
            #         if "Stage" in df.columns:
            #             new_row["Stage"] = crm_updates["stage"]
            #         if "Amount" in df.columns:
            #             new_row["Amount"] = ""  # Leave blank for seller inquiry
            #         if "Close Date" in df.columns:
            #             new_row["Close Date"] = crm_updates["due_date"]
            #         if "Next Steps" in df.columns:
            #             # Put full agent recommendations in Next Steps column
            #             new_row["Next Steps"] = agent_next_steps_text
            #         if "Products" in df.columns:
            #             new_row["Products"] = products_str
            #
            #         # Add the new row to the dataframe
            #         df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            #
            #         # Save the updated dataframe
            #         df.to_excel(self.crm_file_path, sheet_name="Sheet1", index=False)
            #         crm_updates["crm_file_updated"] = True
            #         crm_updates["new_row_added"] = True
            #     else:
            #         crm_updates["crm_file_updated"] = False
            # except Exception as e:
            #     crm_updates["crm_file_updated"] = False
            #     crm_updates["crm_update_error"] = str(e)
            
            # Mark as disabled
            crm_updates["crm_file_updated"] = False
            crm_updates["crm_update_disabled"] = True

            # ----------------------------------------------------------------
            # EMAIL GENERATION
            # Builds a rich, CRM-next-steps-aware context block so the LLM
            # can write a single, clean, targeted email.
            # ----------------------------------------------------------------
            recipient_info = {"role": "CPO", "name": None, "title": "CPO"}

            # ---- Pull structured data for the top-priority action ----------
            contract_products_str = "IBM software portfolio"
            contract_value_str    = "significant value"
            contract_end_date_str = "recently"
            contract_status_str   = "requires attention"
            crm_next_steps_str    = ""
            crm_owner_str         = ""
            crm_stage_str         = ""
            crm_opp_name_str      = ""
            has_crm_match         = False

            if detailed_actions:
                top_action   = detailed_actions[0]
                recipient_info = top_action.get("recipient_info", {"role": "CPO", "name": None, "title": "CPO"})
                crm_owner_str  = top_action.get("crm_owner") or ""

                # Locate the source contract object for rich metadata
                all_portfolio_contracts = (
                    contract_summary.get("portfolio_summary", {}).get("recently_expired_contracts", []) +
                    contract_summary.get("portfolio_summary", {}).get("renewal_candidates", []) +
                    contract_summary.get("portfolio_summary", {}).get("active_contracts", [])
                )
                source_contract = next(
                    (c for c in all_portfolio_contracts
                     if c.get("file_name") == top_action.get("contract")),
                    {}
                )

                if source_contract:
                    prods = self._extract_products(source_contract)
                    contract_products_str = ", ".join(prods) if prods else contract_products_str
                    val = self._extract_contract_value(source_contract)
                    if val and val not in ("Unknown", None):
                        # Format as currency if numeric, else use as-is
                        if isinstance(val, (int, float)):
                            contract_value_str = f"${val:,.0f}"
                        else:
                            contract_value_str = str(val)
                    end_date = source_contract.get("end_date", "")
                    if end_date and end_date != "Unknown":
                        contract_end_date_str = end_date
                    status = source_contract.get("status", "")
                    if status:
                        contract_status_str = status.upper()

                # Pull CRM match details
                crm_match = self._find_matching_data_for_contract(
                    source_contract or {"file_name": top_action.get("contract", "")},
                    matched_contracts
                )
                if crm_match:
                    has_crm_match     = True
                    crm_next_steps_str = crm_match.get("next_steps", "") or ""
                    crm_owner_str      = crm_match.get("owner", crm_owner_str) or crm_owner_str
                    crm_stage_str      = crm_match.get("stage", "") or ""
                    crm_opp_name_str   = crm_match.get("opportunity_name", "") or ""

            recipient_name_str = recipient_info["name"] if recipient_info["name"] else None
            recipient_role_str = recipient_info["role"]
            
            # Extract first name only for greeting
            if recipient_name_str:
                first_name = recipient_name_str.split()[0]  # Get first word (first name)
                greeting = first_name
            else:
                greeting = recipient_role_str
            
            recipient_display  = recipient_name_str if recipient_name_str else recipient_role_str

            # ---- Build a tightly-scoped CRM next-steps guidance block ------
            # If CRM next steps exist, the email should directly address them.
            crm_guidance_block = ""
            if crm_next_steps_str.strip():
                crm_guidance_block = (
                    f"CRM Next Steps on file: \"{crm_next_steps_str}\"\n"
                    f"CRM Opportunity: {crm_opp_name_str}\n"
                    f"CRM Stage: {crm_stage_str}\n"
                    f"Account Owner in CRM: {crm_owner_str}\n\n"
                    "The email MUST directly address what the CRM next steps say. "
                    "If the next steps mention sizing, address that. If they mention a specific meeting or decision, reference it. "
                    "If there is an account owner noted above, mention coordinating with them."
                )
            else:
                crm_guidance_block = (
                    "No CRM next steps are on file. The email should re-engage the contact from scratch, "
                    "noting the gap since the contract ended and proposing a specific meeting to restart discussions."
                )

            email_prompt = ChatPromptTemplate.from_template(
                "You are a senior IBM seller writing ONE professional outreach email.\n\n"
                "════════════════════════════════════════════\n"
                "RECIPIENT\n"
                "  Name : {recipient_name}\n"
                "  Role : {recipient_role}\n"
                "  Company: {partner_name}\n\n"
                "CONTRACT FACTS\n"
                "  Product(s)   : {products}\n"
                "  Contract Value: {value}\n"
                "  End / Expiry : {end_date}\n"
                "  Status       : {status}\n\n"
                "CRM INTELLIGENCE\n"
                "{crm_guidance}\n\n"
                "OVERALL RISK: {risk_level}\n"
                "TOP PRIORITY ACTION: {top_step}\n"
                "════════════════════════════════════════════\n\n"
                "WRITING RULES — follow every one:\n"
                "1. Output EXACTLY ONE email. No alternatives, no commentary, no markdown fences.\n"
                "2. Open with 'Subject:' on the first line, blank line, then 'Dear {greeting},' (use ONLY the first name provided, never full name or title).\n"
                "3. Write 2–3 short paragraphs (total 150–200 words):\n"
                "   • Para 1 – What you are reaching out about (contract, product, timing).\n"
                "   • Para 2 – Address the CRM next steps directly and concretely.\n"
                "   • Para 3 – Specific call-to-action with a proposed timeline or meeting ask.\n"
                "4. Reference the exact products, contract value, and expiry date.\n"
                "5. If a CRM account owner is named, mention coordinating with them.\n"
                "6. Tone: confident, concise, executive-appropriate — no fluff.\n"
                "7. Close with:\n"
                "   Best regards,\n"
                "   [Your Name]\n"
                "   IBM Seller\n\n"
                "STOP immediately after 'IBM Seller'. Output nothing else."
            )

            llm = WatsonxLLM(
                model_id=self.model_id,
                url=self.url,
                apikey=self.apikey,
                project_id=self.project_id,
                params={
                    "max_new_tokens": 500,
                    "temperature": 0.2,
                    "decoding_method": "greedy"
                }
            )

            formatted_prompt = email_prompt.invoke({
                "recipient_name": recipient_name_str if recipient_name_str else f"the {recipient_role_str}",
                "recipient_role": recipient_role_str,
                "greeting": greeting,
                "partner_name": partner_name,
                "products": contract_products_str,
                "value": contract_value_str,
                "end_date": contract_end_date_str,
                "status": contract_status_str,
                "crm_guidance": crm_guidance_block,
                "risk_level": risk_assessment.get("risk_level", "Medium"),
                "top_step": top_step,
            })

            result = llm.invoke(formatted_prompt)
            draft_email_raw = result.content if hasattr(result, "content") else str(result)

            # POST-PROCESS: strip duplicate emails and stray preamble
            draft_email = self._extract_first_email(draft_email_raw)

            return {
                "crm_updates": crm_updates,
                "draft_email": draft_email,
                "email_recipient_info": recipient_info,
                "email_recipient_display": recipient_display,
                "messages": [f"Artifacts created: CRM update and draft email to {recipient_display}"]
            }
        
        def generate_final_output_node(state: ActionState) -> dict:
            """Generate final comprehensive output with detailed reasoning"""
            recommended_action = state.get("recommended_action", {})
            risk_assessment = state.get("risk_assessment", {})
            crm_updates = state.get("crm_updates", {})
            draft_email = state.get("draft_email", "")
            email_recipient_info = state.get("email_recipient_info", {"role": "CPO", "name": None})
            email_recipient_display = state.get("email_recipient_display", "CPO")
            historical_patterns = state.get("historical_patterns", {})
            detailed_actions = recommended_action.get("detailed_actions", [])
            
            output_parts = [
                "=" * 80,
                "ACTION AGENT - DETAILED NEXT BEST STEP RECOMMENDATION",
                "=" * 80,
                "",
            ]
            
            # Add detailed recommendation with full reasoning
            if recommended_action.get("raw_recommendation"):
                output_parts.extend([
                    recommended_action["raw_recommendation"],
                    ""
                ])
            else:
                output_parts.extend([
                    "RECOMMENDED ACTION:",
                    "-" * 80,
                    "No urgent actions identified. Review portfolio for upcoming renewals.",
                    ""
                ])
            
            # Add summary of all priority items
            if detailed_actions:
                output_parts.extend([
                    "",
                    "PRIORITY SUMMARY:",
                    "-" * 80,
                ])
                for i, action in enumerate(detailed_actions, 1):
                    recipient_info = action.get('recipient_info', {"role": "CPO", "name": None})
                    recipient_display = recipient_info["name"] if recipient_info["name"] else recipient_info["role"]
                    
                    output_parts.extend([
                        f"{i}. {action['contract']} - {action['urgency_level']} urgency",
                        f"   Recipient: {recipient_display} ({recipient_info['role']})",
                        f"   Action: {action['specific_action']}",
                    ])
                    if action.get('crm_owner'):
                        output_parts.append(f"   CRM Owner: {action['crm_owner']}")
                    output_parts.append("")
            
            output_parts.extend([
                "",
                "RISK ASSESSMENT:",
                "-" * 80,
                f"Risk Level: {risk_assessment.get('risk_level', 'Unknown')}",
                f"Risk Score: {risk_assessment.get('risk_score', 0)}/100",
                "Risk Factors:",
            ])
            
            for factor in risk_assessment.get('risk_factors', []):
                output_parts.append(f"  - {factor}")
            
            output_parts.extend([
                "",
                "HISTORICAL CONTEXT:",
                "-" * 80,
                f"Pattern Confidence: {historical_patterns.get('pattern_confidence', 'Unknown')}",
                f"Based on {historical_patterns.get('total_won_deals', 0)} won deals",
                f"Average Deal Size: ${historical_patterns.get('average_deal_size', 0):,.0f}" if isinstance(historical_patterns.get('average_deal_size', 0), (int, float)) else f"Average Deal Size: {historical_patterns.get('average_deal_size', '$0')}",
                "",
                "CRM UPDATE STATUS:",
                "-" * 80,
                f"Opportunity Created: {crm_updates.get('opportunity_name', 'N/A')}",
                f"Priority: {crm_updates.get('priority', 'N/A')}",
                f"Due Date: {crm_updates.get('due_date', 'N/A')}",
                f"CRM File Updated: {crm_updates.get('crm_file_updated', False)}",
                "",
                f"DRAFT EMAIL TO {email_recipient_display}:",
                "-" * 80,
                draft_email,
                "",
                "=" * 80,
                "NEXT STEPS FOR SELLER:",
                "=" * 80,
            ])
            
            # Add actionable next steps
            if detailed_actions:
                output_parts.append("1. Review the detailed contract analysis above")
                output_parts.append(f"2. Send the draft email to {email_recipient_display}")
                if detailed_actions[0].get('crm_owner'):
                    output_parts.append(f"3. Coordinate with {detailed_actions[0]['crm_owner']} on CRM opportunity")
                output_parts.append("4. Update CRM with progress and next meeting date")
                output_parts.append("5. Set follow-up reminder for 7 days if no response")
            else:
                output_parts.extend([
                    "1. Review contract portfolio for upcoming renewals",
                    "2. Update CRM with current partnership status",
                    "3. Schedule quarterly business review with partner"
                ])
            
            output_parts.extend([
                "",
                "=" * 80,
                "Action recommendation complete. All details provided for immediate execution.",
                "=" * 80
            ])
            
            final_output = "\n".join(output_parts)
            
            return {
                "final_output": final_output,
                "messages": ["Final output generated with detailed reasoning"]
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
    
    def run(self, contract_summary: dict, partner_profile: dict, seller_query: str = "", matching_data: Optional[dict] = None) -> dict:
        """
        Run the action agent to determine next best action.
        
        Args:
            contract_summary: Output from Contract Agent
            partner_profile: Output from Research Agent
            seller_query: Seller's original query
            matching_data: Output from Matching Agent (optional)
            
        Returns:
            Final state with action recommendation and artifacts
        """
        if self.graph is None:
            self.build_agent()
        
        initial_state = {
            "contract_summary": contract_summary,
            "partner_profile": partner_profile,
            "seller_query": seller_query,
            "matching_data": matching_data or {},
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