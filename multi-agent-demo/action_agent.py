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
        
        Args:
            contract_summary: Contract analysis from Contract Agent
            partner_profile: Partner profile from Research Agent
            
        Returns:
            Risk assessment dictionary
        """
        risk_factors = []
        risk_score = 0  # 0-100, higher is riskier
        
        # Check partner maturity
        maturity = partner_profile.get('maturity_level', 'Unknown')
        if maturity == 'New Partner':
            risk_factors.append("New partner with no prior engagement history")
            risk_score += 30
        elif maturity == 'Early Stage':
            risk_factors.append("Limited successful engagement history")
            risk_score += 20
        
        # Check deal blockers
        deal_blockers = partner_profile.get('deal_blockers', [])
        if deal_blockers:
            risk_factors.append(f"Historical deal blockers: {len(deal_blockers)} identified")
            risk_score += min(30, len(deal_blockers) * 10)
        
        # Check sales velocity
        velocity = partner_profile.get('sales_velocity', 'Unknown')
        if velocity == 'Low':
            risk_factors.append("Low historical sales velocity")
            risk_score += 15
        
        # Check contract complexity (if available in structured summary)
        structured = contract_summary.get('structured_summary', {})
        if isinstance(structured, dict):
            contract_risk = structured.get('risk_level', 'Medium')
            if contract_risk == 'High':
                risk_factors.append("High-risk contract terms identified")
                risk_score += 25
        
        # Determine overall risk level
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
            """Determine recommended next best action using LLM"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            historical_patterns = state.get("historical_patterns", {})
            risk_assessment = state.get("risk_assessment", {})
            
            # Create comprehensive context for LLM including scenario actions
            context = f"""
CONTRACT SUMMARY:
{json.dumps(contract_summary.get('structured_summary', {}), indent=2)}

PARTNER PROFILE:
- Maturity Level: {partner_profile.get('maturity_level', 'Unknown')}
- Sales Velocity: {partner_profile.get('sales_velocity', 'Unknown')}
- Deal Blockers: {len(partner_profile.get('deal_blockers', []))}

HISTORICAL PATTERNS:
- Total Won Deals: {historical_patterns.get('total_won_deals', 0)}
- Pattern Confidence: {historical_patterns.get('pattern_confidence', 'Unknown')}
- Successful Next Steps: {historical_patterns.get('successful_next_steps', [])}

RISK ASSESSMENT:
- Risk Level: {risk_assessment.get('risk_level', 'Unknown')}
- Risk Factors: {risk_assessment.get('risk_factors', [])}

SCENARIO-BASED ACTION GUIDELINES:
{self.scenario_actions_text[:3000] if self.scenario_actions_text else 'No scenario guidelines available'}
"""
            
            action_prompt = ChatPromptTemplate.from_template(
                "You are a sales strategy expert. Based on the following context and scenario-based action guidelines, determine the SINGLE most important next best action.\n\n"
                "Use the SCENARIO-BASED ACTION GUIDELINES to identify which scenario best matches the current situation and follow the recommended next steps from those guidelines.\n\n"
                "{context}\n\n"
                "Provide your recommendation in this exact format:\n"
                "MATCHING_SCENARIO: [Which scenario from guidelines best matches]\n"
                "ACTION: [One clear, specific action from the guidelines]\n"
                "PRIORITY: [High/Medium/Low]\n"
                "TIMELINE: [Immediate/This Week/This Month]\n"
                "RATIONALE: [Why this action is most important based on the scenario]\n"
                "SUCCESS_CRITERIA: [How to measure success]\n"
                "OWNER: [Who should execute this]\n"
            )
            
            llm = WatsonxLLM(
                model_id=self.model_id,
                url=self.url,
                apikey=self.apikey,
                project_id=self.project_id,
                params={
                    "max_new_tokens": 600,
                    "temperature": 0.2,
                    "decoding_method": "greedy"
                }
            )
            
            formatted_prompt = action_prompt.invoke({"context": context})
            result = llm.invoke(formatted_prompt)
            action_text = result.content if hasattr(result, "content") else str(result)
            
            # Parse the action recommendation
            recommended_action = {
                "raw_recommendation": action_text,
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
            
            # Extract partner name
            partner_name = partner_profile.get("partner_name", "Partner")
            
            # Create CRM update
            structured = contract_summary.get("structured_summary", {})
            crm_updates = {
                "opportunity_name": f"{partner_name} - Post-ESA Follow-up",
                "stage": "Onboarding",
                "next_step": recommended_action.get("raw_recommendation", "Follow up on contract"),
                "owner": "Sales Team",
                "due_date": (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                "priority": risk_assessment.get("risk_level", "Medium"),
                "updated_timestamp": datetime.now().isoformat()
            }
            
            # Create draft email using LLM
            email_prompt = ChatPromptTemplate.from_template(
                "You are a professional sales representative. Write a personalized follow-up email based on:\n\n"
                "Partner: {partner_name}\n"
                "Contract Type: {contract_type}\n"
                "Recommended Action: {action}\n"
                "Partner Maturity: {maturity}\n\n"
                "Write a professional, concise email (3-4 paragraphs) that:\n"
                "1. References the signed contract\n"
                "2. Proposes the next step\n"
                "3. Provides clear value proposition\n"
                "4. Includes a specific call-to-action\n\n"
                "Format as a complete email with subject line."
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
            
            contract_type = structured.get("contract_type", "Agreement") if isinstance(structured, dict) else "Agreement"
            
            formatted_prompt = email_prompt.invoke({
                "partner_name": partner_name,
                "contract_type": contract_type,
                "action": recommended_action.get("raw_recommendation", "Schedule onboarding"),
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
    
    def run(self, contract_summary: dict, partner_profile: dict) -> dict:
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
