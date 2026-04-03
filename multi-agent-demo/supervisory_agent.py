"""
Supervisory Agent - Orchestrates Contract, Research, and Action agents

This agent:
1. Interprets seller intent from natural language query
2. Identifies required agents based on the workflow
3. Executes agents in sequence: Contract → Research → Action
4. Aggregates context and passes between agents
5. Returns final result with next best action to seller
"""

from typing import TypedDict, Annotated, Optional, Dict, List, Literal
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
import operator
from datetime import datetime
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our specialized agents
from contract_agent import ContractAgent
from research_agent import research_partner
from action_agent import ActionAgent


# ============================================================================
# State Definition
# ============================================================================

class SupervisoryState(TypedDict):
    """State for Supervisory Agent workflow"""
    seller_query: str
    contract_file_path: Optional[str]
    partner_name: Optional[str]
    messages: Annotated[list, operator.add]
    
    # Agent outputs
    contract_summary: NotRequired[dict]
    partner_profile: NotRequired[dict]
    action_recommendation: NotRequired[dict]
    
    # Workflow control
    workflow_stage: NotRequired[str]
    required_agents: NotRequired[List[str]]
    
    # Final output
    final_result: NotRequired[str]


# ============================================================================
# Supervisory Agent Class
# ============================================================================

class SupervisoryAgent:
    """
    Supervisory Agent using LangGraph to orchestrate the sales workflow.
    
    This agent manages the complete workflow:
    1. Seller query interpretation
    2. Contract Agent execution (ESA/contract ingestion)
    3. Research Agent execution (partner intelligence)
    4. Action Agent execution (next best action)
    5. Final result presentation
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/llama-3-3-70b-instruct",
        url: str = "https://us-south.ml.cloud.ibm.com",
        apikey: Optional[str] = None,
        project_id: Optional[str] = None,
        contract_vector_store_path: str = "./contract_vector_store",
        crm_file_path: str = "docs/Confluent Sales Cloud Infor.xlsx"
    ):
        """
        Initialize Supervisory Agent.
        
        Args:
            model_id: Watsonx model ID for text generation
            url: Watsonx API URL
            apikey: IBM Cloud API key (defaults to WATSONX_APIKEY env var)
            project_id: Watsonx project ID (defaults to WATSONX_PROJECT_ID env var)
            contract_vector_store_path: Path for contract vector store
            crm_file_path: Path to CRM Excel file
        """
        self.model_id = model_id
        self.url = url
        # Use environment variables if not provided
        self.apikey = apikey or os.getenv("WATSONX_APIKEY")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.contract_vector_store_path = contract_vector_store_path
        self.crm_file_path = crm_file_path
        self.graph = None
        
        # Validate credentials
        if not self.apikey:
            raise ValueError("WATSONX_APIKEY must be provided or set in environment variables")
        if not self.project_id:
            raise ValueError("WATSONX_PROJECT_ID must be provided or set in environment variables")
        
        # Initialize specialized agents
        self.contract_agent = ContractAgent(
            model_id=model_id,
            url=url,
            apikey=self.apikey,
            project_id=self.project_id,
            vector_store_path=contract_vector_store_path
        )
        
        self.action_agent = ActionAgent(
            model_id=model_id,
            url=url,
            apikey=apikey,
            project_id=project_id,
            crm_file_path=crm_file_path
        )
    
    def _interpret_intent(self, seller_query: str) -> dict:
        """
        Interpret seller's intent and extract key information.
        
        Args:
            seller_query: Natural language query from seller
            
        Returns:
            Dictionary with intent, partner name, and workflow type
        """
        intent_prompt = ChatPromptTemplate.from_template(
            "You are a sales workflow assistant. Analyze this seller query and extract:\n\n"
            "Query: {query}\n\n"
            "Provide in this exact format:\n"
            "INTENT: [What the seller wants to do]\n"
            "PARTNER_NAME: [Name of partner/company mentioned, or 'Unknown']\n"
            "WORKFLOW_TYPE: [contract_analysis/next_action/general_inquiry]\n"
            "CONTRACT_MENTIONED: [yes/no]\n"
            "KEY_ENTITIES: [Any other important entities mentioned]\n"
        )
        
        llm = WatsonxLLM(
            model_id=self.model_id,
            url=self.url,
            apikey=self.apikey,
            project_id=self.project_id,
            params={
                "max_new_tokens": 300,
                "temperature": 0.1,
                "decoding_method": "greedy"
            }
        )
        
        formatted_prompt = intent_prompt.invoke({"query": seller_query})
        result = llm.invoke(formatted_prompt)
        intent_text = result.content if hasattr(result, "content") else str(result)
        
        # Parse the intent
        intent_data = {
            "raw_interpretation": intent_text,
            "timestamp": datetime.now().isoformat()
        }
        
        # Extract partner name from interpretation
        for line in intent_text.split('\n'):
            if 'PARTNER_NAME:' in line:
                partner_name = line.split('PARTNER_NAME:')[1].strip()
                if partner_name and partner_name.lower() != 'unknown':
                    intent_data['partner_name'] = partner_name
        
        return intent_data
    
    def build_agent(self):
        """Build the LangGraph for supervisory workflow"""
        
        def initialize_workflow_node(state: SupervisoryState) -> dict:
            """Initialize workflow by interpreting seller query"""
            seller_query = state["seller_query"]
            
            print(f"\n{'='*80}")
            print("SUPERVISORY AGENT - Workflow Initialization")
            print(f"{'='*80}")
            print(f"Seller Query: {seller_query}")
            
            # Interpret intent
            intent_data = self._interpret_intent(seller_query)
            
            # Determine required agents
            required_agents = ["contract", "research", "action"]
            
            # Extract partner name if available
            partner_name = intent_data.get('partner_name')
            if not partner_name and state.get("partner_name"):
                partner_name = state["partner_name"]
            
            print(f"\nIntent Analysis:")
            print(intent_data.get('raw_interpretation', 'No interpretation'))
            print(f"\nRequired Agents: {', '.join(required_agents)}")
            print(f"Partner Name: {partner_name or 'To be extracted from contract'}")
            
            return {
                "workflow_stage": "initialized",
                "required_agents": required_agents,
                "partner_name": partner_name,
                "messages": [f"Workflow initialized: {', '.join(required_agents)} agents required"]
            }
        
        def execute_contract_agent_node(state: SupervisoryState) -> dict:
            """Execute Contract Agent to ingest and analyze contract"""
            contract_file_path = state.get("contract_file_path")
            
            if not contract_file_path:
                return {
                    "contract_summary": {"error": "No contract file path provided"},
                    "messages": ["Contract Agent skipped - no file path"]
                }
            
            print(f"\n{'='*80}")
            print("EXECUTING CONTRACT AGENT")
            print(f"{'='*80}")
            print(f"Processing: {contract_file_path}")
            
            try:
                # Run Contract Agent
                result = self.contract_agent.run(contract_file_path)
                
                print("\nContract Agent completed successfully")
                print(f"Document length: {len(result.get('raw_text', ''))} characters")
                
                # Extract partner name from contract if not already set
                partner_name = state.get("partner_name")
                if not partner_name and result.get("structured_summary"):
                    structured = result["structured_summary"]
                    if isinstance(structured, dict) and "parties" in structured:
                        parties = structured["parties"]
                        if isinstance(parties, list) and len(parties) > 0:
                            # Use the first party that's not "Confluent"
                            for party in parties:
                                if "Confluent" not in party:
                                    partner_name = party
                                    break
                
                return {
                    "contract_summary": result,
                    "partner_name": partner_name,
                    "workflow_stage": "contract_complete",
                    "messages": ["Contract Agent executed successfully"]
                }
            
            except Exception as e:
                print(f"\nContract Agent error: {str(e)}")
                return {
                    "contract_summary": {"error": str(e)},
                    "messages": [f"Contract Agent error: {str(e)}"]
                }
        
        def execute_research_agent_node(state: SupervisoryState) -> dict:
            """Execute Research Agent to enrich partner context"""
            partner_name = state.get("partner_name")
            
            if not partner_name:
                # Try to extract from contract summary
                contract_summary = state.get("contract_summary", {})
                structured = contract_summary.get("structured_summary", {})
                if isinstance(structured, dict) and "parties" in structured:
                    parties = structured["parties"]
                    if isinstance(parties, list) and len(parties) > 0:
                        for party in parties:
                            if "Confluent" not in party:
                                partner_name = party
                                break
            
            if not partner_name:
                return {
                    "partner_profile": {"error": "No partner name available"},
                    "messages": ["Research Agent skipped - no partner name"]
                }
            
            print(f"\n{'='*80}")
            print("EXECUTING RESEARCH AGENT")
            print(f"{'='*80}")
            print(f"Researching: {partner_name}")
            
            try:
                # Run Research Agent
                partner_profile = research_partner(partner_name)
                
                print("\nResearch Agent completed successfully")
                print(f"Maturity Level: {partner_profile.get('maturity_level', 'Unknown')}")
                print(f"Sales Velocity: {partner_profile.get('sales_velocity', 'Unknown')}")
                
                return {
                    "partner_profile": partner_profile,
                    "workflow_stage": "research_complete",
                    "messages": ["Research Agent executed successfully"]
                }
            
            except Exception as e:
                print(f"\nResearch Agent error: {str(e)}")
                return {
                    "partner_profile": {"error": str(e), "partner_name": partner_name},
                    "messages": [f"Research Agent error: {str(e)}"]
                }
        
        def execute_action_agent_node(state: SupervisoryState) -> dict:
            """Execute Action Agent to determine next best action"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            
            if not contract_summary or not partner_profile:
                return {
                    "action_recommendation": {"error": "Missing required context"},
                    "messages": ["Action Agent skipped - insufficient context"]
                }
            
            print(f"\n{'='*80}")
            print("EXECUTING ACTION AGENT")
            print(f"{'='*80}")
            
            try:
                # Run Action Agent
                result = self.action_agent.run(contract_summary, partner_profile)
                
                print("\nAction Agent completed successfully")
                risk_level = result.get("risk_assessment", {}).get("risk_level", "Unknown")
                print(f"Risk Level: {risk_level}")
                
                return {
                    "action_recommendation": result,
                    "workflow_stage": "action_complete",
                    "messages": ["Action Agent executed successfully"]
                }
            
            except Exception as e:
                print(f"\nAction Agent error: {str(e)}")
                return {
                    "action_recommendation": {"error": str(e)},
                    "messages": [f"Action Agent error: {str(e)}"]
                }
        
        def generate_final_result_node(state: SupervisoryState) -> dict:
            """Generate final result for seller"""
            seller_query = state["seller_query"]
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            action_recommendation = state.get("action_recommendation", {})
            
            output_parts = [
                "=" * 80,
                "SALES ASSIST TOOL - WORKFLOW COMPLETE",
                "=" * 80,
                "",
                f"Original Query: {seller_query}",
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "=" * 80,
                "EXECUTIVE SUMMARY",
                "=" * 80,
            ]
            
            # Add partner information
            partner_name = partner_profile.get("partner_name", "Unknown")
            maturity = partner_profile.get("maturity_level", "Unknown")
            velocity = partner_profile.get("sales_velocity", "Unknown")
            
            output_parts.extend([
                "",
                f"Partner: {partner_name}",
                f"Maturity Level: {maturity}",
                f"Sales Velocity: {velocity}",
                ""
            ])
            
            # Add risk assessment
            risk_assessment = action_recommendation.get("risk_assessment", {})
            if risk_assessment:
                output_parts.extend([
                    "RISK ASSESSMENT:",
                    f"  Risk Level: {risk_assessment.get('risk_level', 'Unknown')}",
                    f"  Risk Score: {risk_assessment.get('risk_score', 0)}/100",
                    ""
                ])
            
            # Add recommended action (most important)
            recommended_action = action_recommendation.get("recommended_action", {})
            if recommended_action:
                output_parts.extend([
                    "=" * 80,
                    "RECOMMENDED NEXT STEP",
                    "=" * 80,
                    "",
                    recommended_action.get("raw_recommendation", "No recommendation available"),
                    ""
                ])
            
            # Add draft email
            draft_email = action_recommendation.get("draft_email", "")
            if draft_email:
                output_parts.extend([
                    "=" * 80,
                    "DRAFT FOLLOW-UP EMAIL",
                    "=" * 80,
                    "",
                    draft_email,
                    ""
                ])
            
            # Add CRM update
            crm_updates = action_recommendation.get("crm_updates", {})
            if crm_updates:
                output_parts.extend([
                    "=" * 80,
                    "CRM UPDATE (Demo)",
                    "=" * 80,
                    "",
                    json.dumps(crm_updates, indent=2),
                    ""
                ])
            
            output_parts.extend([
                "=" * 80,
                "WORKFLOW COMPLETE",
                "=" * 80,
                "",
                "✓ Contract analyzed and ingested",
                "✓ Partner research completed",
                "✓ Next best action determined",
                "✓ Artifacts created (CRM update, draft email)",
                "",
                "Seller can now execute the recommended action directly from this tool.",
                "=" * 80
            ])
            
            final_result = "\n".join(output_parts)
            
            return {
                "final_result": final_result,
                "workflow_stage": "complete",
                "messages": ["Final result generated"]
            }
        
        # Build the graph
        graph = StateGraph(SupervisoryState)
        
        # Add nodes
        graph.add_node("initialize", initialize_workflow_node)
        graph.add_node("contract_agent", execute_contract_agent_node)
        graph.add_node("research_agent", execute_research_agent_node)
        graph.add_node("action_agent", execute_action_agent_node)
        graph.add_node("generate_result", generate_final_result_node)
        
        # Add edges - linear workflow
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "contract_agent")
        graph.add_edge("contract_agent", "research_agent")
        graph.add_edge("research_agent", "action_agent")
        graph.add_edge("action_agent", "generate_result")
        graph.add_edge("generate_result", END)
        
        self.graph = graph.compile()
        return self.graph
    
    def run(
        self,
        seller_query: str,
        contract_file_path: str,
        partner_name: Optional[str] = None
    ) -> dict:
        """
        Run the complete supervisory workflow.
        
        Args:
            seller_query: Natural language query from seller
            contract_file_path: Path to contract document
            partner_name: Optional partner name (will be extracted if not provided)
            
        Returns:
            Final state with complete workflow results
        """
        if self.graph is None:
            self.build_agent()
        
        initial_state = {
            "seller_query": seller_query,
            "contract_file_path": contract_file_path,
            "partner_name": partner_name,
            "messages": []
        }
        
        result = self.graph.invoke(initial_state)
        return result


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Initialize Supervisory Agent
    supervisor = SupervisoryAgent(
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )
    
    # Example workflow
    seller_query = "I just received a signed ESA from IBM. What should I do next?"
    contract_file_path = "docs/Confluent_IBM-1.30.2024.docx"
    
    print("\n" + "="*80)
    print("SALES ASSIST TOOL - SUPERVISORY AGENT")
    print("="*80)
    print(f"\nSeller Query: {seller_query}")
    print(f"Contract File: {contract_file_path}")
    
    # Run the workflow
    result = supervisor.run(
        seller_query=seller_query,
        contract_file_path=contract_file_path,
        partner_name="IBM"  # Optional - will be extracted from contract if not provided
    )
    
    # Display final result
    print("\n" + result["final_result"])

# Made with Bob
