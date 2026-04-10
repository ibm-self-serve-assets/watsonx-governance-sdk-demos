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
from matching_agent import MatchingAgent
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
    matching_data: NotRequired[dict]  # NEW: From Matching Agent
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
        
        self.matching_agent = MatchingAgent(
            model_id=model_id,
            url=url,
            apikey=self.apikey,
            project_id=self.project_id
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
        Interpret seller intent with a rule-based fallback to avoid workflow failure
        when Watsonx rate limits are hit.
        """
        normalized_query = seller_query.lower()

        fallback_partner = "Confluent" if "confluent" in normalized_query else "Unknown"
        fallback_workflow = "contract_analysis" if any(
            token in normalized_query for token in ["contract", "renewal", "expired", "cpo", "email", "next step"]
        ) else "general_inquiry"

        intent_data = {
            "raw_interpretation": "RULE_BASED_FALLBACK",
            "timestamp": datetime.now().isoformat(),
            "partner_name": fallback_partner,
            "workflow_type": fallback_workflow
        }

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

        try:
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
            intent_data["raw_interpretation"] = intent_text

            for line in intent_text.split('\n'):
                if 'PARTNER_NAME:' in line:
                    partner_name = line.split('PARTNER_NAME:')[1].strip()
                    if partner_name and partner_name.lower() != 'unknown':
                        intent_data['partner_name'] = partner_name
                elif 'WORKFLOW_TYPE:' in line:
                    workflow_type = line.split('WORKFLOW_TYPE:')[1].strip()
                    if workflow_type:
                        intent_data['workflow_type'] = workflow_type

        except Exception as e:
            intent_data["fallback_reason"] = str(e)

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
            lowered_query = seller_query.lower()
            if "next month" in lowered_query or "next 30 days" in lowered_query:
                workflow_type = "portfolio_overview_30_day_actions"
                required_agents = ["contract", "action"]
            elif "renewal" in lowered_query or "expired" in lowered_query or "expir" in lowered_query:
                workflow_type = "renewal_expiration_awareness"
                required_agents = ["contract", "action"]
            elif "draft me an email" in lowered_query or "reach out to the cpo" in lowered_query or "draft me email" in lowered_query:
                workflow_type = "executive_outreach"
                required_agents = ["contract", "action"]
            else:
                workflow_type = intent_data.get("workflow_type", "contract_analysis")
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
                "workflow_stage": workflow_type,
                "required_agents": required_agents,
                "partner_name": partner_name,
                "messages": [f"Workflow initialized: {workflow_type} using {', '.join(required_agents)}"]
            }
        
        def execute_contract_agent_node(state: SupervisoryState) -> dict:
            """Execute Contract Agent as a portfolio-wide pre-step across all partner contracts"""
            partner_name = state.get("partner_name") or "Confluent"

            print(f"\n{'='*80}")
            print("EXECUTING CONTRACT AGENT")
            print(f"{'='*80}")
            print(f"Preloading contract portfolio for partner: {partner_name}")
            print("Contract scope: all files in docs/ beginning with Confluent_IBM")

            try:
                discovered_paths = self.contract_agent.discover_partner_contracts(partner_name)

                result = self.contract_agent.run_portfolio(
                    partner_name=partner_name,
                    contract_paths=discovered_paths
                )

                print("\nContract Agent completed successfully")
                print(f"Contracts processed: {len(result.get('contract_paths', []))}")

                return {
                    "contract_summary": result,
                    "partner_name": partner_name,
                    "workflow_stage": "contract_complete",
                    "messages": ["Contract portfolio preloaded successfully"]
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
        def execute_matching_agent_node(state: SupervisoryState) -> dict:
            """Execute Matching Agent to correlate contracts with CRM opportunities"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            
            print(f"\nDEBUG - Supervisory Agent passing to Matching Agent:")
            print(f"  contract_summary keys: {list(contract_summary.keys()) if contract_summary else 'None'}")
            print(f"  partner_profile keys: {list(partner_profile.keys()) if partner_profile else 'None'}")
            
            if not contract_summary:
                return {
                    "matching_data": {"error": "Missing contract context"},
                    "messages": ["Matching Agent skipped - missing contract context"]
                }
            
            print(f"\n{'='*80}")
            print("EXECUTING MATCHING AGENT")
            print(f"{'='*80}")
            
            try:
                # Extract CRM opportunities from partner profile
                # CRM data is in internal_data.sales_history.opportunities
                internal_data = partner_profile.get("internal_data", {})
                sales_history = internal_data.get("sales_history", {})
                
                print(f"DEBUG - internal_data keys: {list(internal_data.keys()) if internal_data else 'None'}")
                print(f"DEBUG - sales_history keys: {list(sales_history.keys()) if sales_history else 'None'}")
                
                crm_opportunities = sales_history.get("opportunities", [])
                print(f"DEBUG - CRM opportunities extracted: {len(crm_opportunities)}")
                
                if crm_opportunities and len(crm_opportunities) > 0:
                    print(f"DEBUG - First opportunity keys: {list(crm_opportunities[0].keys())}")
                
                # Run Matching Agent
                result = self.matching_agent.run(
                    contract_portfolio=contract_summary,
                    crm_opportunities=crm_opportunities
                )
                
                print("\nMatching Agent completed successfully")
                matched_count = len(result.get("matched_contracts", []))
                unmatched_count = len(result.get("unmatched_contracts", []))
                print(f"Matched: {matched_count}, Unmatched: {unmatched_count}")
                
                return {
                    "matching_data": result,
                    "workflow_stage": "matching_complete",
                    "messages": ["Matching Agent executed successfully"]
                }
            
            except Exception as e:
                print(f"\nMatching Agent error: {str(e)}")
                return {
                    "matching_data": {"error": str(e)},
                    "messages": [f"Matching Agent error: {str(e)}"]
                }
        
        
        def execute_action_agent_node(state: SupervisoryState) -> dict:
            """Execute Action Agent to determine next best action"""
            contract_summary = state.get("contract_summary", {})
            partner_profile = state.get("partner_profile", {})
            matching_data = state.get("matching_data", {})  # NEW: Get matching data
            required_agents = state.get("required_agents", [])
            
            if not contract_summary:
                return {
                    "action_recommendation": {"error": "Missing contract context"},
                    "messages": ["Action Agent skipped - missing contract context"]
                }

            if "research" in required_agents and not partner_profile:
                return {
                    "action_recommendation": {"error": "Missing required context"},
                    "messages": ["Action Agent skipped - insufficient context"]
                }
            
            print(f"\n{'='*80}")
            print("EXECUTING ACTION AGENT")
            print(f"{'='*80}")
            
            try:
                # Run Action Agent with matching data
                result = self.action_agent.run(
                    contract_summary,
                    partner_profile or {"partner_name": state.get("partner_name", "Confluent")},
                    state.get("seller_query", ""),
                    matching_data  # NEW: Pass matching data
                )
                
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
            
            # Add comprehensive contract and CRM analysis
            portfolio_summary = contract_summary.get("portfolio_summary", {})
            matching_data = state.get("matching_data", {})
            
            if portfolio_summary or matching_data:
                output_parts.extend([
                    "=" * 80,
                    "COMPREHENSIVE CONTRACT & CRM ANALYSIS",
                    "=" * 80,
                    ""
                ])
                
                # Contract Portfolio Summary
                if portfolio_summary:
                    output_parts.extend([
                        "CONTRACT PORTFOLIO SUMMARY:",
                        f"  Total Contracts: {portfolio_summary.get('total_contracts', 0)}",
                        f"  Active Contracts: {len(portfolio_summary.get('active_contracts', []))}",
                        f"  Renewal Candidates: {len(portfolio_summary.get('renewal_candidates', []))}",
                        f"  Recently Expired: {len(portfolio_summary.get('recently_expired_contracts', []))}",
                        ""
                    ])
                    
                    # Detail each contract
                    all_contracts = portfolio_summary.get("contract_results", [])
                    if all_contracts:
                        output_parts.append("CONTRACT DETAILS:")
                        for i, contract in enumerate(all_contracts, 1):
                            structured = contract.get("structured_summary", {})
                            output_parts.extend([
                                f"\n{i}. {contract.get('file_name', 'Unknown')}",
                                f"   Product(s): {', '.join(structured.get('products', ['Unknown']))}",
                                f"   Amount: {structured.get('amount', 'Not specified')}",
                                f"   Start Date: {contract.get('effective_date', 'Unknown')}",
                                f"   End Date: {contract.get('end_date', 'Unknown')}",
                                f"   Status: {contract.get('status', 'Unknown').upper()}",
                                f"   Days to End: {contract.get('days_to_end', 'N/A')}"
                            ])
                        output_parts.append("")
                
                # CRM Opportunities Summary
                internal_data = partner_profile.get("internal_data", {})
                sales_history = internal_data.get("sales_history", {})
                opportunities = sales_history.get("opportunities", [])
                
                if opportunities:
                    output_parts.extend([
                        "CRM OPPORTUNITIES:",
                        f"  Total Opportunities: {len(opportunities)}",
                        ""
                    ])
                    
                    for i, opp in enumerate(opportunities, 1):
                        # Format amount safely - handle both numeric and string values
                        amount = opp.get('amount', 0)
                        if isinstance(amount, (int, float)):
                            amount_str = f"${amount:,.0f}"
                        else:
                            # Amount is already a string like "$1,000,000.00"
                            amount_str = str(amount) if amount else "$0"
                        
                        output_parts.extend([
                            f"{i}. {opp.get('opportunity_name', 'Unknown')}",
                            f"   Owner: {opp.get('owner', 'Unknown')}",
                            f"   Stage: {opp.get('stage', 'Unknown')}",
                            f"   Amount: {amount_str}",
                            f"   Close Date: {opp.get('close_date', 'Unknown')}",
                            f"   Products: {opp.get('products', 'Unknown')}",
                            f"   Next Steps: {opp.get('next_steps', 'None specified')}",
                            ""
                        ])
                
                # Contract-CRM Matching Summary
                if matching_data:
                    matched = matching_data.get("matched_contracts", [])
                    unmatched = matching_data.get("unmatched_contracts", [])
                    
                    output_parts.extend([
                        "CONTRACT-CRM CORRELATION:",
                        f"  Matched Contracts: {len(matched)}",
                        f"  Unmatched Contracts: {len(unmatched)}",
                        ""
                    ])
                    
                    if matched:
                        output_parts.append("MATCHED CONTRACTS:")
                        for match in matched:
                            contract_file = match.get("contract", {}).get("file_name", "Unknown")
                            product = match.get("product", "Unknown")
                            opps = match.get("opportunities", [])
                            
                            output_parts.append(f"\n  • {contract_file} ({product})")
                            for opp in opps:
                                output_parts.extend([
                                    f"    → CRM: {opp.get('opportunity_name', 'Unknown')}",
                                    f"      Owner: {opp.get('owner', 'Unknown')}",
                                    f"      Next Steps: {opp.get('next_steps', 'None')}"
                                ])
                        output_parts.append("")
                    
                    if unmatched:
                        output_parts.append("UNMATCHED CONTRACTS (No CRM Entry):")
                        for unmatch in unmatched:
                            contract_file = unmatch.get("contract", {}).get("file_name", "Unknown")
                            product = unmatch.get("product", "Unknown")
                            output_parts.append(f"  • {contract_file} ({product})")
                        output_parts.append("")
                
                # Recommended Next Steps
                if portfolio_summary.get("recommended_next_steps"):
                    output_parts.append("RECOMMENDED NEXT STEPS:")
                    for step in portfolio_summary.get("recommended_next_steps", []):
                        output_parts.append(f"  - {step}")
                    output_parts.append("")

            output_parts.extend([
                "=" * 80,
                "WORKFLOW COMPLETE",
                "=" * 80,
                "",
                "Contract portfolio analyzed and ingested",
                "Partner research completed",
                "Next best action determined",
                "Artifacts created (CRM update, draft email)",
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
        graph.add_node("matching_agent", execute_matching_agent_node)  # NEW
        graph.add_node("action_agent", execute_action_agent_node)
        graph.add_node("generate_result", generate_final_result_node)
        
        # Add edges - updated workflow with matching agent
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "contract_agent")
        graph.add_edge("contract_agent", "research_agent")
        graph.add_edge("research_agent", "matching_agent")  # NEW
        graph.add_edge("matching_agent", "action_agent")    # NEW
        graph.add_edge("action_agent", "generate_result")
        graph.add_edge("generate_result", END)
        
        self.graph = graph.compile()
        return self.graph
    
    def run(
        self,
        seller_query: str,
        contract_file_path: Optional[str] = None,
        partner_name: Optional[str] = None
    ) -> dict:
        """
        Run the complete supervisory workflow.

        Args:
            seller_query: Natural language query from seller
            contract_file_path: Optional legacy contract path; portfolio workflow uses all docs/Confluent_IBM*.docx
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

