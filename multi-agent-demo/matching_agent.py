"""
Matching Agent - Correlates contract data with CRM opportunities

This agent:
1. Takes contract portfolio data from Contract Agent
2. Takes CRM opportunity data from Research Agent
3. Matches contracts to CRM opportunities based on:
   - Product names (e.g., Cognos, watsonx)
   - Dollar amounts
   - Dates
4. Returns enriched contract-opportunity mappings with next steps
"""

from typing import TypedDict, Annotated, Optional, Dict, List
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
import operator
from datetime import datetime
import json
import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# State Definition
# ============================================================================

class MatchingState(TypedDict):
    """State for Matching Agent workflow"""
    contract_portfolio: dict  # From Contract Agent
    crm_opportunities: list   # From Research Agent
    messages: Annotated[list, operator.add]
    matched_contracts: NotRequired[List[dict]]
    unmatched_contracts: NotRequired[List[dict]]
    final_output: NotRequired[str]


# ============================================================================
# Matching Agent Class
# ============================================================================

class MatchingAgent:
    """
    Matching Agent using LangGraph to correlate contracts with CRM opportunities.
    
    This agent:
    1. Analyzes contract portfolio for key products and amounts
    2. Analyzes CRM opportunities for matching products and amounts
    3. Creates enriched mappings with next steps from CRM
    4. Identifies unmatched contracts that need attention
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/llama-3-3-70b-instruct",
        url: str = "https://us-south.ml.cloud.ibm.com",
        apikey: Optional[str] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize Matching Agent.
        
        Args:
            model_id: Watsonx model ID for text generation
            url: Watsonx API URL
            apikey: IBM Cloud API key (defaults to WATSONX_APIKEY env var)
            project_id: Watsonx project ID (defaults to WATSONX_PROJECT_ID env var)
        """
        self.model_id = model_id
        self.url = url
        self.apikey = apikey or os.getenv("WATSONX_APIKEY")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.graph = None
        
        if not self.apikey:
            raise ValueError("WATSONX_APIKEY must be provided or set in environment variables")
        if not self.project_id:
            raise ValueError("WATSONX_PROJECT_ID must be provided or set in environment variables")
    
    def _extract_product_from_filename(self, filename: str) -> str:
        """Extract product name from contract filename"""
        filename_lower = filename.lower()
        
        # Check for specific products
        if "cognos" in filename_lower:
            return "Cognos"
        elif "watsonx" in filename_lower:
            return "watsonx"
        elif "watson" in filename_lower:
            return "Watson"
        elif "orchestrate" in filename_lower:
            return "watsonx Orchestrate"
        elif "governance" in filename_lower:
            return "watsonx.governance"
        
        return "Unknown"
    
    def _match_contracts_to_opportunities(
        self,
        contracts: List[dict],
        opportunities: List[dict]
    ) -> tuple[List[dict], List[dict]]:
        """
        Match contracts to CRM opportunities based on product names, amounts, and dates.
        
        Returns:
            Tuple of (matched_contracts, unmatched_contracts)
        """
        matched = []
        unmatched = []
        
        for contract in contracts:
            filename = contract.get("file_name", "")
            product = self._extract_product_from_filename(filename)
            contract_amount = contract.get("structured_summary", {}).get("amount")
            end_date = contract.get("end_date", "Unknown")
            status = contract.get("status", "unknown")
            
            # Find matching opportunities
            matching_opps = []
            for opp in opportunities:
                opp_name = opp.get("opportunity_name", "").lower()
                opp_products = str(opp.get("products", "")).lower()
                
                # Check if product matches
                if product.lower() in opp_name or product.lower() in opp_products:
                    matching_opps.append(opp)
            
            if matching_opps:
                matched.append({
                    "contract": contract,
                    "product": product,
                    "opportunities": matching_opps,
                    "status": status,
                    "end_date": end_date,
                    "match_confidence": "high" if len(matching_opps) == 1 else "medium"
                })
            else:
                unmatched.append({
                    "contract": contract,
                    "product": product,
                    "status": status,
                    "end_date": end_date,
                    "reason": "No matching CRM opportunity found"
                })
        
        return matched, unmatched
    
    def build_agent(self):
        """Build the LangGraph for contract-CRM matching"""
        
        def match_contracts_node(state: MatchingState) -> dict:
            """Match contracts to CRM opportunities"""
            portfolio = state.get("contract_portfolio", {})
            opportunities = state.get("crm_opportunities", [])
            
            # Get contracts from portfolio
            portfolio_summary = portfolio.get("portfolio_summary", {})
            all_contracts = []
            
            # Include active contracts
            all_contracts.extend(portfolio_summary.get("active_contracts", []))
            
            # Include renewal candidates
            all_contracts.extend(portfolio_summary.get("renewal_candidates", []))
            
            # Include recently expired
            all_contracts.extend(portfolio_summary.get("recently_expired_contracts", []))
            
            # Perform matching
            matched, unmatched = self._match_contracts_to_opportunities(
                all_contracts,
                opportunities
            )
            
            return {
                "matched_contracts": matched,
                "unmatched_contracts": unmatched,
                "messages": [f"Matched {len(matched)} contracts to CRM opportunities, {len(unmatched)} unmatched"]
            }
        
        def generate_output_node(state: MatchingState) -> dict:
            """Generate final matching report"""
            matched = state.get("matched_contracts", [])
            unmatched = state.get("unmatched_contracts", [])
            
            output_parts = [
                "=" * 80,
                "MATCHING AGENT - CONTRACT-CRM CORRELATION",
                "=" * 80,
                "",
                f"Total Matched: {len(matched)}",
                f"Total Unmatched: {len(unmatched)}",
                ""
            ]
            
            if matched:
                output_parts.extend([
                    "MATCHED CONTRACTS:",
                    "-" * 80
                ])
                
                for match in matched:
                    contract = match["contract"]
                    product = match["product"]
                    opportunities = match["opportunities"]
                    end_date = match["end_date"]
                    status = match["status"]
                    
                    output_parts.extend([
                        "",
                        f"Product: {product}",
                        f"Contract: {contract.get('file_name', 'Unknown')}",
                        f"Status: {status}",
                        f"End Date: {end_date}",
                        f"Matching Opportunities: {len(opportunities)}"
                    ])
                    
                    for opp in opportunities:
                        output_parts.extend([
                            f"  - {opp.get('opportunity_name', 'Unknown')}",
                            f"    Owner: {opp.get('owner', 'Unknown')}",
                            f"    Stage: {opp.get('stage', 'Unknown')}",
                            f"    Amount: ${opp.get('amount', 0):,.0f}",
                            f"    Next Steps: {opp.get('next_steps', 'None')}",
                        ])
            
            if unmatched:
                output_parts.extend([
                    "",
                    "UNMATCHED CONTRACTS (Need Attention):",
                    "-" * 80
                ])
                
                for item in unmatched:
                    contract = item["contract"]
                    output_parts.extend([
                        "",
                        f"Product: {item['product']}",
                        f"Contract: {contract.get('file_name', 'Unknown')}",
                        f"Status: {item['status']}",
                        f"End Date: {item['end_date']}",
                        f"Reason: {item['reason']}"
                    ])
            
            output_parts.extend([
                "",
                "=" * 80,
                "Matching complete. Use this information to prioritize seller actions.",
                "=" * 80
            ])
            
            final_output = "\n".join(output_parts)
            
            return {
                "final_output": final_output,
                "messages": ["Matching report generated"]
            }
        
        # Build the graph
        graph = StateGraph(MatchingState)
        
        # Add nodes
        graph.add_node("match_contracts", match_contracts_node)
        graph.add_node("generate_output", generate_output_node)
        
        # Add edges
        graph.add_edge(START, "match_contracts")
        graph.add_edge("match_contracts", "generate_output")
        graph.add_edge("generate_output", END)
        
        self.graph = graph.compile()
        return self.graph
    
    def run(self, contract_portfolio: dict, crm_opportunities: list) -> dict:
        """
        Run the matching agent to correlate contracts with CRM opportunities.
        
        Args:
            contract_portfolio: Output from Contract Agent
            crm_opportunities: List of CRM opportunities from Research Agent
            
        Returns:
            Final state with matched and unmatched contracts
        """
        if self.graph is None:
            self.build_agent()
        
        initial_state = {
            "contract_portfolio": contract_portfolio,
            "crm_opportunities": crm_opportunities,
            "messages": []
        }
        
        result = self.graph.invoke(initial_state)
        return result


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Mock data for testing
    mock_portfolio = {
        "portfolio_summary": {
            "active_contracts": [
                {
                    "file_name": "Confluent_IBM_Cognos_2023.docx",
                    "status": "expiring_soon",
                    "end_date": "2026-05-31",
                    "structured_summary": {"amount": 1000000}
                }
            ],
            "renewal_candidates": [
                {
                    "file_name": "Confluent_IBM_watsonx_2024.docx",
                    "status": "active",
                    "end_date": "2026-07-31",
                    "structured_summary": {"amount": 400000}
                }
            ],
            "recently_expired_contracts": [
                {
                    "file_name": "Confluent_IBM_watsonx_2025.docx",
                    "status": "expired",
                    "end_date": "2026-01-31",
                    "structured_summary": {"amount": 250000}
                }
            ]
        }
    }
    
    mock_opportunities = [
        {
            "opportunity_name": "Confluent Cognos Renewal",
            "owner": "Kylie Brittz",
            "stage": "Qualified",
            "amount": 1000000,
            "products": "Cognos",
            "next_steps": "Quote for renewal being shared with team and discussing expansion"
        },
        {
            "opportunity_name": "Confluent watsonx ESA",
            "owner": "Anand Das",
            "stage": "Qualified",
            "amount": 250000,
            "products": "watsonx",
            "next_steps": "watsonx ESA expired working with team on size needed to renew"
        }
    ]
    
    agent = MatchingAgent(
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )
    
    result = agent.run(mock_portfolio, mock_opportunities)
    
    print(result["final_output"])
