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
    
    def _match_with_llm(
        self,
        contract: dict,
        opportunities: List[dict]
    ) -> tuple[List[dict], str, str]:
        """
        Use LLM to intelligently match a contract to CRM opportunities.
        
        Returns:
            Tuple of (matching_opportunities, confidence_level, reasoning)
        """
        # Extract contract details
        filename = contract.get("file_name", "")
        structured_summary = contract.get("structured_summary", {})
        
        products_list = structured_summary.get("products", [])
        product = ", ".join(products_list) if products_list and products_list != ["Not specified"] else self._extract_product_from_filename(filename)
        contract_amount = structured_summary.get("amount", "Not specified")
        start_date = structured_summary.get("start_date", "Unknown")
        end_date = structured_summary.get("end_date") or contract.get("end_date", "Unknown")
        
        # Format contract info for LLM
        contract_info = f"""
Contract File: {filename}
Products: {product}
Amount: {contract_amount}
Start Date: {start_date}
End Date: {end_date}
Status: {contract.get("status", "unknown")}
"""
        
        # Format opportunities for LLM
        opp_info = "\n\n".join([
            f"""Opportunity {i+1}:
Name: {opp.get('opportunity_name', 'Unknown')}
Owner: {opp.get('owner', 'Unknown')}
Products: {opp.get('products', 'Unknown')}
Amount: ${opp.get('amount', 0):,.0f}
Stage: {opp.get('stage', 'Unknown')}
Next Steps: {opp.get('next_steps', 'None')}"""
            for i, opp in enumerate(opportunities)
        ])
        
        # Create LLM for matching
        llm = WatsonxLLM(
            model_id=self.model_id,
            url=self.url,
            apikey=self.apikey,
            project_id=self.project_id,
            params={
                "max_new_tokens": 500,
                "temperature": 0.1,
                "decoding_method": "greedy"
            }
        )
        
        # Matching prompt
        prompt = f"""You are a contract-CRM matching expert. Analyze this contract and determine which CRM opportunities (if any) are related to it.

CONTRACT INFORMATION:
{contract_info}

CRM OPPORTUNITIES:
{opp_info}

MATCHING CRITERIA:
1. Product/service alignment (e.g., "watsonx ESA" matches "watsonx as a Service", "Cognos" matches "Cognos Analytics")
2. Dollar amount proximity (within reasonable range)
3. Timeline alignment (contract dates vs opportunity close dates)
4. Semantic relationships (understand product families and variations)

Provide your analysis in this EXACT format:

MATCHED_OPPORTUNITIES: [comma-separated list of opportunity numbers that match, or "NONE"]
CONFIDENCE: [high/medium/low]
REASONING: [Brief explanation of why these opportunities match or don't match]

Be thorough - consider semantic relationships, not just exact string matches."""

        try:
            response = llm.invoke(prompt)
            
            # Parse LLM response
            matched_indices = []
            confidence = "low"
            reasoning = "Unable to parse LLM response"
            
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith("MATCHED_OPPORTUNITIES:"):
                    matched_str = line.split(":", 1)[1].strip()
                    if matched_str.upper() != "NONE":
                        # Extract numbers
                        import re
                        numbers = re.findall(r'\d+', matched_str)
                        matched_indices = [int(n) - 1 for n in numbers if 0 <= int(n) - 1 < len(opportunities)]
                elif line.startswith("CONFIDENCE:"):
                    confidence = line.split(":", 1)[1].strip().lower()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            # Get matched opportunities
            matching_opps = [opportunities[i] for i in matched_indices]
            
            print(f"\nLLM Matching for {filename}:")
            print(f"  Matched: {len(matching_opps)} opportunities")
            print(f"  Confidence: {confidence}")
            print(f"  Reasoning: {reasoning[:100]}...")
            
            return matching_opps, confidence, reasoning
            
        except Exception as e:
            print(f"LLM matching error: {e}")
            # Fallback to simple string matching
            return self._fallback_string_match(contract, opportunities)
    
    def _fallback_string_match(
        self,
        contract: dict,
        opportunities: List[dict]
    ) -> tuple[List[dict], str, str]:
        """Fallback to simple string matching if LLM fails"""
        filename = contract.get("file_name", "")
        structured_summary = contract.get("structured_summary", {})
        
        products_list = structured_summary.get("products", [])
        product = ", ".join(products_list) if products_list and products_list != ["Not specified"] else self._extract_product_from_filename(filename)
        
        matching_opps = []
        for opp in opportunities:
            opp_name = opp.get("opportunity_name", "").lower()
            opp_products = str(opp.get("products", "")).lower()
            product_lower = product.lower()
            
            if product_lower in opp_name or product_lower in opp_products:
                matching_opps.append(opp)
        
        confidence = "high" if len(matching_opps) == 1 else "medium" if matching_opps else "low"
        reasoning = f"String match on product '{product}'"
        
        return matching_opps, confidence, reasoning
    
    def _match_contracts_to_opportunities(
        self,
        contracts: List[dict],
        opportunities: List[dict]
    ) -> tuple[List[dict], List[dict]]:
        """
        Match contracts to CRM opportunities using LLM-based intelligent matching.
        
        Returns:
            Tuple of (matched_contracts, unmatched_contracts)
        """
        matched = []
        unmatched = []
        
        for contract in contracts:
            filename = contract.get("file_name", "")
            structured_summary = contract.get("structured_summary", {})
            
            # Get product info
            products_list = structured_summary.get("products", [])
            product = ", ".join(products_list) if products_list and products_list != ["Not specified"] else self._extract_product_from_filename(filename)
            
            # Extract all contract details
            contract_amount = structured_summary.get("amount", "Not specified")
            start_date = structured_summary.get("start_date", "Unknown")
            end_date = structured_summary.get("end_date") or contract.get("end_date", "Unknown")
            term_length = structured_summary.get("term_length", "Unknown")
            status = contract.get("status", "unknown")
            
            # Use LLM to find matching opportunities
            matching_opps, confidence, reasoning = self._match_with_llm(contract, opportunities)
            
            if matching_opps:
                matched.append({
                    "contract": contract,
                    "contract_file": filename,
                    "contract_product": product,
                    "contract_value": contract_amount,
                    "contract_start": start_date,
                    "contract_end": end_date,
                    "contract_term": term_length,
                    "contract_status": status,
                    "opportunities": matching_opps,
                    "match_confidence": confidence,
                    "match_reasoning": reasoning,
                    # Add CRM details from first matching opportunity
                    "crm_opportunity": matching_opps[0].get("opportunity_name", "Unknown"),
                    "crm_owner": matching_opps[0].get("owner", "Unknown"),
                    "crm_stage": matching_opps[0].get("stage", "Unknown"),
                    "crm_amount": matching_opps[0].get("amount", "Unknown"),
                    "crm_close_date": matching_opps[0].get("close_date", "Unknown"),
                    "crm_next_steps": matching_opps[0].get("next_steps", "None")
                })
            else:
                unmatched.append({
                    "contract": contract,
                    "contract_file": filename,
                    "contract_product": product,
                    "contract_value": contract_amount,
                    "contract_start": start_date,
                    "contract_end": end_date,
                    "contract_status": status,
                    "reason": f"No matching CRM opportunity found. {reasoning}"
                })
        
        return matched, unmatched
    
    def build_agent(self):
        """Build the LangGraph for contract-CRM matching"""
        
        def match_contracts_node(state: MatchingState) -> dict:
            """Match contracts to CRM opportunities"""
            portfolio = state.get("contract_portfolio", {})
            opportunities = state.get("crm_opportunities", [])
            
            print(f"\nDEBUG - Matching Agent:")
            print(f"  Portfolio keys: {list(portfolio.keys())}")
            print(f"  Opportunities count: {len(opportunities)}")
            
            # Get contracts from portfolio
            portfolio_summary = portfolio.get("portfolio_summary", {})
            all_contracts = []
            
            # Include active contracts
            active = portfolio_summary.get("active_contracts", [])
            print(f"  Active contracts: {len(active)}")
            all_contracts.extend(active)
            
            # Include renewal candidates
            renewal = portfolio_summary.get("renewal_candidates", [])
            print(f"  Renewal candidates: {len(renewal)}")
            all_contracts.extend(renewal)
            
            # Include recently expired
            expired = portfolio_summary.get("recently_expired_contracts", [])
            print(f"  Recently expired: {len(expired)}")
            all_contracts.extend(expired)
            
            print(f"  Total contracts to match: {len(all_contracts)}")
            
            if all_contracts and len(all_contracts) > 0:
                print(f"  Sample contract keys: {list(all_contracts[0].keys())}")
            
            # Perform matching
            matched, unmatched = self._match_contracts_to_opportunities(
                all_contracts,
                opportunities
            )
            
            print(f"  Matching complete: {len(matched)} matched, {len(unmatched)} unmatched")
            
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
                    product = match.get("contract_product", "Unknown")
                    opportunities = match["opportunities"]
                    end_date = match.get("contract_end", "Unknown")
                    status = match.get("contract_status", "unknown")
                    confidence = match.get("match_confidence", "unknown")
                    reasoning = match.get("match_reasoning", "No reasoning provided")
                    
                    output_parts.extend([
                        "",
                        f"Product: {product}",
                        f"Contract: {contract.get('file_name', 'Unknown')}",
                        f"Status: {status}",
                        f"End Date: {end_date}",
                        f"Match Confidence: {confidence}",
                        f"Match Reasoning: {reasoning[:100]}..." if len(reasoning) > 100 else f"Match Reasoning: {reasoning}",
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
                        f"Product: {item.get('contract_product', 'Unknown')}",
                        f"Contract: {contract.get('file_name', 'Unknown')}",
                        f"Status: {item.get('contract_status', 'unknown')}",
                        f"End Date: {item.get('contract_end', 'Unknown')}",
                        f"Reason: {item.get('reason', 'No reason provided')}"
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
