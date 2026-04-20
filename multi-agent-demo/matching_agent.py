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
            f"""Opportunity {opp.get('opportunity_number', i+1)} (CRM #{opp.get('opportunity_number', i+1)}):
Name: {opp.get('opportunity_name', 'Unknown')}
Owner: {opp.get('owner', 'Unknown')}
Products: {opp.get('products', 'Unknown')}
Amount: {opp.get('amount', '$0')}
Stage: {opp.get('stage', 'Unknown')}
Close Date: {opp.get('close_date', 'Unknown')}
Contract Expiration Date: {opp.get('contract_expiration_date', 'N/A')}
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
        
        # Matching prompt with business logic
        prompt = f"""Analyze this contract and identify which CRM opportunities match it.

CONTRACT:
{contract_info}

CRM OPPORTUNITIES:
{opp_info}

IMPORTANT BUSINESS RULES FOR MATCHING:

1. AMOUNT MATCHING:
   - CRM amounts are ROUNDED (e.g., $250,000) while contracts have EXACT amounts (e.g., $250,003.20)
   - Use +/-7% tolerance when comparing amounts
   - Example: $250,000 (CRM) matches $250,003.20 (contract) [MATCH]

2. PRODUCT MATCHING:
   - "watsonx" or "watsonx as a service" in contracts should match ANY watsonx product in CRM
   - Examples: "watsonx" matches "watsonx.ai", "watsonx.data", "watsonx.governance", "watsonx Orchestrate" [MATCH]
   - "Cognos" matches "Cognos Analytics", "Cognos BI" [MATCH]

3. DATE MATCHING FOR RENEWALS:
   - +/-2 days between contract end date and opportunity close date is a STRONG indicator of renewal
   - However, renewals can happen OUTSIDE the +/-2 day window (e.g., months before expiration)
   - Look at the full context: products, amounts, and "Next Steps" mentioning renewal

4. MATCHING LOGIC:
   - For EXPIRED contracts: Look for ACTIVE opportunities (Stage=Qualify/Design/Engage/Negotiate) with:
     * Same/similar products (use rules above)
     * Similar amounts (+/-7% tolerance)
     * Next Steps mentioning "renewal", "expired", or "renew"
   
   - For ACTIVE contracts: Look for opportunities with:
     * Contract Expiration Date matching the contract's End Date (+/-2 days is strong, but not required)
     * Same products (use product matching rules)
     * Similar amounts (+/-7% tolerance)

5. MULTIPLE MATCHES:
   - A contract can match MULTIPLE opportunities (e.g., renewal + expansion)
   - List ALL matching opportunities, not just the best one

RESPOND IN THIS EXACT FORMAT (fill in all fields):

MATCHED_OPPORTUNITIES: [list CRM opportunity numbers like "1, 6" or write "NONE"]
CONFIDENCE: [write "high", "medium", or "low"]
REASONING: [explain your decision, referencing the business rules above]

Now analyze and respond:"""

        try:
            response = llm.invoke(prompt)
            
            # Parse LLM response
            matching_opps = []
            confidence = "low"
            reasoning = "Unable to parse LLM response"
            
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith("MATCHED_OPPORTUNITIES:"):
                    matched_str = line.split(":", 1)[1].strip()
                    if matched_str.upper() != "NONE":
                        # Extract numbers - these are CRM opportunity numbers
                        import re
                        numbers = re.findall(r'\d+', matched_str)
                        # Match by opportunity_number field
                        matched_numbers = [int(n) for n in numbers]
                        matching_opps = [opp for opp in opportunities if opp.get('opportunity_number') in matched_numbers]
                elif line.startswith("CONFIDENCE:"):
                    confidence = line.split(":", 1)[1].strip().lower()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
            
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
        Enhanced with renewal detection, CRM stage analysis, and urgency flags.
        
        Returns:
            Tuple of (matched_contracts, unmatched_contracts)
        """
        from datetime import datetime
        
        matched = []
        unmatched = []
        now = datetime.now().date()
        
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
            
            # Calculate days until renewal/expiration
            days_until_renewal = None
            renewal_urgency = "NONE"
            is_renewal = False
            
            if end_date != "Unknown":
                try:
                    end_date_obj = datetime.fromisoformat(end_date).date()
                    days_until_renewal = (end_date_obj - now).days
                    
                    if days_until_renewal < 0:
                        renewal_urgency = "EXPIRED"
                    elif days_until_renewal <= 30:
                        renewal_urgency = "CRITICAL"
                    elif days_until_renewal <= 90:
                        renewal_urgency = "HIGH"
                    elif days_until_renewal <= 180:
                        renewal_urgency = "MEDIUM"
                    else:
                        renewal_urgency = "LOW"
                except:
                    pass
            
            # Check if this is a renewal based on filename or title
            filename_lower = filename.lower()
            is_renewal = "renewal" in filename_lower or "renew" in filename_lower
            
            # Use LLM to find matching opportunities
            matching_opps, confidence, reasoning = self._match_with_llm(contract, opportunities)
            
            # Analyze CRM stage for matched opportunities
            crm_stage_analysis = None
            active_engagement = False
            
            if matching_opps:
                primary_opp = matching_opps[0]
                stage = primary_opp.get("stage", "Unknown")
                
                # CRM stage analysis: lost = failed, won = contract signed, anything else = active engagement
                if stage == "Lost":
                    crm_stage_analysis = "FAILED - Deal was lost"
                    active_engagement = False
                elif stage == "Won":
                    crm_stage_analysis = "CONTRACT SIGNED - Deal won"
                    active_engagement = False
                else:
                    crm_stage_analysis = f"ACTIVE ENGAGEMENT - Stage: {stage}"
                    active_engagement = True
                
                matched.append({
                    "contract": contract,
                    "contract_file": filename,
                    "contract_product": product,
                    "contract_value": contract_amount,
                    "contract_start": start_date,
                    "contract_end": end_date,
                    "contract_term": term_length,
                    "contract_status": status,
                    "is_renewal": is_renewal,
                    "days_until_renewal": days_until_renewal,
                    "renewal_urgency": renewal_urgency,
                    "opportunities": matching_opps,
                    "match_confidence": confidence,
                    "match_reasoning": reasoning,
                    # Add CRM details from first matching opportunity
                    "crm_opportunity": primary_opp.get("opportunity_name", "Unknown"),
                    "crm_owner": primary_opp.get("owner", "Unknown"),
                    "crm_stage": stage,
                    "crm_stage_analysis": crm_stage_analysis,
                    "active_engagement": active_engagement,
                    "crm_amount": primary_opp.get("amount", "Unknown"),
                    "crm_close_date": primary_opp.get("close_date", "Unknown"),
                    "crm_next_steps": primary_opp.get("next_steps", "None")
                })
            else:
                # CRITICAL: No CRM match found - ACTION REQUIRED NOW
                urgency_message = "CRITICAL: No CRM opportunity found - ACTION REQUIRED NOW!"
                if renewal_urgency == "EXPIRED" and days_until_renewal is not None:
                    urgency_message = f"URGENT: Contract EXPIRED {abs(days_until_renewal)} days ago with NO CRM tracking!"
                elif renewal_urgency == "CRITICAL" and days_until_renewal is not None:
                    urgency_message = f"URGENT: Contract expires in {days_until_renewal} days with NO CRM tracking!"
                elif renewal_urgency == "HIGH" and days_until_renewal is not None:
                    urgency_message = f"HIGH PRIORITY: Contract expires in {days_until_renewal} days with NO CRM tracking!"
                
                unmatched.append({
                    "contract": contract,
                    "contract_file": filename,
                    "contract_product": product,
                    "contract_value": contract_amount,
                    "contract_start": start_date,
                    "contract_end": end_date,
                    "contract_status": status,
                    "is_renewal": is_renewal,
                    "days_until_renewal": days_until_renewal,
                    "renewal_urgency": renewal_urgency,
                    "reason": f"No matching CRM opportunity found. {reasoning}",
                    "urgency_message": urgency_message,
                    "action_required": "CREATE CRM OPPORTUNITY IMMEDIATELY"
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
            seen_files = set()
            
            # Include active contracts
            active = portfolio_summary.get("active_contracts", [])
            for contract in active:
                file_path = contract.get("file_path", "")
                if file_path and file_path not in seen_files:
                    all_contracts.append(contract)
                    seen_files.add(file_path)
            
            # Include renewal candidates (skip if already added as active)
            renewal = portfolio_summary.get("renewal_candidates", [])
            for contract in renewal:
                file_path = contract.get("file_path", "")
                if file_path and file_path not in seen_files:
                    all_contracts.append(contract)
                    seen_files.add(file_path)
            
            # Include recently expired
            expired = portfolio_summary.get("recently_expired_contracts", [])
            for contract in expired:
                file_path = contract.get("file_path", "")
                if file_path and file_path not in seen_files:
                    all_contracts.append(contract)
                    seen_files.add(file_path)
            
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
            """Generate final matching report with enhanced renewal and CRM analysis"""
            matched = state.get("matched_contracts", [])
            unmatched = state.get("unmatched_contracts", [])
            
            # Count critical items
            expired_no_crm = sum(1 for u in unmatched if u.get("renewal_urgency") == "EXPIRED")
            critical_no_crm = sum(1 for u in unmatched if u.get("renewal_urgency") in ["EXPIRED", "CRITICAL"])
            active_engagement = sum(1 for m in matched if m.get("active_engagement", False))
            
            output_parts = [
                "=" * 80,
                "MATCHING AGENT - CONTRACT-CRM CORRELATION & RENEWAL ANALYSIS",
                "=" * 80,
                "",
                "SUMMARY:",
                f"  Total Matched: {len(matched)} (with CRM tracking)",
                f"  Total Unmatched: {len(unmatched)} (NO CRM TRACKING - ACTION REQUIRED)",
                f"  Active Engagements: {active_engagement}",
                f"  Expired without CRM: {expired_no_crm}",
                f"  Critical/Expired without CRM: {critical_no_crm}",
                ""
            ]
            
            if matched:
                output_parts.extend([
                    "=" * 80,
                    "MATCHED CONTRACTS (CRM Tracking Active):",
                    "=" * 80
                ])
                
                for match in matched:
                    contract = match["contract"]
                    product = match.get("contract_product", "Unknown")
                    opportunities = match["opportunities"]
                    end_date = match.get("contract_end", "Unknown")
                    status = match.get("contract_status", "unknown")
                    is_renewal = match.get("is_renewal", False)
                    days_until = match.get("days_until_renewal")
                    urgency = match.get("renewal_urgency", "NONE")
                    crm_stage = match.get("crm_stage_analysis", "Unknown")
                    active_eng = match.get("active_engagement", False)
                    
                    output_parts.extend([
                        "",
                        f"{product} - {contract.get('file_name', 'Unknown')}",
                        f"   Contract Status: {status.upper()}",
                        f"   End Date: {end_date}",
                    ])
                    
                    if days_until is not None:
                        if days_until < 0:
                            output_parts.append(f"   EXPIRED {abs(days_until)} days ago")
                        else:
                            output_parts.append(f"   {days_until} days until renewal")
                    
                    output_parts.extend([
                        f"   Renewal Urgency: {urgency}",
                        f"   {'RENEWAL CONTRACT' if is_renewal else 'Initial Contract'}",
                        "",
                        f"   CRM LINKAGE:",
                        f"   {'ACTIVE ENGAGEMENT' if active_eng else 'NOT ACTIVE'} - {crm_stage}",
                    ])
                    
                    for opp in opportunities:
                        stage = opp.get('stage', 'Unknown')
                        stage_status = "WON" if stage == "Won" else "LOST" if stage == "Lost" else "IN PROGRESS"
                        opp_num = opp.get('opportunity_number', '?')
                        output_parts.extend([
                            f"   [{stage_status}] CRM #{opp_num}: {opp.get('opportunity_name', 'Unknown')}",
                            f"      Owner: {opp.get('owner', 'Unknown')}",
                            f"      Stage: {stage}",
                            f"      Amount: {opp.get('amount', '$0')}",
                            f"      Next Steps: {opp.get('next_steps', 'None')}",
                        ])
            
            if unmatched:
                output_parts.extend([
                    "",
                    "=" * 80,
                    "UNMATCHED CONTRACTS - IMMEDIATE ACTION REQUIRED!",
                    "=" * 80,
                    "These contracts have NO CRM opportunity tracking.",
                    "Someone needs to act on these NOW!",
                    ""
                ])
                
                # Sort by urgency
                unmatched_sorted = sorted(unmatched, key=lambda x: {
                    "EXPIRED": 0, "CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4, "NONE": 5
                }.get(x.get("renewal_urgency", "NONE"), 5))
                
                for item in unmatched_sorted:
                    contract = item["contract"]
                    urgency = item.get("renewal_urgency", "NONE")
                    days_until = item.get("days_until_renewal")
                    urgency_msg = item.get("urgency_message", "")
                    
                    urgency_prefix = "[URGENT]" if urgency in ["EXPIRED", "CRITICAL"] else "[HIGH]" if urgency == "HIGH" else "[NORMAL]"
                    
                    output_parts.extend([
                        "",
                        f"{urgency_prefix} {item.get('contract_product', 'Unknown')} - {contract.get('file_name', 'Unknown')}",
                        f"   Contract Status: {item.get('contract_status', 'unknown').upper()}",
                        f"   End Date: {item.get('contract_end', 'Unknown')}",
                    ])
                    
                    if days_until is not None:
                        if days_until < 0:
                            output_parts.append(f"   EXPIRED {abs(days_until)} days ago")
                        else:
                            output_parts.append(f"   {days_until} days until renewal")
                    
                    output_parts.extend([
                        f"   Urgency Level: {urgency}",
                        f"   NO CRM OPPORTUNITY FOUND",
                        f"   {urgency_msg}",
                        f"   Action Required: {item.get('action_required', 'Create CRM opportunity')}",
                        f"   Why no match: {item.get('reason', 'Unknown')}"
                    ])
            
            output_parts.extend([
                "",
                "=" * 80,
                "KEY INSIGHTS:",
                "=" * 80,
                f"- {len(matched)} contracts have CRM tracking (someone is working on them)",
                f"- {len(unmatched)} contracts have NO CRM tracking (need immediate attention)",
                f"- {active_engagement} contracts have ACTIVE engagement (not lost/won)",
            ])
            
            if critical_no_crm > 0:
                output_parts.append(f"- {critical_no_crm} CRITICAL contracts without CRM - ACT NOW!")
            
            output_parts.extend([
                "",
                "NEXT STEPS:",
                "1. Review unmatched contracts and create CRM opportunities",
                "2. Contact owners of active engagements for status updates",
                "3. Prioritize expired/critical contracts for immediate outreach",
                "=" * 80
            ])
            
            final_output = "\n".join(output_parts)
            
            return {
                "final_output": final_output,
                "messages": ["Enhanced matching report with renewal analysis generated"]
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
