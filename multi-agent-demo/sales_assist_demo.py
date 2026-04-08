"""
Sales Assist Tool - Complete Workflow Demo

This demo shows the complete workflow:
1. Seller uploads/specifies a contract (ESA)
2. Supervisory Agent orchestrates:
   - Contract Agent: Reads and analyzes contract
   - Research Agent: Enriches partner context
   - Action Agent: Determines next best action
3. Seller receives actionable guidance with draft email and CRM updates

Usage:
    python sales_assist_demo.py
"""

import os
from dotenv import load_dotenv
from supervisory_agent import SupervisoryAgent
import sys

# Load environment variables
load_dotenv()


def print_banner(text: str, char: str = "="):
    """Print a formatted banner"""
    print(f"\n{char * 80}")
    print(text.center(80))
    print(f"{char * 80}\n")


def demo_workflow_1():
    """
    Demo Workflow 1: Signed ESA from Partner
    
    Scenario: Seller just received a signed ESA from IBM and wants to know next steps
    """
    print_banner("SALES ASSIST TOOL - DEMO WORKFLOW 1", "=")
    
    print("SCENARIO:")
    print("-" * 80)
    print("A seller opens the sales-assist tool to determine the next best step")
    print("for an active opportunity. They have just received a signed ESA from IBM.")
    print("-" * 80)
    
    # Seller query (natural language)
    seller_query = "I just received a signed ESA from IBM. What should I do next?"
    
    # Contract file path (in demo, we read from filesystem)
    contract_file_path = "docs/Confluent_IBM-1.30.2024.docx"
    
    # Partner name (optional - will be extracted from contract)
    partner_name = "IBM"
    
    print(f"\nSeller Query: \"{seller_query}\"")
    print(f"Contract File: {contract_file_path}")
    print(f"Partner: {partner_name}")
    
    # Check if required environment variables are set
    if not os.getenv("WATSONX_APIKEY") or not os.getenv("WATSONX_PROJECT_ID"):
        print("\n" + "!" * 80)
        print("ERROR: Missing required environment variables")
        print("!" * 80)
        print("\nPlease set the following environment variables:")
        print("  - WATSONX_APIKEY")
        print("  - WATSONX_PROJECT_ID")
        print("  - TAVILY_API_KEY (for web search)")
        print("\nSet them in a .env file or export them in the shell.")
        return
    
    # Check if contract file exists
    if not os.path.exists(contract_file_path):
        print("\n" + "!" * 80)
        print(f"ERROR: Contract file not found: {contract_file_path}")
        print("!" * 80)
        print("\nPlease ensure the contract file exists at the specified path.")
        return
    
    try:
        # Initialize Supervisory Agent
        print("\nInitializing Supervisory Agent...")
        supervisor = SupervisoryAgent(
            apikey=os.getenv("WATSONX_APIKEY"),
            project_id=os.getenv("WATSONX_PROJECT_ID")
        )
        
        print("✓ Supervisory Agent initialized")
        print("\nStarting workflow execution...")
        print("This will execute: Contract Agent → Research Agent → Action Agent")
        
        # Run the complete workflow
        result = supervisor.run(
            seller_query=seller_query,
            contract_file_path=contract_file_path,
            partner_name=partner_name
        )
        
        # Display final result
        print_banner("WORKFLOW RESULTS", "=")
        print(result["final_result"])
        
        # Display workflow messages
        print_banner("WORKFLOW EXECUTION LOG", "-")
        for i, msg in enumerate(result.get("messages", []), 1):
            print(f"{i}. {msg}")
        
        print_banner("DEMO COMPLETE", "=")
        print("The seller can now:")
        print("  1. Review the recommended next step")
        print("  2. Edit the draft email if needed")
        print("  3. Execute the action directly from the tool")
        print("  4. CRM is automatically updated with next steps")
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("ERROR during workflow execution")
        print("!" * 80)
        print(f"\n{str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()


def demo_workflow_2():
    """
    Demo Workflow 2: Different contract scenario
    
    Scenario: Seller has a different IBM contract and wants guidance
    """
    print_banner("SALES ASSIST TOOL - DEMO WORKFLOW 2", "=")
    
    print("SCENARIO:")
    print("-" * 80)
    print("A seller has received a different contract from IBM (dated 3/29/2024)")
    print("and wants to understand the next best action.")
    print("-" * 80)
    
    seller_query = "I have a new contract from IBM dated March 2024. What are the recommended next steps?"
    contract_file_path = "docs/Confluent_IBM-3.29.2024.docx"
    partner_name = "IBM"
    
    print(f"\nSeller Query: \"{seller_query}\"")
    print(f"Contract File: {contract_file_path}")
    print(f"Partner: {partner_name}")
    
    # Check prerequisites
    if not os.getenv("WATSONX_APIKEY") or not os.getenv("WATSONX_PROJECT_ID"):
        print("\n" + "!" * 80)
        print("ERROR: Missing required environment variables")
        print("!" * 80)
        return
    
    if not os.path.exists(contract_file_path):
        print("\n" + "!" * 80)
        print(f"ERROR: Contract file not found: {contract_file_path}")
        print("!" * 80)
        return
    
    try:
        supervisor = SupervisoryAgent(
            apikey=os.getenv("WATSONX_APIKEY"),
            project_id=os.getenv("WATSONX_PROJECT_ID")
        )
        
        print("\nExecuting workflow...")
        result = supervisor.run(
            seller_query=seller_query,
            contract_file_path=contract_file_path,
            partner_name=partner_name
        )
        
        print_banner("WORKFLOW RESULTS", "=")
        print(result["final_result"])
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("ERROR during workflow execution")
        print("!" * 80)
        print(f"\n{str(e)}")


def interactive_demo():
    """
    Interactive demo - allows user to input their own query
    """
    print_banner("SALES ASSIST TOOL - INTERACTIVE MODE", "=")
    
    print("This interactive mode allows you to test the workflow with custom inputs.")
    print("\nAvailable contract files:")
    print("  1. docs/Confluent_IBM-1.30.2024.docx")
    print("  2. docs/Confluent_IBM-3.29.2024.docx")
    print("  3. docs/Confluent_IBM-5.30.2023.docx")
    print("  4. docs/Confluent_IBM-1.30.2025.docx")
    
    # Get user input
    print("\n" + "-" * 80)
    seller_query = input("Enter query (or press Enter for default): ").strip()
    if not seller_query:
        seller_query = "I just received a signed ESA. What should I do next?"
    
    contract_file = input("Enter contract file path (or press Enter for default): ").strip()
    if not contract_file:
        contract_file = "docs/Confluent_IBM-1.30.2024.docx"
    
    partner_name = input("Enter partner name (or press Enter to auto-detect): ").strip()
    if not partner_name:
        partner_name = None
    
    print("-" * 80)
    
    # Check prerequisites
    if not os.getenv("WATSONX_APIKEY") or not os.getenv("WATSONX_PROJECT_ID"):
        print("\n" + "!" * 80)
        print("ERROR: Missing required environment variables")
        print("!" * 80)
        return
    
    if not os.path.exists(contract_file):
        print("\n" + "!" * 80)
        print(f"ERROR: Contract file not found: {contract_file}")
        print("!" * 80)
        return
    
    try:
        supervisor = SupervisoryAgent(
            apikey=os.getenv("WATSONX_APIKEY"),
            project_id=os.getenv("WATSONX_PROJECT_ID")
        )
        
        print("\nExecuting workflow...")
        result = supervisor.run(
            seller_query=seller_query,
            contract_file_path=contract_file,
            partner_name=partner_name
        )
        
        print_banner("WORKFLOW RESULTS", "=")
        print(result["final_result"])
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("ERROR during workflow execution")
        print("!" * 80)
        print(f"\n{str(e)}")


def main():
    """Main demo function"""
    print_banner("SALES ASSIST TOOL - MULTI-AGENT WORKFLOW DEMO", "█")
    
    print("This demo showcases the complete sales assist workflow:")
    print("\n1. Contract Agent - Reads and analyzes ESA/contracts using OCR")
    print("2. Research Agent - Enriches partner context with internal + external data")
    print("3. Action Agent - Determines next best action based on historical patterns")
    print("4. Supervisory Agent - Orchestrates all agents and presents results")
    
    print("\n" + "=" * 80)
    print("DEMO OPTIONS")
    print("=" * 80)
    print("\n1. Demo Workflow 1: Signed ESA from IBM (1/30/2024)")
    print("2. Demo Workflow 2: Different IBM contract (3/29/2024)")
    print("3. Interactive Mode: Custom query and contract")
    print("4. Exit")
    
    choice = input("\nSelect demo option (1-4): ").strip()
    
    if choice == "1":
        demo_workflow_1()
    elif choice == "2":
        demo_workflow_2()
    elif choice == "3":
        interactive_demo()
    elif choice == "4":
        print("\nExiting demo. Thank you!")
        sys.exit(0)
    else:
        print("\nInvalid choice. Running default demo (Workflow 1)...")
        demo_workflow_1()


if __name__ == "__main__":
    main()

