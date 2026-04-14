#!/usr/bin/env python3
"""
Test the full workflow to verify LLM parsing fix
"""

import os
from dotenv import load_dotenv
from supervisory_agent import SupervisoryAgent

# Load environment
load_dotenv()

def main():
    print("=" * 80)
    print("TESTING FULL WORKFLOW WITH LLM PARSING FIX")
    print("=" * 80)
    
    # Initialize supervisory agent
    print("\nInitializing Supervisory Agent...")
    supervisor = SupervisoryAgent(
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )
    
    # Run workflow with a question that triggers matching
    question = "Which contracts need renewal attention and do we have CRM opportunities for them?"
    
    print(f"\nRunning workflow with question: '{question}'")
    print("=" * 80)
    
    result = supervisor.run(question)
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    
    # Check if July 2024 contract was mentioned in the output
    final_answer = result.get('final_answer', '')
    
    if 'Confluent_IBM-7.31.2024.docx' in final_answer or 'July 2024' in final_answer:
        print("\n✅ July 2024 contract was processed in the workflow")
        
        # Check if it was matched or unmatched
        if 'matched' in final_answer.lower() and '7.31.2024' in final_answer:
            print("✅ July 2024 contract appears to be MATCHED")
        elif 'unmatched' in final_answer.lower() and '7.31.2024' in final_answer:
            print("❌ July 2024 contract appears to be UNMATCHED")
    else:
        print("\n⚠️  July 2024 contract not clearly mentioned in output")
    
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(final_answer)
    
    return 0

if __name__ == "__main__":
    main()

# Made with Bob
