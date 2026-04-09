"""
Cache Contract Processing Results with Rate Limiting

This script:
1. Extracts text from contracts (no LLM)
2. Uses regex to extract structured fields (amount, products, dates)
3. Caches everything so you don't re-process

Usage:
    python cache_contracts.py                    # Process and cache all contracts
    python cache_contracts.py --load             # View cached data
    python cache_contracts.py --text-only        # Extract text only (no structured fields)
"""

import os
import json
import time
import re
from datetime import datetime
from dotenv import load_dotenv
from docx import Document
import PyPDF2
from tqdm import tqdm

load_dotenv()

CACHE_FILE = "contracts_cache.json"
PROGRESS_FILE = "contracts_progress.json"
BATCH_SIZE = 1  # Process 1 contract at a time
DELAY_BETWEEN_CONTRACTS = 15  # Wait 15 seconds between LLM calls


def extract_text_from_docx(file_path):
    """Extract text from DOCX without using LLM"""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"  Error extracting from {file_path}: {e}")
        return None


def extract_text_from_pdf(file_path):
    """Extract text from PDF without using LLM"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"  Error extracting from {file_path}: {e}")
        return None


def extract_structured_fields(text):
    """
    Extract structured fields from contract text using regex (no LLM).
    This is a fast fallback that works without API calls.
    """
    structured = {}
    
    # Extract amount - look for "committed order value of $X"
    amount_match = re.search(r'committed order value of \$[\d,]+\.?\d*', text, re.IGNORECASE)
    if amount_match:
        structured['amount'] = amount_match.group(0).split('of ')[-1]
    else:
        # Fallback: find any dollar amount
        amount_match = re.search(r'\$[\d,]+\.?\d*', text)
        structured['amount'] = amount_match.group(0) if amount_match else "Not specified"
    
    # Extract products
    products = []
    if "watsonx" in text.lower():
        if "orchestrate" in text.lower():
            products.append("watsonx Orchestrate")
        if "governance" in text.lower():
            products.append("watsonx.governance")
        if "watsonx as a service" in text.lower() or "watsonx.ai" in text.lower():
            products.append("watsonx as a Service")
        if not products:
            products.append("watsonx ESA")
    if "cognos" in text.lower():
        products.append("Cognos")
    structured['products'] = products if products else ["Not specified"]
    
    # Extract dates
    effective_date_match = re.search(r'effective as (\w+ \d{1,2}, \d{4})', text, re.IGNORECASE)
    structured['start_date'] = effective_date_match.group(1) if effective_date_match else "Not specified"
    
    # Extract term length
    term_match = re.search(r'term of this TD will be (\w+) \((\d+)\) years? from the Effective Date', text, re.IGNORECASE)
    if term_match:
        term_years = term_match.group(2)
        structured['term_length'] = f"{term_years} year(s)"
        structured['end_date'] = f"{term_years} year(s) from effective date"
    else:
        structured['term_length'] = "Not specified"
        structured['end_date'] = "Not specified"
    
    # Extract parties
    parties = []
    if "confluent" in text.lower():
        parties.append("Confluent")
    if "ibm" in text.lower():
        parties.append("IBM")
    structured['parties'] = parties if parties else ["Not specified"]
    
    # Contract type
    structured['contract_type'] = "ESA" if "ESA" in text else "Sales Agreement"
    
    return structured


def save_progress(processed_contracts):
    """Save progress incrementally"""
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "processed_contracts": processed_contracts
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)
    print(f"  ✓ Progress saved ({len(processed_contracts)} contracts)")


def load_progress():
    """Load previous progress"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('processed_contracts', [])
    return []


def process_contracts_batch(partner_name="Confluent", text_only=False):
    """
    Process contracts in batches.
    
    Args:
        partner_name: Partner to process contracts for
        text_only: If True, only extract text (no structured fields)
    """
    
    print(f"\n{'='*80}")
    if text_only:
        print("CONTRACT TEXT EXTRACTION (No LLM)")
    else:
        print("CONTRACT PROCESSING (Text + Structured Fields)")
    print(f"{'='*80}")
    print(f"Partner: {partner_name}")
    if not text_only:
        print(f"Extracting: Text + Amount + Products + Dates")
    print(f"{'='*80}\n")
    
    # Find contract files
    docs_dir = "docs"
    contract_files = []
    for file in os.listdir(docs_dir):
        if file.startswith(f"{partner_name}_") and (file.endswith('.docx') or file.endswith('.pdf')):
            contract_files.append(os.path.join(docs_dir, file))
    
    print(f"Found {len(contract_files)} contract files\n")
    
    # Load previous progress
    processed_contracts = load_progress()
    processed_paths = [c['file_path'] for c in processed_contracts]
    
    if processed_contracts:
        print(f"Resuming from previous run ({len(processed_contracts)} already processed)\n")
    
    # Process remaining contracts with progress bar
    remaining_files = [f for f in contract_files if f not in processed_paths]
    
    if not remaining_files:
        print("All contracts already processed!")
        return processed_contracts
    
    with tqdm(total=len(remaining_files), desc="Processing contracts", unit="contract") as pbar:
        for file_path in remaining_files:
            pbar.set_description(f"Processing {os.path.basename(file_path)}")
            
            # Extract text based on file type
            if file_path.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            elif file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            else:
                text = None
            
            if text:
                contract_data = {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "extracted_text": text,
                    "text_length": len(text),
                    "processed_at": datetime.now().isoformat()
                }
                
                # Extract structured fields if not text-only mode
                if not text_only:
                    structured = extract_structured_fields(text)
                    contract_data["structured_summary"] = structured
                    pbar.set_postfix({"amount": structured.get('amount', 'N/A')})
                else:
                    pbar.set_postfix({"chars": f"{len(text):,}"})
                
                processed_contracts.append(contract_data)
                
                # Save progress after each contract
                save_progress(processed_contracts)
                
                # Wait before next contract (except for last one)
                if file_path != remaining_files[-1]:
                    pbar.set_description(f"Waiting {DELAY_BETWEEN_CONTRACTS}s...")
                    time.sleep(DELAY_BETWEEN_CONTRACTS)
            else:
                pbar.set_postfix({"status": "FAILED"})
            
            pbar.update(1)
    
    # Create final cache
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "partner_name": partner_name,
        "total_contracts": len(processed_contracts),
        "contracts": processed_contracts
    }
    
    # Save final cache
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✓ ALL CONTRACTS PROCESSED AND CACHED")
    print(f"{'='*80}")
    print(f"Total contracts: {len(processed_contracts)}")
    print(f"Cache file: {CACHE_FILE}")
    print(f"Total text extracted: {sum(c['text_length'] for c in processed_contracts):,} characters")
    
    # Clean up progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print(f"✓ Cleaned up progress file")
    
    return cache_data


def load_cached_contracts():
    """Load cached contract data"""
    if not os.path.exists(CACHE_FILE):
        print(f"No cache found. Run 'python cache_contracts.py' first.")
        return None
    
    with open(CACHE_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {data['total_contracts']} cached contracts")
    return data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and cache contract data")
    parser.add_argument("--partner", default="Confluent", help="Partner name")
    parser.add_argument("--load", action="store_true", help="Load and display cache")
    parser.add_argument("--force", action="store_true", help="Force re-process (delete progress)")
    parser.add_argument("--text-only", action="store_true", help="Extract text only (no structured fields)")
    
    args = parser.parse_args()
    
    if args.load:
        data = load_cached_contracts()
        if data:
            print(f"\nCached at: {data['timestamp']}")
            print(f"Partner: {data['partner_name']}")
            print(f"Contracts: {data['total_contracts']}")
            for contract in data['contracts']:
                print(f"  - {contract['file_name']}: {contract['text_length']:,} chars")
                if 'structured_summary' in contract:
                    print(f"    Amount: {contract['structured_summary'].get('amount', 'N/A')}")
                    print(f"    Products: {', '.join(contract['structured_summary'].get('products', []))}")
        return
    
    if args.force and os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("✓ Deleted progress file - starting fresh")
    
    try:
        process_contracts_batch(partner_name=args.partner, text_only=args.text_only)
        
        print(f"\n{'='*80}")
        print("USAGE IN YOUR CODE")
        print(f"{'='*80}")
        print("\n```python")
        print("from cache_contracts import load_cached_contracts")
        print("")
        print("# Load all contract text")
        print("data = load_cached_contracts()")
        print("for contract in data['contracts']:")
        print("    print(f\"{contract['file_name']}: {contract['text_length']} chars\")")
        print("    text = contract['extracted_text']")
        print("    if 'structured_summary' in contract:")
        print("        print(f\"Amount: {contract['structured_summary']['amount']}\")")
        print("```")
        
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted - Progress saved!")
        print(f"Run 'python cache_contracts.py' again to resume")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Progress has been saved. Run again to resume.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# Made with Bob
