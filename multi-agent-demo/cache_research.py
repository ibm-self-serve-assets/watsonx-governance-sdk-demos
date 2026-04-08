"""
Cache Research Agent Results

This script pre-runs the Research Agent for a partner and caches the results
to avoid slow Tavily API calls and LLM processing during demos.

Usage:
    python cache_research.py
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from research_agent import research_partner

load_dotenv()

RESEARCH_CACHE_FILE = "research_cache.json"


def cache_partner_research(partner_name="Confluent"):
    """Cache research results for a partner"""
    
    print(f"\n{'='*80}")
    print("CACHING RESEARCH AGENT RESULTS")
    print(f"{'='*80}")
    print(f"Partner: {partner_name}")
    print(f"This will take 1-2 minutes (Tavily API + LLM calls)")
    print(f"{'='*80}\n")
    
    try:
        print(f"Running Research Agent for {partner_name}...")
        profile = research_partner(partner_name)
        
        # Save to cache
        cache_data = {
            "partner_name": partner_name,
            "timestamp": datetime.now().isoformat(),
            "profile": profile
        }
        
        with open(RESEARCH_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"\n✓ Research results cached successfully!")
        print(f"  Maturity Level: {profile.get('maturity_level', 'Unknown')}")
        print(f"  Sales Velocity: {profile.get('sales_velocity', 'Unknown')}")
        print(f"  Deal Blockers: {len(profile.get('deal_blockers', []))}")
        print(f"\nCache saved to: {RESEARCH_CACHE_FILE}")
        
    except Exception as e:
        print(f"\n✗ Error caching research: {str(e)}")
        print("You may need to wait for API rate limits to reset")


def load_cached_research(partner_name="Confluent"):
    """Load cached research results"""
    if not os.path.exists(RESEARCH_CACHE_FILE):
        return None
    
    try:
        with open(RESEARCH_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)
        
        # Check if it's for the right partner
        if cache_data.get('partner_name') == partner_name:
            print(f"✓ Loaded cached research for {partner_name}")
            print(f"  Cached at: {cache_data.get('timestamp')}")
            return cache_data.get('profile')
        else:
            return None
    except Exception as e:
        print(f"Error loading research cache: {e}")
        return None


if __name__ == "__main__":
    cache_partner_research("Confluent")
