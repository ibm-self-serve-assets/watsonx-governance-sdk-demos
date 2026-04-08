# Enhanced Multi-Agent Workflow for Sales Assist Tool

## Overview

This document describes the enhanced multi-agent workflow that supports comprehensive seller queries with contract-CRM matching, pre-acquisition executive research, and conversational follow-ups.

## Primary Use Case

**IBM Seller Query:**
> "I'm a new seller at IBM. I recently got Confluent as a new customer and I want to understand what contracts are coming up for renewal. Are there any contracts that have already expired. Based on the CRM, Contracts and webscraped information can you then put a plan of next steps"

## Enhanced Agent Architecture

### 1. Supervisory Agent
**Role:** Orchestrates the entire workflow and manages agent execution sequence

**Workflow:**
1. Interprets seller intent
2. Calls Contract Agent to analyze portfolio
3. Calls Research Agent to get CRM and web data
4. Calls Matching Agent to correlate contracts with CRM
5. Calls Action Agent to generate recommendations
6. Presents comprehensive results to seller

### 2. Contract Agent (Enhanced)
**Role:** Analyzes contract portfolio with focus on renewals and expirations

**Key Features:**
- Processes all contracts in the portfolio
- Identifies contracts expiring in next 90 days
- Identifies contracts expired in last year
- Extracts key terms, amounts, and dates
- Returns structured portfolio summary

**Example Output:**
```
- Cognos expiring on 5/31/2026 for $1M over 3 years
- watsonx expiring on 7/31/2026 for $400K over 2 years
- watsonx expired on 1/31/2026 for $250K (needs renewal)
```

### 3. Research Agent (Enhanced)
**Role:** Enriches context with internal CRM data and external intelligence

**Internal Data (CRM):**
- Opportunity names and owners
- Dollar amounts and stages
- Next steps and action items
- Historical deal patterns

**External Data (Web Scraping via Tavily):**
- **Pre-acquisition executives:** CPO, CTO, CEO names
- **Revenue data:** Current and historical revenue, growth trends
- **Key announcements:** Product releases, strategic initiatives
- **Market signals:** Industry trends, competitive positioning

**Example Output:**
```
Executives (Pre-acquisition):
- CPO: Shaun Clowes
- CTO: Stephen Deasy
- CEO: Jay Kreps

Revenue:
- 2025: $1.167 Bn (up from $963.6M in 2024)
- Growth: 21% YoY
- Status: Strong growth trajectory
```

### 4. Matching Agent (NEW)
**Role:** Correlates contracts with CRM opportunities

**Matching Logic:**
- Matches by product name (Cognos, watsonx, etc.)
- Validates by dollar amounts
- Correlates by dates and timelines
- Enriches with CRM next steps

**Example Output:**
```
MATCHED: Cognos Contract ($1M, expires 5/31/2026)
├─ CRM Opportunity: "Confluent Cognos Renewal"
│  ├─ Owner: Kylie Brittz
│  ├─ Stage: Qualified
│  ├─ Amount: $1M
│  └─ Next Steps: "Quote for renewal being shared with team and discussing expansion"
└─ CRM Opportunity: "Cognos Expansion"
   ├─ Owner: Kylie Brittz
   ├─ Amount: $500K
   └─ Next Steps: "Meeting with CPO on 4/1/2026 to discuss possible expansion"

MATCHED: watsonx Contract ($250K, expired 1/31/2026)
└─ CRM Opportunity: "Confluent watsonx ESA"
   ├─ Owner: Anand Das
   ├─ Stage: Qualified
   ├─ Amount: $250K
   └─ Next Steps: "watsonx ESA expired working with team on size needed to renew"
```

### 5. Action Agent (Enhanced for Conversations)
**Role:** Generates recommendations and supports conversational follow-ups

**Initial Capabilities:**
- Risk assessment based on renewals/expirations
- Prioritized action plan
- Draft emails to executives
- CRM updates

**Conversational Capabilities (Enhanced):**
- Responds to seller follow-up questions
- Drafts customized emails based on context
- Updates CRM with new actions
- Tracks conversation history

**Example Conversation Flow:**

**Seller:** "I have reached out to them and I am now leading these 2 opportunities"

**Action Agent:** 
```
Great! In that case your next steps are:
1. Follow up with the CPO on the Cognos meeting scheduled for 4/1/2026
2. Address the watsonx contract that expired on 1/31/2026
3. Prepare for the watsonx renewal discussion for 7/31/2026
```

**Seller:** "Great can you draft a note to the CPO specifically on the watsonx contract that has expired asking if he has finalized sizing and say there will be a penalty of 5% if we don't get this done shortly"

**Action Agent:**
```
Subject: watsonx Contract Renewal - Sizing Update Needed

Hi Shaun,

It was great connecting with you. I just wanted to check in if you were able to get 
the sizing for the watsonx renewal finalized. Considering the contract expired on 
January 31st, I want to make sure we can get it in soon so I can make the exception 
not to charge the 5% penalty. 

Please let me know if it would be helpful to schedule a call and if you have any questions.

Regards,
[Seller Name]
```

**Seller:** "This is great, I just sent that to Shaun, can you please update CRM accordingly"

**Action Agent:**
```
✓ Updated CRM
  - Opportunity: "Confluent watsonx ESA"
  - Next Steps: "Emailed CPO on 4/10/2026 asking for sizing update to finalize renewal"
  - Owner: [Seller Name]
  - Status: Awaiting Response
```

## Complete Workflow Example

### Initial Query
**Seller:** "I'm a new seller at IBM. I recently got Confluent as a new customer and I want to understand what contracts are coming up for renewal. Are there any contracts that have already expired. Based on the CRM, Contracts and webscraped information can you then put a plan of next steps"

### System Response

**EXECUTIVE SUMMARY**
```
Partner: Confluent
Pre-acquisition Leadership:
- CPO: Shaun Clowes
- CTO: Stephen Deasy  
- CEO: Jay Kreps

Financial Performance:
- 2025 Revenue: $1.167 Bn (↑21% from $963.6M)
- Status: Strong growth trajectory

CONTRACT PORTFOLIO STATUS
Total Contracts: 3 key agreements
- 1 expiring soon (Cognos - 5/31/2026)
- 1 expiring in 90 days (watsonx - 7/31/2026)
- 1 expired (watsonx - 1/31/2026)
```

**CONTRACT-CRM CORRELATION**
```
1. Cognos Contract ($1M, expires 5/31/2026)
   ├─ Renewal Opportunity: $1M (Kylie Brittz)
   │  └─ Next: Quote being shared with team
   └─ Expansion Opportunity: $500K (Kylie Brittz)
      └─ Next: Meeting with CPO on 4/1/2026

2. watsonx Contract ($400K, expires 7/31/2026)
   └─ Renewal Opportunity: $400K (Kylie Brittz)
      └─ Next: Meeting with CPO to discuss larger partnership

3. watsonx Contract ($250K, expired 1/31/2026) ⚠️
   └─ Renewal Opportunity: $250K (Anand Das)
      └─ Next: Working with team on sizing needed to renew
```

**RECOMMENDED NEXT STEPS**
```
Priority 1: Address Expired Contract
- watsonx contract expired 1/31/2026 - needs immediate attention
- Contact: Anand Das (current owner)
- Action: Finalize sizing and expedite renewal

Priority 2: Cognos Renewal & Expansion
- Contract expires 5/31/2026 ($1M renewal + $500K expansion)
- Contact: Kylie Brittz (current owner)
- Action: Support CPO meeting on 4/1/2026

Priority 3: watsonx Renewal Planning
- Contract expires 7/31/2026 ($400K)
- Contact: Kylie Brittz (current owner)
- Action: Prepare for larger partnership discussion

Recommendation: Reach out to Kylie Brittz and Anand Das to coordinate handoff
```

## Technical Implementation

### File Structure
```
multi-agent-demo/
├── supervisory_agent.py      # Orchestrates workflow
├── contract_agent.py          # Contract analysis
├── research_agent.py          # CRM + web research (enhanced)
├── matching_agent.py          # Contract-CRM correlation (NEW)
├── action_agent.py            # Recommendations + conversations (enhanced)
└── multiagent_workflow.ipynb # Demo notebook
```

### Key Enhancements

1. **Matching Agent** (`matching_agent.py`)
   - New agent for contract-CRM correlation
   - Intelligent matching by product, amount, date
   - Enriches contracts with CRM next steps

2. **Research Agent** (`research_agent.py`)
   - New tools for pre-acquisition executive data
   - Revenue and growth analysis
   - Key announcements tracking
   - Enhanced Tavily integration

3. **Action Agent** (`action_agent.py`)
   - Conversational state management
   - Context-aware email drafting
   - Dynamic CRM updates
   - Multi-turn dialogue support

4. **Supervisory Agent** (`supervisory_agent.py`)
   - Enhanced orchestration logic
   - Matching agent integration
   - Conversation flow management

## Usage

### Basic Workflow
```python
from supervisory_agent import SupervisoryAgent

supervisor = SupervisoryAgent(
    apikey=os.getenv("WATSONX_APIKEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID")
)

result = supervisor.run(
    seller_query="I'm a new seller at IBM. I recently got Confluent as a new customer...",
    partner_name="Confluent"
)

print(result["final_result"])
```

### Conversational Follow-up
```python
# After initial workflow
action_agent = ActionAgent(...)

# Seller follow-up
response = action_agent.handle_conversation(
    seller_message="Can you draft an email to the CPO about the expired contract?",
    context=previous_result
)

print(response["draft_email"])
```

## Benefits

1. **Comprehensive Context:** Combines contracts, CRM, and web intelligence
2. **Intelligent Matching:** Automatically correlates contracts with opportunities
3. **Executive Intelligence:** Provides pre-acquisition leadership information
4. **Conversational:** Supports natural follow-up questions and actions
5. **Actionable:** Generates specific next steps with owner assignments
6. **Automated:** Updates CRM automatically based on actions

## Future Enhancements

1. **Real-time CRM Integration:** Direct API connections to Salesforce/Dynamics
2. **Email Automation:** Send emails directly from the tool
3. **Calendar Integration:** Schedule meetings automatically
4. **Sentiment Analysis:** Analyze email responses and meeting notes
5. **Predictive Analytics:** Forecast renewal likelihood and expansion opportunities