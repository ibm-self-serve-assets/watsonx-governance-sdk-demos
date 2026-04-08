# Sales Assist Tool - Multi-Agent Workflow

A sophisticated sales assistance system that uses multiple AI agents to analyze contracts, research partners, and recommend next best actions for sales opportunities.

## Overview

This system implements a multi-agent workflow orchestrated by a supervisory agent that coordinates three specialized agents:

1. **Contract Agent** - Reads and analyzes ESA/contracts using OCR and NLP
2. **Research Agent** - Enriches partner context with internal CRM data and external web research
3. **Action Agent** - Determines next best actions based on historical patterns and risk assessment

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Supervisory Agent                         │
│  (Orchestrates workflow and aggregates context)             │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Contract   │   │   Research   │   │    Action    │
│    Agent     │──▶│    Agent     │──▶│    Agent     │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
   OCR + NLP         CRM + Web Search    Historical Analysis
   Vector DB         Partner Profile     Risk Assessment
   Metadata          External Signals    Next Best Action
```

## Workflow

### Granular Workflow Steps

1. **Seller Query** (Natural Language)
   - Seller: "I just received a signed ESA from Partner X. What should I do next?"

2. **Supervisory Agent Initializes**
   - Interprets intent: Contract uploaded → determine next best sales action
   - Identifies required agents: Contract, Research, Action

3. **Contract Agent Execution**
   - Reads document from file path (read-only access)
   - Performs OCR + text extraction
   - Extracts metadata: parties, dates, terms, obligations, milestones
   - Ingests into vector database for semantic search
   - Returns structured contract summary

4. **Research Agent Execution**
   - Retrieves internal sales data from CRM (Excel)
     - Prior opportunities, stage history, stakeholders
     - Historical sales velocity, deal blockers
   - Retrieves external context via Tavily web search
     - Partner background, recent announcements
     - Technology alignment, market signals
   - Returns enriched partner profile

5. **Action Agent Execution**
   - Analyzes historical patterns from CRM
     - Successful post-ESA actions
     - Win rates and deal patterns
   - Assesses risk based on:
     - Partner maturity level
     - Historical deal blockers
     - Contract complexity
   - Determines recommended next step
   - Creates artifacts:
     - CRM update with next steps
     - Draft follow-up email

6. **Final Result Presentation**
   - Displays next best step (clear, ranked)
   - Shows draft follow-up email
   - Confirms CRM updated
   - Seller can execute directly from tool

## Files

### Core Agent Files

- **`contract_agent.py`** - Contract ingestion and analysis agent
  - OCR and text extraction from DOCX, PDF, Excel
  - Metadata extraction (parties, dates, terms)
  - Vector database ingestion (Chroma)
  - Semantic search capabilities

- **`research_agent.py`** - Partner intelligence agent
  - Internal CRM data retrieval
  - Partner maturity analysis
  - External web research (Tavily)
  - Technology alignment assessment

- **`action_agent.py`** - Next best action determination agent
  - Historical pattern analysis
  - Risk assessment
  - Action recommendation
  - Artifact creation (CRM updates, emails)

- **`supervisory_agent.py`** - Orchestration agent
  - Intent interpretation
  - Agent coordination
  - Context aggregation
  - Final result synthesis

### Demo and Documentation

- **`sales_assist_demo.py`** - Interactive demo script
- **`SALES_ASSIST_README.md`** - This file

### Supporting Files

- **`docs/Confluent_IBM-*.docx`** - Sample contract files
- **`docs/Confluent Sales Cloud Infor.xlsx`** - Sample CRM data
- **`docs/ScenarioActions.pdf`** - Workflow specification

## Installation

### Prerequisites

```bash
# Python 3.9+
python --version

# Install required packages
pip install -r requirements.txt
```

### Required Packages

```
langgraph
langchain-ibm
langchain-core
langchain-community
langchain-tavily
python-docx
openpyxl
pandas
chromadb
python-dotenv
```

### Environment Variables

Create a `.env` file with:

```bash
# IBM Watsonx credentials
WATSONX_APIKEY=your_watsonx_api_key
WATSONX_PROJECT_ID=your_project_id

# Tavily API key (for web search)
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

### Quick Start

```bash
# Run the demo
python sales_assist_demo.py
```

### Demo Options

1. **Demo Workflow 1**: Signed ESA from IBM (1/30/2024)
2. **Demo Workflow 2**: Different IBM contract (3/29/2024)
3. **Interactive Mode**: Custom query and contract

### Programmatic Usage

```python
from supervisory_agent import SupervisoryAgent
import os

# Initialize
supervisor = SupervisoryAgent(
    apikey=os.getenv("WATSONX_APIKEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID")
)

# Run workflow
result = supervisor.run(
    seller_query="I just received a signed ESA from IBM. What should I do next?",
    contract_file_path="docs/Confluent_IBM-1.30.2024.docx",
    partner_name="IBM"  # Optional - auto-detected if not provided
)

# Access results
print(result["final_result"])
print(result["action_recommendation"])
print(result["partner_profile"])
```

### Individual Agent Usage

#### Contract Agent

```python
from contract_agent import ContractAgent

agent = ContractAgent(
    apikey=os.getenv("WATSONX_APIKEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID")
)

result = agent.run("docs/Confluent_IBM-1.30.2024.docx")
print(result["generated_text"])

# Query contracts
results = agent.query_contracts("What are the payment terms?")
```

#### Research Agent

```python
from research_agent import research_partner

profile = research_partner("IBM")
print(f"Maturity: {profile['maturity_level']}")
print(f"Sales Velocity: {profile['sales_velocity']}")
```

#### Action Agent

```python
from action_agent import ActionAgent

agent = ActionAgent(
    apikey=os.getenv("WATSONX_APIKEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID")
)

result = agent.run(contract_summary, partner_profile)
print(result["final_output"])
```

## Key Features

### Contract Agent
- Multi-format support (DOCX, PDF, Excel, TXT)
- OCR and text extraction
- Metadata extraction and normalization
- Vector database ingestion (Chroma)
- Semantic search capabilities
- Read-only file access

### Research Agent
- Internal CRM data retrieval (Excel)
- Partner maturity analysis
- Sales velocity calculation
- Deal blocker identification
- External web research (Tavily)
- Technology alignment assessment
- Comprehensive partner profile synthesis

### Action Agent
- Historical pattern analysis
- Risk assessment (0-100 score)
- Next best action determination
- CRM update generation
- Draft email creation
- Success criteria definition

### Supervisory Agent
- Natural language intent interpretation
- Multi-agent orchestration
- Context aggregation
- Error handling and recovery
- Comprehensive result synthesis

## Output Example

```
================================================================================
                    SALES ASSIST TOOL - WORKFLOW COMPLETE
================================================================================

Original Query: I just received a signed ESA from IBM. What should I do next?
Timestamp: 2024-04-01 10:30:00

================================================================================
                            EXECUTIVE SUMMARY
================================================================================

Partner: IBM
Maturity Level: Strategic Partner
Sales Velocity: High

RISK ASSESSMENT:
  Risk Level: Low
  Risk Score: 15/100

================================================================================
                        RECOMMENDED NEXT STEP
================================================================================

ACTION: Schedule technical onboarding session within 5 business days
PRIORITY: High
TIMELINE: This Week
RATIONALE: Based on 15 similar won deals, immediate onboarding scheduling 
           correlates with 85% faster time-to-value and higher customer 
           satisfaction scores.
SUCCESS_CRITERIA: Onboarding session scheduled with technical stakeholders
OWNER: Sales Engineer + Account Executive

================================================================================
                        DRAFT FOLLOW-UP EMAIL
================================================================================

Subject: Next Steps: IBM ESA - Technical Onboarding

Dear [Stakeholder],

Thank you for signing the Enterprise Subscription Agreement. We're excited 
to begin our partnership and help you achieve your data streaming goals.

As the next step, I'd like to schedule a technical onboarding session with 
your team. This session will cover:
- Platform setup and configuration
- Best practices for your use cases
- Integration with your existing technology stack
- Q&A with our solutions architects

Could you please share your availability for a 90-minute session this week?

Best regards,
[Your Name]

================================================================================
                            CRM UPDATE (Demo)
================================================================================

{
  "opportunity_name": "IBM - Post-ESA Follow-up",
  "stage": "Onboarding",
  "next_step": "Schedule technical onboarding session",
  "owner": "Sales Team",
  "due_date": "2024-04-04",
  "priority": "Low"
}

================================================================================
                            WORKFLOW COMPLETE
================================================================================

✓ Contract analyzed and ingested
✓ Partner research completed
✓ Next best action determined
✓ Artifacts created (CRM update, draft email)

Seller can now execute the recommended action directly from this tool.
================================================================================
```

## Technical Details

### LangGraph Workflow

Each agent is implemented as a LangGraph StateGraph with nodes representing processing steps:

```python
# Example: Contract Agent Graph
graph = StateGraph(ContractState)
graph.add_node("read_document", read_document_node)
graph.add_node("extract_metadata", extract_metadata_node)
graph.add_node("normalize_structure", normalize_and_structure_node)
graph.add_node("ingest_vector_db", ingest_to_vector_db_node)
graph.add_node("generate_summary", generate_summary_node)

graph.add_edge(START, "read_document")
graph.add_edge("read_document", "extract_metadata")
# ... etc
```

### State Management

Each agent maintains typed state dictionaries:

```python
class ContractState(TypedDict):
    file_path: str
    raw_text: Optional[str]
    contract_metadata: Optional[Dict]
    structured_summary: Optional[Dict]
    vector_store_status: Optional[str]
    generated_text: str
    messages: Annotated[list, operator.add]
```

### LLM Integration

Uses IBM Watsonx for:
- Intent interpretation
- Metadata extraction
- Partner profile synthesis
- Action recommendation
- Email generation

### Vector Database

Chroma for contract storage and semantic search:
- Persistent storage
- Metadata filtering
- Similarity search
- Chunk-based retrieval

## Extending the System

### Adding New Agents

1. Create agent class with StateGraph
2. Define state TypedDict
3. Implement processing nodes
4. Add to supervisory agent workflow

### Custom Tools

Add tools to Research Agent:

```python
@tool
def custom_research_tool(query: str) -> str:
    """Your custom research logic"""
    return results
```

### Alternative LLMs

Replace Watsonx with other providers:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
```

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   - Ensure `.env` file exists with all required keys
   - Check `WATSONX_APIKEY`, `WATSONX_PROJECT_ID`, `TAVILY_API_KEY`

2. **Contract File Not Found**
   - Verify file path is correct
   - Ensure file exists in `docs/` directory

3. **Import Errors**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.9+)

4. **Vector Store Issues**
   - Delete `./contract_vector_store` directory to reset
   - Ensure write permissions in working directory

## Future Enhancements

- [ ] SendGrid integration for actual email sending
- [ ] Salesforce/Salesloft CRM integration
- [ ] Real-time contract upload UI
- [ ] Multi-language support
- [ ] Advanced risk scoring models
- [ ] A/B testing for action recommendations
- [ ] Analytics dashboard
- [ ] Mobile app integration

## License

Proprietary - Internal Use Only

## Support

For questions or issues, contact the development team.

---

**Version**: 1.0.0  
**Last Updated**: 2024-04-01  
**Authors**: AI Development Team