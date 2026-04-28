# Multi-Agent Sales Governance Demo

A comprehensive demonstration of a multi-agent AI system with integrated watsonx.governance monitoring for intelligent sales contract management and partner engagement.

## Overview

This demo showcases a sophisticated 5-agent workflow that automates sales contract analysis, partner research, CRM correlation, and action recommendations. Each agent interaction is monitored using watsonx.governance decorators to track performance metrics including faithfulness, context relevance, latency, and cost.

### Key Features

- **Multi-Agent Orchestration**: Five specialized AI agents working together via LangGraph
- **Contract Intelligence**: Automated contract ingestion, parsing, and portfolio analysis
- **Partner Research**: Internal CRM data enrichment with external web intelligence
- **CRM Correlation**: Intelligent matching of contracts to CRM opportunities
- **Action Recommendations**: Context-aware next best actions with draft artifacts
- **Governance Monitoring**: Real-time tracking of AI quality metrics and performance
- **End-to-End Workflow**: From contract upload to actionable sales recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SUPERVISORY AGENT                          │
│              (Orchestrates entire workflow)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │     CONTRACT AGENT                │
         │  • Ingests contract documents     │
         │  • Extracts metadata & terms      │
         │  • Builds vector store            │
         │  • Analyzes portfolio             │
         │  [GOVERNANCE: Faithfulness]       │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │     RESEARCH AGENT                │
         │  • Retrieves CRM data             │
         │  • Web search for partner intel   │
         │  • Analyzes maturity & velocity   │
         │  • Identifies executives          │
         │  [GOVERNANCE: Context Relevance]  │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │     MATCHING AGENT                │
         │  • Correlates contracts with CRM  │
         │  • Identifies renewal gaps        │
         │  • Flags untracked contracts      │
         │  [GOVERNANCE: Latency & Cost]     │
         └───────────────┬───────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │     ACTION AGENT                  │
         │  • Assesses risk levels           │
         │  • Recommends next steps          │
         │  • Drafts emails & CRM updates    │
         │  [GOVERNANCE: All Metrics]        │
         └───────────────┬───────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │   OUTPUT    │
                  │  • Report   │
                  │  • Email    │
                  │  • Metrics  │
                  └─────────────┘
```

**Governance Evaluation Points:**
- After each agent execution
- Metrics tracked: faithfulness scores, context relevance, latency, token usage, cost
- Results exported to CSV for analysis

## Prerequisites

- **Python**: 3.9 or higher
- **IBM watsonx Account**: Required for LLM access
- **API Keys**:
  - IBM watsonx API key
  - Tavily API key (for web search)
- **Environment**: Virtual environment recommended

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd watsonx-governance-sdk-demos/multi-agent-demo
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the `multi-agent-demo` directory:

```bash
# IBM watsonx Configuration
WATSONX_APIKEY=your_watsonx_api_key_here
WATSONX_PROJECT_ID=your_project_id_here

# Tavily API Key (for web search)
TAVILY_API_KEY=your_tavily_api_key_here
```

### Getting API Keys

1. **watsonx API Key**:
   - Log in to [IBM Cloud](https://cloud.ibm.com)
   - Navigate to watsonx.ai
   - Create or select a project
   - Copy your API key and Project ID

2. **Tavily API Key**:
   - Sign up at [Tavily](https://tavily.com)
   - Generate an API key from your dashboard

### Example .env Template

```env
# IBM watsonx Credentials
WATSONX_APIKEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here

# Tavily Search API
TAVILY_API_KEY=your_tavily_key_here
```

## Usage

### Running the Notebook

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open the Demo Notebook**:
   - Navigate to `workflow_agentic_governance.ipynb`
   - Run cells sequentially

3. **Expected Output**:
   - Contract portfolio analysis
   - Partner intelligence report
   - Contract-CRM correlation
   - Risk assessment and recommendations
   - Draft email for executive outreach
   - Governance metrics CSV export

### Execution Time

- **Full workflow**: ~3-5 minutes
- **Contract ingestion**: 30-60 seconds
- **Research phase**: 60-90 seconds
- **Matching & action**: 30-60 seconds

### Sample Query

```python
seller_query = "What contracts are expiring in the next 30 days and what should I do?"
```

The system will:
1. Analyze all contracts in the portfolio
2. Identify expiring contracts
3. Match with CRM opportunities
4. Recommend prioritized actions
5. Generate draft communications

## Workflow Details

### 1. Supervisory Agent

**Role**: Orchestrates the entire workflow

**Responsibilities**:
- Interprets seller's natural language query
- Determines required agents based on intent
- Manages state between agents
- Aggregates final results

**Key Functions**:
- `_interpret_intent()`: Analyzes seller query
- `build_agent()`: Constructs LangGraph workflow
- `run()`: Executes complete workflow

### 2. Contract Agent

**Role**: Contract document intelligence

**Responsibilities**:
- Reads DOCX/Excel contract files
- Extracts metadata (parties, dates, amounts, products)
- Normalizes and structures contract data
- Builds vector store for semantic search
- Analyzes portfolio for renewals and expirations

**Key Functions**:
- `_read_document()`: Parses contract files
- `run()`: Single contract analysis
- `run_portfolio()`: Multi-contract portfolio analysis
- `discover_partner_contracts()`: Finds all partner contracts

**Governance Metrics**:
- Faithfulness: Accuracy of extracted contract terms
- Latency: Document processing time

### 3. Research Agent

**Role**: Partner intelligence gathering

**Responsibilities**:
- Retrieves internal CRM sales history
- Performs web search for partner background
- Identifies key executives (CPO, CTO, CEO)
- Analyzes partner maturity and sales velocity
- Synthesizes comprehensive partner profile

**Key Functions**:
- `retrieve_sales_history()`: CRM data extraction
- `search_pre_acquisition_executives()`: Executive identification
- `analyze_partner_maturity()`: Maturity assessment
- `research_partner()`: Complete research workflow

**Governance Metrics**:
- Context Relevance: Quality of research results
- Latency: Search and synthesis time

### 4. Matching Agent

**Role**: Contract-CRM correlation

**Responsibilities**:
- Matches contracts to CRM opportunities
- Uses LLM for intelligent product/amount matching
- Identifies renewal gaps and untracked contracts
- Flags urgent action items
- Analyzes CRM stage and engagement status

**Key Functions**:
- `_match_with_llm()`: Intelligent contract matching
- `_match_contracts_to_opportunities()`: Portfolio correlation
- `run()`: Complete matching workflow

**Governance Metrics**:
- Latency: Matching operation time
- Cost: Token usage for LLM matching

### 5. Action Agent

**Role**: Next best action determination

**Responsibilities**:
- Assesses contract and partner risk
- Analyzes historical success patterns
- Recommends prioritized next steps
- Drafts executive outreach emails
- Creates CRM update payloads

**Key Functions**:
- `_assess_risk()`: Risk level calculation
- `_determine_action()`: Action recommendation
- `_draft_email()`: Email generation
- `run()`: Complete action workflow

**Governance Metrics**:
- Faithfulness: Accuracy of recommendations
- Context Relevance: Appropriateness of actions
- Latency: Decision-making time
- Cost: Total token usage

## Governance Integration

### watsonx.governance Decorators

The demo uses decorators to monitor AI agent performance:

```python
from ibm_watsonx_gov import WatsonxGovernance

@WatsonxGovernance.monitor(
    task_type="contract_analysis",
    metrics=["faithfulness", "context_relevance", "latency"]
)
def analyze_contract(contract_text):
    # Agent logic here
    pass
```

### Tracked Metrics

1. **Faithfulness Score** (0-1):
   - Measures accuracy of AI-generated content
   - Compares output against source documents
   - Higher is better (>0.8 is excellent)

2. **Context Relevance Score** (0-1):
   - Evaluates relevance of retrieved information
   - Ensures AI uses appropriate context
   - Higher is better (>0.7 is good)

3. **Latency** (seconds):
   - Time taken for each agent operation
   - Helps identify performance bottlenecks
   - Lower is better (<5s is optimal)

4. **Token Usage**:
   - Input and output tokens consumed
   - Tracks LLM API usage
   - Used for cost calculation

5. **Cost** (USD):
   - Estimated cost per operation
   - Based on token usage and model pricing
   - Helps optimize budget

### Interpreting Metrics

**Good Performance**:
- Faithfulness: >0.8
- Context Relevance: >0.7
- Latency: <5 seconds per agent
- Cost: <$0.10 per workflow

**Needs Attention**:
- Faithfulness: <0.6 (check prompts)
- Context Relevance: <0.5 (improve retrieval)
- Latency: >10 seconds (optimize code)
- Cost: >$0.50 per workflow (reduce tokens)

## Output

### Final Report Includes

1. **Executive Summary**:
   - Partner name and maturity level
   - Sales velocity assessment
   - Risk level and score

2. **Recommended Next Step**:
   - Prioritized action with rationale
   - Timeline and urgency indicators

3. **Draft Follow-Up Email**:
   - Personalized to executive (CPO/CTO)
   - Context-aware messaging
   - Ready to send

4. **CRM Update Payload**:
   - Structured data for CRM system
   - Next steps and follow-up dates
   - Risk flags and notes

5. **Contract & CRM Analysis**:
   - Portfolio summary (active, expiring, expired)
   - CRM opportunities list
   - Contract-CRM correlation matrix
   - Unmatched contracts requiring attention

### CSV Export

Governance metrics are exported to `sales_agent_governance_results.csv`:

```csv
timestamp,agent,task,faithfulness,context_relevance,latency,tokens_in,tokens_out,cost
2026-04-28 10:30:15,contract_agent,extract_metadata,0.92,0.88,2.3,450,320,0.015
2026-04-28 10:30:45,research_agent,partner_search,0.85,0.91,3.1,380,280,0.012
...
```

### Sample Metrics

Typical workflow metrics:
- **Total Latency**: 180-300 seconds
- **Total Cost**: $0.05-$0.15
- **Average Faithfulness**: 0.85-0.92
- **Average Context Relevance**: 0.80-0.90

## File Structure

```
multi-agent-demo/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables (create this)
├── workflow_agentic_governance.ipynb   # Main demo notebook
│
├── supervisory_agent.py                # Orchestration agent
├── contract_agent.py                   # Contract analysis agent
├── research_agent.py                   # Partner research agent
├── matching_agent.py                   # Contract-CRM matching agent
├── action_agent.py                     # Action recommendation agent
│
├── cache_contracts.py                  # Contract caching utility
├── contracts_cache.json                # Cached contract data
├── sales_agent_governance_results.csv  # Governance metrics output
│
├── docs/                               # Sample contract documents
│   ├── Confluent_IBM-1.30.2024.docx
│   ├── Confluent_IBM-1.30.2025.docx
│   ├── Confluent_IBM-5.30.2023.docx
│   ├── Confluent_IBM-7.31.2024.docx
│   └── Confluent Sales Cloud Infor.xlsx  # CRM data
│
└── AGENT_PROMPTS_GUIDE.txt             # Prompt engineering guide
```

### Key Files

- **`workflow_agentic_governance.ipynb`**: Main entry point, demonstrates full workflow
- **`supervisory_agent.py`**: Orchestrates all agents using LangGraph
- **`contract_agent.py`**: Handles document parsing and vector storage
- **`research_agent.py`**: Performs CRM and web research
- **`matching_agent.py`**: Correlates contracts with CRM opportunities
- **`action_agent.py`**: Generates recommendations and artifacts
- **`contracts_cache.json`**: Speeds up repeated runs by caching parsed contracts

## Troubleshooting

### Common Issues

#### 1. API Key Errors

**Problem**: `ValueError: WATSONX_APIKEY must be provided`

**Solution**:
```bash
# Verify .env file exists and contains keys
cat .env

# Ensure keys are properly formatted (no quotes)
WATSONX_APIKEY=your_key_here
```

#### 2. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'langgraph'`

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep langgraph
```

#### 3. Contract Parsing Failures

**Problem**: "Error reading document"

**Solution**:
- Ensure contract files are in `docs/` directory
- Verify files are valid DOCX format
- Check file permissions
- Regenerate cache: delete `contracts_cache.json` and rerun

#### 4. Tavily Search Errors

**Problem**: "Tavily API key not found"

**Solution**:
```bash
# Add Tavily key to .env
echo "TAVILY_API_KEY=your_key_here" >> .env

# Restart Jupyter kernel
```

#### 5. Rate Limiting

**Problem**: "Rate limit exceeded"

**Solution**:
- Wait 60 seconds between runs
- Use cached contracts (don't delete `contracts_cache.json`)
- Reduce number of contracts processed

#### 6. Cache Regeneration

**Problem**: Need to refresh contract data

**Solution**:
```bash
# Delete cache file
rm contracts_cache.json

# Run cache generation script
python cache_contracts.py

# Or let the notebook regenerate automatically
```

### Performance Optimization

1. **Use Contract Cache**: Keep `contracts_cache.json` to avoid re-parsing
2. **Limit Portfolio Size**: Start with 2-3 contracts for testing
3. **Adjust Token Limits**: Reduce `max_new_tokens` in agent configs
4. **Batch Operations**: Process multiple queries in one session

## Additional Resources

### Documentation

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [watsonx.ai Documentation](https://www.ibm.com/docs/en/watsonx-as-a-service)
- [watsonx.governance SDK](https://github.com/IBM/watsonx-governance-sdk)
- [LangChain IBM Integration](https://python.langchain.com/docs/integrations/providers/ibm)

### Agent Development Guides

- **`AGENT_PROMPTS_GUIDE.txt`**: Prompt engineering best practices
- **`AGENT_FUNCTIONS_GUIDE.txt`**: Function implementation patterns

### Related Demos

- **`../rag-agent-demo/`**: RAG-based Q&A agent with governance
- **`../evaluate-metrics/`**: Standalone metrics evaluation

### Support

For issues or questions:
1. Check this README's troubleshooting section
2. Review agent guide files in this directory
3. Consult watsonx.governance documentation
4. Open an issue in the repository

---

## License

This demo is provided as-is for educational and demonstration purposes.

## Contributing

Contributions welcome! Please follow standard pull request procedures.

---

**Built with**: IBM watsonx.ai, LangGraph, LangChain, watsonx.governance SDK