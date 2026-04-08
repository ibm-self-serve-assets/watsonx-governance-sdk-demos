# Agent Functions Guide

This document explains what each agent does and breaks down their key functions in simple, practical terms.

---

## 1. Supervisory Agent (`supervisory_agent.py`)

**Purpose**: Orchestrates the entire workflow by coordinating all other agents.

### Key Functions:

#### `__init__()`
- **What it does**: Sets up the supervisor with access to all other agents
- **Simple explanation**: Creates the "manager" that will tell other agents when to work

#### `run(seller_query, contract_file_path, partner_name)`
- **What it does**: Executes the complete multi-agent workflow
- **Simple explanation**: Takes a seller's question like "What contracts are expiring?" and coordinates all agents to provide a complete answer with email drafts and CRM updates
- **Example**: Seller asks "Show me Confluent contracts" → Supervisor calls Contract Agent → Research Agent → Action Agent → Returns complete analysis

---

## 2. Contract Agent (`contract_agent.py`)

**Purpose**: Reads and analyzes contract documents to extract key information.

### Key Functions:

#### `_read_docx(file_path)`
- **What it does**: Opens a Word document and extracts all text
- **Simple explanation**: Reads a contract file like "Confluent_IBM-1.30.2025.docx" and pulls out all the words so we can analyze them
- **Example**: Takes "Confluent_IBM-1.30.2025.docx" → Returns "This Agreement is between IBM and Confluent..."

#### `_extract_contract_metadata(text, filename)`
- **What it does**: Finds important details in the contract text
- **Simple explanation**: Looks through the contract to find the partner name, start date, end date, products, and dollar amounts
- **Example**: From contract text → Extracts "Partner: Confluent, Start: 2023-01-30, End: 2025-01-30, Amount: $250,000"

#### `_load_contracts_to_vector_store()`
- **What it does**: Stores all contracts in a searchable database
- **Simple explanation**: Takes all Confluent contracts and puts them in a special database so we can quickly search through them later
- **Example**: Loads 4 Confluent contracts → Creates searchable index → Can now find "watsonx" mentions across all contracts

#### `contract_node(state)`
- **What it does**: Main function that processes contracts for the workflow
- **Simple explanation**: Reads all Confluent contracts, extracts key info, and prepares a summary of what's expiring, what's active, and what needs attention
- **Example**: Processes 4 contracts → Returns "1 expired, 2 active, 1 renewal needed"

---

## 3. Research Agent (`research_agent.py`)

**Purpose**: Gathers external intelligence about the partner company from the web and CRM data.

### Key Functions:

#### `search_pre_acquisition_executives(partner_name)`
- **What it does**: Searches the web for executive names before IBM acquired the company
- **Simple explanation**: Finds who the CPO, CTO, and CEO were at Confluent before IBM bought them
- **Example**: Searches "Confluent CPO CTO CEO before IBM acquisition" → Returns "CPO: John Smith, CTO: Jane Doe"

#### `search_revenue_and_growth(partner_name)`
- **What it does**: Searches for financial performance data
- **Simple explanation**: Finds how much money Confluent is making and if they're growing
- **Example**: Searches "Confluent revenue growth 2025" → Returns "$298.5M revenue, 19% growth"

#### `search_key_announcements(partner_name)`
- **What it does**: Searches for recent news and product launches
- **Simple explanation**: Finds what new products or partnerships Confluent announced recently
- **Example**: Searches "Confluent product launches 2025" → Returns "Launched Confluent Intelligence, partnered with Jio"

#### `_read_crm_data(partner_name)`
- **What it does**: Reads the Excel CRM file to get deal history
- **Simple explanation**: Opens "Confluent Sales Cloud Infor.xlsx" and finds all past deals, who owns them, and what stage they're in
- **Example**: Reads Excel → Returns "2 won deals: $1M Cognos (Kylie), $250K watsonx (Anand)"

#### `research_external_node(state)`
- **What it does**: Main function that combines web research and CRM data
- **Simple explanation**: Gathers everything we know about Confluent from the internet and our CRM, then creates a partner profile
- **Example**: Combines web searches + CRM data → Returns "Strategic Partner, High velocity, 2 deal blockers, $1.25M total value"

---

## 4. Matching Agent (`matching_agent.py`)

**Purpose**: Correlates contract data with CRM opportunities to find matches and gaps.

### Key Functions:

#### `_extract_product_from_filename(filename)`
- **What it does**: Figures out what product a contract is for based on the filename
- **Simple explanation**: Looks at "Confluent_IBM-Cognos-2024.docx" and knows it's a Cognos contract
- **Example**: "Confluent_IBM-1.30.2025.docx" → Checks content → Identifies "watsonx" product

#### `_match_contracts_to_opportunities(portfolio, opportunities)`
- **What it does**: Connects contracts to CRM deals
- **Simple explanation**: Takes the 4 Confluent contracts and matches them to the 2 CRM opportunities to see which contracts have corresponding deals
- **Example**: Contract "watsonx-2025" matches CRM opportunity "Confluent watsonx ESA" → Creates matched pair

#### `matching_node(state)`
- **What it does**: Main function that creates the correlation report
- **Simple explanation**: Compares all contracts to all CRM opportunities and creates lists of what matches, what doesn't, and what needs attention
- **Example**: 4 contracts + 2 opportunities → Returns "2 matched, 2 unmatched contracts, 0 unmatched opportunities"

---

## 5. Action Agent (`action_agent.py`)

**Purpose**: Determines next best actions and creates seller artifacts (emails, CRM updates).

### Key Functions:

#### `_determine_matching_scenario(contract_summary, partner_profile)`
- **What it does**: Figures out what type of situation the seller is in
- **Simple explanation**: Looks at contracts and partner info to decide if this is a "renewal situation", "expansion opportunity", or "risk mitigation" scenario
- **Example**: Sees 1 expired contract + high-value partner → Identifies "renewal_expiration_awareness" scenario

#### `_rank_next_steps(contract_summary, partner_profile, matching_scenario)`
- **What it does**: Creates a prioritized list of what the seller should do
- **Simple explanation**: Makes a to-do list for the seller, putting the most urgent items first
- **Example**: Returns ["1. Review expired contract by Friday", "2. Prepare renewal dashboard", "3. Schedule meeting with account team"]

#### `_draft_email(partner_profile, recommended_action, contract_summary)`
- **What it does**: Writes a professional email for the seller
- **Simple explanation**: Creates a ready-to-send email to the CPO about the contract situation
- **Example**: Generates email: "Dear CPO, I wanted to discuss our recently expired watsonx contract..."

#### `_update_crm(seller_query, partner_name, recommended_action, contract_summary)`
- **What it does**: Adds a new row to the CRM Excel file
- **Simple explanation**: Opens the Excel file and writes the seller's inquiry, next steps, and timestamp as a new opportunity
- **Example**: Adds row: "Confluent - Seller Inquiry 2026-04-08 | Review expired contract | IBM Seller | 2026-04-15"

#### `action_node(state)`
- **What it does**: Main function that creates all seller artifacts
- **Simple explanation**: Takes all the analysis and creates everything the seller needs: risk assessment, recommended actions, draft email, and CRM update
- **Example**: Processes analysis → Returns complete package with email draft, CRM update, and action plan

---

## How They Work Together

### Example Workflow:

1. **Seller asks**: "I'm new to Confluent. What contracts are expiring?"

2. **Supervisory Agent**: "Let me coordinate everyone to answer that"

3. **Contract Agent**: 
   - Reads 4 Confluent contract files
   - Extracts: "1 expired (watsonx-2025), 2 active, 1 renewal coming"

4. **Research Agent**:
   - Searches web: "Confluent acquired by IBM, $298M revenue, growing 19%"
   - Reads CRM: "2 won deals, $1.25M total, Kylie and Anand are owners"

5. **Matching Agent**:
   - Matches contracts to CRM deals
   - Finds: "2 contracts match CRM, 2 don't have CRM entries"

6. **Action Agent**:
   - Determines: "This is a renewal situation with medium risk"
   - Creates: Draft email to CPO, CRM update, prioritized action list
   - Recommends: "Review expired contract this week, prepare renewal dashboard"

7. **Supervisory Agent**: Returns complete package to seller

---

## Quick Reference

| Agent | Main Job | Key Output |
|-------|----------|------------|
| **Supervisory** | Coordinate workflow | Complete analysis package |
| **Contract** | Read contract files | Contract metadata & summaries |
| **Research** | Gather external intel | Partner profile & deal history |
| **Matching** | Connect contracts to CRM | Correlation report |
| **Action** | Create seller artifacts | Email draft, CRM update, action plan |

---

## File Locations

- Contracts: `docs/Confluent_IBM-*.docx`
- CRM Data: `docs/Confluent Sales Cloud Infor.xlsx`
- Vector Store: `contract_vector_store/`
- Research Cache: `contracts_cache.json`