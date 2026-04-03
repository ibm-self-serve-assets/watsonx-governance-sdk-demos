"""
Contract Agent for ESA / Contract Ingestion

This agent performs OCR, contract extraction, and ingestion into vector database.
It reads contracts from file paths, extracts key information, and enables semantic queries.
"""

from typing import TypedDict, Annotated, Optional, Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import operator
import os
from pathlib import Path
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# For document processing
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False


class ContractState(TypedDict):
    """State for Contract Agent"""
    file_path: str
    raw_text: Optional[str]
    contract_metadata: Optional[Dict]
    structured_summary: Optional[Dict]
    vector_store_status: Optional[str]
    generated_text: str
    messages: Annotated[list, operator.add]


class ContractAgent:
    """
    Contract Agent using LangGraph for ESA/Contract ingestion.
    
    This agent:
    1. Reads document from file path (read-only access)
    2. Performs OCR/text extraction
    3. Extracts and normalizes contract metadata:
       - Parties
       - Effective date
       - Term length
       - Key obligations/milestones
    4. Ingests into in-memory vector database (Chroma)
    5. Returns structured contract summary
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/llama-3-3-70b-instruct",
        embedding_model_id: str = "ibm/slate-125m-english-rtrvr-v2",
        url: str = "https://us-south.ml.cloud.ibm.com",
        apikey: Optional[str] = None,
        project_id: Optional[str] = None,
        max_new_tokens: int = 800,
        vector_store_path: str = "./contract_vector_store"
    ):
        """
        Initialize Contract Agent.
        
        Args:
            model_id: Watsonx model ID for text generation
            embedding_model_id: Watsonx model ID for embeddings
            url: Watsonx API URL
            apikey: IBM Cloud API key (defaults to WATSONX_APIKEY env var)
            project_id: Watsonx project ID (defaults to WATSONX_PROJECT_ID env var)
            max_new_tokens: Maximum tokens for generation
            vector_store_path: Path for persistent vector store
        """
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.url = url
        # Use environment variables if not provided
        self.apikey = apikey or os.getenv("WATSONX_APIKEY")
        self.project_id = project_id or os.getenv("WATSONX_PROJECT_ID")
        self.max_new_tokens = max_new_tokens
        self.vector_store_path = vector_store_path
        self.graph = None
        self.vector_store = None
        
        # Validate credentials
        if not self.apikey:
            raise ValueError("WATSONX_APIKEY must be provided or set in environment variables")
        if not self.project_id:
            raise ValueError("WATSONX_PROJECT_ID must be provided or set in environment variables")
        
    def _initialize_vector_store(self):
        """Initialize or load the vector store"""
        if self.vector_store is None:
            embeddings = WatsonxEmbeddings(
                model_id=self.embedding_model_id,
                url=self.url,
                apikey=self.apikey,
                project_id=self.project_id
            )
            
            # Create or load persistent vector store
            self.vector_store = Chroma(
                collection_name="contracts",
                embedding_function=embeddings,
                persist_directory=self.vector_store_path
            )
    
    def _read_document(self, file_path: str) -> str:
        """
        Read document content from file path (read-only).
        Supports .txt, .docx, and .xlsx files.
        
        Args:
            file_path: Path to the contract document
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read based on file extension
        if file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_path.suffix.lower() == '.docx':
            if not DOCX_AVAILABLE:
                raise ImportError("python-docx not installed. Install with: pip install python-docx")
            doc = DocxDocument(file_path)
            
            # Extract text from paragraphs
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = '\t'.join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        text_parts.append(row_text)
            
            full_text = '\n'.join(text_parts)
            
            # If still empty, try alternative extraction methods
            if not full_text.strip():
                # Fallback 1: Try to get text from all runs in all paragraphs
                all_text = []
                for paragraph in doc.paragraphs:
                    for run in paragraph.runs:
                        if run.text and run.text.strip():
                            all_text.append(run.text)
                
                # Fallback 2: Try to extract from document body elements
                if not all_text:
                    try:
                        from docx.oxml.text.paragraph import CT_P
                        from docx.oxml.table import CT_Tbl
                        from docx.table import Table
                        from docx.text.paragraph import Paragraph
                        
                        for element in doc.element.body:
                            if isinstance(element, CT_P):
                                para = Paragraph(element, doc)
                                if para.text and para.text.strip():
                                    all_text.append(para.text)
                            elif isinstance(element, CT_Tbl):
                                table = Table(element, doc)
                                for row in table.rows:
                                    row_text = '\t'.join([cell.text.strip() for cell in row.cells])
                                    if row_text.strip():
                                        all_text.append(row_text)
                    except Exception as e:
                        # If advanced extraction fails, log but continue
                        pass
                
                full_text = '\n'.join(all_text)
            
            # Final check - if still empty, raise an error
            if not full_text.strip():
                raise ValueError(f"Could not extract any text from DOCX file: {file_path}")
            
            return full_text
        
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            if not EXCEL_AVAILABLE:
                raise ImportError("openpyxl not installed. Install with: pip install openpyxl")
            wb = openpyxl.load_workbook(file_path, read_only=True)
            text_parts = []
            for sheet in wb.worksheets:
                text_parts.append(f"\n=== Sheet: {sheet.title} ===\n")
                for row in sheet.iter_rows(values_only=True):
                    row_text = '\t'.join([str(cell) if cell is not None else '' for cell in row])
                    if row_text.strip():
                        text_parts.append(row_text)
            return '\n'.join(text_parts)
        
        else:
            # Fallback: try to read as text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def build_agent(self):
        """Build the LangGraph for contract processing"""
        
        def read_document_node(state: ContractState) -> dict:
            """Read document from file path (read-only access)"""
            try:
                print(f"DEBUG: Reading document from: {state['file_path']}")
                raw_text = self._read_document(state["file_path"])
                print(f"DEBUG: Extracted {len(raw_text) if raw_text else 0} characters")
                
                if not raw_text or len(raw_text.strip()) == 0:
                    print(f"DEBUG: Document appears empty!")
                    return {
                        "raw_text": None,
                        "messages": [f"Warning: Document appears empty or could not extract text from {state['file_path']}"]
                    }
                
                print(f"DEBUG: Returning raw_text with {len(raw_text)} characters")
                return {
                    "raw_text": raw_text,
                    "messages": [f"Document read successfully: {len(raw_text)} characters"]
                }
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"DEBUG: Error reading document: {str(e)}")
                print(error_details)
                return {
                    "raw_text": None,
                    "messages": [f"Error reading document: {str(e)}\n{error_details}"]
                }
        
        def extract_metadata_node(state: ContractState) -> dict:
            """Extract and normalize contract metadata using LLM"""
            
            if not state["raw_text"]:
                return {
                    "contract_metadata": None,
                    "messages": ["Skipping metadata extraction - no text available"]
                }
            
            # Truncate text if too long (keep first 3000 chars for metadata extraction)
            text_sample = state["raw_text"][:3000]
            
            metadata_prompt = ChatPromptTemplate.from_template(
                "You are a contract analysis expert. Extract key metadata from this contract:\n\n"
                "{contract_text}\n\n"
                "Extract and provide in this exact format:\n"
                "PARTIES: [List all parties involved]\n"
                "EFFECTIVE_DATE: [Contract effective date]\n"
                "TERM_LENGTH: [Duration/term of contract]\n"
                "KEY_OBLIGATIONS: [Main obligations and responsibilities]\n"
                "MILESTONES: [Key milestones or deliverables]\n\n"
                "If information is not found, write 'Not specified'."
            )
            
            llm = WatsonxLLM(
                model_id=self.model_id,
                url=self.url,
                apikey=self.apikey,
                project_id=self.project_id,
                params={
                    "max_new_tokens": 500,
                    "decoding_method": "greedy",
                }
            )
            
            formatted_prompt = metadata_prompt.invoke({"contract_text": text_sample})
            result = llm.invoke(formatted_prompt)
            extracted_metadata = result.content if hasattr(result, "content") else str(result)
            
            # Parse the extracted metadata
            metadata = {
                "raw_extraction": extracted_metadata,
                "file_path": state["file_path"],
                "extraction_timestamp": datetime.now().isoformat(),
                "document_length": len(state["raw_text"])
            }
            
            return {
                "contract_metadata": metadata,
                "messages": ["Contract metadata extracted"]
            }
        
        def normalize_and_structure_node(state: ContractState) -> dict:
            """Normalize extracted data and create structured summary"""
            
            if not state["contract_metadata"]:
                return {
                    "structured_summary": None,
                    "messages": ["Skipping normalization - no metadata available"]
                }
            
            normalize_prompt = ChatPromptTemplate.from_template(
                "Based on this extracted contract metadata:\n\n"
                "{metadata}\n\n"
                "Create a structured JSON summary with these fields:\n"
                "- parties: array of party names\n"
                "- effective_date: normalized date (YYYY-MM-DD format if possible)\n"
                "- term_length: normalized duration\n"
                "- key_obligations: array of main obligations\n"
                "- milestones: array of key milestones\n"
                "- contract_type: inferred type (e.g., 'Sales Agreement', 'Service Contract')\n"
                "- risk_level: assessed risk level (Low/Medium/High)\n\n"
                "Provide ONLY valid JSON, no additional text."
            )
            
            llm = WatsonxLLM(
                model_id=self.model_id,
                url=self.url,
                apikey=self.apikey,
                project_id=self.project_id,
                params={
                    "max_new_tokens": 400,
                    "decoding_method": "greedy",
                }
            )
            
            formatted_prompt = normalize_prompt.invoke({
                "metadata": state["contract_metadata"]["raw_extraction"]
            })
            result = llm.invoke(formatted_prompt)
            structured_data = result.content if hasattr(result, "content") else str(result)
            
            # Try to parse as JSON, fallback to dict if fails
            try:
                structured_summary = json.loads(structured_data)
            except json.JSONDecodeError:
                structured_summary = {
                    "raw_structured_data": structured_data,
                    "parsing_note": "Could not parse as JSON"
                }
            
            return {
                "structured_summary": structured_summary,
                "messages": ["Contract data normalized and structured"]
            }
        
        def ingest_to_vector_db_node(state: ContractState) -> dict:
            """Ingest contract into vector database for semantic search"""
            
            if not state["raw_text"]:
                return {
                    "vector_store_status": "Failed - no text available",
                    "messages": ["Vector ingestion skipped - no text"]
                }
            
            try:
                # Initialize vector store
                self._initialize_vector_store()
                
                # Create document with metadata
                doc_metadata = {
                    "file_path": state["file_path"],
                    "ingestion_timestamp": datetime.now().isoformat(),
                    "document_length": len(state["raw_text"])
                }
                
                # Add structured summary to metadata if available
                if state["structured_summary"]:
                    doc_metadata.update({
                        "contract_type": state["structured_summary"].get("contract_type", "Unknown"),
                        "parties": str(state["structured_summary"].get("parties", [])),
                        "effective_date": state["structured_summary"].get("effective_date", "Unknown")
                    })
                
                # Split text into chunks for better retrieval
                chunk_size = 1000
                chunks = []
                for i in range(0, len(state["raw_text"]), chunk_size):
                    chunk_text = state["raw_text"][i:i+chunk_size]
                    chunk_metadata = doc_metadata.copy()
                    chunk_metadata["chunk_index"] = i // chunk_size
                    chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
                
                # Add to vector store
                self.vector_store.add_documents(chunks)
                
                return {
                    "vector_store_status": f"Success - {len(chunks)} chunks ingested",
                    "messages": [f"Contract ingested into vector DB: {len(chunks)} chunks"]
                }
            
            except Exception as e:
                return {
                    "vector_store_status": f"Failed - {str(e)}",
                    "messages": [f"Vector ingestion error: {str(e)}"]
                }
        
        def generate_summary_node(state: ContractState) -> dict:
            """Generate final structured contract summary"""
            
            # Build comprehensive summary
            summary_parts = [
                "="*70,
                "CONTRACT ANALYSIS SUMMARY",
                "="*70,
                "",
                f"File: {state['file_path']}",
                f"Document Length: {len(state['raw_text']) if state['raw_text'] else 0} characters",
                ""
            ]
            
            if state["contract_metadata"]:
                summary_parts.extend([
                    "EXTRACTED METADATA:",
                    "-"*70,
                    state["contract_metadata"]["raw_extraction"],
                    ""
                ])
            
            if state["structured_summary"]:
                summary_parts.extend([
                    "STRUCTURED SUMMARY:",
                    "-"*70,
                    json.dumps(state["structured_summary"], indent=2),
                    ""
                ])
            
            summary_parts.extend([
                "VECTOR DATABASE STATUS:",
                "-"*70,
                state.get("vector_store_status", "Not processed"),
                "",
                "="*70,
                "Contract ingestion complete. Document is now searchable via semantic queries.",
                "="*70
            ])
            
            final_output = "\n".join(summary_parts)
            
            return {
                "generated_text": final_output,
                "messages": ["Contract summary generated"]
            }
        
        # Build the graph
        graph = StateGraph(ContractState)
        
        # Add nodes
        graph.add_node("read_document", read_document_node)
        graph.add_node("extract_metadata", extract_metadata_node)
        graph.add_node("normalize_structure", normalize_and_structure_node)
        graph.add_node("ingest_vector_db", ingest_to_vector_db_node)
        graph.add_node("generate_summary", generate_summary_node)
        
        # Add edges - linear workflow
        graph.add_edge(START, "read_document")
        graph.add_edge("read_document", "extract_metadata")
        graph.add_edge("extract_metadata", "normalize_structure")
        graph.add_edge("normalize_structure", "ingest_vector_db")
        graph.add_edge("ingest_vector_db", "generate_summary")
        graph.add_edge("generate_summary", END)
        
        self.graph = graph.compile()
        return self.graph
    
    def run(self, file_path: str) -> dict:
        """
        Run the contract agent on a document.
        
        Args:
            file_path: Path to the contract document
            
        Returns:
            Final state with contract analysis
        """
        if self.graph is None:
            self.build_agent()
        
        initial_state = {
            "file_path": file_path,
            "raw_text": None,
            "contract_metadata": None,
            "structured_summary": None,
            "vector_store_status": None,
            "generated_text": "",
            "messages": []
        }
        
        result = self.graph.invoke(initial_state)
        return result
    
    def query_contracts(self, query: str, k: int = 3) -> List[Document]:
        """
        Query ingested contracts using semantic search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks
        """
        if self.vector_store is None:
            self._initialize_vector_store()
        
        results = self.vector_store.similarity_search(query, k=k)
        return results


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    agent = ContractAgent(
        apikey=os.getenv("WATSONX_APIKEY"),
        project_id=os.getenv("WATSONX_PROJECT_ID")
    )
    
    # Process a contract
    result = agent.run("sales-demo/docs/Confluent_IBM-1.30.2024.docx")
    
    print(result["generated_text"])
    
    # Example semantic query
    print("\n" + "="*70)
    print("SEMANTIC QUERY EXAMPLE")
    print("="*70)
    query_results = agent.query_contracts("What are the payment terms?")
    for i, doc in enumerate(query_results, 1):
        print(f"\nResult {i}:")
        print(doc.page_content[:200] + "...")
