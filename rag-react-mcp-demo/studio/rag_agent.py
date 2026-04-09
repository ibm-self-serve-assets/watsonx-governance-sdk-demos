"""
Demo 1: RAG Agent
A simple Retrieval Augmented Generation agent using LangGraph that answers questions
about municipal electric utility documents stored in a Qdrant vector store.
"""

from typing import Annotated, NotRequired
from typing_extensions import TypedDict

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv


load_dotenv()


class RAGState(TypedDict):
    """State passed between agent nodes."""
    question: str
    context: NotRequired[list[Document]]
    response: NotRequired[str]


def tiktoken_len(text: str) -> int:
    """Calculate token length using tiktoken for gpt-4.1-nano model."""
    tokens = tiktoken.encoding_for_model("gpt-4.1-nano").encode(text)
    return len(tokens)


def load_and_chunk_documents() -> list[Document]:
    """Load PDF documents from data directory and chunk them."""
    directory_loader = DirectoryLoader("../data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    docs = directory_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=0,
        length_function=tiktoken_len,
    )
    return text_splitter.split_documents(docs)


def create_vector_store(chunks: list[Document]):
    """Create a Qdrant vector store from document chunks."""
    collection_name = "utility-docs"
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize the Qdrant client
    qdrant_client = QdrantClient(location=":memory:")

    # Create a collection in Qdrant
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # Initialize QdrantVectorStore with the Qdrant client
    qdrant_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    # Add the docs to the vector store
    qdrant_vector_store.add_documents(chunks)

    return qdrant_vector_store.as_retriever()


def create_chat_prompt() -> ChatPromptTemplate:
    """Create the chat prompt template for RAG."""
    human_template = """CONTEXT:
{context}

QUERY:
{query}

Use the provided context to answer the provided user query. Only use the provided context to answer the query. If you do not know the answer, or it's not contained in the provided context respond with "I don't know"."""
    return ChatPromptTemplate.from_messages([("human", human_template)])


def create_rag_graph():
    """Build and compile the RAG graph."""
    chunks = load_and_chunk_documents()
    retriever = create_vector_store(chunks)
    chat_model = ChatOpenAI(model="gpt-4.1-nano")
    chat_prompt = create_chat_prompt()

    def retrieve(state: RAGState) -> dict:
        """Retrieve relevant documents for the question."""
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    def generate(state: RAGState) -> dict:
        """Generate response using retrieved context."""
        generator_chain = chat_prompt | chat_model | StrOutputParser()
        response = generator_chain.invoke({
            "query": state["question"],
            "context": state.get("context", []),
        })
        return {"response": response}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")

    return builder.compile()


@tool
def rag_tool(query: Annotated[str, "query to ask the retrieve information tool"]) -> str:
    """Use Retrieval Augmented Generation to retrieve information about municipal electric utility documents."""
    rag_agent = create_rag_graph()
    result = rag_agent.invoke({"question": query})

    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return str(result)


rag_agent_2 = create_react_agent("openai:gpt-4.1-nano", [rag_tool])