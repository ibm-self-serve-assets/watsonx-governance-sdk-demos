"""
Demo 3: ReAct Agent with Multiple Tools
A ReAct agent that combines RAG with web search capabilities.
The agent uses a tool belt containing:
- RAG tool for municipal electric utility documents
- Tavily web search for general queries
"""

from typing import Annotated

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from rag_agent import create_rag_graph


load_dotenv()

rag_agent = create_rag_graph()


@tool
def rag_tool(query: Annotated[str, "query to ask the retrieve information tool"]) -> str:
    """Use Retrieval Augmented Generation to retrieve information about municipal electric utility documents."""
    result = rag_agent.invoke({"question": query})

    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return str(result)


def get_tool_belt() -> list:
    """Return the list of tools available to the agent."""
    tavily_tool = TavilySearch(max_results=5)
    return [tavily_tool, rag_tool]


tools_agent = create_react_agent("openai:gpt-4.1-nano", get_tool_belt())