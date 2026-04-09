"""
Demo 2: ReAct Agent with Tavily Search
A ReAct agent with web search capabilities using Tavily for answering general queries.
"""

from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv


load_dotenv()


def get_tool_belt() -> list:
    """Return the list of tools available to the agent."""
    tavily_tool = TavilySearch(max_results=5)
    return [tavily_tool]


tavily_agent = create_react_agent("openai:gpt-4.1-nano", get_tool_belt())