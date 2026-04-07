# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Outage")

@mcp.tool()
def check_outage(address: str) -> str:
    """Check for power outage at the address provided"""
    if address is None:
        outage = "No address provided!"
    else:
        outage = f"No power outage reported at {address}"
    
    return outage

if __name__ == "__main__":
    mcp.run(transport="stdio")