"""Official MCP stdio transport for the local read-only API."""
from . import config
from .read_api import Reader


def main():
    import mcp.server
    from mcp.types import ToolAnnotations
    if hasattr(mcp.server, "MCPServer"):
        server = mcp.server.MCPServer("LatentSelf")
    else:
        from mcp.server.fastmcp import FastMCP
        server = FastMCP("LatentSelf")
    reader = Reader(config.DB_PATH)
    annotations = ToolAnnotations(readOnlyHint=True,destructiveHint=False)
    for tool in (reader.search,reader.show_conversation,reader.timeline,reader.digest,reader.facts,
                 reader.attachment,reader.stats):
        server.tool(annotations=annotations)(tool)
    server.run(transport="stdio")


if __name__=="__main__":
    main()
