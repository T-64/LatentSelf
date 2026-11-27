"""Official MCP stdio transport for the local read-only API."""
from . import config
from .read_api import Reader


def main():
    from mcp.server import MCPServer
    from mcp.types import ToolAnnotations
    server = MCPServer("LatentSelf")
    reader = Reader(config.DB_PATH)
    annotations = ToolAnnotations(readOnlyHint=True,destructiveHint=False)
    for tool in (reader.search,reader.show_conversation,reader.timeline,reader.digest,reader.facts,
                 reader.attachment,reader.stats):
        server.tool(annotations=annotations)(tool)
    server.run(transport="stdio")


if __name__=="__main__":
    main()
