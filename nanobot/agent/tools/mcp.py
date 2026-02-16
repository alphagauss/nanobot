"""MCP client: connects to MCP servers and wraps their tools as native nanobot tools."""

from contextlib import AsyncExitStack
import os
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import MCPServerConfig


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as a nanobot Tool."""

    def __init__(self, session, server_name: str, tool_def):
        self._session = session
        self._original_name = tool_def.name
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        from mcp import types
        result = await self._session.call_tool(self._original_name, arguments=kwargs)
        parts = []
        for block in result.content:
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts) or "(no output)"


async def connect_mcp_servers(
    mcp_servers: dict[str, MCPServerConfig], registry: ToolRegistry, stack: AsyncExitStack
) -> None:
    """Connect to configured MCP servers and register their tools."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.client.sse import sse_client

    for name, cfg in mcp_servers.items():
        try:
            if cfg.transport == "stdio":
                if not cfg.command:
                    logger.warning(f"MCP '{name}': missing command, skipping")
                    continue

                env = {**os.environ, **cfg.env} if cfg.env else None
                params = StdioServerParameters(
                    command=cfg.command, args=cfg.args, env=env
                )
                read, write = await stack.enter_async_context(stdio_client(params))

            elif cfg.transport == "streamable-http":
                if not cfg.url:
                    logger.warning(f"MCP '{name}': missing url, skipping")   
                    continue

                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(cfg.url)
                )
            elif cfg.transport == "sse":
                if not cfg.url:
                    logger.warning(f"MCP '{name}': missing url, skipping")   
                    continue
                         
                read, write, _ = await stack.enter_async_context(
                    sse_client(cfg.url)
                )
            else:
                logger.warning(f"MCP server '{name}': no command or url configured, skipping")
                continue

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools = await session.list_tools()
            for tool_def in tools.tools:
                wrapper = MCPToolWrapper(session, name, tool_def)
                registry.register(wrapper)
                logger.debug(f"MCP: registered tool '{wrapper.name}' from server '{name}'")

            logger.info(f"MCP server '{name}': connected, {len(tools.tools)} tools registered")
        except Exception as e:
            logger.error(f"MCP server '{name}': failed to connect: {e}")
