"""Agent loop: the core processing engine."""

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
import json
import json_repair
from pathlib import Path
import re
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider, LLMResponse
from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools import A2A_TOOL_AVAILABLE
from nanobot.agent.memory import MemoryStore
from nanobot.agent.reasoning import ReasoningEngine, TaskPlan
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import Session, SessionManager
from nanobot.agent.tools.offloader import (
    ToolResponseOffloader,
    ReadArtifactTool, TailArtifactTool, SearchArtifactTool, ListArtifactsTool
)

from nanobot.config.schema import OffloadConfig

class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        memory_window: int = 50,
        reasoning_enabled: bool = True,
        reasoning_complexity_min: int = 2,
        reasoning_verify_always: bool = True,
        brave_api_key: str | None = None,
        offload_config: OffloadConfig | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        from nanobot.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_enabled = reasoning_enabled
        self.reasoning_complexity_min = reasoning_complexity_min
        self.reasoning_verify_always = reasoning_verify_always
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.offload_config = offload_config
        
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            brave_api_key=brave_api_key,
            offload_config=offload_config,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        self.reasoning = ReasoningEngine(provider=provider, model=self.model)

        # Initialize output offloader
        self.offloader = ToolResponseOffloader(workspace, config=offload_config)

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._register_default_tools()

    def _should_plan(self, content: str, session: Session) -> bool:
        """Determine whether task planning should be enabled for this request."""
        if not self.reasoning_enabled:
            return False

        complexity = 0
        text = content.strip()

        if len(text) >= 120:
            complexity += 1

        lowered = text.lower()
        planning_keywords = ("计划", "分步骤", "先分析", "plan", "step by step")
        if any(keyword in lowered for keyword in planning_keywords):
            complexity += 1

        multi_step_markers = ("并且", "然后", "同时", "先", "再", "并行", "and then")
        marker_hits = sum(1 for marker in multi_step_markers if marker in lowered)
        if marker_hits >= 2:
            complexity += 1

        recent_tools_used = [
            msg.get("tools_used", [])
            for msg in session.messages[-8:]
            if isinstance(msg, dict)
        ]
        if any(len(used) >= 2 for used in recent_tools_used):
            complexity += 1

        return complexity >= self.reasoning_complexity_min

    def _should_verify(self, tools_used: list[str], content: str) -> bool:
        """Determine whether completion verification should run."""
        if not self.reasoning_enabled:
            return False
        if self.reasoning_verify_always:
            return True

        verify_tools = {"write_file", "edit_file", "spawn"}
        if any(tool in verify_tools for tool in tools_used):
            return True

        lowered = content.lower()
        verify_keywords = ("完成了吗", "验收", "检查", "verify", "double-check")
        return any(keyword in lowered for keyword in verify_keywords)

    async def _maybe_create_task_plan(
        self,
        initial_messages: list[dict[str, Any]],
        content: str,
        session: Session,
        channel: str,
        chat_id: str,
    ) -> TaskPlan | None:
        """Create and inject a task plan when complexity warrants it."""
        if not self._should_plan(content, session):
            return None

        task_plan = await self.reasoning.create_plan(
            messages=initial_messages,
            task=content,
            available_tools=self.tools.get_simple_definitions(),
            context=f"channel={channel}, chat_id={chat_id}",
        )
        if task_plan:
            initial_messages.append({
                "role": "system",
                "content": (
                    "Use this plan as a guide. Adapt if tool outputs prove otherwise.\n\n"
                    f"{task_plan.to_readable_string()}"
                ),
            })
        return task_plan

    async def _maybe_reflect_step(
        self,
        messages: list[dict[str, Any]],
        task_plan: TaskPlan | None,
        plan_step_index: int,
        tools_used_in_step: list[str],
        step_tool_results: list[str],
    ) -> int:
        """Reflect on the current planned step and return the updated step index."""
        if not task_plan or plan_step_index >= len(task_plan.steps) or not tools_used_in_step:
            return plan_step_index

        step = task_plan.steps[plan_step_index]
        expected_tool = (step.tool or "").strip().lower()
        used_tool_set = {name.strip().lower() for name in tools_used_in_step}
        if expected_tool not in used_tool_set:
            return plan_step_index

        reflection = await self.reasoning.reflect_on_step(
            messages=messages,
            step=step,
            actual_tools_used=tools_used_in_step,
            actual_result="\n".join(step_tool_results),
        )
        if reflection.needs_adjustment and reflection.suggested_adjustment:
            messages.append({
                "role": "user",
                "content": f"Plan adjustment hint: {reflection.suggested_adjustment}",
            })
        return plan_step_index + 1

    async def _maybe_verify_and_repair(
        self,
        initial_messages: list[dict[str, Any]],
        content: str,
        final_content: str,
        tools_used: list[str],
        stream_callback: Callable[[str], Any] | None,
        task_plan: TaskPlan | None,
    ) -> tuple[str, list[str]]:
        """Verify completion and run one repair pass if needed."""
        if not self._should_verify(tools_used, content):
            return final_content, tools_used

        verification = await self.reasoning.verify_completion(
            messages=initial_messages,
            original_task=content,
            plan=task_plan or TaskPlan(
                goal=content,
                analysis="No explicit plan generated.",
                steps=[],
                success_criteria="Satisfy user request with accurate and complete output.",
                estimated_iterations=max(1, len(tools_used)),
            ),
            final_result=final_content,
        )
        if verification.task_completed:
            return final_content, tools_used

        logger.info("Reasoning verification requested one repair pass")
        repair_messages = initial_messages + [
            {"role": "assistant", "content": final_content},
            {
                "role": "user",
                "content": (
                    "The previous result is incomplete. Fix it now.\n"
                    f"Missing items: {verification.missing_items}\n"
                    f"Issues: {verification.issues}\n"
                    "Return the corrected final answer."
                ),
            },
        ]
        retry_content, retry_tools_used = await self._run_agent_loop(
            repair_messages,
            stream_callback,
            task_plan=task_plan,
        )
        if retry_content:
            final_content = retry_content
        tools_used.extend(retry_tools_used)
        return final_content, tools_used
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # Initialize output offloader
        self.offloader = ToolResponseOffloader(self.workspace, config=self.offload_config)

        # Artifact tools (for offloaded content)
        self.tools.register(ReadArtifactTool(self.offloader))
        self.tools.register(TailArtifactTool(self.offloader))
        self.tools.register(SearchArtifactTool(self.offloader))
        self.tools.register(ListArtifactsTool(self.offloader))

        # A2A client tool (for calling other A2A agents)
        if A2A_TOOL_AVAILABLE:
            from nanobot.agent.tools.a2a_client import A2AClientTool
            self.tools.register(A2AClientTool())
    
    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or not self._mcp_servers:
            return
        self._mcp_connected = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        self._mcp_stack = AsyncExitStack()
        await self._mcp_stack.__aenter__()
        await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id)

        if spawn_tool := self.tools.get("spawn"):
            if isinstance(spawn_tool, SpawnTool):
                spawn_tool.set_context(channel, chat_id)

        if cron_tool := self.tools.get("cron"):
            if isinstance(cron_tool, CronTool):
                cron_tool.set_context(channel, chat_id)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            val = next(iter(tc.arguments.values()), None) if tc.arguments else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        stream_callback: Callable[[str], Any] | None = None,
        task_plan: TaskPlan | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str]]:
        """
        Run the agent iteration loop.

        Args:
            initial_messages: Starting messages for the LLM conversation.
            on_progress: Optional callback to push intermediate content to the user.

        Returns:
            Tuple of (final_content, list_of_tools_used).
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        plan_step_index = 0

        while iteration < self.max_iterations:
            iteration += 1

            if stream_callback:
                # Use streaming provider
                full_content = ""
                full_reasoning = ""
                tool_calls: list = []

                async for chunk in self.provider.stream(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                ):
                    if chunk.content:
                        full_content += chunk.content
                        if not chunk.tool_calls:
                            res = stream_callback(chunk.content)
                            if asyncio.iscoroutine(res):
                                await res
                    if chunk.reasoning_content:
                        full_reasoning += chunk.reasoning_content
                    if chunk.tool_calls:
                        tool_calls.extend(chunk.tool_calls)

                response = LLMResponse(
                    content=full_content if full_content else None,
                    reasoning_content=full_reasoning if full_reasoning else None,
                    tool_calls=tool_calls,
                )
            
            else: 
                # Call LLM normally

                response = await self.provider.chat(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

            # Handle tool calls
            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    await on_progress(clean or self._tool_hint(response.tool_calls))
                if stream_callback:
                    await stream_callback(self._tool_hint(response.tool_calls))

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )

                tools_used_in_step: list[str] = []
                step_tool_results: list[str] = []
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    tools_used_in_step.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")

                    result = await self.tools.execute(tool_call.name, tool_call.arguments)

                    # 执行工具流式反馈
                    if stream_callback:
                        section = str(result)[:30].replace("\n", "")
                        res = stream_callback(f"【执行工具】：{tool_call.name}({args_str}) -> {section}...\n")
                        if asyncio.iscoroutine(res):
                            await res

                    if self.offloader.should_offload(tool_call.name, result):
                        offload_res = self.offloader.offload(tool_call.name, result)
                        result = offload_res.context_message
                        # Notify user of offload (non-blocking)
                        logger.info(f"📁 **Tool Response Offloaded**\nSaved `{offload_res.original_tokens}` tokens from `{tool_call.name}` to `{offload_res.artifact_id}`.")

                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    step_tool_results.append(result)

                plan_step_index = await self._maybe_reflect_step(
                    messages=messages,
                    task_plan=task_plan,
                    plan_step_index=plan_step_index,
                    tools_used_in_step=tools_used_in_step,
                    step_tool_results=step_tool_results,
                )
                messages.append({"role": "user", "content": "Reflect on the results and decide next steps."})
            else:
                final_content = self._strip_think(response.content)
                break

        return final_content, tools_used
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )

                # Check for streaming callback
                stream_callback = None
                if msg.stream_id:
                    stream_callback = self.bus.get_stream_callback(msg.stream_id)

                try:
                    response = await self._process_message(msg, stream_callback=stream_callback)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(
        self,
        msg: InboundMessage,
        stream_callback: Callable[[str], Any] | None = None, 
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            stream_callback: Optional callback for streaming content chunks.
            session_key: Override session key (used by process_direct).
            on_progress: Optional callback for intermediate output (defaults to bus publish).
        
        Returns:
            The response message, or None if no response needed.
        """
        # System messages route back via chat_id ("channel:chat_id")
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)
        
        # Handle slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            # Capture messages before clearing (avoid race condition with background task)
            messages_to_archive = session.messages.copy()
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            async def _consolidate_and_cleanup():
                temp_session = Session(key=session.key)
                temp_session.messages = messages_to_archive
                await self._consolidate_memory(temp_session, archive_all=True)

            asyncio.create_task(_consolidate_and_cleanup())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started. Memory consolidation in progress.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")
        
        if len(session.messages) > self.memory_window:
            asyncio.create_task(self._consolidate_memory(session))

        self._set_tool_context(msg.channel, msg.chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        task_plan = await self._maybe_create_task_plan(
            initial_messages=initial_messages,
            content=msg.content,
            session=session,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str) -> None:
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content,
                metadata=msg.metadata or {},
            ))

        final_content, tools_used = await self._run_agent_loop(
            initial_messages,
            stream_callback,
            task_plan=task_plan,
            on_progress=on_progress or _bus_progress,
            )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        final_content, tools_used = await self._maybe_verify_and_repair(
            initial_messages=initial_messages,
            content=msg.content,
            final_content=final_content,
            tools_used=tools_used,
            stream_callback=stream_callback,
            task_plan=task_plan,
        )
        
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        # Mark stream as done so channel can close streaming session
        if msg.stream_id:
            self.bus.mark_stream_done(msg.stream_id)

        # If streaming was used, content was already delivered via callback
        # Return None to skip sending a duplicate OutboundMessage
        if stream_callback:
            return None
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        self._set_tool_context(origin_channel, origin_chat_id)
        initial_messages = self.context.build_messages(
            history=session.get_history(max_messages=self.memory_window),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
        )
        final_content, _ = await self._run_agent_loop(initial_messages)

        if final_content is None:
            final_content = "Background task completed."
        
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def _consolidate_memory(self, session, archive_all: bool = False) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md.

        Args:
            archive_all: If True, clear all messages and reset session (for /new command).
                       If False, only write to files without modifying session.
        """
        memory = MemoryStore(self.workspace)

        if archive_all:
            old_messages = session.messages
            keep_count = 0
            logger.info(f"Memory consolidation (archive_all): {len(session.messages)} total messages archived")
        else:
            keep_count = self.memory_window // 2
            if len(session.messages) <= keep_count:
                logger.debug(f"Session {session.key}: No consolidation needed (messages={len(session.messages)}, keep={keep_count})")
                return

            messages_to_process = len(session.messages) - session.last_consolidated
            if messages_to_process <= 0:
                logger.debug(f"Session {session.key}: No new messages to consolidate (last_consolidated={session.last_consolidated}, total={len(session.messages)})")
                return

            old_messages = session.messages[session.last_consolidated:-keep_count]
            if not old_messages:
                return
            logger.info(f"Memory consolidation started: {len(session.messages)} total, {len(old_messages)} new to consolidate, {keep_count} keep")

        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            text = (response.content or "").strip()
            if not text:
                logger.warning("Memory consolidation: LLM returned empty response, skipping")
                return
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json_repair.loads(text)
            if not isinstance(result, dict):
                logger.warning(f"Memory consolidation: unexpected response type, skipping. Response: {text[:200]}")
                return

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            if archive_all:
                session.last_consolidated = 0
            else:
                session.last_consolidated = len(session.messages) - keep_count
            logger.info(f"Memory consolidation done: {len(session.messages)} messages, last_consolidated={session.last_consolidated}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        stream_callback: Callable[[str], Any] | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or cron usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
            on_progress: Optional callback for intermediate output.
            stream_callback: Optional callback for streaming content chunks.
        
        Returns:
            The agent's response.
        """
        await self._connect_mcp()
        msg = InboundMessage(
            channel=channel,
            sender_id="user",
            chat_id=chat_id,
            content=content
        )
        
        response = await self._process_message(msg, stream_callback=stream_callback, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
