"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.agent.tools import A2A_TOOL_AVAILABLE
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.agent.reasoning import ReasoningEngine, TaskPlan
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager
from nanobot.agent.tools.offloader import (
    ToolResponseOffloader,
    ReadArtifactTool, TailArtifactTool, SearchArtifactTool, ListArtifactsTool
)
from nanobot.config.schema import OffloadConfig

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService



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
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_enabled: bool = True,
        reasoning_complexity_min: int = 2,
        reasoning_verify_always: bool = True,
        brave_api_key: str | None = None,
        offload_config: OffloadConfig | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
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
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
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
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
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
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.set_context(channel, chat_id, message_id)

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
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
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
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)
                if stream_callback:
                    await stream_callback(self._tool_hint(response.tool_calls))

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
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
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
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

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

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
                    elif msg.channel == "cli":
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id, content="", metadata=msg.metadata or {},
                        ))
                except Exception as e:
                    logger.error("Error processing message: {}", e)
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

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    def _prune_consolidation_lock(self, session_key: str, lock: asyncio.Lock) -> None:
        """Drop lock entry if no longer in use."""
        if not lock.locked():
            self._consolidation_locks.pop(session_key, None)

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
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            lock = self._get_consolidation_lock(session.key)
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)
                self._prune_consolidation_lock(session.key, lock)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — Start a new conversation\n/help — Show available commands")

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._get_consolidation_lock(session.key)

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    self._prune_consolidation_lock(session.key, lock)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, tools_used, all_msgs = await self._run_agent_loop(
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
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        # Mark stream as done so channel can close streaming session
        if msg.stream_id:
            self.bus.mark_stream_done(msg.stream_id)

        # If streaming was used, content was already delivered via callback
        # Return None to skip sending a duplicate OutboundMessage
        if stream_callback:
            return None
        
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
                return None

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    _TOOL_RESULT_MAX_CHARS = 500

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = {k: v for k, v in m.items() if k != "reasoning_content"}
            if entry.get("role") == "tool" and isinstance(entry.get("content"), str):
                content = entry["content"]
                if len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )

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
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, stream_callback=stream_callback, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
