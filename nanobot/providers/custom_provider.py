"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import json_repair
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class CustomProvider(LLMProvider):

    def __init__(self, api_key: str = "no-key", api_base: str = "http://localhost:8000/v1", default_model: str = "default"):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self._client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                   model: str | None = None, max_tokens: int = 4096, temperature: float = 0.7) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: ChatCompletion) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator["LLMResponse"]:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")

        try:
            stream: AsyncIterator[ChatCompletionChunk] = await self._client.chat.completions.create(**kwargs)
            tool_calls_map: dict[int, dict[str, Any]] = {}
            finish_reason = "stop"

            async for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield LLMResponse(content=None, reasoning_content=reasoning)

                if delta.content:
                    yield LLMResponse(content=delta.content)

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc_delta.id,
                                "name": tc_delta.function.name if tc_delta.function else "",
                                "arguments": "",
                            }
                        if tc_delta.id:
                            tool_calls_map[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_calls_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_calls_map[idx]["arguments"] += tc_delta.function.arguments

            if tool_calls_map:
                final_tool_calls: list[ToolCallRequest] = []
                for tc in tool_calls_map.values():
                    raw_args = tc["arguments"]
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args = json_repair.loads(raw_args)
                    else:
                        args = raw_args
                    final_tool_calls.append(ToolCallRequest(id=tc["id"], name=tc["name"], arguments=args))
                yield LLMResponse(content=None, tool_calls=final_tool_calls, finish_reason=finish_reason)
        except Exception as e:
            yield LLMResponse(content=f"Error (stream): {e}", finish_reason="error")

if __name__ == "__main__":
    import asyncio

    OPENCODE_ZEN_BASE = "https://opencode.ai/zen/v1"
    OPENCODE_FREE_MODELS_ = [
        "minimax-m2.5-free",
        "minimax-m2.1-free",
        "kimi-k2.5-free",
        "glm-5-free",
        "trinity-large-preview-free"
    ]
    TEST_MODEL = "minimax-m2.5-free"

    test_msgs_plain = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁?"}
    ]
    test_msgs_tool = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": "请告诉我北京时间，并调用工具。"},
    ]
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get current time in a specific timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "IANA timezone, e.g. Asia/Shanghai"}
                    },
                    "required": ["timezone"],
                },
            },
        }
    ]

    async def run_chat_test(fun_call=True) -> None:
        provider = CustomProvider(api_key="", api_base=OPENCODE_ZEN_BASE, default_model=TEST_MODEL)

        if not fun_call:

            print("\n=== CustomProvider chat: plain ===")
            plain = await provider.chat(test_msgs_plain)
            print("reasoning_content:", plain.reasoning_content)
            print("Content:", plain.content)
            print("Tool Calls:", plain.tool_calls)
            print("Finish Reason:", plain.finish_reason)
            print("Usage:", plain.usage)

        else:
            print("\n=== CustomProvider chat: with tool ===")
            with_tool = await provider.chat(test_msgs_tool, tools=test_tools)
            print("reasoning_content:", with_tool.reasoning_content)
            print("Content:", with_tool.content)
            print("Tool Calls:", with_tool.tool_calls)
            print("Finish Reason:", with_tool.finish_reason)
            print("Usage:", with_tool.usage)

    async def run_stream_test(fun_call = False) -> None:
        provider = CustomProvider(api_key="", api_base=OPENCODE_ZEN_BASE, default_model=TEST_MODEL)
        
        if not fun_call:
            print("\n=== CustomProvider stream: plain ===")
            async for chunk in provider.stream(test_msgs_plain):
                if chunk.reasoning_content:
                    print("[reasoning]", chunk.reasoning_content, end="", flush=True)
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                if chunk.tool_calls:
                    print("\nTool Calls:", chunk.tool_calls, flush=True)
            print("\n")

        else:
            print("\n=== CustomProvider stream: with tool ===")
            async for chunk in provider.stream(test_msgs_tool, tools=test_tools):
                if chunk.reasoning_content:
                    print("[reasoning]", chunk.reasoning_content, end="", flush=True)
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                if chunk.tool_calls:
                    print("\nTool Calls:", chunk.tool_calls, flush=True)
            print("\n")

    # asyncio.run(run_chat_test(fun_call=True))
    asyncio.run(run_stream_test(True))
