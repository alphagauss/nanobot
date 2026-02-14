"""FastAPI web server for nanobot frontend."""

from __future__ import annotations

import asyncio
import json
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from nanobot.config.loader import load_config, get_config_path, get_data_dir
from nanobot.config.schema import Config
from nanobot.bus.queue import MessageBus
from nanobot.session.manager import SessionManager
from nanobot.cron.service import CronService
from nanobot.cron.types import CronSchedule, CronJob
from nanobot.providers.registry import PROVIDERS

if TYPE_CHECKING:
    from nanobot.channels.web import WebChannel


# ============================================================================
# Request/Response models
# ============================================================================


class ChatRequest(BaseModel):
    message: str
    session_id: str = "web:default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


class AddCronJobRequest(BaseModel):
    name: str
    message: str
    every_seconds: int | None = None
    cron_expr: str | None = None
    at_iso: str | None = None
    deliver: bool = False
    channel: str | None = None
    to: str | None = None


class ToggleCronJobRequest(BaseModel):
    enabled: bool


# ============================================================================
# App factory
# ============================================================================


def create_app(
    *,
    bus: MessageBus | None = None,
    web_channel: "WebChannel | None" = None,
    session_manager: SessionManager | None = None,
    config: Config | None = None,
    cron_service: CronService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Two modes:
    - **Gateway mode** (bus + web_channel provided): messages go through the
      MessageBus; the WebChannel's ``_handle_message`` publishes inbound
      messages and the AgentLoop processes them asynchronously.
    - **Standalone mode** (no bus): creates its own AgentLoop and uses
      ``process_direct()`` for synchronous request-response (legacy).
    """
    if config is None:
        config = load_config()

    app = FastAPI(title="nanobot", version="0.1.0")

    # CORS for frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Standalone fallback: create an isolated AgentLoop when no bus provided
    if bus is None:
        from nanobot.agent.loop import AgentLoop
        from nanobot.providers.litellm_provider import LiteLLMProvider

        bus = MessageBus()
        provider = _make_provider(config)
        session_manager = SessionManager(config.workspace_path)
        cron_store_path = get_data_dir() / "cron" / "jobs.json"
        cron_service = CronService(cron_store_path)

        agent = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=config.workspace_path,
            model=config.agents.defaults.model,
            max_iterations=config.agents.defaults.max_tool_iterations,
            brave_api_key=config.tools.web.search.api_key or None,
            exec_config=config.tools.exec,
            cron_service=cron_service,
            restrict_to_workspace=config.tools.restrict_to_workspace,
            session_manager=session_manager,
        )
        app.state.agent = agent
    else:
        app.state.agent = None  # gateway mode – no standalone agent

    if session_manager is None:
        session_manager = SessionManager(config.workspace_path)
    if cron_service is None:
        cron_store_path = get_data_dir() / "cron" / "jobs.json"
        cron_service = CronService(cron_store_path)

    app.state.config = config
    app.state.session_manager = session_manager
    app.state.cron_service = cron_service
    app.state.bus = bus
    app.state.web_channel = web_channel  # may be None in standalone

    _register_routes(app)
    return app


def _make_provider(config: Config):
    """Create LLM provider from config."""
    from nanobot.providers.litellm_provider import LiteLLMProvider

    model = config.agents.defaults.model

    if config.is_proxy_mode:
        return LiteLLMProvider(
            default_model=model,
            proxy_url=config.proxy.url,
            proxy_token=config.proxy.token,
        )

    p = config.get_provider()
    if not (p and p.api_key) and not model.startswith("bedrock/"):
        raise RuntimeError("No API key configured. Set one in ~/.nanobot/config.json")
    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=config.get_provider_name(),
    )


# ============================================================================
# Routes
# ============================================================================


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    # ------ Chat ------

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        """Send a message.

        Gateway mode: publishes to the bus and returns immediately.
        Standalone mode: processes synchronously and returns the response.
        """
        session_key = req.session_id
        chat_id = session_key.split(":", 1)[-1] if ":" in session_key else session_key

        web_channel: "WebChannel | None" = app.state.web_channel

        if web_channel is not None:
            # Gateway mode – async via bus
            await web_channel._handle_message(
                sender_id="web_user",
                chat_id=chat_id,
                content=req.message,
            )
            # Notify connected clients that processing started
            await web_channel.notify_thinking(chat_id)
            return {"status": "accepted", "session_id": session_key}
        else:
            # Standalone fallback
            from nanobot.agent.loop import AgentLoop

            agent: AgentLoop = app.state.agent
            response = await agent.process_direct(
                content=req.message,
                session_key=session_key,
                channel="web",
                chat_id=chat_id,
            )
            return ChatResponse(response=response, session_id=session_key)

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest):
        """Send a message and stream the response via SSE (standalone mode only)."""
        from nanobot.agent.loop import AgentLoop

        agent: AgentLoop | None = app.state.agent
        if agent is None:
            raise HTTPException(
                status_code=400,
                detail="Streaming not available in gateway mode. Use WebSocket.",
            )

        session_key = req.session_id

        async def event_generator():
            yield f"data: {json.dumps({'type': 'start'})}\n\n"
            try:
                response = await agent.process_direct(
                    content=req.message,
                    session_key=session_key,
                    channel="web",
                    chat_id=session_key.split(":", 1)[-1] if ":" in session_key else session_key,
                )
                chunk_size = 20
                for i in range(0, len(response), chunk_size):
                    chunk = response[i : i + chunk_size]
                    yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                    await asyncio.sleep(0.02)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ------ WebSocket ------

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for real-time chat.

        Clients send: {"type":"message","content":"..."}
        Server sends: {"type":"message","role":"assistant","content":"..."}
                      {"type":"status","status":"thinking"}
        """
        web_channel: "WebChannel | None" = app.state.web_channel

        await websocket.accept()

        if web_channel is not None:
            web_channel.register_connection(session_id, websocket)

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    continue

                if data.get("type") == "message":
                    content = data.get("content", "").strip()
                    if not content:
                        continue

                    if web_channel is not None:
                        # Gateway mode – publish via bus
                        await web_channel._handle_message(
                            sender_id="web_user",
                            chat_id=session_id,
                            content=content,
                        )
                        await web_channel.notify_thinking(session_id)
                    else:
                        # Standalone fallback – process directly
                        from nanobot.agent.loop import AgentLoop

                        agent: AgentLoop = app.state.agent
                        session_key = f"web:{session_id}"
                        response = await agent.process_direct(
                            content=content,
                            session_key=session_key,
                            channel="web",
                            chat_id=session_id,
                        )
                        await websocket.send_text(json.dumps({
                            "type": "message",
                            "role": "assistant",
                            "content": response,
                        }))

        except WebSocketDisconnect:
            logger.debug(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error for session {session_id}: {e}")
        finally:
            if web_channel is not None:
                web_channel.unregister_connection(session_id, websocket)

    # ------ Sessions ------

    @app.get("/api/sessions")
    async def list_sessions():
        """List all conversation sessions."""
        sm: SessionManager = app.state.session_manager
        return sm.list_sessions()

    @app.get("/api/sessions/{key:path}")
    async def get_session(key: str):
        """Get a session's message history."""
        sm: SessionManager = app.state.session_manager
        session = sm.get_or_create(key)
        return {
            "key": session.key,
            "messages": [
                {"role": m["role"], "content": m["content"], "timestamp": m.get("timestamp")}
                for m in session.messages
            ],
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }

    @app.delete("/api/sessions/{key:path}")
    async def delete_session(key: str):
        """Delete a session."""
        sm: SessionManager = app.state.session_manager
        if sm.delete(key):
            return {"ok": True}
        raise HTTPException(status_code=404, detail="Session not found")

    # ------ Status ------

    @app.get("/api/status")
    async def get_status():
        """Get system status."""
        config: Config = app.state.config
        config_path = get_config_path()

        providers_status = []
        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_local:
                providers_status.append({
                    "name": spec.label,
                    "has_key": bool(p.api_base),
                    "detail": p.api_base or "",
                })
            else:
                providers_status.append({
                    "name": spec.label,
                    "has_key": bool(p.api_key),
                })

        channels_status = []
        for ch_name in ["whatsapp", "telegram", "discord", "feishu", "dingtalk", "email", "slack", "qq", "web"]:
            ch_cfg = getattr(config.channels, ch_name, None)
            if ch_cfg:
                channels_status.append({
                    "name": ch_name,
                    "enabled": getattr(ch_cfg, "enabled", False),
                })

        cron: CronService = app.state.cron_service
        cron_status = cron.status()

        return {
            "config_path": str(config_path),
            "config_exists": config_path.exists(),
            "workspace": str(config.workspace_path),
            "workspace_exists": config.workspace_path.exists(),
            "model": config.agents.defaults.model,
            "max_tokens": config.agents.defaults.max_tokens,
            "temperature": config.agents.defaults.temperature,
            "max_tool_iterations": config.agents.defaults.max_tool_iterations,
            "providers": providers_status,
            "channels": channels_status,
            "cron": cron_status,
        }

    # ------ Cron Jobs ------

    @app.get("/api/cron/jobs")
    async def list_cron_jobs(include_disabled: bool = False):
        """List cron jobs."""
        cron: CronService = app.state.cron_service
        jobs = cron.list_jobs(include_disabled=include_disabled)
        return [_serialize_job(j) for j in jobs]

    @app.post("/api/cron/jobs")
    async def add_cron_job(req: AddCronJobRequest):
        """Add a new cron job."""
        cron: CronService = app.state.cron_service

        if req.every_seconds:
            schedule = CronSchedule(kind="every", every_ms=req.every_seconds * 1000)
        elif req.cron_expr:
            schedule = CronSchedule(kind="cron", expr=req.cron_expr)
        elif req.at_iso:
            import datetime
            dt = datetime.datetime.fromisoformat(req.at_iso)
            schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
        else:
            raise HTTPException(status_code=400, detail="Must specify every_seconds, cron_expr, or at_iso")

        job = cron.add_job(
            name=req.name,
            schedule=schedule,
            message=req.message,
            deliver=req.deliver,
            channel=req.channel,
            to=req.to,
        )
        return _serialize_job(job)

    @app.delete("/api/cron/jobs/{job_id}")
    async def remove_cron_job(job_id: str):
        """Remove a cron job."""
        cron: CronService = app.state.cron_service
        if cron.remove_job(job_id):
            return {"ok": True}
        raise HTTPException(status_code=404, detail="Job not found")

    @app.put("/api/cron/jobs/{job_id}/toggle")
    async def toggle_cron_job(job_id: str, req: ToggleCronJobRequest):
        """Enable or disable a cron job."""
        cron: CronService = app.state.cron_service
        job = cron.enable_job(job_id, enabled=req.enabled)
        if job:
            return _serialize_job(job)
        raise HTTPException(status_code=404, detail="Job not found")

    @app.post("/api/cron/jobs/{job_id}/run")
    async def run_cron_job(job_id: str):
        """Manually run a cron job."""
        cron: CronService = app.state.cron_service
        if await cron.run_job(job_id, force=True):
            return {"ok": True}
        raise HTTPException(status_code=404, detail="Job not found")

    # ------ Health ------

    @app.get("/api/ping")
    async def ping():
        return {"message": "pong"}


def _serialize_job(job: CronJob) -> dict[str, Any]:
    """Serialize a CronJob to a JSON-friendly dict."""
    sched_str = ""
    if job.schedule.kind == "every":
        secs = (job.schedule.every_ms or 0) // 1000
        if secs >= 3600:
            sched_str = f"every {secs // 3600}h"
        elif secs >= 60:
            sched_str = f"every {secs // 60}m"
        else:
            sched_str = f"every {secs}s"
    elif job.schedule.kind == "cron":
        sched_str = job.schedule.expr or ""
    else:
        sched_str = "one-time"

    next_run = None
    if job.state.next_run_at_ms:
        next_run = job.state.next_run_at_ms

    last_run = None
    if job.state.last_run_at_ms:
        last_run = job.state.last_run_at_ms

    return {
        "id": job.id,
        "name": job.name,
        "enabled": job.enabled,
        "schedule_kind": job.schedule.kind,
        "schedule_display": sched_str,
        "schedule_expr": job.schedule.expr,
        "schedule_every_ms": job.schedule.every_ms,
        "message": job.payload.message,
        "deliver": job.payload.deliver,
        "channel": job.payload.channel,
        "to": job.payload.to,
        "next_run_at_ms": next_run,
        "last_run_at_ms": last_run,
        "last_status": job.state.last_status,
        "last_error": job.state.last_error,
        "created_at_ms": job.created_at_ms,
    }
