"""Textual-based TUI client for nanobot web backend."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from rich.markdown import Markdown
import websockets
from websockets.exceptions import ConnectionClosed
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, RichLog


class GatewayResponse(Message):
    """Message posted from background polling to the UI thread."""

    def __init__(self, data: Any) -> None:
        super().__init__()
        self.data = data


class SelectDialog(ModalScreen):
    """Modal dialog for select-style responses."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, message: str, options: list[Any], action_type: str | None = None):
        super().__init__()
        self.message = message
        self.options = options
        self.action_type = action_type
        self._option_values: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.message, id="dialog_title")
            items = []
            for idx, opt in enumerate(self.options):
                label = opt.get("label", opt.get("name", str(opt))) if isinstance(opt, dict) else str(opt)
                value = opt.get("value", opt) if isinstance(opt, dict) else opt
                item_id = f"opt_{idx}"
                self._option_values[item_id] = value
                items.append(ListItem(Label(label), id=item_id))
            yield ListView(*items, id="option_list")
            yield Label("Enter to select, Esc to cancel", id="dialog_footer")

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        raw_val = self._option_values.get(event.item.id, "")
        if self.action_type == "session":
            result = f"/session {raw_val}"
        elif self.action_type == "model":
            result = f"/model {raw_val}"
        else:
            result = raw_val
        self.dismiss(result)


class NanobotTUI(App):
    """Interactive terminal UI client using WebSocket endpoint."""

    CSS = """
    RichLog {
        height: 1fr;
        border: tall $primary;
        background: $surface;
        padding: 1 2;
    }
    Input {
        dock: bottom;
        width: 100%;
        border: double $accent;
    }
    #dialog {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: round $primary;
        padding: 1;
        align: center middle;
    }
    #dialog_title { text-align: left; text-style: bold; margin: 0 0 1 0; }
    #option_list { height: auto; max-height: 20; overflow-y: auto; }
    #dialog_footer { text-align: right; text-style: dim; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_log", "Clear", show=True),
    ]

    def __init__(self, gateway_url: str, session_id: str):
        super().__init__()
        self.gateway_url = gateway_url.rstrip("/")
        self.session_id = session_id
        self.history: list[str] = []
        self.history_index = 0
        self._ws_worker = None
        self._send_queue: asyncio.Queue[str] = asyncio.Queue()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield RichLog(id="chat_log", wrap=True, markup=True, auto_scroll=True)
        yield Input(placeholder="Enter command or message...", id="user_input")
        yield Footer()

    def on_mount(self) -> None:
        self.chat_log = self.query_one("#chat_log", RichLog)
        self.chat_log.write("[bold green]System ready, live sync enabled...[/bold green]")
        self._ws_worker = self.run_worker(self.websocket_loop(), exclusive=False)

    def on_unmount(self) -> None:
        if self._ws_worker:
            self._ws_worker.cancel()

    def on_key(self, event: events.Key) -> None:
        input_widget = self.query_one("#user_input", Input)
        if self.focused is not input_widget:
            return
        if event.key == "up" and self.history:
            self.history_index = max(0, self.history_index - 1)
            input_widget.value = self.history[self.history_index]
            input_widget.cursor_position = len(input_widget.value)
        elif event.key == "down" and self.history:
            self.history_index = min(len(self.history), self.history_index + 1)
            input_widget.value = self.history[self.history_index] if self.history_index < len(self.history) else ""
            input_widget.cursor_position = len(input_widget.value)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return

        self.history.append(text)
        self.history_index = len(self.history)
        self.chat_log.write(f"[bold blue]You:[/bold blue] {text}")
        event.input.clear()

        await self._send_queue.put(text)

    def _ws_url(self) -> str:
        base = self.gateway_url
        if base.startswith("https://"):
            base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://") :]
        elif not base.startswith(("ws://", "wss://")):
            base = "ws://" + base
        return f"{base}/ws/{self.session_id}"

    async def websocket_loop(self) -> None:
        ws_url = self._ws_url()
        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    self.post_message(
                        GatewayResponse({"type": "status", "status": "connected"})
                    )
                    while True:
                        recv_task = asyncio.create_task(ws.recv())
                        send_task = asyncio.create_task(self._send_queue.get())
                        done, pending = await asyncio.wait(
                            {recv_task, send_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in pending:
                            task.cancel()

                        if recv_task in done:
                            raw = recv_task.result()
                            try:
                                data = json.loads(raw)
                            except Exception:
                                data = raw
                            self.post_message(GatewayResponse(data))

                        if send_task in done:
                            content = send_task.result()
                            await ws.send(
                                json.dumps({"type": "message", "content": content})
                            )
            except ConnectionClosed:
                self.post_message(
                    GatewayResponse({"type": "status", "status": "disconnected"})
                )
            except Exception:
                self.post_message(
                    GatewayResponse({"type": "status", "status": "reconnecting"})
                )
            await asyncio.sleep(1.0)

    def on_gateway_response(self, message: GatewayResponse) -> None:
        data = message.data

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                self.chat_log.write(Markdown(data))
                return

        if isinstance(data, dict):
            msg_type = data.get("type", "message")
            if msg_type == "select":
                def handle_selected(choice):
                    if choice:
                        input_widget = self.query_one(Input)
                        input_widget.value = str(choice)
                        input_widget.focus()

                self.push_screen(
                    SelectDialog(data.get("message", "Please select:"), data.get("options", []), data.get("action")),
                    handle_selected,
                )
                return

            if msg_type == "status":
                status = data.get("status", "")
                if status == "thinking":
                    self.chat_log.write("[dim]nanobot is thinking...[/dim]")
                elif status == "connected":
                    self.chat_log.write("[green]Connected[/green]")
                elif status in {"disconnected", "reconnecting"}:
                    self.chat_log.write("[yellow]Connection lost, retrying...[/yellow]")
                return

            if msg_type == "message":
                content = data.get("content") or data.get("message")
                if content:
                    self.chat_log.write(Markdown(content))
                return

        self.chat_log.write(str(data))

    def action_clear_log(self) -> None:
        self.query_one(RichLog).clear()


class TUIInput:
    """Entry wrapper for TUI mode."""

    def __init__(self, gateway_url: str = "http://localhost:18790"):
        self.gateway_url = gateway_url
        self._session_id = "tui:richlog:stable"

    async def start(self) -> None:
        app = NanobotTUI(gateway_url=self.gateway_url, session_id=self._session_id)
        await app.run_async()
