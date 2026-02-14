"""CLI entrypoint for running the TUI client with `python -m tui`."""

from __future__ import annotations

import argparse
import asyncio

from tui.app import TUIInput


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nanobot TUI client")
    parser.add_argument(
        "--api",
        default="http://localhost:18790",
        help="Backend API base URL",
    )
    args = parser.parse_args()

    tui_input = TUIInput(gateway_url=args.api)
    asyncio.run(tui_input.start())


if __name__ == "__main__":
    main()
