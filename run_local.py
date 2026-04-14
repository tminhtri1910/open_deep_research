"""Simple terminal runner for the Open Deep Research graph."""

from __future__ import annotations

import asyncio
from datetime import datetime
import uuid
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

console = Console()


def _timestamp() -> str:
    """Return a compact local timestamp."""
    return datetime.now().strftime("%H:%M:%S")


def _log(message: str, style: str = "white") -> None:
    """Print a timestamped log line."""
    console.print(f"[dim]{_timestamp()}[/dim] [{style}]{message}[/{style}]")


def _title(message: str, style: str = "bold cyan") -> None:
    """Print a section title with timestamp."""
    console.print()
    console.print(f"[dim]{_timestamp()}[/dim] [{style}]{message}[/{style}]")


async def run_once(question: str) -> None:
    """Run a single research question and print the result."""
    try:
        from open_deep_research.deep_researcher import deep_researcher_builder
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: langgraph.\n"
            "This usually means you're running the script with the wrong Python "
            "interpreter or the repo environment was not installed.\n\n"
            "Try:\n"
            "  1. Install Python 3.11\n"
            "  2. Recreate the virtualenv\n"
            "  3. Run: uv sync\n"
            "  4. Start with: uv run python run_local.py\n"
        ) from exc

    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config,
    )

    final_report = result.get("final_report")
    if final_report:
        _title("Final Report", "bold green")
        console.print(final_report)
        return

    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        content = getattr(last_message, "content", str(last_message))
        _title("Response", "bold green")
        console.print(content)
        return

    _log("No response was returned.", "yellow")


def _print_value(value, indent: str = "  ") -> None:
    """Pretty-print an update payload from the graph."""
    if isinstance(value, BaseMessage):
        label = value.__class__.__name__.replace("Message", "")
        style = {
            "Human": "green",
            "AI": "cyan",
            "Tool": "yellow",
            "System": "red",
        }.get(label, "white")
        console.print(f"{indent}[bold {style}]{label}[/bold {style}]: {getattr(value, 'content', value)}")
        return

    if isinstance(value, str):
        console.print(f"{indent}{value}")
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            console.print(f"{indent}[bold magenta]{key}[/bold magenta]:")
            _print_value(item, indent + "  ")
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _print_value(item, indent)
        return

    console.print(f"{indent}{value}")


async def run_with_updates(question: str, graph, thread_id: str) -> None:
    """Run a single question and print intermediate graph updates."""
    _title("Reasearching...", "bold cyan")
    final_report = None
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        for node_name, payload in chunk.items():
            _title(f"=={node_name}==", "bold blue")
            _print_value(payload)
            console.print()
            if isinstance(payload, Mapping) and payload.get("final_report"):
                final_report = payload["final_report"]

    if final_report:
        _title("Final Report", "bold green")
        console.print(Panel(final_report, border_style="green"))
        return
    _log("No final report was returned.", "yellow")


async def main() -> None:
    """Run a tiny interactive prompt loop in the terminal."""
    load_dotenv()

    try:
        from open_deep_research.deep_researcher import deep_researcher_builder
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: langgraph.\n"
            "This usually means you're running the script with the wrong Python "
            "interpreter or the repo environment was not installed.\n\n"
            "Try:\n"
            "  1. Install Python 3.11\n"
            "  2. Recreate the virtualenv\n"
            "  3. Run: uv sync\n"
            "  4. Start with: uv run python run_local.py\n"
        ) from exc

    graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
    thread_id = str(uuid.uuid4())

    console.print(
        Panel(
            "[bold cyan]Open Deep Research local runner[/bold cyan]\n"
            "Type a question and press Enter. Type 'exit' to quit.",
            border_style="cyan",
        )
    )
    _log(f"Session thread_id: {thread_id}", "dim")

    while True:
        question = console.input("[bold cyan]Research> [/bold cyan]").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        if question == "/new":
            thread_id = str(uuid.uuid4())
            _log(f"Started new session thread_id: {thread_id}", "dim")
            continue

        try:
            await run_with_updates(question, graph, thread_id)
        except Exception as exc:
            _log(f"Error: {exc}", "bold red")


if __name__ == "__main__":
    asyncio.run(main())
