"""Simple terminal runner for the Open Deep Research graph."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
import uuid
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence
import re
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUTPUT_DIR = ROOT / "outputs"
SESSION_EXPORTS: dict[str, dict[str, Any]] = {}
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


def _safe_topic_name(text: str, max_length: int = 80) -> str:
    """Create a filesystem-friendly folder name while preserving Vietnamese text."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        return "research"
    return cleaned[:max_length]


def _json_safe(value):
    """Convert graph payloads into JSON-serializable data."""
    if isinstance(value, BaseMessage):
        return {
            "type": value.__class__.__name__,
            "content": getattr(value, "content", None),
            "additional_kwargs": getattr(value, "additional_kwargs", {}),
            "response_metadata": getattr(value, "response_metadata", {}),
            "tool_calls": getattr(value, "tool_calls", []),
        }

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    return str(value)


def _ensure_output_dir() -> Path:
    """Create the export directory if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _get_session_export(thread_id: str, topic: str) -> dict[str, Any]:
    """Return the export metadata for a session thread, creating it if needed."""
    session = SESSION_EXPORTS.get(thread_id)
    if session is not None:
        return session

    output_dir = _ensure_output_dir()
    topic_dir = output_dir / _safe_topic_name(topic)
    topic_dir.mkdir(parents=True, exist_ok=True)

    session = {
        "thread_id": thread_id,
        "topic": topic,
        "created_at": datetime.now().isoformat(),
        "output_dir": str(topic_dir),
        "json_path": str(topic_dir / f"{thread_id}.json"),
        "markdown_path": str(topic_dir / f"{thread_id}.md"),
        "runs": [],
    }
    SESSION_EXPORTS[thread_id] = session
    return session


def _write_session_export(session: dict[str, Any]) -> None:
    """Write the current session export to disk."""
    json_path = Path(session["json_path"])
    json_path.write_text(
        json.dumps(_json_safe(session), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_session_markdown(session: dict[str, Any], content: str) -> None:
    """Write the session markdown report to disk."""
    markdown_path = Path(session["markdown_path"])
    markdown_path.write_text(
        content if content.endswith("\n") else content + "\n",
        encoding="utf-8",
    )


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
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    session = _get_session_export(thread_id, question)

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config,
    )

    final_report = result.get("final_report")
    if final_report:
        session["runs"].append(
            {
                "question": question,
                "started_at": datetime.now().isoformat(),
                "result": result,
                "final_report": final_report,
            }
        )
        _write_session_export(session)
        _write_session_markdown(session, final_report)
        _log(f"Exported JSON: {Path(session['json_path']).relative_to(ROOT)}", "dim")
        _log(f"Exported markdown: {Path(session['markdown_path']).relative_to(ROOT)}", "dim")
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
    session = _get_session_export(thread_id, question)
    run_record: dict[str, Any] = {
        "question": question,
        "started_at": datetime.now().isoformat(),
        "node_results": [],
    }
    session["runs"].append(run_record)
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": question}]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        for node_name, payload in chunk.items():
            run_record["node_results"].append(
                {
                    "node": node_name,
                    "payload": payload,
                }
            )
            _title(f"=={node_name}==", "bold blue")
            _print_value(payload)
            console.print()
            if isinstance(payload, Mapping) and payload.get("final_report"):
                final_report = payload["final_report"]

    if final_report:
        run_record["final_report"] = final_report
        _write_session_export(session)
        _write_session_markdown(session, final_report)
        _log(f"Exported JSON: {Path(session['json_path']).relative_to(ROOT)}", "dim")
        _log(f"Exported markdown: {Path(session['markdown_path']).relative_to(ROOT)}", "dim")
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
