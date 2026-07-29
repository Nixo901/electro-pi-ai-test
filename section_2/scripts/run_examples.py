"""Run three example questions through the RAG pipeline and display the results.

Usage (from the section_2/ directory):
    python scripts/run_examples.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running directly without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.WARNING)  # suppress debug noise in demo output

DOCS_DIR = Path(__file__).parent.parent / "docs"

EXAMPLE_QUESTIONS = [
    "Compare FAISS and Chroma vector stores: which is better for a local prototype?",
    "How does cross-encoder re-ranking improve retrieval quality?",
    "What happens in a RAG pipeline when no relevant context is found?",
]

# Off-topic question to demonstrate the no-context guard
OFF_TOPIC_QUESTION = "What is the capital of France and what is its population?"

console = Console()


def print_result(idx: int, result) -> None:
    """Pretty-print a single RAGResult."""
    colour = "red" if result.no_context else "green"
    label = "[NO CONTEXT]" if result.no_context else "[ANSWERED]"

    console.print(Rule(f"[bold]Question {idx}[/bold]", style="blue"))
    console.print(f"[bold cyan]Q:[/bold cyan] {result.question}\n")

    console.print(
        Panel(
            result.answer,
            title=f"{label} Answer",
            border_style=colour,
            expand=False,
        )
    )

    if result.citations:
        table = Table(
            "Source", "Chunk", "Score", "Excerpt",
            box=box.SIMPLE,
            title="Citations",
            show_lines=True,
        )
        for c in result.citations:
            table.add_row(
                c["source"],
                str(c["chunk_index"]),
                f"{c['score']:.4f}",
                c["excerpt"][:120] + ("…" if len(c["excerpt"]) > 120 else ""),
            )
        console.print(table)
    else:
        console.print("[dim]No citations (no-context response).[/dim]\n")


def main() -> None:
    console.print(
        Panel.fit(
            "[bold magenta]Task 2.1 — LangChain RAG Pipeline[/bold magenta]\n"
            "[dim]Building index from docs/ …[/dim]",
            border_style="magenta",
        )
    )

    pipeline = RAGPipeline()
    pipeline.build(DOCS_DIR)

    console.print(f"\n[green][OK] Index built successfully.[/green]\n")

    all_questions = EXAMPLE_QUESTIONS + [OFF_TOPIC_QUESTION]

    for i, question in enumerate(all_questions, start=1):
        result = pipeline.query(question)
        print_result(i, result)
        console.print()

    console.print(Rule("[bold green]Done[/bold green]"))


if __name__ == "__main__":
    main()
