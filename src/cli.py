"""Ask questions against the indexed corpus.

    python -m src.cli "how long is the SLA credit window?"
    python -m src.cli            # interactive
"""
from __future__ import annotations

import sys

# Windows consoles default to cp1252, which cannot encode characters the model
# legitimately produces -- non-breaking hyphens, typographic quotes, en dashes. The
# answer generates fine and then dies on the way to the screen. Force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel

from .config import get_settings
from .rag import answer_question, get_retriever

console = Console()


def ask(question: str, retriever) -> None:
    ans = answer_question(question, retriever)

    style = "yellow" if ans.refused else "green"
    console.print(Panel(ans.text, title="Answer", border_style=style))

    if ans.hits:
        console.print("\n[bold]Retrieved:[/bold]")
        for i, hit in enumerate(ans.hits, start=1):
            used = "[green]cited[/green]" if i in ans.cited_indices else "[dim]unused[/dim]"
            console.print(
                f"  [{i}] {hit.chunk.citation}  "
                f"score={hit.score:.3f}  via={hit.origin}  {used}"
            )

    console.print(
        f"\n[dim]{ans.prompt_tokens} in / {ans.completion_tokens} out  "
        f"~${ans.estimated_cost_usd:.6f}[/dim]\n"
    )


def main() -> None:
    s = get_settings()
    retriever = get_retriever()
    console.print(f"[dim]backend={s.retriever}  top_k={s.top_k}  model={s.chat_deployment}[/dim]\n")

    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]), retriever)
        return

    while True:
        try:
            q = console.input("[bold cyan]?[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        ask(q, retriever)


if __name__ == "__main__":
    main()
