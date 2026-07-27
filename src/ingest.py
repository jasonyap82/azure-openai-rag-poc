"""Ingest: chunk the corpus, embed it, write the index. Run this before asking anything."""
from __future__ import annotations

import logging

from rich.console import Console
from rich.table import Table

from .chunking import chunk_corpus
from .config import get_settings
from .rag import get_retriever

console = Console()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = get_settings()

    chunks = chunk_corpus(s.data_dir, s.chunk_tokens, s.chunk_overlap_tokens)
    if not chunks:
        console.print(f"[red]No .md files found in {s.data_dir}[/red]")
        return

    token_counts = [c.token_count for c in chunks]
    table = Table(title="Corpus", show_header=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("files", str(len({c.source for c in chunks})))
    table.add_row("chunks", str(len(chunks)))
    table.add_row("total tokens", f"{sum(token_counts):,}")
    table.add_row("mean tokens/chunk", f"{sum(token_counts) / len(chunks):.0f}")
    table.add_row("max tokens/chunk", str(max(token_counts)))
    # text-embedding-3-small list price, per 1M tokens.
    table.add_row("est. embedding cost", f"${sum(token_counts) / 1_000_000 * 0.02:.5f}")
    console.print(table)

    retriever = get_retriever()
    retriever.build(chunks)
    console.print(f"[green]Index built using the '{s.retriever}' backend.[/green]")


if __name__ == "__main__":
    main()
