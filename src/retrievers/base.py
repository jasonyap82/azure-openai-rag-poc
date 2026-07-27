"""Retriever interface.

The whole point of this abstraction: retrieval strategy is the part of a RAG system
you will change most often, and it is also the part with the biggest cost implications.
Keeping it behind a Protocol means swapping numpy for Azure AI Search (or pgvector, or
Cosmos DB) is a config change, not a rewrite -- and the evals in eval/ can score both
backends with the same harness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from ..chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    # Where the hit came from, so the CLI can show whether semantic reranking fired.
    origin: str = "vector"


@runtime_checkable
class Retriever(Protocol):
    def build(self, chunks: Sequence[Chunk]) -> None:
        """Index the corpus. Idempotent -- safe to re-run."""

    def search(self, query: str, top_k: int) -> list[Hit]:
        """Return the top_k most relevant chunks, highest score first."""
