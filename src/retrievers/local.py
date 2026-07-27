"""Local numpy retriever -- brute-force cosine similarity over a flat vector array.

This is the default because it costs nothing to run and, at POC corpus sizes
(hundreds to low thousands of chunks), an exhaustive scan is genuinely faster than
an ANN index: no graph to build, no recall/latency tradeoff to tune. It is the
correct engineering choice here, not a compromise.

Where it stops being correct: roughly 10^5 chunks, or the moment you need filtering,
keyword matching, incremental updates, or more than one process reading the index.
That is the point where you move to azure_search.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from ..aoai import build_client, embed
from ..chunking import Chunk
from ..config import get_settings
from .base import Hit

log = logging.getLogger(__name__)


class LocalRetriever:
    def __init__(self, index_dir: Path | None = None) -> None:
        s = get_settings()
        self.index_dir = index_dir or s.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = build_client()
        return self._client

    # ---- persistence -------------------------------------------------
    @property
    def _vec_path(self) -> Path:
        return self.index_dir / "vectors.npy"

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    def build(self, chunks: Sequence[Chunk]) -> None:
        texts = [c.text for c in chunks]
        vectors = np.asarray(embed(self.client, texts), dtype=np.float32)
        # L2-normalise once at write time so query-time cosine similarity is a dot product.
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        np.save(self._vec_path, vectors)
        with self._meta_path.open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c.__dict__) + "\n")
        self._vectors, self._chunks = vectors, list(chunks)
        log.info("Built local index: %d chunks, dim=%d", *vectors.shape)

    def load(self) -> None:
        if not self._vec_path.exists():
            raise FileNotFoundError("No index found. Run: python -m src.ingest")
        self._vectors = np.load(self._vec_path)
        self._chunks = [
            Chunk(**json.loads(line))
            for line in self._meta_path.read_text(encoding="utf-8").splitlines()
        ]

    # ---- query -------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[Hit]:
        if self._vectors is None:
            self.load()
        qv = np.asarray(embed(self.client, [query])[0], dtype=np.float32)
        qv /= np.linalg.norm(qv)

        scores = self._vectors @ qv
        # argpartition beats a full sort when top_k << n.
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [Hit(chunk=self._chunks[i], score=float(scores[i])) for i in idx]
