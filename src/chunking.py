"""Token-aware chunking with hierarchical heading paths.

Naive character splitting is the most common reason a RAG demo retrieves garbage:
it cuts mid-sentence and destroys the semantic unit the embedding represents. This
splits on paragraph boundaries, packs paragraphs to a token budget, and only
hard-splits a paragraph that exceeds the budget alone.

The heading *path* (rather than just the nearest heading) is what makes citations
usable in a legal or tax corpus. "Division 3 > Section 8-1" tells a reader exactly
where to verify a claim; "sla.md" does not. The path is also prepended to the
embedded text, so hierarchy influences the vector rather than only the label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from .loaders import discover, load_document

# cl100k_base is the tokenizer for the text-embedding-3-* family.
_ENC = tiktoken.get_encoding("cl100k_base")

_PAGE_RE = re.compile(r"<!--page:(\d+)-->")
PATH_SEPARATOR = " › "


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    heading: str = ""
    heading_path: str = ""
    page: int | None = None
    token_count: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = count_tokens(self.text)

    @property
    def citation(self) -> str:
        """Human-readable location, precise enough for a reader to go and verify."""
        parts = [self.source]
        if self.heading_path:
            parts.append(self.heading_path)
        if self.page:
            parts.append(f"p.{self.page}")
        return " · ".join(parts)


class _HeadingStack:
    """Tracks the current position in the document hierarchy."""

    def __init__(self) -> None:
        self._levels: dict[int, str] = {}

    def push(self, level: int, title: str) -> None:
        self._levels[level] = title
        # A new heading at level N invalidates everything nested beneath it.
        for deeper in [lv for lv in self._levels if lv > level]:
            del self._levels[deeper]

    @property
    def path(self) -> str:
        return PATH_SEPARATOR.join(self._levels[lv] for lv in sorted(self._levels))

    @property
    def leaf(self) -> str:
        return self._levels[max(self._levels)] if self._levels else ""


def _split_long_paragraph(para: str, max_tokens: int) -> list[str]:
    """Hard-split an oversized paragraph, preferring sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out, buf, buf_tokens = [], [], 0
    for sent in sentences:
        st = count_tokens(sent)
        if buf and buf_tokens + st > max_tokens:
            out.append(" ".join(buf))
            buf, buf_tokens = [], 0
        buf.append(sent)
        buf_tokens += st
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_document(path: Path, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    raw = load_document(path)
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

    chunks: list[Chunk] = []
    stack = _HeadingStack()
    page: int | None = None
    buf: list[str] = []
    buf_tokens = 0
    seq = 0

    def flush() -> None:
        nonlocal buf, buf_tokens, seq
        if not buf:
            return
        body = "\n\n".join(buf).strip()
        if not body:
            buf, buf_tokens = [], 0
            return
        path_prefix = stack.path
        text = f"{path_prefix}\n\n{body}" if path_prefix else body
        chunks.append(
            Chunk(
                id=f"{path.stem}-{seq:03d}",
                text=text,
                source=path.name,
                heading=stack.leaf,
                heading_path=path_prefix,
                page=page,
            )
        )
        seq += 1
        if overlap_tokens > 0:
            tail = _ENC.encode(body)[-overlap_tokens:]
            buf = [_ENC.decode(tail)] if tail else []
        else:
            buf = []
        buf_tokens = count_tokens(buf[0]) if buf else 0

    for block in blocks:
        page_match = _PAGE_RE.search(block)
        if page_match:
            page = int(page_match.group(1))
            block = _PAGE_RE.sub("", block).strip()
            if not block:
                continue

        if block.lstrip().startswith("#"):
            flush()
            level = len(block) - len(block.lstrip("#"))
            stack.push(level, block.strip().lstrip("# ").strip())
            buf, buf_tokens = [], 0
            continue

        bt = count_tokens(block)
        if bt > max_tokens:
            flush()
            for piece in _split_long_paragraph(block, max_tokens):
                buf, buf_tokens = [piece], count_tokens(piece)
                flush()
            continue

        if buf_tokens + bt > max_tokens:
            flush()
        buf.append(block)
        buf_tokens += bt

    flush()
    return [c for c in chunks if c.text.strip() and c.token_count > 20]


def chunk_corpus(data_dir: Path, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in discover(data_dir):
        chunks.extend(chunk_document(path, max_tokens, overlap_tokens))
    return chunks


# Backwards-compatible alias -- ingest.py and the evals still call this name.
chunk_markdown = chunk_document
