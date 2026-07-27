"""Token-aware chunking.

Naive character splitting is the single most common reason a RAG demo retrieves
garbage: it cuts mid-sentence and destroys the semantic unit the embedding is
supposed to represent. This splits on paragraph boundaries first, packs paragraphs
up to a token budget, and only hard-splits a paragraph that exceeds the budget alone.

Overlap exists so a fact that straddles a boundary appears whole in at least one chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

# cl100k_base is the tokenizer for the text-embedding-3-* and gpt-4o families.
_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    heading: str = ""
    token_count: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = count_tokens(self.text)

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.heading}" if self.heading else self.source


def _split_long_paragraph(para: str, max_tokens: int) -> list[str]:
    """Hard-split a paragraph that will not fit, at sentence boundaries where possible."""
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


def chunk_markdown(path: Path, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Chunk a markdown file, carrying the nearest '#' heading onto every chunk.

    Carrying the heading matters: a chunk reading 'Refunds are prorated to the day'
    is nearly useless without knowing it sits under '## Billing'. The heading gets
    prepended to the embedded text so it influences the vector, not just the citation.
    """
    raw = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]

    chunks: list[Chunk] = []
    heading = ""
    buf: list[str] = []
    buf_tokens = 0
    seq = 0

    def flush() -> None:
        nonlocal buf, buf_tokens, seq
        if not buf:
            return
        body = "\n\n".join(buf)
        text = f"{heading}\n\n{body}" if heading else body
        chunks.append(
            Chunk(id=f"{path.stem}-{seq:03d}", text=text, source=path.name, heading=heading)
        )
        seq += 1
        # Carry the tail of this chunk into the next one for overlap.
        if overlap_tokens > 0:
            tail_ids = _ENC.encode(body)[-overlap_tokens:]
            buf = [_ENC.decode(tail_ids)] if tail_ids else []
        else:
            buf = []
        buf_tokens = count_tokens(buf[0]) if buf else 0

    for block in blocks:
        if block.lstrip().startswith("#"):
            flush()
            heading = block.strip().lstrip("# ").strip()
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
    for path in sorted(data_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path, max_tokens, overlap_tokens))
    return chunks
