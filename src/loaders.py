"""Document loaders: normalise .md / .txt / .pdf / .docx into headed markdown.

Everything downstream assumes markdown with '#' headings, because the heading
hierarchy is what makes a citation useful. "Section 8-1(2), Division 3" is a
citation a tax professional can verify; "chunk 47" is not.

PDF heading detection is necessarily heuristic -- a PDF has no semantic structure,
only glyphs at coordinates. The patterns below target legal and tax drafting
conventions (Part / Division / Subdivision / numbered sections). They are declared
in one place so they can be tuned per corpus rather than buried in parsing code.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

# (level, pattern). Lower level = higher in the hierarchy. Order matters: the first
# match wins, so more specific patterns must come first.
LEGAL_HEADING_PATTERNS: list[tuple[int, re.Pattern]] = [
    (1, re.compile(r"^\s*(PART\s+[IVXLC\d]+[A-Z]?\b.{0,80})$", re.I)),
    (2, re.compile(r"^\s*(DIVISION\s+[\d.]+[A-Z]?\b.{0,80})$", re.I)),
    (2, re.compile(r"^\s*(CHAPTER\s+[\d.]+[A-Z]?\b.{0,80})$", re.I)),
    (3, re.compile(r"^\s*(SUBDIVISION\s+[\w.\-]+\b.{0,80})$", re.I)),
    (3, re.compile(r"^\s*(SECTION\s+[\d\w.\-]+\b.{0,80})$", re.I)),
    # Bare section numbers as used in tax acts: "8-1 General deductions"
    (3, re.compile(r"^\s*(\d+[A-Z]?[-–]\d+[A-Z]?\s+\S.{2,80})$")),
    # Plain numbered clauses: "12.4 Deductible expenses"
    (3, re.compile(r"^\s*(\d+\.\d+(?:\.\d+)?\s+\S.{2,80})$")),
]

PAGE_MARKER = "<!--page:{n}-->"


def _detect_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return None
    for level, pattern in LEGAL_HEADING_PATTERNS:
        m = pattern.match(stripped)
        if m:
            return level, m.group(1).strip()
    # ALL-CAPS short lines are conventionally headings in legislative drafting.
    if 4 <= len(stripped) <= 70 and stripped.isupper() and any(c.isalpha() for c in stripped):
        return 2, stripped.title()
    return None


def _apply_heading_patterns(text: str) -> str:
    """Insert markdown heading markers into unstructured text."""
    out: list[str] = []
    for line in text.splitlines():
        detected = _detect_heading(line)
        if detected:
            level, title = detected
            out.extend(["", "#" * level + " " + title, ""])
        else:
            out.append(line)
    return "\n".join(out)


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_text(path: Path) -> str:
    return _apply_heading_patterns(path.read_text(encoding="utf-8", errors="replace"))


def load_pdf(path: Path) -> str:
    """Extract text page by page, tagging each page so citations can name it."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support needs pypdf: pip install pypdf") from exc

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page_no, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        if not raw.strip():
            continue
        parts.append(PAGE_MARKER.format(n=page_no))
        parts.append(_apply_heading_patterns(raw))
    if not parts:
        log.warning("%s produced no extractable text -- it may be a scanned image.", path.name)
    return "\n\n".join(parts)


def load_docx(path: Path) -> str:
    """Map Word heading styles straight to markdown levels -- no guessing needed."""
    try:
        import docx
    except ImportError as exc:
        raise RuntimeError("DOCX support needs python-docx: pip install python-docx") from exc

    document = docx.Document(str(path))
    parts: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        if style.startswith("heading"):
            try:
                level = int(style.split()[-1])
            except (ValueError, IndexError):
                level = 2
            parts.append("#" * min(level, 4) + " " + text)
        elif style.startswith("title"):
            parts.append("# " + text)
        else:
            parts.append(text)
    return "\n\n".join(parts)


LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
    ".pdf": load_pdf,
    ".docx": load_docx,
}

SUPPORTED_SUFFIXES = tuple(LOADERS)


def load_document(path: Path) -> str:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader(path)


def discover(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
