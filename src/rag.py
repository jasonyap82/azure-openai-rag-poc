"""RAG orchestration: retrieve -> ground -> generate -> attribute.

Design decisions worth defending:

  * The model is instructed to answer ONLY from context and to say so when the context
    is insufficient. An ungrounded RAG system is worse than no RAG system, because it
    launders a hallucination through the credibility of a citation.

  * A relevance floor is applied BEFORE the LLM call. If nothing clears it we refuse
    without spending a chat token. Most demos skip this and cheerfully answer
    out-of-scope questions from parametric memory.

  * Every source is numbered and the model must cite [1], [2]. Citations are then
    resolved back to real chunks so the caller can verify them -- an unverifiable
    citation is decoration.

  * Token usage is returned on every call. If you cannot answer "what does a query
    cost?" you cannot capacity-plan, and that question comes up in every real review.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .aoai import build_client, chat_kwargs
from .config import get_settings
from .retrievers.base import Hit, Retriever

SYSTEM_PROMPT = """You are a support assistant for Contoso Cloud Services.

Rules, in priority order:
1. Answer using ONLY the numbered sources provided. Never use outside knowledge.
2. Cite the source number in square brackets after each claim, e.g. [2].
3. If the sources do not contain the answer, reply exactly:
   "I don't have that in my sources." Then state what related information you do have.
4. If sources conflict, surface the conflict rather than silently picking one.
5. Be concise. No preamble, no restating the question."""

# Pricing is per 1M tokens and changes; treat these as illustrative and override via
# the deployment you actually use. Kept here so cost is visible at the call site.
PRICE_PER_MTOK = {"input": 0.15, "output": 0.60}


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    refused: bool = False
    cited_indices: list[int] = field(default_factory=list)

    @property
    def estimated_cost_usd(self) -> float:
        return (
            self.prompt_tokens * PRICE_PER_MTOK["input"]
            + self.completion_tokens * PRICE_PER_MTOK["output"]
        ) / 1_000_000

    @property
    def sources(self) -> list[str]:
        return [h.chunk.citation for h in self.hits]


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.chunk.citation})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    )


def _extract_citations(text: str, n_sources: int) -> list[int]:
    import re

    found = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return sorted(i for i in found if 1 <= i <= n_sources)


def answer_question(question: str, retriever: Retriever, top_k: int | None = None) -> Answer:
    s = get_settings()
    top_k = top_k or s.top_k

    hits = retriever.search(question, top_k)
    hits = [h for h in hits if h.score >= s.min_relevance]

    if not hits:
        return Answer(
            text="I don't have that in my sources.",
            hits=[],
            refused=True,
        )

    client = build_client()
    resp = client.chat.completions.create(
        model=s.chat_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Sources:\n{format_context(hits)}\n\nQuestion: {question}",
            },
        ],
        **chat_kwargs(s.chat_deployment, 500),
    )

    text = resp.choices[0].message.content or ""
    usage = resp.usage
    return Answer(
        text=text,
        hits=hits,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        refused=text.strip().startswith("I don't have that in my sources"),
        cited_indices=_extract_citations(text, len(hits)),
    )


def get_retriever() -> Retriever:
    """Factory -- the only place the backend choice is made."""
    s = get_settings()
    if s.retriever == "azure_search":
        from .retrievers.azure_search import AzureSearchRetriever

        return AzureSearchRetriever()
    from .retrievers.local import LocalRetriever

    return LocalRetriever()
