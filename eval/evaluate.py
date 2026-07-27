"""Evaluation harness.

Most RAG demos stop at "it answered my question." That proves nothing: you cannot
tune chunk size, top_k, or the retrieval backend without a number that moves. This
measures four things, deliberately separating retrieval failure from generation failure:

  1. Retrieval recall@k -- did the right document make it into the context at all?
     If this is low, no amount of prompt engineering will save the answer.
  2. Answer correctness -- does the answer contain the required fact? Substring
     matching, which is crude but deterministic and free.
  3. Groundedness -- LLM-as-judge, asking whether every claim is supported by the
     retrieved context. This is what catches the dangerous failure: a fluent,
     confident, wrong answer.
  4. Refusal accuracy -- does it correctly decline the out-of-scope questions? A
     system that never refuses has simply moved its hallucinations off your test set.

Run:  python -m eval.evaluate
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.aoai import build_client
from src.config import get_settings
from src.rag import Answer, answer_question, format_context, get_retriever

console = Console()
GOLDENS = Path(__file__).parent / "goldens.jsonl"

JUDGE_PROMPT = """You are grading a retrieval-augmented answer for groundedness.

Given CONTEXT and ANSWER, decide whether every factual claim in ANSWER is directly
supported by CONTEXT. Ignore style, tone, and completeness -- judge support only.

Reply with exactly one word:
GROUNDED   - every claim is supported by the context
UNGROUNDED - at least one claim is not supported
REFUSAL    - the answer declines to answer rather than making claims"""


def judge_groundedness(client, context: str, answer: str) -> str:
    s = get_settings()
    resp = client.chat.completions.create(
        model=s.chat_deployment,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    return (resp.choices[0].message.content or "").strip().upper()


def score_one(golden: dict, ans: Answer, client) -> dict:
    retrieved_sources = {h.chunk.source for h in ans.hits}

    if golden["answerable"]:
        recall = golden["expected_source"] in retrieved_sources
        correct = all(t.lower() in ans.text.lower() for t in golden["must_include"])
        refusal_ok = not ans.refused
    else:
        recall = None  # not meaningful for out-of-scope questions
        correct = None
        refusal_ok = ans.refused

    grounded = "N/A"
    if ans.hits and not ans.refused:
        grounded = judge_groundedness(client, format_context(ans.hits), ans.text)

    return {
        "id": golden["id"],
        "recall": recall,
        "correct": correct,
        "refusal_ok": refusal_ok,
        "grounded": grounded,
        "tokens": ans.prompt_tokens + ans.completion_tokens,
        "cost": ans.estimated_cost_usd,
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    s = get_settings()
    retriever = get_retriever()
    client = build_client()

    goldens = [json.loads(line) for line in GOLDENS.read_text().splitlines() if line.strip()]
    rows = []
    for g in goldens:
        ans = answer_question(g["question"], retriever)
        rows.append(score_one(g, ans, client))
        console.print(f"[dim]scored {g['id']}[/dim]")

    table = Table(title=f"Results (backend={s.retriever}, top_k={s.top_k})")
    for col in ("id", "recall@k", "correct", "refusal", "grounded", "tokens"):
        table.add_column(col)
    for r in rows:
        fmt = lambda v: "-" if v is None else ("[green]PASS[/green]" if v else "[red]FAIL[/red]")
        table.add_row(
            r["id"], fmt(r["recall"]), fmt(r["correct"]), fmt(r["refusal_ok"]),
            r["grounded"], str(r["tokens"]),
        )
    console.print(table)

    answerable = [r for r in rows if r["recall"] is not None]
    pct = lambda xs: f"{100 * sum(xs) / len(xs):.0f}%" if xs else "n/a"

    console.print("\n[bold]Summary[/bold]")
    console.print(f"  retrieval recall@{s.top_k}: {pct([r['recall'] for r in answerable])}")
    console.print(f"  answer correctness:  {pct([r['correct'] for r in answerable])}")
    console.print(f"  refusal accuracy:    {pct([r['refusal_ok'] for r in rows])}")
    console.print(f"  ungrounded answers:  {sum(r['grounded'] == 'UNGROUNDED' for r in rows)}")
    console.print(f"  total eval cost:     ${sum(r['cost'] for r in rows):.5f}")


if __name__ == "__main__":
    main()
