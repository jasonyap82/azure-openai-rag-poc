"""Demonstration interface.

The job of this screen is not the chat -- it is making retrieval *visible*. An
executive's real question about any AI answer is "why should I believe this?", so
the evidence apparatus below each answer is the point, and the conversation is
merely the way you provoke it.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import time
from pathlib import Path

import streamlit as st

from src.aoai import build_client, chat_kwargs
from src.config import get_settings
from src.rag import answer_question, format_context, get_retriever

# --------------------------------------------------------------------------- setup

st.set_page_config(page_title="Document Assistant", page_icon="§", layout="wide")

PALETTE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --paper:      #eef1f5;
  --card:       #ffffff;
  --ink:        #17233a;
  --ink-soft:   #5a6884;
  --brass:      #96742b;
  --rule:       #c9d2de;
  --strong:     #1f6f5c;
  --moderate:   #a8641b;
  --weak:       #96313a;
}

.stApp { background: var(--paper); }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }

.masthead {
  border-bottom: 2px solid var(--ink);
  padding-bottom: .6rem;
  margin-bottom: .35rem;
}
.masthead h1 {
  font-family: 'Spectral', serif;
  font-weight: 600;
  font-size: 2.1rem;
  letter-spacing: -.015em;
  margin: 0;
  color: var(--ink);
}
.masthead .sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .72rem;
  letter-spacing: .13em;
  text-transform: uppercase;
  color: var(--brass);
  margin-top: .3rem;
}
.disclaimer {
  font-size: .76rem;
  color: var(--ink-soft);
  margin-bottom: 1.4rem;
}

/* ---- pipeline strip: the live mechanism, stage by stage ---- */
.pipeline {
  display: flex;
  gap: 0;
  margin: .5rem 0 1.1rem 0;
  border: 1px solid var(--rule);
  background: var(--card);
  flex-wrap: wrap;
}
.stage {
  flex: 1 1 130px;
  padding: .6rem .75rem;
  border-right: 1px solid var(--rule);
  position: relative;
}
.stage:last-child { border-right: none; }
.stage .n {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .62rem;
  color: var(--brass);
  letter-spacing: .1em;
}
.stage .v {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.22rem;
  font-weight: 500;
  color: var(--ink);
  line-height: 1.3;
}
.stage .l {
  font-size: .69rem;
  color: var(--ink-soft);
  line-height: 1.25;
}

/* ---- authority cards: the signature element ---- */
.authority {
  display: flex;
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--brass);
  margin-bottom: .5rem;
}
.authority.unused { border-left-color: var(--rule); opacity: .72; }
.authority .margin {
  flex: 0 0 2.6rem;
  padding: .65rem .4rem;
  text-align: center;
  font-family: 'IBM Plex Mono', monospace;
  font-size: .95rem;
  color: var(--brass);
  border-right: 1px solid var(--rule);
}
.authority.unused .margin { color: var(--ink-soft); }
.authority .body { flex: 1; padding: .6rem .8rem; min-width: 0; }
.authority .path {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .74rem;
  color: var(--ink);
  word-break: break-word;
}
.authority .meta {
  font-size: .69rem;
  color: var(--ink-soft);
  margin-top: .18rem;
}
.authority .tag {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .62rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  padding: .08rem .35rem;
  border: 1px solid currentColor;
  margin-left: .4rem;
}
.tag.cited { color: var(--strong); }
.tag.unused { color: var(--ink-soft); }

.bar { height: 4px; background: var(--rule); margin-top: .45rem; position: relative; }
.bar .fill { height: 100%; background: var(--brass); }
.authority.unused .bar .fill { background: var(--ink-soft); }
.bar .threshold {
  position: absolute; top: -3px; width: 1px; height: 10px; background: var(--weak);
}

.verdict {
  background: var(--card);
  border: 1px solid var(--rule);
  border-top: 3px solid var(--ink);
  padding: .7rem .9rem;
}
.verdict .row { display: flex; justify-content: space-between; gap: 1rem; padding: .2rem 0; }
.verdict .k { font-size: .78rem; color: var(--ink-soft); }
.verdict .v { font-family: 'IBM Plex Mono', monospace; font-size: .78rem; font-weight: 500; }
.v.strong { color: var(--strong); }
.v.moderate { color: var(--moderate); }
.v.weak { color: var(--weak); }

.section-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: .66rem;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--brass);
  border-bottom: 1px solid var(--rule);
  padding-bottom: .25rem;
  margin: .3rem 0 .6rem 0;
}
</style>
"""
st.markdown(PALETTE_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------- cached resources


@st.cache_resource(show_spinner=False)
def load_retriever():
    retriever = get_retriever()
    if hasattr(retriever, "load"):
        retriever.load()
    return retriever


@st.cache_resource(show_spinner=False)
def load_client():
    return build_client()


@st.cache_data(show_spinner=False)
def corpus_summary(data_dir_str: str) -> tuple[int, list[str]]:
    from src.loaders import discover

    files = discover(Path(data_dir_str))
    return len(files), [f.name for f in files]


JUDGE_PROMPT = """You are grading a retrieval-augmented answer for groundedness.

Given CONTEXT and ANSWER, decide whether every factual claim in ANSWER is directly
supported by CONTEXT. Ignore style, tone and completeness -- judge support only.

Reply with exactly one word:
GROUNDED   - every claim is supported by the context
UNGROUNDED - at least one claim is not supported
REFUSAL    - the answer declines to answer rather than making claims"""


def check_grounding(context: str, answer: str) -> str:
    s = get_settings()
    client = load_client()
    resp = client.chat.completions.create(
        model=s.chat_deployment,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"},
        ],
        **chat_kwargs(s.chat_deployment, 5),
    )
    return (resp.choices[0].message.content or "").strip().upper()


# ------------------------------------------------------------------- render parts


def render_pipeline(stats: dict) -> None:
    stages = [
        ("01", stats["dim"], "dimensions in the question vector"),
        ("02", stats["corpus"], "passages searched"),
        ("03", stats["retrieved"], "cleared the relevance floor"),
        ("04", f"{stats['context_tokens']:,}", "tokens of source text supplied"),
        ("05", f"{stats['elapsed']:.1f}s", "to retrieve and answer"),
    ]
    cells = "".join(
        f'<div class="stage"><div class="n">{n}</div>'
        f'<div class="v">{v}</div><div class="l">{label}</div></div>'
        for n, v, label in stages
    )
    st.markdown(f'<div class="pipeline">{cells}</div>', unsafe_allow_html=True)


def render_authorities(hits, cited: list[int], threshold: float) -> None:
    st.markdown('<div class="section-label">Sources consulted</div>', unsafe_allow_html=True)
    for i, hit in enumerate(hits, start=1):
        used = i in cited
        pct = max(0.0, min(1.0, hit.score)) * 100
        chunk = hit.chunk
        location = html.escape(chunk.heading_path or chunk.heading or "(no heading)")
        page = f" · p.{chunk.page}" if chunk.page else ""
        st.markdown(
            f'<div class="authority {"" if used else "unused"}">'
            f'<div class="margin">{i}</div>'
            f'<div class="body">'
            f'<div class="path">{location}</div>'
            f'<div class="meta">{html.escape(chunk.source)}{page} · '
            f"relevance {hit.score:.3f} · {hit.origin}"
            f'<span class="tag {"cited" if used else "unused"}">'
            f'{"cited" if used else "not used"}</span></div>'
            f'<div class="bar"><div class="fill" style="width:{pct:.1f}%"></div>'
            f'<div class="threshold" style="left:{threshold * 100:.1f}%"></div></div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Read passage {i} in full"):
            st.text(chunk.text)


def render_verdict(ans, grounding: str | None, threshold: float) -> None:
    top = ans.hits[0].score if ans.hits else 0.0
    margin = top - threshold
    if margin > 0.25:
        conf, cls = "Strong", "strong"
    elif margin > 0.08:
        conf, cls = "Moderate", "moderate"
    else:
        conf, cls = "Marginal", "weak"

    rows = [
        ("Retrieval confidence", f"{conf} — top match {top:.3f} vs floor {threshold:.2f}", cls),
    ]
    if grounding:
        gcls = {"GROUNDED": "strong", "UNGROUNDED": "weak"}.get(grounding, "moderate")
        rows.append(("Grounding check", grounding.title(), gcls))
    rows.append(
        (
            "Cost of this answer",
            f"${ans.estimated_cost_usd:.5f} · {ans.prompt_tokens} in / {ans.completion_tokens} out",
            "",
        )
    )

    body = "".join(
        f'<div class="row"><span class="k">{k}</span>'
        f'<span class="v {cls}">{html.escape(str(v))}</span></div>'
        for k, v, cls in rows
    )
    st.markdown(f'<div class="verdict">{body}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------ sidebar

settings = get_settings()

with st.sidebar:
    st.markdown('<div class="section-label">Assistant</div>', unsafe_allow_html=True)
    assistant_name = st.text_input("Name shown to users", value="Taxation Assistant")

    n_files, filenames = corpus_summary(str(settings.data_dir))
    st.caption(f"{n_files} document(s) indexed")
    for name in filenames:
        st.caption(f"· {name}")

    st.markdown('<div class="section-label">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Passages to retrieve", 1, 10, settings.top_k)
    threshold = st.slider("Relevance floor", 0.0, 0.8, settings.min_relevance, 0.05)
    st.caption("Below this score a passage is discarded. Raise it to make the assistant more willing to say it doesn't know.")

    run_grounding = st.toggle("Run grounding check", value=True)
    st.caption("A second model call that verifies every claim traces to a retrieved passage. Roughly doubles the cost per question.")

    if st.button("Rebuild index"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# ------------------------------------------------------------------------- header

st.markdown(
    f'<div class="masthead"><h1>{html.escape(assistant_name)}</h1>'
    f'<div class="sub">Retrieval-augmented · every answer traced to source</div></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="disclaimer">Demonstration system. Answers are drawn only from the '
    "indexed documents and are not professional advice.</div>",
    unsafe_allow_html=True,
)

with st.expander("How this assistant produces an answer"):
    st.markdown(
        """
The assistant has **no knowledge of these documents**. It was never trained on them.

Each document is split into passages and converted into a numeric vector capturing its
meaning. When you ask a question, the question is converted the same way, and the
passages with the closest meaning are pulled out and placed into the model's prompt
alongside your question. The model is instructed to answer **only** from those
passages, and to say so when they don't contain the answer.

Everything below each answer is the actual evidence used — the passages retrieved,
how closely each matched, and which ones the answer drew on. Nothing is reconstructed
after the fact.

**On accuracy:** no system can report its own accuracy on a question whose answer
isn't already known. What is shown per answer is *retrieval confidence* (how strongly
the sources matched) and a *grounding check* (a second model verifying each claim
traces to a retrieved passage). Actual accuracy is measured separately, offline,
against a fixed set of questions with known answers.
        """
    )

# --------------------------------------------------------------------------- chat

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    retriever = load_retriever()
except FileNotFoundError:
    st.error("No index found. Run `python -m src.ingest` in the project folder first.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            trace = msg["trace"]
            render_pipeline(trace["stats"])
            if trace["hits"]:
                render_authorities(trace["hits"], trace["cited"], trace["threshold"])
            render_verdict(trace["answer"], trace.get("grounding"), trace["threshold"])

if question := st.chat_input("Ask a question about the indexed documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            started = time.perf_counter()
            settings.min_relevance = threshold
            ans = answer_question(question, retriever, top_k=top_k)
            elapsed = time.perf_counter() - started

            grounding = None
            if run_grounding and ans.hits and not ans.refused:
                grounding = check_grounding(format_context(ans.hits), ans.text)

        st.markdown(ans.text)

        corpus_size = len(getattr(retriever, "_chunks", []) or [])
        stats = {
            "dim": 1536,
            "corpus": corpus_size or "—",
            "retrieved": len(ans.hits),
            "context_tokens": sum(h.chunk.token_count for h in ans.hits),
            "elapsed": elapsed,
        }
        render_pipeline(stats)
        if ans.hits:
            render_authorities(ans.hits, ans.cited_indices, threshold)
        else:
            st.info(
                "No passage in the indexed documents scored above the relevance floor, "
                "so the assistant declined rather than guessing."
            )
        render_verdict(ans, grounding, threshold)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": ans.text,
            "trace": {
                "stats": stats,
                "hits": ans.hits,
                "cited": ans.cited_indices,
                "answer": ans,
                "grounding": grounding,
                "threshold": threshold,
            },
        }
    )
