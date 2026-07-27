# Azure OpenAI RAG — proof of concept

A grounded question-answering system over a document corpus, built on Azure OpenAI.
Small on purpose: the interesting part is not that it answers questions, it's the
retrieval strategy, the cost controls, and the fact that it's measured.

**Runs for well under $0.05 end-to-end, including the eval suite.**

```bash
python -m src.ingest                          # chunk, embed, index
python -m src.cli "how long do I have to file an SLA credit claim?"
python -m eval.evaluate                       # score the whole thing
```

---

## What this demonstrates

| Area | Where to look |
|---|---|
| Entra ID auth over API keys | `src/aoai.py` — `DefaultAzureCredential` is the default path |
| Rate-limit handling | `src/aoai.py` — batched embeddings, exponential backoff on 429 |
| Token-aware chunking | `src/chunking.py` — semantic boundaries, heading propagation, overlap |
| Hybrid retrieval + semantic reranking | `src/retrievers/azure_search.py` |
| Pluggable retrieval backends | `src/retrievers/base.py` — Protocol, swapped by config |
| Grounding & refusal | `src/rag.py` — relevance floor, citation enforcement |
| Evaluation | `eval/evaluate.py` — recall@k, correctness, groundedness, refusal |
| Cost instrumentation | token counts and USD on every single query |

---

## Architecture

```
data/*.md
    │
    ▼
chunking.py ──── token-aware split (400 tok, 60 overlap)
    │            headings prepended so they influence the embedding
    ▼
aoai.py ──────── text-embedding-3-small, batched 128/req, retried on 429
    │
    ▼
Retriever (Protocol)
    ├── LocalRetriever ......... numpy cosine, $0 standing cost      [default]
    └── AzureSearchRetriever ... hybrid BM25+vector → semantic rerank
    │
    ▼
rag.py ───────── relevance floor → grounded prompt → gpt-4o-mini @ temp 0
    │            citations resolved back to chunks; usage returned
    ▼
Answer(text, hits, tokens, cost, refused, cited_indices)
```

---

## Design decisions

Each of these is a question an interviewer is likely to ask.

### Why two retrieval backends?

Retrieval strategy is the part of a RAG system you change most often, and the part
with the largest cost implications. Putting it behind a `Protocol` means moving from
brute-force numpy to Azure AI Search is a one-line config change, and the same eval
harness scores both — so the decision is made on measured recall, not vibes.

It also solves a practical problem: Azure AI Search bills **hourly, whether you query
it or not**. Defaulting to the local backend means a reviewer can clone this repo and
run it without provisioning a search service.

### Why brute-force cosine similarity instead of an ANN index?

At POC scale (hundreds to low thousands of chunks) an exhaustive scan over a
normalised matrix is a single BLAS call — genuinely faster than HNSW, with perfect
recall and no graph build. ANN indexes trade recall for latency, and that trade only
pays above roughly 10⁵ vectors.

Vectors are L2-normalised at write time so query-time cosine similarity is a plain dot
product, and `argpartition` avoids a full sort when `top_k << n`.

The local backend stops being the right answer the moment you need filtering, keyword
matching, incremental updates, or multi-process access. That's the handoff to Azure AI
Search — not corpus size alone.

### Why hybrid search rather than pure vector?

Pure vector search reliably fails on exact-match tokens. In this corpus, `SLA-4402` is
the claim type you file service credits under. It is a near-meaningless string to an
embedding model — it has no semantic neighbourhood — but BM25 matches it exactly.
Golden `g02` exists specifically to catch this regression.

Hybrid runs both queries and fuses the ranked lists with Reciprocal Rank Fusion. The
semantic ranker then re-scores the fused shortlist with a cross-encoder: a bi-encoder
embeds query and document independently, whereas a cross-encoder sees both together
and is substantially better at relevance — too slow for the full corpus, ideal over a
shortlist of ~50. Standard two-stage retrieval.

### Why prepend headings to chunk text?

A chunk reading *"Credits are applied against the following month's invoice"* is
ambiguous alone. Prepending `## Claiming service credits` disambiguates it, and
because the heading is part of the **embedded** text — not just the citation metadata
— it shifts the vector toward the right semantic region.

### Why a relevance floor before the LLM call?

If nothing clears `MIN_RELEVANCE`, the system refuses without spending a chat token.
Most demos skip this and will happily answer out-of-scope questions from the model's
parametric memory, wearing a citation it didn't use. Goldens `g13` and `g14` are
unanswerable by construction and exist to measure exactly that.

An ungrounded RAG system is worse than no RAG system: it launders a hallucination
through the credibility of a citation.

### Why temperature 0?

This is extraction, not composition. Non-determinism here buys nothing and makes the
eval suite noisy.

### Why is chunk overlap doing nothing on this corpus?

Honest answer: with 400-token chunks and a corpus whose sections are all under ~90
tokens, every section becomes exactly one chunk and overlap never engages. That's the
correct outcome — the section boundary *is* the semantic boundary. Overlap matters on
long unstructured documents, and the code path is exercised there (`_split_long_paragraph`).
I'd rather explain this than tune the numbers to make a feature look busy.

---

## Cost

This was designed around a personal Azure subscription, so cost control is a
first-class constraint rather than an afterthought.

**Verify current rates before quoting these — Azure pricing moves.**

| Item | Rate (approx., mid-2026) | This POC |
|---|---|---|
| `text-embedding-3-small` | ~$0.02 / 1M tokens | corpus is ~1.5K tokens → **~$0.00003** |
| `gpt-4o-mini` | ~$0.15 in / $0.60 out per 1M | ~1.5K tokens/query → **~$0.0005/query** |
| Full eval run (14 goldens × 2 calls) | — | **~$0.02** |
| Azure AI Search **Free** tier | $0 | 3 indexes, 50 MB, **no semantic ranker** |
| Azure AI Search **Basic** | ~$74/month, billed hourly | **~$0.10/hour** |

The token costs are rounding errors. **The only thing that can hurt you is a search
service left running.** Hence the local-first default.

Three ways to keep the Azure AI Search bill near zero:

1. **Stay on the local backend** (default). Everything except hybrid/semantic ranking works.
2. **Free tier** for structural work — index schema, hybrid queries, BM25. Semantic
   reranking is a separately-billed premium feature and isn't available here.
3. **Basic for one afternoon, then delete it.** At roughly $0.10/hour, provisioning it,
   running the eval on both backends, screenshotting the comparison, and tearing it
   down costs under a dollar. `infra/teardown.sh` deletes the whole resource group.

Azure AI Search also now offers a **Serverless** (consumption-based) pricing model
alongside the hourly Dedicated tiers — worth checking, since consumption billing suits
a POC far better than an hourly SKU.

New Azure subscriptions typically include trial credit that covers all of this. Azure
OpenAI itself has no permanent free tier — it's pay-per-token from the first request.

**Set a budget alert on the subscription before you deploy anything.** Note the
deliberate irony that `data/billing.md` documents that budget alerts don't actually
stop consumption — only sandbox subscriptions support hard caps.

---

## Evaluation

`python -m eval.evaluate` scores four things, deliberately separating *retrieval*
failure from *generation* failure:

- **Retrieval recall@k** — did the correct document reach the context at all? If this
  is low, no prompt engineering will fix the answer.
- **Answer correctness** — is the required fact present? Substring matching: crude,
  but deterministic and free.
- **Groundedness** — LLM-as-judge over (context, answer). Catches the dangerous
  failure mode: fluent, confident, unsupported.
- **Refusal accuracy** — does it decline the two out-of-scope questions? A system that
  never refuses has just moved its hallucinations off the test set.

This is the piece that separates a POC from a demo. Without it, "should chunks be 400
or 800 tokens?" has no answer beyond taste, and you can't tell whether swapping to
hybrid retrieval actually helped.

Suggested experiment to run and report: score `RETRIEVER=local` against
`RETRIEVER=azure_search`, and show `g02` (`SLA-4402`) flipping from FAIL to PASS.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # set AZURE_OPENAI_ENDPOINT
az login                  # Entra ID auth — no key needed
```

You need two Azure OpenAI deployments: a chat model (`gpt-4o-mini` or `gpt-4.1-mini`)
and `text-embedding-3-small`. For Entra ID auth, assign yourself the **Cognitive
Services OpenAI User** role on the resource. If you can't assign roles, set
`AZURE_OPENAI_API_KEY` instead — the code falls back and logs a warning.

To try the Azure AI Search backend: set `RETRIEVER=azure_search` and
`AZURE_SEARCH_ENDPOINT`, then re-run `python -m src.ingest`.

---

## What I'd change for production

Being explicit about the gap between a POC and a real system is usually the point of
the exercise.

- **Ingestion** — this reads local markdown. Real ingestion means Azure AI Search
  indexers or Document Intelligence for PDF layout, plus change detection so you
  re-embed only what moved.
- **Security trimming** — the hard problem nobody demos. Retrieval must filter by the
  *caller's* permissions at query time, or RAG becomes a data-exfiltration channel.
  In Azure AI Search that's a security-filter field populated with group OIDs and an
  `$filter` built from the caller's token.
- **Query rewriting** — multi-turn conversation breaks naive retrieval, because
  "what about Premium?" embeds to nothing useful. Rewrite to a standalone query first.
- **Observability** — App Insights traces per stage: retrieve, rerank, generate.
  Log retrieved chunk IDs so a bad answer can be traced to bad retrieval.
- **Content safety** — Azure AI Content Safety on input and output, plus prompt-shield
  for injection attempts embedded in indexed documents. A poisoned document is a real
  attack surface in RAG specifically.
- **Caching** — semantic cache on near-duplicate queries; typical support workloads
  are extremely repetitive.
- **Evaluation at scale** — this harness is the right shape but 14 goldens is not a
  test set. Move to Azure AI Foundry evaluations, run it in CI, and gate deploys on
  a groundedness threshold.
