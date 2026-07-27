"""Azure AI Search retriever -- hybrid (vector + BM25) retrieval with semantic reranking.

This is the part to talk through in an interview, because it is where Azure AI Search
earns its cost over a plain vector store:

  * Hybrid search runs a vector query and a BM25 keyword query in parallel and fuses
    the result lists with Reciprocal Rank Fusion. Pure vector search reliably fails on
    exact-match tokens -- error codes, SKU names, surnames, "SLA-4402" -- because those
    strings carry almost no semantic signal. BM25 nails them. Neither alone is enough.

  * The semantic ranker then re-scores the fused top ~50 with a cross-encoder. A
    bi-encoder embeds query and document independently; a cross-encoder sees both at
    once and is markedly better at relevance, but is too slow to run over a whole
    corpus. Running it over a shortlist is the standard two-stage retrieval pattern.

  * Semantic reranking is billed separately from the service tier and is not available
    on the Free tier -- see the README cost notes before enabling this backend.
"""
from __future__ import annotations

import logging
from typing import Sequence

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from ..aoai import build_client, embed
from ..chunking import Chunk
from ..config import get_settings
from .base import Hit

log = logging.getLogger(__name__)

EMBEDDING_DIM = 1536  # text-embedding-3-small
SEMANTIC_CONFIG = "default-semantic"


def _credential():
    s = get_settings()
    if s.azure_search_api_key:
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(s.azure_search_api_key)
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


class AzureSearchRetriever:
    def __init__(self) -> None:
        s = get_settings()
        if not s.azure_search_endpoint:
            raise RuntimeError("AZURE_SEARCH_ENDPOINT is not set.")
        self.endpoint = s.azure_search_endpoint
        self.index_name = s.azure_search_index
        self._client = None

    @property
    def aoai(self):
        if self._client is None:
            self._client = build_client()
        return self._client

    def _search_client(self) -> SearchClient:
        return SearchClient(self.endpoint, self.index_name, _credential())

    # ---- index management --------------------------------------------
    def _ensure_index(self) -> None:
        index_client = SearchIndexClient(self.endpoint, _credential())
        index = SearchIndex(
            name=self.index_name,
            fields=[
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                # Analyzed for BM25 -- this is the keyword half of hybrid search.
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchableField(name="heading", type=SearchFieldDataType.String),
                SimpleField(
                    name="source", type=SearchFieldDataType.String, filterable=True, facetable=True
                ),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=EMBEDDING_DIM,
                    vector_search_profile_name="hnsw-profile",
                ),
            ],
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
                profiles=[
                    VectorSearchProfile(
                        name="hnsw-profile", algorithm_configuration_name="hnsw-config"
                    )
                ],
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name=SEMANTIC_CONFIG,
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="heading"),
                            content_fields=[SemanticField(field_name="content")],
                        ),
                    )
                ]
            ),
        )
        index_client.create_or_update_index(index)
        log.info("Index '%s' created or updated", self.index_name)

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._ensure_index()
        vectors = embed(self.aoai, [c.text for c in chunks])
        docs = [
            {
                "id": c.id,
                "content": c.text,
                "heading": c.heading,
                "source": c.source,
                "embedding": v,
            }
            for c, v in zip(chunks, vectors)
        ]
        client = self._search_client()
        # Upload in batches; the service caps payload size per request.
        for start in range(0, len(docs), 100):
            client.upload_documents(documents=docs[start : start + 100])
        log.info("Uploaded %d documents to Azure AI Search", len(docs))

    # ---- query --------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[Hit]:
        qv = embed(self.aoai, [query])[0]
        client = self._search_client()

        results = client.search(
            search_text=query,  # BM25 half
            vector_queries=[  # vector half
                VectorizedQuery(vector=qv, k_nearest_neighbors=top_k * 5, fields="embedding")
            ],
            query_type="semantic",
            semantic_configuration_name=SEMANTIC_CONFIG,
            top=top_k,
            select=["id", "content", "heading", "source"],
        )

        hits: list[Hit] = []
        for r in results:
            # Prefer the reranker score when present; it is on a 0-4 scale, so normalise
            # it to roughly 0-1 to stay comparable with cosine scores from the local path.
            rerank = r.get("@search.reranker_score")
            score = (rerank / 4.0) if rerank is not None else r["@search.score"]
            hits.append(
                Hit(
                    chunk=Chunk(
                        id=r["id"],
                        text=r["content"],
                        source=r["source"],
                        heading=r.get("heading", ""),
                    ),
                    score=float(score),
                    origin="semantic-rerank" if rerank is not None else "hybrid",
                )
            )
        return hits
