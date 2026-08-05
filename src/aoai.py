"""Azure OpenAI client construction and embedding helpers.

Two things here are worth calling out in an interview:
  1. Entra ID (managed identity / az login) is the default auth path; API keys are the
     fallback. Keys in .env files are the #1 way Azure secrets end up in git history.
  2. Embeddings are batched and retried. Azure OpenAI enforces per-deployment TPM/RPM
     quotas and returns 429s with a Retry-After header well before you hit any real limit.
"""
from __future__ import annotations

import logging
from typing import Sequence

from openai import AzureOpenAI, RateLimitError, APITimeoutError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import get_settings

log = logging.getLogger(__name__)

# text-embedding-3-* accepts up to 2048 inputs per request; stay well under it so a
# single oversized batch cannot fail the whole ingest run.
EMBED_BATCH_SIZE = 128


def _http_client():
    """Build an HTTP client that validates TLS against the operating system trust store.

    By default the SDK verifies certificates against certifi's bundle of public CAs.
    That breaks on any machine where a TLS-inspecting middlebox sits in the path --
    endpoint protection suites, corporate proxies, and some VPNs all terminate TLS and
    re-sign traffic with a private root. That root is installed in the OS trust store,
    so browsers work, but Python does not consult the OS store and rejects it.

    truststore delegates verification to the OS, which is what every other application
    on the machine already does. Certificates are still fully verified -- this is not
    `verify=False`, which would disable the check and defeat the point.
    """
    try:
        import ssl

        import httpx
        import truststore

        return httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
    except ImportError:
        log.debug("truststore unavailable; using the default certifi bundle")
        return None


def build_client() -> AzureOpenAI:
    s = get_settings()
    if not s.azure_openai_endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set. Copy .env.example to .env.")

    http_client = _http_client()

    if s.use_entra_id:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
        log.info("Authenticating to Azure OpenAI with Entra ID")
        return AzureOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_version=s.azure_openai_api_version,
            azure_ad_token_provider=token_provider,
            timeout=30.0,
            http_client=http_client,
        )

    log.warning("Authenticating with an API key. Prefer Entra ID for anything real.")
    return AzureOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        api_version=s.azure_openai_api_version,
        api_key=s.azure_openai_api_key,
        timeout=30.0,
        http_client=http_client,
    )


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _embed_batch(client: AzureOpenAI, texts: Sequence[str]) -> list[list[float]]:
    s = get_settings()
    resp = client.embeddings.create(model=s.embedding_deployment, input=list(texts))
    # The API does not guarantee ordering; sort by index before returning.
    return [item.embedding for item in sorted(resp.data, key=lambda d: d.index)]


def embed(client: AzureOpenAI, texts: Sequence[str]) -> list[list[float]]:
    """Embed an arbitrary number of texts, batching to respect request limits."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors.extend(_embed_batch(client, batch))
        log.info("Embedded %d/%d chunks", len(vectors), len(texts))
    return vectors


# GPT-5 and the o-series changed the chat API contract: max_tokens became
# max_completion_tokens, and temperature is fixed at 1. Older families still use the
# old names. Normalising it here means the rest of the codebase doesn't need to care
# which model family it is pointed at -- which is the whole problem during a migration.
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(deployment: str) -> bool:
    return deployment.lower().startswith(REASONING_PREFIXES)


def chat_kwargs(deployment: str, max_output_tokens: int, temperature: float = 0.0) -> dict:
    """Build generation parameters appropriate to the deployed model family."""
    if is_reasoning_model(deployment):
        # Reasoning tokens are consumed against this budget BEFORE any visible output,
        # so a ceiling sized for the answer alone will return an empty string.
        return {"max_completion_tokens": max(max_output_tokens, 2000)}
    return {"max_tokens": max_output_tokens, "temperature": temperature}
