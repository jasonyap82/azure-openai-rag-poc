"""Central configuration. Everything tunable lives here, nothing is hardcoded downstream."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    chat_deployment: str = "gpt-4o-mini"
    embedding_deployment: str = "text-embedding-3-small"

    # Retrieval backend
    retriever: Literal["local", "azure_search"] = "local"
    azure_search_endpoint: str = ""
    azure_search_index: str = "rag-poc"
    azure_search_api_key: str = ""

    # Chunking / retrieval
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60
    top_k: int = 4
    min_relevance: float = 0.25

    # Paths
    data_dir: Path = ROOT / "data"
    index_dir: Path = ROOT / "index"

    @property
    def use_entra_id(self) -> bool:
        """No key configured => authenticate with the caller's Entra ID identity."""
        return not self.azure_openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
