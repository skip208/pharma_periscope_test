"""
Application configuration loaded from environment variables.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised application settings."""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_model_name: str = Field(default="gpt-4.1-mini", alias="LLM_MODEL_NAME")
    embedding_model_name: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL_NAME")

    vector_store_backend: str = Field(default="chroma", alias="VECTOR_STORE_BACKEND")
    vector_store_path: str = Field(default="./data/vector_store", alias="VECTOR_STORE_PATH")

    corpus_dir: str = Field(default="./data/corpus", alias="CORPUS_DIR")

    relevance_threshold: float = Field(default=0.78, alias="RELEVANCE_THRESHOLD")
    min_good_chunks: int = Field(default=2, alias="MIN_GOOD_CHUNKS")
    max_context_chunks: int = Field(default=5, alias="MAX_CONTEXT_CHUNKS")

    chunk_size_chars: int = Field(default=1000, alias="CHUNK_SIZE_CHARS")
    chunk_overlap_chars: int = Field(default=200, alias="CHUNK_OVERLAP_CHARS")

    admin_token: SecretStr | None = Field(default=None, alias="ADMIN_TOKEN")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")


settings = Settings()


def setup_logging() -> logging.Logger:
    """
    Configure base logging for the app.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )
    return logging.getLogger("app")


def public_settings() -> Dict[str, Any]:
    """
    Return settings without secrets for safe logging/inspection.
    """
    return settings.model_dump(
        exclude={"openai_api_key", "admin_token"},
        exclude_none=True,
    )


__all__ = ["Settings", "settings", "setup_logging", "public_settings"]

