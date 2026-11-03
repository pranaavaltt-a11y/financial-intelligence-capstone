"""Runtime settings, read from environment variables.

Nothing here is required for the offline parts of the pipeline. GEMINI_API_KEY enables the LLM and
embedding model; SEC_USER_AGENT is required by the SEC for any download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("FININTEL_DATA_DIR", "data")))
    gemini_api_key: str | None = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY") or None)
    llm_model: str = field(default_factory=lambda: os.environ.get("FININTEL_LLM_MODEL", "gemini-2.5-flash"))
    embed_model: str = field(default_factory=lambda: os.environ.get("FININTEL_EMBED_MODEL", "gemini-embedding-001"))
    # The SEC asks automated clients to identify themselves, e.g. "Team Name contact@example.edu".
    sec_user_agent: str | None = field(default_factory=lambda: os.environ.get("SEC_USER_AGENT") or None)
    top_k: int = 8          # passages kept after reranking (Multi-HyDE paper: K = 8)
    candidates: int = 25    # passages fused from BM25 + dense before reranking
    max_retries: int = 1    # verification-triggered re-retrievals

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "processed" / "chunks.jsonl"

    @property
    def warehouse_path(self) -> Path:
        return self.data_dir / "warehouse.duckdb"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "processed" / "embeddings.npy"


def get_settings() -> Settings:
    return Settings()
