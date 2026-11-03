"""Dense (semantic) vectors for chunks and queries.

GeminiEmbedder uses Google's free-tier embedding model. TfidfEmbedder needs no network or key, so the
pipeline and the tests run anywhere; it is a weaker, lexical stand-in, not a semantic model.

The Gemini free tier is rate-limited (about 100 embedding requests per minute, plus a daily cap), so
GeminiEmbedder waits and retries when told to slow down, caches every finished batch on disk so an
interrupted build resumes where it stopped, and raises EmbeddingQuotaExhausted when retrying is pointless.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)


class EmbeddingQuotaExhausted(RuntimeError):
    pass


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, stop_words="english")

    def fit(self, texts: list[str]) -> np.ndarray:
        return _normalise(self.vec.fit_transform(texts).toarray().astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self.vec.transform([text]).toarray().astype(np.float32))[0]


def _retry_delay(exc: Exception) -> float | None:
    """Seconds Google asks us to wait, or None if the error is not a rate limit (or is a daily cap)."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc)
    if code != 429 and "RESOURCE_EXHAUSTED" not in msg:
        return None
    if re.search(r"PerDay|per day|daily", msg, re.I):
        raise EmbeddingQuotaExhausted("Gemini daily embedding quota used up - try again tomorrow, "
                                      "or build with --no-gemini") from exc
    m = re.search(r"retry in ([\d.]+)s|'retryDelay': '([\d.]+)s'", msg)
    return float(next(g for g in m.groups() if g)) if m else 30.0


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", batch: int = 50,
                 dims: int = 768, cache_dir: Path | None = None, max_waits: int = 30,
                 sleep=time.sleep, client=None):
        if client is None:
            from google import genai  # imported lazily so the offline path has no dependency on it
            from google.genai import types

            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
        self.client = client
        self.model = model
        self.batch = batch
        self.dims = dims
        self.cache_dir = cache_dir
        self.max_waits = max_waits
        self.sleep = sleep

    def _call(self, texts: list[str], task: str) -> list[list[float]]:
        from google.genai import types

        waits = 0
        while True:
            try:
                res = self.client.models.embed_content(
                    model=self.model, contents=texts,
                    config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dims))
                return [e.values for e in res.embeddings]
            except EmbeddingQuotaExhausted:
                raise
            except Exception as exc:
                delay = _retry_delay(exc)
                if delay is None:
                    raise
                waits += 1
                if waits > self.max_waits:
                    raise EmbeddingQuotaExhausted("Gemini kept rate-limiting; giving up") from exc
                log.info("Gemini rate limit - waiting %.0fs (retry %d/%d)", delay + 1, waits, self.max_waits)
                self.sleep(delay + 1)

    def _cache_file(self, texts: list[str], task: str) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha1(f"{self.model}|{self.dims}|{task}|".encode() + "\x00".join(texts).encode()).hexdigest()
        return self.cache_dir / f"{h[:20]}.npy"

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        out: list[np.ndarray] = []
        total = len(texts)
        for i in range(0, total, self.batch):
            part = texts[i:i + self.batch]
            cf = self._cache_file(part, task)
            if cf and cf.exists():
                out.append(np.load(cf))
                continue
            vec = np.asarray(self._call(part, task), dtype=np.float32)
            if cf:
                cf.parent.mkdir(parents=True, exist_ok=True)
                np.save(cf, vec)
            out.append(vec)
            if total > self.batch:
                log.info("Embedded %d / %d passages", min(i + self.batch, total), total)
        return _normalise(np.vstack(out))

    def fit(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        saved, self.max_waits = self.max_waits, 2      # a user is waiting: don't sit out long rate limits
        try:
            return self._embed([text], "RETRIEVAL_QUERY")[0]
        except Exception as exc:   # keep answering with BM25 alone rather than failing the question
            log.warning("Query embedding failed (%s); using keyword search only for this question", exc)
            return np.zeros(self.dims, dtype=np.float32)
        finally:
            self.max_waits = saved
