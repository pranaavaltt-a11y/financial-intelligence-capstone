"""Dense (semantic) vectors for chunks and queries.

GeminiEmbedder uses Google's free-tier embedding model. TfidfEmbedder needs no network or key, so the
pipeline and the tests run anywhere; it is a weaker, lexical stand-in, not a semantic model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


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


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", batch: int = 50):
        from google import genai  # imported lazily so the offline path has no dependency on it

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.batch = batch

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import types

        vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            res = self.client.models.embed_content(model=self.model, contents=texts[i:i + self.batch],
                                                   config=types.EmbedContentConfig(task_type=task))
            vecs.extend(e.values for e in res.embeddings)
        return _normalise(np.asarray(vecs, dtype=np.float32))

    def fit(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
