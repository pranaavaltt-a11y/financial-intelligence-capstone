"""Hybrid retrieval: BM25 + dense search fused with Reciprocal Rank Fusion, then reranked.

    query -> BM25 top-N  ┐
                         ├-> RRF fusion -> metadata filter -> rerank -> top-k passages
    query -> dense top-N ┘

The reranker is pluggable. LexicalReranker is the offline default; it rewards query-term coverage,
exact year/number matches and the section a question is likely to be answered in (MD&A for "why").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from finintel.models import Chunk
from finintel.retrieval.dense import Embedder
from finintel.retrieval.text import tokenize

WHY_WORDS = {"why", "reason", "reasons", "driver", "drivers", "explain", "cause", "caused", "due"}
CAUSE_RE = re.compile(r"primarily due to|driven by|due to|as a result of|offset by|attributable to", re.I)
RISK_WORDS = {"risk", "risks", "threat", "uncertainty", "exposure"}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


class LexicalReranker:
    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        q = set(tokenize(query))
        q_nums = {t for t in q if t[0].isdigit()}
        q_words = q - q_nums
        prefer = []
        if q_words & WHY_WORDS:
            prefer.append("MD&A")
        if q_words & RISK_WORDS:
            prefer.append("Risk Factors")
        scored = []
        for h in hits:
            toks = set(tokenize(h.chunk.text))
            coverage = len(q_words & toks) / len(q_words) if q_words else 0.0
            num_match = len(q_nums & toks) / len(q_nums) if q_nums else 0.0
            section = 0.3 if any(p in h.chunk.section for p in prefer) else 0.0
            if "MD&A" in prefer and h.chunk.kind == "text":
                section += 0.2       # "why" questions are answered by narrative, not tables
                if CAUSE_RE.search(h.chunk.text):
                    section += 0.3   # passages that state a cause
            scored.append(Hit(h.chunk, coverage + 0.5 * num_match + section + 0.2 * h.score,
                              h.bm25_rank, h.dense_rank))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, reranker: Reranker | None = None,
                 vectors: np.ndarray | None = None, rrf_k: int = 60):
        if not chunks:
            raise ValueError("No chunks to index - run ingestion and indexing first.")
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = reranker or LexicalReranker()
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])
        self.vectors = vectors if vectors is not None else embedder.fit([c.text for c in chunks])

    def _allowed(self, tickers: set[str] | None, years: set[int] | None) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        if tickers:
            mask &= np.array([c.ticker in tickers for c in self.chunks])
        if years:
            mask &= np.array([c.fiscal_year in years for c in self.chunks])
        return mask

    def search(self, query: str, k: int = 8, candidates: int = 25, tickers: set[str] | None = None,
               years: set[int] | None = None) -> list[Hit]:
        mask = self._allowed(tickers, years)
        if not mask.any():
            return []
        bm = np.asarray(self.bm25.get_scores(tokenize(query) or ["_"]), dtype=float)
        dn = self.vectors @ self.embedder.embed_query(query)
        bm[~mask] = -np.inf
        dn[~mask] = -np.inf
        n = int(min(candidates, mask.sum()))
        bm_order = [i for i in np.argsort(-bm)[:n] if np.isfinite(bm[i])]
        dn_order = [i for i in np.argsort(-dn)[:n] if np.isfinite(dn[i])]

        fused: dict[int, Hit] = {}
        for rank, i in enumerate(bm_order):
            h = fused.setdefault(i, Hit(self.chunks[i], 0.0))
            h.score += 1 / (self.rrf_k + rank + 1)
            h.bm25_rank = rank + 1
        for rank, i in enumerate(dn_order):
            h = fused.setdefault(i, Hit(self.chunks[i], 0.0))
            h.score += 1 / (self.rrf_k + rank + 1)
            h.dense_rank = rank + 1
        hits = sorted(fused.values(), key=lambda h: h.score, reverse=True)
        top = max((h.score for h in hits), default=1.0) or 1.0
        for h in hits:
            h.score /= top      # 0..1, so the reranker can mix it with its own features
        return self.reranker.rerank(query, hits, k)
