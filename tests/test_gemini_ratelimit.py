"""Gemini embedding rate-limit handling, with a fake client (no network)."""

from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors

from finintel.retrieval.dense import EmbeddingQuotaExhausted, GeminiEmbedder


def rate_limit(per_day=False):
    quota = "EmbedContentRequestsPerDayPerUser" if per_day else "EmbedContentRequestsPerMinutePerUser"
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                              "message": f"Quota exceeded ({quota}). Please retry in 2.5s."}})


class FakeModels:
    def __init__(self, fail_first=0, per_day=False):
        self.calls = 0
        self.fail_first = fail_first
        self.per_day = per_day

    def embed_content(self, model, contents, config):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise rate_limit(self.per_day)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[float(len(t)), 1.0, 0.0]) for t in contents])


def make(models, tmp_path=None, **kw):
    slept = []
    emb = GeminiEmbedder("k", client=SimpleNamespace(models=models), batch=2, dims=3,
                         cache_dir=tmp_path, sleep=slept.append, **kw)
    return emb, slept


def test_waits_and_retries_on_per_minute_limit():
    models = FakeModels(fail_first=2)
    emb, slept = make(models)
    vecs = emb.fit(["a", "bb", "ccc"])
    assert vecs.shape == (3, 3) and np.allclose(np.linalg.norm(vecs, axis=1), 1)
    assert slept == [3.5, 3.5]          # Google's retry delay + 1s, twice
    assert models.calls == 4            # 2 failures + 2 batches


def test_daily_quota_stops_immediately():
    emb, slept = make(FakeModels(fail_first=99, per_day=True))
    with pytest.raises(EmbeddingQuotaExhausted):
        emb.fit(["a", "b"])
    assert slept == []


def test_gives_up_after_max_waits():
    emb, slept = make(FakeModels(fail_first=99), max_waits=3)
    with pytest.raises(EmbeddingQuotaExhausted):
        emb.fit(["a"])
    assert len(slept) == 3


def test_finished_batches_are_cached_and_resumed(tmp_path):
    first = FakeModels(fail_first=0)
    make(first, tmp_path)[0].fit(["a", "bb", "ccc", "dddd"])
    assert first.calls == 2
    second = FakeModels(fail_first=0)
    make(second, tmp_path)[0].fit(["a", "bb", "ccc", "dddd", "eeeee"])
    assert second.calls == 1            # only the new batch is sent


def test_query_embedding_failure_degrades_to_keyword_search():
    emb, _ = make(FakeModels(fail_first=99))
    assert not emb.embed_query("revenue").any()
    assert emb.max_waits == 30          # restored after the query


def test_build_falls_back_to_tfidf_when_quota_exhausted(settings, monkeypatch):
    import finintel.system as system

    class Exhausted:
        name = "gemini"

        def fit(self, texts):
            raise EmbeddingQuotaExhausted("Gemini daily embedding quota used up")

    monkeypatch.setattr(system, "choose_embedder", lambda s, use_gemini=None: Exhausted())
    out = system.build(settings, use_gemini=True)
    assert out["search"] == "tfidf" and "resume" in out["note"]
    monkeypatch.undo()
    qa = system.load(settings, use_llm=False)
    assert qa.ask("What was Example Corp's revenue in FY2023?").verification.passed
