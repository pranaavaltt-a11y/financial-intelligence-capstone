"""Wire the layers together: ingest -> index -> load a ready-to-use FinancialQA system."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from finintel.agent.llm import make_llm
from finintel.agent.pipeline import FinancialQA
from finintel.config import Settings
from finintel.data.warehouse import Warehouse
from finintel.index.parse import parse_file
from finintel.ingest.xbrl import extract_facts
from finintel.models import read_chunks, write_jsonl
from finintel.retrieval.dense import Embedder, EmbeddingQuotaExhausted, GeminiEmbedder, TfidfEmbedder
from finintel.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)


def ingest(tickers: list[str], years: list[int], settings: Settings) -> dict[str, list[str]]:
    """Download 10-Ks and company facts from SEC EDGAR into data/raw/<TICKER>/."""
    from finintel.ingest.edgar import EdgarClient

    client = EdgarClient(settings.sec_user_agent or "", settings.raw_dir)
    report: dict[str, list[str]] = {}
    for t in tickers:
        t = t.upper()
        got: list[str] = []
        (settings.raw_dir / t).mkdir(parents=True, exist_ok=True)
        (settings.raw_dir / t / "company.json").write_text(json.dumps({"ticker": t, "name": client.company_name(t)}))
        client.download_companyfacts(t)
        got.append("companyfacts.json")
        for ref in client.list_10k(t, set(years)):
            got.append(client.download_10k(ref).name)
        report[t] = got
    return report


def build(settings: Settings, years: list[int] | None = None, use_gemini: bool | None = None) -> dict[str, int]:
    """Parse every downloaded filing into chunks, load XBRL facts into the warehouse, embed chunks."""
    chunks = []
    wh = Warehouse(settings.warehouse_path)
    n_facts = 0
    for tdir in sorted(p for p in settings.raw_dir.glob("*") if p.is_dir()):
        t = tdir.name.upper()
        meta = tdir / "company.json"
        name = json.loads(meta.read_text()).get("name", t) if meta.exists() else t
        wh.add_company(t, name)
        facts_file = tdir / "companyfacts.json"
        if facts_file.exists():
            n_facts += wh.load_facts(extract_facts(json.loads(facts_file.read_text()), t,
                                                   set(years) if years else None))
        for f in sorted(tdir.glob("*_10k.html")):
            fy = int(f.name[:4])
            if years and fy not in years:
                continue
            chunks.extend(parse_file(f, t, fy))
    wh.close()
    write_jsonl(settings.chunks_path, chunks)
    emb = choose_embedder(settings, use_gemini)
    note = None
    if chunks:
        try:
            vecs = emb.fit([c.text for c in chunks])
        except EmbeddingQuotaExhausted as exc:
            note = (f"{exc}. Finished batches are cached - run build again later to resume. "
                    "Using offline TF-IDF search for now.")
            log.warning(note)
            emb = TfidfEmbedder()
            vecs = emb.fit([c.text for c in chunks])
        if emb.name != "tfidf":   # TF-IDF is refit on load; only store real embeddings
            np.save(settings.embeddings_path, vecs)
        settings.chunks_path.with_suffix(".embedder").write_text(emb.name)
    out = {"chunks": len(chunks), "facts": n_facts, "search": emb.name}
    if note:
        out["note"] = note
    return out


def choose_embedder(settings: Settings, use_gemini: bool | None = None) -> Embedder:
    want = bool(settings.gemini_api_key) if use_gemini is None else use_gemini
    if want and settings.gemini_api_key:
        try:
            return GeminiEmbedder(settings.gemini_api_key, settings.embed_model,
                                  cache_dir=settings.data_dir / "processed" / "emb_cache")
        except Exception as exc:
            log.warning("Gemini embeddings unavailable (%s); using TF-IDF", exc)
    return TfidfEmbedder()


def load(settings: Settings, use_llm: bool = True) -> FinancialQA:
    wh = Warehouse(settings.warehouse_path)
    chunks = read_chunks(settings.chunks_path)
    retriever = None
    if chunks:
        marker = settings.chunks_path.with_suffix(".embedder")
        kind = marker.read_text().strip() if marker.exists() else "tfidf"
        emb = choose_embedder(settings, use_gemini=(kind == "gemini"))
        vectors = None
        if emb.name == "gemini" and settings.embeddings_path.exists():
            vectors = np.load(settings.embeddings_path)
        elif emb.name != kind:
            log.warning("Index was built with %s but loading with %s; refitting", kind, emb.name)
        retriever = HybridRetriever(chunks, emb, vectors=vectors)
    llm = make_llm(settings.gemini_api_key, settings.llm_model) if use_llm else None
    return FinancialQA(wh, retriever, llm, top_k=settings.top_k, max_retries=settings.max_retries)


def data_status(settings: Settings) -> dict:
    out: dict = {"data_dir": str(Path(settings.data_dir).resolve()), "companies": {}, "chunks": 0}
    if settings.warehouse_path.exists():
        wh = Warehouse(settings.warehouse_path)
        out["companies"] = {t: {"name": n, "years": wh.years(t)} for t, n in wh.companies().items()}
        wh.close()
    chunks = read_chunks(settings.chunks_path)
    out["chunks"] = len(chunks)
    for t, info in out["companies"].items():   # years that have filing text, not just XBRL numbers
        info["text_years"] = sorted({c.fiscal_year for c in chunks if c.ticker == t})
    out["llm"] = "gemini" if settings.gemini_api_key else "none (template answers)"
    return out
