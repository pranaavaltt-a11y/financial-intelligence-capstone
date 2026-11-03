"""Compare systems on a question set (the baseline ladder from the Review slides).

Systems:
    llm_only   - the LLM answers from its own knowledge, no retrieval (skipped without GEMINI_API_KEY)
    bm25_rag   - keyword retrieval only; LLM answers from the passages, or the top passage is returned
    full       - the integrated pipeline (planner + warehouse + tools + hybrid retrieval + verifier)

Question file (JSON list):
    {"id": "q1", "question": "...", "answer_value": 41.3, "answer_unit": "%", "tolerance_pct": 1.0,
     "ticker": "AAPL", "year": 2023, "evidence_contains": ["services", "gross margin"]}
answer_value / evidence_contains may be null; those metrics are then skipped for that question.
Gold answers must come from the filing or a benchmark such as FinanceBench - never from this system.
"""

from __future__ import annotations

import json
import time
from statistics import mean

from rank_bm25 import BM25Okapi

from finintel.agent.verifier import extract_numbers
from finintel.config import Settings
from finintel.models import read_chunks
from finintel.retrieval.text import tokenize
from finintel.system import load


def numeric_correct(text: str, gold: float, tol_pct: float = 1.0) -> bool:
    for _, v, _, s in extract_numbers(text):
        for cand in (v * s, v):
            if gold == 0:
                if abs(cand) < 1e-9:
                    return True
            elif abs(cand - gold) / abs(gold) * 100 <= tol_pct:
                return True
    return False


def evidence_hit(passages: list[str], keywords: list[str]) -> bool:
    return any(all(k.lower() in p.lower() for k in keywords) for p in passages)


def run(questions_path: str, settings: Settings, k: int = 8) -> dict:
    questions = json.loads(open(questions_path, encoding="utf-8").read())
    qa = load(settings)
    chunks = read_chunks(settings.chunks_path)
    bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks]) if chunks else None
    systems = ["bm25_rag", "full"] + (["llm_only"] if qa.llm else [])
    rows = []
    for q in questions:
        row = {"id": q.get("id"), "question": q["question"]}
        gold, tol, kw = q.get("answer_value"), q.get("tolerance_pct", 1.0), q.get("evidence_contains")
        for sysname in systems:
            t0 = time.perf_counter()
            passages: list[str] = []
            conf = None
            if sysname == "full":
                ans = qa.ask(q["question"])
                text, conf = ans.text, ans.verification.confidence
                passages = [c.snippet for c in ans.citations if c.ref.startswith("S")]
            elif sysname == "bm25_rag" and bm25:
                scores = bm25.get_scores(tokenize(q["question"]) or ["_"])
                top = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
                passages = [chunks[i].text for i in top]
                ctx = "\n\n".join(f"[S{j + 1}] {p[:1200]}" for j, p in enumerate(passages))
                text = (qa.llm.generate(f"Answer using only these passages.\n{ctx}\n\nQuestion: {q['question']}")
                        if qa.llm else None) or (passages[0] if passages else "")
            elif sysname == "llm_only":
                text = qa.llm.generate(q["question"]) or ""
            else:
                text = ""
            r = {"latency_s": round(time.perf_counter() - t0, 2), "answer": text[:600]}
            if gold is not None:
                r["numeric_correct"] = numeric_correct(text, float(gold), tol)
            if kw and sysname != "llm_only":
                r["evidence_hit"] = evidence_hit(passages, kw)
            if conf is not None:
                r["confidence"] = conf
            row[sysname] = r
        rows.append(row)

    summary = {}
    for sysname in systems:
        rs = [r[sysname] for r in rows if sysname in r]
        acc = [r["numeric_correct"] for r in rs if "numeric_correct" in r]
        hit = [r["evidence_hit"] for r in rs if "evidence_hit" in r]
        summary[sysname] = {
            "questions": len(rs),
            "numeric_accuracy": round(mean(acc), 3) if acc else None,
            "evidence_hit_rate": round(mean(hit), 3) if hit else None,
            "mean_latency_s": round(mean(r["latency_s"] for r in rs), 2) if rs else None,
            "mean_confidence": round(mean(r["confidence"] for r in rs if "confidence" in r), 3)
            if any("confidence" in r for r in rs) else None,
        }
    return {"summary": summary, "rows": rows}
