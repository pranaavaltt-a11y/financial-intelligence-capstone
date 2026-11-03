# Evidence-Grounded Financial Intelligence

Capstone project **PW26_DG_01** - B.Tech CSE (AI & ML), PES University.
Guide: Deepti Gupta. Team: Pranaav Prabhakaran, Raghunandan Kotary, Aditya Shetty, Shahid Tahsildar.

A question-answering system over company 10-K filings that:

- takes **numbers from SEC XBRL data** (a DuckDB warehouse: company x year x metric), not from text,
- does **all arithmetic in Python** (margins, growth, point changes, CAGR),
- retrieves **explanations from the filing text** with hybrid search (BM25 + dense vectors, fused and reranked),
- **verifies every answer in code** - each number must match a fact, calculation or cited passage, and each
  cited sentence must be supported by its passage - and shows a **confidence score**,
- cites every claim ([F#] fact, [C#] calculation, [S#] passage).

## Architecture

```
SEC EDGAR 10-K HTML ──► ingest ──► index/parse (sections + tables) ──► retrieval (BM25 + dense ─► RRF ─► rerank)
SEC XBRL companyfacts ─► ingest/xbrl ──► data/warehouse (DuckDB, OLAP: slice / dice / pivot)
                                                   │                                   │
question ─► agent/planner ─► facts [F#] ─► agent/tools calculations [C#] ─► passages [S#]
                                   └──► composer (Gemini, or deterministic template) ─► agent/verifier ─► answer
                                                          ▲                                   │
                                                          └──── retry with wider retrieval ◄──┘ (if verification fails)
```

| Package | Module | What it does | Owner |
|---|---|---|---|
| `finintel/ingest` | `edgar.py`, `xbrl.py` | SEC download client; XBRL -> one clean value per metric and fiscal year | Raghunandan |
| `finintel/index` | `parse.py` | 10-K HTML -> section-tagged text and table chunks | Raghunandan |
| `finintel/retrieval` | `hybrid.py`, `dense.py`, `text.py` | BM25 + dense search, Reciprocal Rank Fusion, reranking | Aditya |
| `finintel/agent` | `planner.py`, `tools.py`, `pipeline.py`, `llm.py` | question plan, calculation tools, orchestration, Gemini wrapper | Pranaav |
| `finintel/agent` | `verifier.py` | number and claim verification, confidence score | Shahid |
| `finintel/data` | `warehouse.py` | DuckDB warehouse and OLAP queries | Shahid |
| `finintel/app` | `streamlit_app.py` | Q&A page and comparison dashboard | Shahid / Raghunandan |
| `finintel/eval` | `run_eval.py` | LLM-only vs BM25-RAG vs full system | All |

(The owners above are a suggested split; change as the team decides.)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the values, or export them in your shell
```

| Variable | Needed for | How to get it |
|---|---|---|
| `SEC_USER_AGENT` | downloading filings | Your name and email, e.g. `"PES Capstone you@example.edu"` (SEC requirement) |
| `GEMINI_API_KEY` | LLM answers + semantic embeddings (optional) | Free at [Google AI Studio](https://aistudio.google.com) -> Get API key |

Without `GEMINI_API_KEY` everything still runs: search uses TF-IDF instead of semantic embeddings and answers
use a deterministic template. With the key, Gemini writes the answer and the verifier checks it; an answer that
fails verification is regenerated once, and if it still fails the verified template answer is shown instead.

## Run

```bash
# 1. Download filings + XBRL facts (needs internet access to sec.gov and data.sec.gov)
python -m finintel ingest --tickers AAPL MSFT NVDA AMZN --years 2022 2023 2024

# 2. Parse filings, load the warehouse, build the search index
python -m finintel build

# 3. Ask from the command line
python -m finintel ask "How did Apple's operating margin change from FY2022 to FY2024, and why?"

# 4. Web app
streamlit run finintel/app/streamlit_app.py

# 5. Evaluation (fill gold answers in the question file first - from the filings or FinanceBench)
python -m finintel eval --questions finintel/eval/questions.example.json
```

`python -m finintel status` shows which companies and years are loaded.

### Dry run without internet

```bash
python scripts/load_sample_data.py
FININTEL_DATA_DIR=sample_data streamlit run finintel/app/streamlit_app.py
```

This loads two **fictional** companies (Example Corp, Sample Industries) with made-up numbers - for checking the
software only. Never present them as results.

## Tests

```bash
pytest
```

The tests run offline on the fictional fixtures in `tests/conftest.py` and cover XBRL extraction (restatements,
quarterly noise, alternative tags), parsing, hybrid retrieval, the warehouse, calculations, the planner, the
verifier and the full pipeline, including an LLM that hallucinates a number and must be corrected.

## Status

| Part | State |
|---|---|
| Ingestion (EDGAR + XBRL) | Written and unit-tested; **not yet run against live SEC data** |
| Parsing, hybrid retrieval, reranking | Done, tested |
| DuckDB warehouse + OLAP queries | Done, tested |
| Planner, calculation tools, verifier, retry loop | Done, tested |
| Gemini LLM + embeddings | Written; needs `GEMINI_API_KEY` to try |
| Streamlit app: Q&A + comparison dashboard | Done |
| Evaluation script | Written; needs gold answers |
| Year-over-year report, risk tracker, PDF export | Phase-II |

## References

Design choices follow the literature reviewed in the project: hybrid BM25 + dense retrieval and reranking to the
top 8 passages (Srinivasan et al., FinNLP 2025; Lumer et al., 2025), executing arithmetic as code (Program of
Thoughts, PAL), verification before answering (Chain-of-Verification), and numbers checked by code rather than an
LLM judge (RAGAS limitations reported by Srinivasan et al.).
