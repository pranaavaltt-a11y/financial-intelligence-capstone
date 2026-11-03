"""Evidence-grounded financial intelligence over SEC filings.

Layers (one sub-package each):
    ingest     - download 10-K filings and XBRL facts from SEC EDGAR
    index      - parse filings into section-tagged text and table chunks
    retrieval  - BM25 + dense hybrid search with reranking
    data       - DuckDB warehouse of XBRL facts (company x year x metric)
    agent      - planner, calculation tools, answer composer and verifier
    app        - Streamlit web app (Q&A and comparison dashboard)
    eval       - baseline comparison and metrics
"""

__version__ = "0.1.0"
