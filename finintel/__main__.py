"""Command line:

    python -m finintel ingest --tickers AAPL MSFT NVDA --years 2022 2023 2024
    python -m finintel build
    python -m finintel ask "How did Apple's operating margin change from FY2022 to FY2024, and why?"
    python -m finintel status
    python -m finintel check-llm
    python -m finintel eval --questions finintel/eval/questions.example.json
    streamlit run finintel/app/streamlit_app.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from finintel.config import get_settings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="finintel")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest", help="download filings + XBRL facts from SEC EDGAR")
    pi.add_argument("--tickers", nargs="+", required=True)
    pi.add_argument("--years", nargs="+", type=int, required=True)
    pb = sub.add_parser("build", help="parse filings, load warehouse, build search index")
    pb.add_argument("--years", nargs="*", type=int)
    pb.add_argument("--no-gemini", action="store_true", help="use TF-IDF instead of Gemini embeddings")
    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("--no-llm", action="store_true")
    sub.add_parser("status", help="show what data is loaded")
    sub.add_parser("check-llm", help="test which Gemini models your GEMINI_API_KEY can use")
    pe = sub.add_parser("eval", help="compare baselines on a question set")
    pe.add_argument("--questions", required=True)
    pe.add_argument("--out", default="eval_results.json")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()

    if args.cmd == "ingest":
        from finintel.system import ingest
        print(json.dumps(ingest(args.tickers, args.years, s), indent=2))
    elif args.cmd == "build":
        from finintel.system import build
        print(json.dumps(build(s, args.years, use_gemini=False if args.no_gemini else None), indent=2))
    elif args.cmd == "ask":
        from finintel.system import load
        print(load(s, use_llm=not args.no_llm).ask(args.question).to_markdown())
    elif args.cmd == "status":
        from finintel.system import data_status
        print(json.dumps(data_status(s), indent=2))
    elif args.cmd == "check-llm":
        if not s.gemini_api_key:
            print("GEMINI_API_KEY is not set in this window. Run: set GEMINI_API_KEY=your-key")
            return 1
        from finintel.agent.llm import check_models
        results = check_models(s.gemini_api_key)
        for r in results:
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['model']:<32} {r['seconds']:>5}s  {r.get('reply') or r.get('error')}")
        good = [r for r in results if r["ok"]]
        if good:
            best = min(good, key=lambda r: r["seconds"])["model"]
            print(f"\nUse this model:  set FININTEL_LLM_MODEL={best}")
        else:
            print("\nNo model answered. If every error is 503, Google is overloaded - try later. "
                  "If you see 400/403, the key is wrong or not enabled for the Gemini API.")
    elif args.cmd == "eval":
        from finintel.eval.run_eval import run
        res = run(args.questions, s)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
