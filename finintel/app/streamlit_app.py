"""Web app: ask questions, compare companies, inspect loaded data.

    streamlit run finintel/app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # allow `streamlit run` from the repo root

from finintel.agent.tools import DERIVED, LABELS  # noqa: E402
from finintel.config import get_settings  # noqa: E402
from finintel.models import fmt_value  # noqa: E402
from finintel.system import data_status, load  # noqa: E402

st.set_page_config(page_title="Financial Intelligence", layout="wide")
settings = get_settings()


@st.cache_resource(show_spinner="Loading index and warehouse...")
def get_qa(use_llm: bool):
    return load(settings, use_llm=use_llm)


status = data_status(settings)
st.title("Evidence-Grounded Financial Intelligence")
st.caption("Answers over SEC 10-K filings - numbers from XBRL data, calculations in code, every claim cited.")

with st.sidebar:
    st.subheader("Data")
    if not status["companies"]:
        st.warning("No data loaded. Run `python -m finintel ingest ...` then `python -m finintel build`.")
    for t, info in status["companies"].items():
        yrs = info["years"]
        span = f"FY{yrs[0]}-FY{yrs[-1]}" if yrs else "none"
        txt = ", ".join(map(str, info.get("text_years", []))) or "none"
        st.write(f"**{t}** - {info['name']}  \nNumbers: {span}  \n10-K text: FY {txt}")
    st.write(f"Passages indexed: {status['chunks']:,}")
    use_llm = st.toggle("Use LLM (Gemini)", value=bool(settings.gemini_api_key),
                        disabled=not settings.gemini_api_key,
                        help="Without a GEMINI_API_KEY, answers use the deterministic template.")

if not status["companies"]:
    st.stop()

qa = get_qa(use_llm)
tab_ask, tab_compare = st.tabs(["Ask a question", "Comparison dashboard"])

with tab_ask:
    examples = ["How did {t}'s operating margin change from FY{y0} to FY{y1}, and why?",
                "What was {t}'s revenue in FY{y1}?", "What are {t}'s main risk factors about competition?"]
    first = next(iter(status["companies"]))
    info = status["companies"][first]
    ys = info.get("text_years") or info["years"][-3:] or [0]      # years with filing text to explain changes
    ex = [e.format(t=first, y0=ys[0], y1=ys[-1]) for e in examples]
    q = st.text_input("Question", value=ex[0])
    st.caption("Examples: " + " | ".join(ex[1:]))
    if st.button("Analyse", type="primary") and q.strip():
        with st.spinner("Planning, retrieving, calculating, verifying..."):
            ans = qa.ask(q)
        v = ans.verification
        badge = {"High": "green", "Medium": "orange", "Low": "red"}[v.confidence_label]
        st.markdown(f"### Answer  :{badge}[Confidence: {v.confidence_label} ({v.confidence:.2f})]")
        st.write(ans.text)
        st.caption(f"{v.numbers_supported}/{v.numbers_checked} numbers verified - "
                   f"{v.claims_supported}/{v.claims_checked} cited claims supported - "
                   f"{'LLM' if ans.used_llm else 'template'} answer - attempt(s): {ans.attempts}")
        for n in ans.notes:
            st.info(n)
        if v.unsupported_numbers or v.unsupported_claims:
            st.error(f"Unverified: {v.unsupported_numbers + v.unsupported_claims}")
        if ans.calculations:
            st.markdown("#### Calculations")
            st.dataframe(pd.DataFrame([{"Calculation": c.label, "Result": fmt_value(c.value, c.unit),
                                        "Formula": c.formula, "Inputs": ", ".join(c.inputs)}
                                       for c in ans.calculations]), hide_index=True, width="stretch")
        st.markdown("#### Sources")
        for c in ans.citations:
            with st.expander(f"[{c.ref}] {c.description}"):
                st.write(c.snippet or "Reported value from SEC XBRL data.")

with tab_compare:
    tickers = list(status["companies"])
    pick = st.multiselect("Companies", tickers, default=tickers[:3])
    metric_opts = sorted(set(qa.wh.metrics()) | set(DERIVED))
    metric = st.selectbox("Metric", metric_opts, index=metric_opts.index("revenue") if "revenue" in metric_opts else 0,
                          format_func=lambda m: LABELS.get(m, m))
    if pick:
        if metric in DERIVED:
            num, den, unit, _ = DERIVED[metric]
            a, b = qa.wh.pivot(num, pick), qa.wh.pivot(den, pick)
            table = (a / b * (100 if unit == "%" else 1)) if not a.empty and not b.empty else pd.DataFrame()
        else:
            table = qa.wh.pivot(metric, pick)
            unit = "USD"
        if table.empty:
            st.info("No data for this selection.")
        else:
            table.index = [f"FY{y}" for y in table.index]
            scaled = table / 1e9 if unit == "USD" else table
            st.markdown(f"**{LABELS.get(metric, metric)}** " + ("(USD billions)" if unit == "USD" else f"({unit})"))
            st.bar_chart(scaled, stack=False)
            st.dataframe(scaled.round(2), width="stretch")
            if unit == "USD":
                parts = []
                for t in table.columns:
                    col = table[t].dropna()
                    if len(col) >= 2 and col.iloc[0]:
                        parts.append(f"{t}: {(col.iloc[-1] / col.iloc[0] - 1) * 100:+.1f}% ({col.index[0]}-{col.index[-1]})")
                if parts:
                    st.caption("Change over period: " + ", ".join(parts))
