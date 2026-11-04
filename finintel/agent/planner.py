"""Planner: turn a question into a structured plan.

    "How did Apple's operating margin change from FY2022 to FY2024, and why?"
      -> companies [AAPL], metrics [operating_margin], years [2022, 2024],
         operation "change", needs_explanation True

The rule-based planner always works offline. When an LLM is available it may propose a plan as JSON;
that plan is validated against the warehouse (known tickers, metrics, years) and the rule-based plan
fills any gaps, so a bad LLM output can never produce an unanswerable plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finintel.agent.tools import DERIVED, LABELS

METRIC_KEYWORDS: list[tuple[str, str]] = [   # longest phrases first
    ("operating margin", "operating_margin"), ("net margin", "net_margin"), ("profit margin", "net_margin"),
    ("gross margin", "gross_margin"), ("debt to equity", "debt_to_equity"), ("debt-to-equity", "debt_to_equity"),
    ("leverage", "debt_to_equity"), ("r&d intensity", "rd_intensity"),
    ("operating cash flow", "operating_cash_flow"), ("cash flow", "operating_cash_flow"),
    ("cost of revenue", "cost_of_revenue"), ("cost of sales", "cost_of_revenue"),
    ("gross profit", "gross_profit"), ("operating income", "operating_income"),
    ("net income", "net_income"), ("earnings per share", "eps_diluted"), ("eps", "eps_diluted"),
    ("research and development", "rd_expense"), ("r&d", "rd_expense"),
    ("total assets", "total_assets"), ("total liabilities", "total_liabilities"), ("liabilities", "total_liabilities"),
    ("stockholders' equity", "stockholders_equity"), ("shareholders' equity", "stockholders_equity"),
    ("equity", "stockholders_equity"), ("assets", "total_assets"), ("cash", "cash"),
    ("profit", "net_income"), ("earnings", "net_income"),
    ("revenue", "revenue"), ("sales", "revenue"), ("turnover", "revenue"),
]
CHANGE_WORDS = re.compile(r"\b(change|changed|changes|grow|grew|grown|growing|growth|increase|increased|decrease|"
                          r"decreased|decline|declined|rise|rose|risen|fall|fell|fallen|drop|dropped|improve|improved|"
                          r"trend|over the|from)\b", re.I)
COMPARE_WORDS = re.compile(r"\b(compare|compared|comparison|versus|vs\.?|against|higher|lower|better|worse)\b", re.I)
WHY_WORDS = re.compile(r"\b(why|reason|reasons|explain|driver|drivers|drove|caused|because|what led)\b", re.I)
# Questions about what the filing says (risks, strategy, competition) are answered from text alone.
TEXT_WORDS = re.compile(r"\b(risk|risks|describe|describes|discuss|discusses|mention|mentions|say|says|strategy|"
                        r"competition|competitors|compete|challenges|threats|outlook|plans?|segments?|products?|"
                        r"customers|supply chain|regulation|regulatory|litigation|legal|cybersecurity|ai|"
                        r"artificial intelligence|employees|business model)\b", re.I)
CAGR_WORDS = re.compile(r"\b(cagr|compound(ed)? annual)\b", re.I)
YEAR_RE = re.compile(r"\b(?:FY\s?|fiscal\s+(?:year\s+)?)?((?:19|20)\d{2})\b", re.I)


@dataclass
class Plan:
    companies: list[str]
    metrics: list[str]
    years: list[int]
    operation: str                      # "lookup" | "change" | "compare" | "cagr"
    needs_explanation: bool
    notes: list[str] = field(default_factory=list)

    def base_metrics(self) -> list[str]:
        out: list[str] = []
        for m in self.metrics:
            for b in (DERIVED[m][:2] if m in DERIVED else (m,)):
                if b not in out:
                    out.append(b)
        return out

    def describe(self) -> str:
        what = ", ".join(LABELS.get(m, m) for m in self.metrics)
        return (f"{self.operation} of {what} for {', '.join(self.companies)} in "
                f"{', '.join(f'FY{y}' for y in self.years)}" + (" + explanation" if self.needs_explanation else ""))


def find_companies(question: str, companies: dict[str, str]) -> list[str]:
    q = question.lower()
    found: list[tuple[int, str]] = []
    for ticker, name in companies.items():
        aliases = {ticker.lower()}
        if name:
            n = name.lower()
            aliases.add(n)
            short = re.sub(r"[,.]|\b(inc|corp|corporation|co|company|ltd|plc|holdings|group|the)\b", "", n).strip()
            if short:
                aliases.add(re.sub(r"\s+", " ", short))
                aliases.add(short.split()[0])
        for a in aliases:
            m = re.search(rf"(?<![a-z]){re.escape(a)}(?:'s)?(?![a-z])", q)
            if m and len(a) > 1:
                found.append((m.start(), ticker))
                break
    return [t for _, t in sorted(found)]


def find_metrics(question: str) -> list[str]:
    q = question.lower()
    out: list[str] = []
    taken: list[tuple[int, int]] = []
    for phrase, metric in METRIC_KEYWORDS:
        for m in re.finditer(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", q):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue      # part of a longer phrase already matched ("operating margin" vs "margin")
            taken.append((m.start(), m.end()))
            if metric not in out:
                out.append(metric)
    return out


def rule_plan(question: str, companies: dict[str, str], years_available: list[int]) -> Plan:
    notes: list[str] = []
    tickers = find_companies(question, companies)
    if not tickers:
        notes.append("No known company named in the question.")
    metrics = find_metrics(question)
    text_only = not metrics and bool(TEXT_WORDS.search(question))
    if text_only:
        notes.append("No financial metric asked for; answering from the filing text.")
    elif not metrics:
        metrics = ["revenue"]
        notes.append("No metric recognised; defaulting to revenue.")
    years = sorted({int(y) for y in YEAR_RE.findall(question)})

    if CAGR_WORDS.search(question):
        op = "cagr"
    elif len(tickers) > 1 or COMPARE_WORDS.search(question):
        op = "compare"
    elif CHANGE_WORDS.search(question) or len(years) > 1:
        op = "change"
    else:
        op = "lookup"

    avail = sorted(years_available)
    if years:
        missing = [y for y in years if y not in avail]
        if missing:
            notes.append(f"No data loaded for FY{', FY'.join(map(str, missing))}.")
    elif avail:
        years = avail[-2:] if op in ("change", "cagr") else avail[-1:]
        notes.append(f"No year given; using {', '.join(f'FY{y}' for y in years)}.")
    if op in ("change", "cagr") and len(years) == 1 and avail:
        prev = [y for y in avail if y < years[0]]
        if prev:
            years = [prev[-1], years[0]]
    if text_only:
        op = "lookup"
        if not YEAR_RE.findall(question):       # search every filing year rather than guess one
            years = []
            notes = [n for n in notes if not n.startswith("No year given")]
    return Plan(tickers, metrics, years, op, bool(WHY_WORDS.search(question)) or text_only, notes)


def merge_llm_plan(rule: Plan, llm_json: dict | None, companies: dict[str, str], metrics_known: set[str],
                   years_available: list[int]) -> Plan:
    """Accept only LLM plan fields that are valid; otherwise keep the rule-based value."""
    if not llm_json:
        return rule
    tick = [t.upper() for t in llm_json.get("companies", []) if isinstance(t, str) and t.upper() in companies]
    mets = [m for m in llm_json.get("metrics", []) if m in metrics_known or m in DERIVED]
    yrs = [int(y) for y in llm_json.get("years", []) if str(y).isdigit() and int(y) in years_available]
    op = llm_json.get("operation")
    return Plan(tick or rule.companies, mets or rule.metrics, sorted(yrs) or rule.years,
                op if op in ("lookup", "change", "compare", "cagr") else rule.operation,
                bool(llm_json.get("needs_explanation", rule.needs_explanation)), rule.notes + ["LLM-assisted plan"])
