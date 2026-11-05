"""End-to-end question answering.

    question
      -> planner            (rules, optionally LLM-assisted)
      -> warehouse facts    (exact numbers per company / year / metric)          [F#]
      -> calculation tools  (margins, growth, point changes, CAGR in Python)    [C#]
      -> hybrid retrieval   (passages explaining the "why")                       [S#]
      -> composer           (LLM if available, else deterministic template)
      -> verifier           (numbers + claims); on failure: retry, then fall back
      -> Answer with calculations, citations and a confidence score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from finintel.agent import tools
from finintel.agent.llm import ANSWER_PROMPT, LLM, PLAN_PROMPT, parse_json
from finintel.agent.planner import Plan, merge_llm_plan, rule_plan
from finintel.agent.tools import DERIVED, LABELS, CalcError
from finintel.agent.verifier import verify
from finintel.data.warehouse import Warehouse
from finintel.models import Answer, Calculation, Chunk, Citation, Fact, fmt_value
from finintel.retrieval.hybrid import HybridRetriever

log = logging.getLogger(__name__)

# Wording 10-K MD&A sections use when discussing each metric (helps keyword search find the explanation).
EXPLAIN_TERMS = {
    "operating_margin": "operating income operating expenses segment",
    "net_margin": "net income income taxes",
    "gross_margin": "gross margin cost of sales products services",
    "revenue": "net sales revenue segment",
    "net_income": "net income operating income",
    "operating_income": "operating income operating expenses segment",
    "rd_expense": "research and development",
    "rd_intensity": "research and development",
    "operating_cash_flow": "cash provided by operating activities",
}


@dataclass
class Evidence:
    facts: dict[str, Fact]
    calcs: dict[str, Calculation]
    passages: dict[str, Chunk]
    requested: int
    found: int
    missing: list[str]


class FinancialQA:
    def __init__(self, warehouse: Warehouse, retriever: HybridRetriever | None, llm: LLM | None = None,
                 top_k: int = 8, passages_in_answer: int = 2, max_retries: int = 1, llm_planner: bool = False,
                 passages_for_llm: int = 5):
        self.wh = warehouse
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.passages_in_answer = passages_in_answer   # quoted by the template answer
        self.passages_for_llm = passages_for_llm       # the LLM reads more and picks the relevant ones
        self.max_retries = max_retries
        self.llm_planner = llm_planner   # off by default: rules plan well and it saves one LLM call per question

    # ---------------- planning ----------------
    def plan(self, question: str) -> Plan:
        companies = self.wh.companies()
        years = self.wh.years()
        plan = rule_plan(question, companies, years)
        if self.llm and self.llm_planner:
            raw = self.llm.generate(PLAN_PROMPT.format(companies=companies, metrics=self.wh.metrics() + list(DERIVED),
                                                       years=years, question=question), as_json=True)
            plan = merge_llm_plan(plan, parse_json(raw), companies, set(self.wh.metrics()), years)
        return plan

    # ---------------- evidence ----------------
    def gather(self, question: str, plan: Plan, widen: bool = False) -> Evidence:
        facts: dict[str, Fact] = {}
        fid: dict[tuple, str] = {}
        missing: list[str] = []
        requested = found = 0
        for t in plan.companies:
            for m in plan.base_metrics():
                for y in plan.years:
                    requested += 1
                    f = self.wh.get(t, m, y)
                    if f is None:
                        missing.append(f"{t} {LABELS.get(m, m)} FY{y}")
                        continue
                    found += 1
                    ref = f"F{len(facts) + 1}"
                    facts[ref] = f
                    fid[(t, m, y)] = ref

        calcs: dict[str, Calculation] = {}

        def add(c: Calculation) -> str:
            ref = f"C{len(calcs) + 1}"
            calcs[ref] = c
            return ref

        for t in plan.companies:
            for m in plan.metrics:
                per_year: dict[int, Calculation] = {}
                if m in DERIVED:
                    num, den = DERIVED[m][:2]
                    for y in plan.years:
                        if (t, num, y) in fid and (t, den, y) in fid:
                            try:
                                c = tools.ratio(m, facts[fid[(t, num, y)]], facts[fid[(t, den, y)]],
                                                (fid[(t, num, y)], fid[(t, den, y)]))
                                add(c)
                                per_year[y] = c
                            except CalcError as e:
                                missing.append(str(e))
                ys = [y for y in plan.years if (t, m, y) in fid or y in per_year]
                if plan.operation in ("change", "cagr", "compare") and len(ys) >= 2:
                    y0, y1 = ys[0], ys[-1]
                    try:
                        if m in DERIVED:
                            if DERIVED[m][2] == "%":
                                add(tools.point_change(m, per_year[y0], per_year[y1]))
                        elif plan.operation == "cagr" and y1 - y0 > 1:
                            add(tools.cagr(m, facts[fid[(t, m, y0)]], facts[fid[(t, m, y1)]],
                                           (fid[(t, m, y0)], fid[(t, m, y1)])))
                        else:
                            add(tools.growth(m, facts[fid[(t, m, y0)]], facts[fid[(t, m, y1)]],
                                             (fid[(t, m, y0)], fid[(t, m, y1)])))
                    except CalcError as e:
                        missing.append(str(e))

        passages: dict[str, Chunk] = {}
        if self.retriever and (plan.needs_explanation or not facts):
            # Search every filing in the period: a FY2022 -> FY2024 change is explained across the FY2022, FY2023
            # and FY2024 10-Ks, not just the end years.
            years = set(range(min(plan.years), max(plan.years) + 1)) if plan.years and not widen else None
            hits = self.retriever.search(self._search_query(question, plan), k=self.top_k,
                                         tickers=set(plan.companies) or None, years=years)
            if plan.needs_explanation:   # narrative explains "why"; tables only if nothing else was found
                hits = [h for h in hits if h.chunk.kind == "text"] + [h for h in hits if h.chunk.kind != "text"]
            n = self.passages_for_llm if self.llm else self.passages_in_answer + (2 if widen else 0)
            for i, h in enumerate(hits[:n]):
                passages[f"S{i + 1}"] = h.chunk
        return Evidence(facts, calcs, passages, requested, found, missing)

    @staticmethod
    def _search_query(question: str, plan: Plan) -> str:
        """Add the words filings use when explaining a change, so keyword search finds the MD&A explanation."""
        if not (plan.needs_explanation and plan.metrics):
            return question
        terms = {LABELS.get(m, m) for m in plan.metrics + plan.base_metrics()}
        extra = EXPLAIN_TERMS.get(plan.metrics[0], "")
        return f"{question} {' '.join(sorted(terms))} {extra} increased decreased primarily due to driven by compared"

    # ---------------- composing ----------------
    @staticmethod
    def _company(t: str, names: dict[str, str]) -> str:
        return names.get(t) or t

    def template_answer(self, plan: Plan, ev: Evidence) -> str:
        names = self.wh.companies()
        by_label = {c.label: (ref, c) for ref, c in ev.calcs.items()}
        fact_ref = {(f.ticker, f.metric, f.fiscal_year): ref for ref, f in ev.facts.items()}
        sents: list[str] = []
        for t in plan.companies:
            co = self._company(t, names)
            for m in plan.metrics:
                label = LABELS.get(m, m)
                vals: list[tuple[int, str, str]] = []           # (year, formatted, ref)
                for y in plan.years:
                    if m in DERIVED:
                        hit = by_label.get(f"{t} {label} FY{y}")
                        if hit:
                            vals.append((y, fmt_value(hit[1].value, hit[1].unit), hit[0]))
                    elif (t, m, y) in fact_ref:
                        f = ev.facts[fact_ref[(t, m, y)]]
                        vals.append((y, fmt_value(f.value, f.unit), fact_ref[(t, m, y)]))
                if not vals:
                    continue
                if len(vals) == 1:
                    y, v, r = vals[0]
                    sents.append(f"{co}'s {label} was {v} in FY{y} [{r}].")
                    continue
                (y0, v0, r0), (y1, v1, r1) = vals[0], vals[-1]
                chg = next(((ref, c) for ref, c in ev.calcs.items()
                            if c.label.startswith(f"{t} {label}") and c.name.endswith(("_growth", "_change", "_cagr"))),
                           None)
                tail = ""
                if chg:
                    ref, c = chg
                    kind = "a compound annual growth rate of" if c.name.endswith("_cagr") else "a change of"
                    tail = f", {kind} {fmt_value(c.value, c.unit) if c.unit != '%' else f'{c.value:+.1f}%'} [{ref}]"
                sents.append(f"{co}'s {label} went from {v0} in FY{y0} [{r0}] to {v1} in FY{y1} [{r1}]{tail}.")
        if plan.operation == "compare" and len(plan.companies) > 1:
            for m in plan.metrics:
                label = LABELS.get(m, m)
                y = plan.years[-1] if plan.years else None
                vals = []
                for t in plan.companies:
                    if m in DERIVED:
                        hit = by_label.get(f"{t} {label} FY{y}")
                        if hit:
                            vals.append((hit[1].value, t, hit[0]))
                    elif (t, m, y) in fact_ref:
                        vals.append((ev.facts[fact_ref[(t, m, y)]].value, t, fact_ref[(t, m, y)]))
                if len(vals) > 1:
                    v, t, r = max(vals)
                    sents.append(f"In FY{y}, {self._company(t, names)} had the highest {label} [{r}].")
        for ref, ch in list(ev.passages.items())[: self.passages_in_answer]:
            words = ch.text.split()
            quote = (" ".join(words[:45]) + ("..." if len(words) > 45 else "")).replace('"', "'")
            sents.append(f"Context from {ch.ticker}'s FY{ch.fiscal_year} 10-K ({ch.section}): \"{quote}\" [{ref}].")
        for m in ev.missing:
            sents.append(f"Not available in the loaded data: {m}.")
        if not sents:
            sents.append("I could not find data for this question in the loaded filings.")
        return " ".join(sents)

    def _llm_answer(self, question: str, ev: Evidence, feedback: str = "") -> str | None:
        if not self.llm:
            return None
        facts = "\n".join(f"[{r}] {f.ticker} {LABELS.get(f.metric, f.metric)} FY{f.fiscal_year} = "
                          f"{fmt_value(f.value, f.unit)}" for r, f in ev.facts.items()) or "(none)"
        calcs = "\n".join(f"[{r}] {c.label} = {fmt_value(c.value, c.unit) if c.unit != '%' else f'{c.value:.1f}%'}"
                          f" ({c.formula})" for r, c in ev.calcs.items()) or "(none)"
        passages = "\n".join(f"[{r}] ({p.cite()}) {p.text[:1200]}" for r, p in ev.passages.items()) or "(none)"
        return self.llm.generate(ANSWER_PROMPT.format(question=question, facts=facts, calcs=calcs,
                                                      passages=passages, feedback=feedback))

    # ---------------- main entry ----------------
    def ask(self, question: str) -> Answer:
        plan = self.plan(question)
        attempts = 0
        notes: list[str] = []
        got_draft = False
        best: tuple[str, Evidence, object, bool] | None = None
        for widen in [False] + [True] * self.max_retries:
            attempts += 1
            ev = self.gather(question, plan, widen=widen)
            candidates: list[tuple[str, bool]] = []
            draft = None if (self.llm and attempts > 1 and not got_draft) else self._llm_answer(question, ev)
            if self.llm and not draft:
                err = getattr(self.llm, "last_error", None) or "no response"
                reason = ("Google's servers were overloaded" if "503" in err or "UNAVAILABLE" in err else
                          "the free-tier rate limit was reached" if "429" in err else err[:120])
                notes.append(f"Gemini did not return an answer ({reason}) - showing the verified template answer.")
            elif self.llm:
                got_draft = True
            if draft:
                rep = verify(draft, ev.facts, ev.calcs, ev.passages, ev.requested, ev.found)
                if not rep.passed and (rep.unsupported_numbers or rep.unsupported_claims):
                    log.warning("Gemini draft rejected by verifier - numbers: %s; sentences: %s",
                                rep.unsupported_numbers, rep.unsupported_claims)
                    notes.append(f"Gemini's first draft failed verification (numbers {rep.unsupported_numbers}, "
                                 f"{len(rep.unsupported_claims)} unsupported sentence(s)); asked it to correct.")
                    fb = ("\nYour previous draft had unsupported content; fix it. Unsupported numbers: "
                          f"{rep.unsupported_numbers}. Unsupported sentences: {rep.unsupported_claims}.")
                    draft2 = self._llm_answer(question, ev, fb)
                    if draft2:
                        draft = draft2
                elif ev.passages and not self._explained(draft, plan):
                    notes.append("Gemini's first draft skipped the explanation; asked it to use the filing passages.")
                    fb = ("\nYour previous draft did not explain WHY. The PASSAGES below come from the company's 10-K "
                          "MD&A; summarise the causes they state for the change, citing [S#] after each sentence.")
                    draft2 = self._llm_answer(question, ev, fb)
                    if draft2:
                        draft = draft2
                candidates.append((draft, True))
            candidates.append((self.template_answer(plan, ev), False))
            for text, used_llm in candidates:
                rep = verify(text, ev.facts, ev.calcs, ev.passages, ev.requested, ev.found)
                key = self._rank(rep, text, used_llm, plan)
                if best is None or key > self._rank(best[2], best[0], best[3], plan):
                    best = (text, ev, rep, used_llm)
                if rep.passed and self._explained(text, plan):
                    break       # verified and complete; no need to look at the template
            if best and best[2].passed and self._explained(best[0], plan):
                break      # verified, and has an explanation if one was asked for
        text, ev, rep, used_llm = best  # type: ignore[misc]
        if self.llm and not used_llm and got_draft:
            log.warning("Showing the template answer: no Gemini draft passed verification")
            notes.append("No Gemini draft passed verification - the verified template answer is shown.")
        return Answer(question, text, list(ev.calcs.values()), self._citations(ev, text), rep, used_llm, attempts,
                      list(dict.fromkeys(notes)))

    @staticmethod
    def _explained(text: str, plan: Plan) -> bool:
        """A "why" question is only answered if the answer cites at least one filing passage."""
        return not plan.needs_explanation or "[S" in text

    @classmethod
    def _rank(cls, rep, text: str, used_llm: bool, plan: Plan) -> tuple:
        """Prefer verified answers, then ones that cite the explanation asked for, then LLM prose, then confidence."""
        return (rep.passed, cls._explained(text, plan), used_llm, rep.confidence)

    @staticmethod
    def _citations(ev: Evidence, text: str) -> list[Citation]:
        out: list[Citation] = []
        cited_calcs = [c for r, c in ev.calcs.items() if f"{r}]" in text or f"{r}," in text]
        via_calcs = {i for c in cited_calcs for i in c.inputs}
        for r, f in ev.facts.items():
            if f"{r}]" in text or f"{r}," in text or r in via_calcs:
                out.append(Citation(r, f.cite()))
        for r, c in ev.calcs.items():
            if c in cited_calcs:
                out.append(Citation(r, f"Calculation: {c.label} = {c.formula} (inputs {', '.join(c.inputs)})"))
        for r, p in ev.passages.items():
            if f"{r}]" in text or f"{r}," in text:
                out.append(Citation(r, p.cite() + (f" ({p.source})" if p.source else ""), p.text[:300]))
        return out
