import json

from finintel.agent.pipeline import FinancialQA
from finintel.system import build, data_status, load


def test_build_and_status(settings):
    stats = build(settings, use_gemini=False)
    assert stats["chunks"] > 0 and stats["facts"] > 10
    st = data_status(settings)
    assert set(st["companies"]) == {"EXMP", "SMPL"} and st["companies"]["SMPL"]["years"] == [2022, 2023]


def test_margin_change_with_explanation(qa):
    ans = qa.ask("How did Example Corp's operating margin change from FY2022 to FY2023, and why?")
    v = ans.verification
    assert v.passed, (ans.text, v)
    assert "26.7%" in ans.text and "24.0%" in ans.text and "-2.7 pp" in ans.text
    assert any(c.ref.startswith("S") and "research and development" in c.snippet for c in ans.citations)
    assert {c.name for c in ans.calculations} >= {"operating_margin", "operating_margin_change"}
    assert v.confidence_label == "High"


def test_lookup_and_growth(qa):
    ans = qa.ask("What was Example Corp's revenue in FY2023?")
    assert "$99.00 billion" in ans.text and ans.verification.passed
    g = qa.ask("How much did Sample Industries revenue grow from 2022 to 2023?")
    assert "+25.0%" in g.text and g.verification.passed


def test_compare_two_companies(qa):
    ans = qa.ask("Compare the net margin of Example Corp and Sample Industries in FY2023")
    assert "18.0%" in ans.text and "12.0%" in ans.text
    assert "Example Corp had the highest net margin" in ans.text


def test_missing_data_is_reported_not_invented(qa):
    ans = qa.ask("What was Sample Industries revenue in FY2021?")
    assert "Not available" in ans.text
    assert ans.verification.coverage == 0 and not ans.verification.passed
    assert ans.verification.confidence_label != "High"


class FakeLLM:
    """Stands in for Gemini: first draft hallucinates a number, second draft is correct."""
    name = "fake"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, as_json=False):
        if as_json:
            return json.dumps({"companies": ["EXMP"], "metrics": ["revenue"], "years": [2022, 2023],
                               "operation": "change", "needs_explanation": False})
        self.calls += 1
        if self.calls == 1:
            return "Example Corp's revenue rose to $120 billion [F2]."
        return "Example Corp's revenue rose from $90.00 billion [F1] to $99.00 billion [F2], up 10.0% [C1]."


def test_llm_answer_is_verified_and_corrected(settings):
    build(settings, use_gemini=False)
    base = load(settings, use_llm=False)
    llm = FakeLLM()
    qa = FinancialQA(base.wh, base.retriever, llm)
    ans = qa.ask("How did Example Corp's revenue change?")
    assert ans.used_llm and ans.verification.passed
    assert "$120" not in ans.text and llm.calls == 2


class BrokenLLM:
    name = "broken"

    def generate(self, prompt, as_json=False):
        return None if as_json else "Revenue was $1 trillion [F1]."


def test_unverifiable_llm_falls_back_to_template(settings):
    build(settings, use_gemini=False)
    base = load(settings, use_llm=False)
    ans = FinancialQA(base.wh, base.retriever, BrokenLLM()).ask("What was Example Corp's revenue in FY2023?")
    assert not ans.used_llm and "$99.00 billion" in ans.text and ans.verification.passed


def test_explanation_found_when_asked_years_have_no_filing_text(qa):
    # Regression: real data has XBRL numbers for ~20 years but 10-K text for only a few.
    ans = qa.ask("How did Example Corp's operating margin change from FY2021 to FY2022, and why?")
    assert ans.verification.passed, (ans.text, ans.verification)
    assert any(c.ref.startswith("S") for c in ans.citations)
    assert "25.0%" in ans.text and "26.7%" in ans.text


def test_notes_explain_why_template_was_used(settings):
    build(settings, use_gemini=False)
    base = load(settings, use_llm=False)
    ans = FinancialQA(base.wh, base.retriever, BrokenLLM()).ask("What was Example Corp's revenue in FY2023?")
    assert any("failed verification" in n for n in ans.notes)
    assert any("template answer is shown" in n for n in ans.notes)


class SilentLLM:
    name = "silent"
    last_error = "503 UNAVAILABLE. This model is currently experiencing high demand."

    def generate(self, prompt, as_json=False):
        return None


def test_note_explains_overload_without_claiming_a_draft_failed(settings):
    build(settings, use_gemini=False)
    base = load(settings, use_llm=False)
    ans = FinancialQA(base.wh, base.retriever, SilentLLM()).ask("What was Example Corp's revenue in FY2023?")
    assert ans.notes == ["Gemini did not return an answer (Google's servers were overloaded) - "
                         "showing the verified template answer."]


def test_text_only_question_answers_from_filing(qa):
    # Regression: "risk" questions used to get a revenue figure instead of filing passages.
    ans = qa.ask("What are Example Corp's main risk factors about competition?")
    assert ans.verification.passed
    cited = [c for c in ans.citations if c.ref.startswith("S")]
    assert cited and "Risk Factors" in cited[0].description
    assert "Competition" in cited[0].snippet and "$" not in ans.text.split("Context")[0]


class NoWhyLLM:
    """Answers the numbers correctly but gives up on the 'why' (what Gemini did on Amazon)."""
    name = "nowhy"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, as_json=False):
        self.prompts.append(prompt)
        return ("Example Corp's operating margin went from 26.7% [C1] to 24.0% [C2]. "
                "The provided evidence does not explain why.")


def test_why_question_prefers_an_answer_that_cites_the_explanation(settings):
    build(settings, use_gemini=False)
    base = load(settings, use_llm=False)
    llm = NoWhyLLM()
    ans = FinancialQA(base.wh, base.retriever, llm).ask(
        "How did Example Corp's operating margin change from FY2022 to FY2023, and why?")
    assert any(c.ref.startswith("S") for c in ans.citations)          # explanation quoted from the filing
    assert "research and development" in ans.text
    assert "primarily due to" in llm.prompts[0] and "PASSAGES:" in llm.prompts[0]   # LLM was told where to look
    assert len(llm.prompts) == 2 and "did not explain WHY" in llm.prompts[1]        # asked again before falling back


def test_why_search_covers_every_year_in_the_period(qa):
    plan = qa.plan("How did Example Corp's operating margin change from FY2021 to FY2023, and why?")
    ev = qa.gather("How did Example Corp's operating margin change from FY2021 to FY2023, and why?", plan)
    assert ev.passages and all(p.fiscal_year == 2023 for p in ev.passages.values())   # middle/end-year text found
