from finintel.agent.planner import find_companies, find_metrics, merge_llm_plan, rule_plan
from finintel.agent.verifier import extract_numbers, verify
from finintel.models import Calculation, Chunk, Fact

COMPANIES = {"EXMP": "Example Corp", "SMPL": "Sample Industries Inc."}
YEARS = [2021, 2022, 2023]


def test_find_companies_by_name_ticker_and_possessive():
    assert find_companies("How did Example Corp's revenue grow?", COMPANIES) == ["EXMP"]
    assert find_companies("compare SMPL and exmp", COMPANIES) == ["SMPL", "EXMP"]
    assert find_companies("What about Sample's margin?", COMPANIES) == ["SMPL"]
    assert find_companies("unrelated question", COMPANIES) == []


def test_find_metrics_prefers_longer_phrase():
    assert find_metrics("operating margin and operating income") == ["operating_margin", "operating_income"]
    assert find_metrics("net profit margin") == ["net_margin"]


def test_rule_plan_change_with_explanation():
    p = rule_plan("How did Example Corp's operating margin change from FY2022 to FY2023, and why?", COMPANIES, YEARS)
    assert p.companies == ["EXMP"] and p.metrics == ["operating_margin"]
    assert p.years == [2022, 2023] and p.operation == "change" and p.needs_explanation
    assert p.base_metrics() == ["operating_income", "revenue"]


def test_rule_plan_defaults_and_compare():
    p = rule_plan("Compare revenue of Example Corp and Sample Industries", COMPANIES, YEARS)
    assert p.operation == "compare" and p.years == [2023] and len(p.companies) == 2
    p2 = rule_plan("How has Example Corp's revenue grown?", COMPANIES, YEARS)
    assert p2.operation == "change" and p2.years == [2022, 2023]
    assert rule_plan("Example Corp revenue CAGR from 2021 to 2023", COMPANIES, YEARS).operation == "cagr"


def test_llm_plan_is_validated():
    rule = rule_plan("Example Corp revenue in 2023", COMPANIES, YEARS)
    bad = {"companies": ["FAKE"], "metrics": ["made_up"], "years": [1990], "operation": "teleport"}
    merged = merge_llm_plan(rule, bad, COMPANIES, {"revenue"}, YEARS)
    assert merged.companies == ["EXMP"] and merged.metrics == ["revenue"] and merged.years == [2023]
    assert merged.operation == rule.operation


def test_extract_numbers_skips_years_and_refs():
    nums = extract_numbers("In FY2023 [F1] revenue was $99.00 billion, up 10.0% [C1] versus 2022; 3 segments.")
    assert [(raw, v, s) for raw, v, _, s in nums] == [("$99.00 billion", 99.0, 1e9), ("10.0%", 10.0, 1.0)]


def _ev():
    facts = {"F1": Fact("EXMP", 2022, "revenue", 90e9, "USD"), "F2": Fact("EXMP", 2023, "revenue", 99e9, "USD")}
    calcs = {"C1": Calculation("revenue_growth", "EXMP revenue growth", 10.0, "%", "")}
    passages = {"S1": Chunk("x", "EXMP", 2023, "Item 7 MD&A", "text",
                            "Revenue increased 10% in 2023, driven by higher subscription software sales.")}
    return facts, calcs, passages


def test_verify_accepts_supported_answer():
    facts, calcs, passages = _ev()
    text = ("Revenue went from $90.00 billion [F1] to $99.00 billion [F2], growth of 10.0% [C1]. "
            "The increase was driven by higher subscription software sales [S1].")
    rep = verify(text, facts, calcs, passages, requested=2, found=2)
    assert rep.passed and rep.numbers_checked == 3 and rep.claims_checked == 1
    assert rep.confidence == 1.0 and rep.confidence_label == "High"


def test_verify_flags_wrong_number_and_unsupported_claim():
    facts, calcs, passages = _ev()
    text = ("Revenue reached $105 billion [F2], growth of 10.0% [C1]. "
            "The increase came from a major acquisition in Europe [S1].")
    rep = verify(text, facts, calcs, passages, requested=2, found=2)
    assert not rep.passed
    assert rep.unsupported_numbers == ["$105 billion"]
    assert len(rep.unsupported_claims) == 1
    assert rep.confidence < 0.8


def test_verify_citation_to_missing_passage():
    facts, calcs, passages = _ev()
    rep = verify("Sales grew because of subscription software [S9].", facts, calcs, passages, 2, 2)
    assert rep.unsupported_claims


def test_form_and_item_names_are_not_figures():
    # Regression: the "10" in "10-K" was flagged as an unsupported number.
    nums = extract_numbers("Context from AAPL's FY2024 10-K (Item 7 MD&A) and the 10-Q, Item 1A.")
    assert nums == []


def test_text_only_plan_has_no_metric_or_year():
    p = rule_plan("What does Example Corp say about supply chain risks?", COMPANIES, YEARS)
    assert p.metrics == [] and p.years == [] and p.needs_explanation
    # a metric question is unaffected
    assert rule_plan("Example Corp revenue in 2023", COMPANIES, YEARS).metrics == ["revenue"]
