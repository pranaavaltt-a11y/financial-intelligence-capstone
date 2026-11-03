import pytest

from finintel.ingest.edgar import EdgarClient, FilingRef
from finintel.ingest.xbrl import extract_facts


def by(facts, metric, year):
    return next(f for f in facts if f.metric == metric and f.fiscal_year == year)


def test_one_value_per_metric_and_year(exmp_facts_json):
    facts = extract_facts(exmp_facts_json, "exmp")
    keys = [(f.metric, f.fiscal_year) for f in facts]
    assert len(keys) == len(set(keys))
    assert by(facts, "revenue", 2023).value == 99e9
    assert by(facts, "revenue", 2023).ticker == "EXMP"


def test_quarterly_and_10q_values_are_ignored(exmp_facts_json):
    rev = [f for f in extract_facts(exmp_facts_json, "EXMP") if f.metric == "revenue"]
    assert {f.value for f in rev} == {80e9, 90e9, 99e9}


def test_latest_filing_wins_for_restatements(exmp_facts_json):
    assert by(extract_facts(exmp_facts_json, "EXMP"), "net_income", 2021).value == 15.5e9


def test_balance_sheet_instants(exmp_facts_json):
    assert by(extract_facts(exmp_facts_json, "EXMP"), "total_assets", 2022).value == 160e9


def test_alternative_revenue_tag(smpl_facts_json):
    f = by(extract_facts(smpl_facts_json, "SMPL"), "revenue", 2023)
    assert f.value == 50e9 and f.tag == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_year_filter(exmp_facts_json):
    assert {f.fiscal_year for f in extract_facts(exmp_facts_json, "EXMP", {2023})} == {2023}


def test_edgar_requires_contact_user_agent(tmp_path):
    with pytest.raises(ValueError):
        EdgarClient("", tmp_path)
    with pytest.raises(ValueError):
        EdgarClient("no contact", tmp_path)
    EdgarClient("Capstone Team team@example.edu", tmp_path)


def test_archive_url():
    ref = FilingRef("AAPL", 320193, 2023, "0000320193-23-000106", "aapl-20230930.htm", "2023-09-30", "2023-11-03")
    assert ref.url == "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
