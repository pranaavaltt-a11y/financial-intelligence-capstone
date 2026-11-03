"""Test fixtures.

FICTIONAL DATA ONLY: "Example Corp" (EXMP) and "Sample Industries" (SMPL) are invented companies with
made-up figures, shaped like real SEC companyfacts JSON and 10-K HTML. They exist to test the code
offline and must never be presented as real financial data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finintel.config import Settings

# fiscal year -> metric -> value (USD); fiscal year ends 31 Dec
EXMP = {
    2021: {"Revenues": 80e9, "OperatingIncomeLoss": 20e9, "NetIncomeLoss": 15e9, "GrossProfit": 36e9,
           "Assets": 150e9, "Liabilities": 90e9, "StockholdersEquity": 60e9},
    2022: {"Revenues": 90e9, "OperatingIncomeLoss": 24e9, "NetIncomeLoss": 18e9, "GrossProfit": 41e9,
           "Assets": 160e9, "Liabilities": 95e9, "StockholdersEquity": 65e9},
    2023: {"Revenues": 99e9, "OperatingIncomeLoss": 23.76e9, "NetIncomeLoss": 17.82e9, "GrossProfit": 44e9,
           "Assets": 170e9, "Liabilities": 100e9, "StockholdersEquity": 70e9},
}
SMPL = {
    2022: {"RevenueFromContractWithCustomerExcludingAssessedTax": 40e9, "OperatingIncomeLoss": 6e9,
           "NetIncomeLoss": 4e9, "Assets": 60e9, "Liabilities": 30e9, "StockholdersEquity": 30e9},
    2023: {"RevenueFromContractWithCustomerExcludingAssessedTax": 50e9, "OperatingIncomeLoss": 9e9,
           "NetIncomeLoss": 6e9, "Assets": 66e9, "Liabilities": 31e9, "StockholdersEquity": 35e9},
}
INSTANT = {"Assets", "Liabilities", "StockholdersEquity"}


def companyfacts(data: dict[int, dict[str, float]], restate: dict | None = None) -> dict:
    """Each 10-K reports its own year and the prior year (as real filings do)."""
    gaap: dict = {}
    years = sorted(data)
    for filing_year in years:
        for y in (filing_year - 1, filing_year):
            if y not in data:
                continue
            for tag, val in data[y].items():
                e = {"end": f"{y}-12-31", "val": val, "accn": f"0000000000-{filing_year % 100:02d}-000001",
                     "fy": filing_year, "fp": "FY", "form": "10-K", "filed": f"{filing_year + 1}-02-15"}
                if tag not in INSTANT:
                    e["start"] = f"{y}-01-01"
                gaap.setdefault(tag, {"units": {"USD": []}})["units"]["USD"].append(e)
    # noise that must be ignored: a quarterly value and a 10-Q
    gaap.setdefault("Revenues", {"units": {"USD": []}})["units"]["USD"] += [
        {"start": f"{years[-1]}-10-01", "end": f"{years[-1]}-12-31", "val": 1.0, "fy": years[-1], "fp": "FY",
         "form": "10-K", "filed": f"{years[-1] + 1}-02-15"},
        {"start": f"{years[-1]}-01-01", "end": f"{years[-1]}-03-31", "val": 2.0, "fy": years[-1], "fp": "Q1",
         "form": "10-Q", "filed": f"{years[-1]}-05-01"},
    ]
    if restate:   # a later filing restating an earlier value
        for tag, (y, val) in restate.items():
            gaap[tag]["units"]["USD"].append({"start": f"{y}-01-01", "end": f"{y}-12-31", "val": val, "fy": y + 2,
                                              "fp": "FY", "form": "10-K", "filed": f"{y + 3}-02-20",
                                              "accn": "0000000000-99-000009"})
    return {"entityName": "fixture", "facts": {"us-gaap": gaap}}


FILING_HTML = """<html><body>
<div style="display:none"><ix:header>hidden xbrl header 123456789</ix:header></div>
<p>EXAMPLE CORP - ANNUAL REPORT (FICTIONAL TEST FIXTURE)</p>
<p>Item 1. Business</p>
<p>Example Corp designs widgets and sells subscription software to small businesses worldwide.</p>
<p>Item 1A. Risk Factors</p>
<p>Competition in the widget market is intense. New entrants with lower prices could reduce our market
share and pressure our gross margin.</p>
<p>We depend on a single supplier for widget housings; a disruption at that supplier could delay shipments.</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue increased 10% in 2023, driven by higher subscription software sales and new enterprise customers.</p>
<p>Operating margin declined in 2023 because research and development spending rose sharply to build our
new cloud platform, and freight costs increased.</p>
<table>
<tr><td>(in millions)</td><td>2023</td><td>2022</td></tr>
<tr><td>Revenue</td><td>$</td><td>99,000</td><td>$</td><td>90,000</td></tr>
<tr><td>Operating income</td><td>23,760</td><td>24,000</td></tr>
</table>
<p>Item 8. Financial Statements</p>
<p>The consolidated financial statements are presented on the following pages.</p>
</body></html>"""


@pytest.fixture
def exmp_facts_json() -> dict:
    return companyfacts(EXMP, restate={"NetIncomeLoss": (2021, 15.5e9)})


@pytest.fixture
def smpl_facts_json() -> dict:
    return companyfacts(SMPL)


@pytest.fixture
def filing_html() -> str:
    return FILING_HTML


@pytest.fixture
def data_dir(tmp_path: Path, exmp_facts_json: dict, smpl_facts_json: dict) -> Path:
    """A data/raw tree like `finintel ingest` would create, filled with the fictional fixtures."""
    raw = tmp_path / "raw"
    for t, name, facts in (("EXMP", "Example Corp", exmp_facts_json), ("SMPL", "Sample Industries Inc.", smpl_facts_json)):
        d = raw / t
        d.mkdir(parents=True)
        (d / "company.json").write_text(json.dumps({"ticker": t, "name": name}))
        (d / "companyfacts.json").write_text(json.dumps(facts))
    (raw / "EXMP" / "2023_10k.html").write_text(FILING_HTML)
    return tmp_path


@pytest.fixture
def settings(data_dir: Path) -> Settings:
    return Settings(data_dir=data_dir, gemini_api_key=None, sec_user_agent=None)


@pytest.fixture
def qa(settings: Settings):
    from finintel.system import build, load

    build(settings, use_gemini=False)
    return load(settings, use_llm=False)
