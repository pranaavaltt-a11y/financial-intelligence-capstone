"""Turn the SEC "companyfacts" XBRL JSON into clean annual facts.

Companies tag the same concept differently (revenue alone has several us-gaap tags), and every 10-K
also repeats prior years' figures. This module picks one value per (metric, fiscal year):

* only annual figures from 10-K filings (fp == "FY"),
* duration metrics must cover roughly one year; balance-sheet metrics are instants,
* fiscal year = calendar year of the period end date (matches how AAPL, MSFT, NVDA name their years),
* when several filings report the same period, the most recently filed value wins (restatements),
* tags are tried in priority order; a higher-priority tag wins for a given year.
"""

from __future__ import annotations

from datetime import date

from finintel.models import Fact

# metric -> (us-gaap tags in priority order, unit, kind)
METRICS: dict[str, tuple[list[str], str, str]] = {
    "revenue": (["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"], "USD", "duration"),
    "cost_of_revenue": (["CostOfRevenue", "CostOfGoodsAndServicesSold"], "USD", "duration"),
    "gross_profit": (["GrossProfit"], "USD", "duration"),
    "operating_income": (["OperatingIncomeLoss"], "USD", "duration"),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "USD", "duration"),
    "rd_expense": (["ResearchAndDevelopmentExpense"], "USD", "duration"),
    "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities"], "USD", "duration"),
    "eps_diluted": (["EarningsPerShareDiluted"], "USD/shares", "duration"),
    "total_assets": (["Assets"], "USD", "instant"),
    "total_liabilities": (["Liabilities"], "USD", "instant"),
    "stockholders_equity": (["StockholdersEquity"], "USD", "instant"),
    "cash": (["CashAndCashEquivalentsAtCarryingValue"], "USD", "instant"),
}


def _days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def extract_facts(companyfacts: dict, ticker: str, years: set[int] | None = None) -> list[Fact]:
    """Return one Fact per (metric, fiscal_year) found in a companyfacts document."""
    gaap = companyfacts.get("facts", {}).get("us-gaap", {})
    out: list[Fact] = []
    for metric, (tags, unit, kind) in METRICS.items():
        chosen: dict[int, Fact] = {}
        for priority, tag in enumerate(tags):
            entries = gaap.get(tag, {}).get("units", {}).get(unit, [])
            best: dict[int, dict] = {}
            for e in entries:
                if e.get("form") != "10-K" or e.get("fp") != "FY" or "end" not in e:
                    continue
                if kind == "duration":
                    if "start" not in e or not 330 <= _days(e["start"], e["end"]) <= 400:
                        continue
                elif "start" in e:
                    continue
                fy = date.fromisoformat(e["end"]).year
                if years and fy not in years:
                    continue
                prev = best.get(fy)
                if prev is None or e.get("filed", "") > prev.get("filed", ""):
                    best[fy] = e
            for fy, e in best.items():
                if fy in chosen:          # a higher-priority tag already supplied this year
                    continue
                chosen[fy] = Fact(ticker=ticker.upper(), fiscal_year=fy, metric=metric, value=float(e["val"]),
                                  unit=unit, tag=tag, accession=e.get("accn", ""), filed=e.get("filed", ""),
                                  period_end=e["end"])
        out.extend(chosen.values())
    return sorted(out, key=lambda f: (f.metric, f.fiscal_year))
