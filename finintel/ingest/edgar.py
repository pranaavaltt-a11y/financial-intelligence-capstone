"""Download 10-K filings and XBRL company facts from SEC EDGAR.

SEC fair-access rules: identify yourself with a User-Agent (SEC_USER_AGENT env var) and stay under
10 requests per second. Files are cached under data/raw/<TICKER>/ so re-runs don't re-download.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


@dataclass
class FilingRef:
    ticker: str
    cik: int
    fiscal_year: int
    accession: str
    primary_doc: str
    report_date: str
    filing_date: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_doc)


class EdgarClient:
    def __init__(self, user_agent: str, raw_dir: Path, min_interval: float = 0.15):
        if not user_agent or "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent with contact details, e.g. 'Team Name you@college.edu'. "
                             "Set SEC_USER_AGENT.")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.raw_dir = raw_dir
        self.min_interval = min_interval
        self._last = 0.0
        self._ciks: dict[str, int] | None = None

    def _get(self, url: str) -> requests.Response:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.get(url, timeout=60)
        self._last = time.monotonic()
        resp.raise_for_status()
        return resp

    def cik(self, ticker: str) -> int:
        if self._ciks is None:
            data = self._get(TICKERS_URL).json()
            self._ciks = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
        try:
            return self._ciks[ticker.upper()]
        except KeyError:
            raise ValueError(f"Unknown ticker {ticker!r}") from None

    def company_name(self, ticker: str) -> str:
        sub = self._get(SUBMISSIONS_URL.format(cik=self.cik(ticker))).json()
        return sub.get("name", ticker)

    def list_10k(self, ticker: str, years: set[int]) -> list[FilingRef]:
        cik = self.cik(ticker)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        refs = []
        for form, acc, doc, rep, filed in zip(recent["form"], recent["accessionNumber"], recent["primaryDocument"],
                                              recent["reportDate"], recent["filingDate"]):
            if form != "10-K" or not rep:
                continue
            fy = int(rep[:4])
            if fy in years:
                refs.append(FilingRef(ticker.upper(), cik, fy, acc, doc, rep, filed))
        return refs

    def download_10k(self, ref: FilingRef) -> Path:
        out = self.raw_dir / ref.ticker / f"{ref.fiscal_year}_10k.html"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(self._get(ref.url).content)
            meta = out.with_suffix(".json")
            meta.write_text(json.dumps(ref.__dict__ | {"url": ref.url}, indent=2))
        return out

    def download_companyfacts(self, ticker: str) -> Path:
        out = self.raw_dir / ticker.upper() / "companyfacts.json"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(self._get(FACTS_URL.format(cik=self.cik(ticker))).content)
        return out
