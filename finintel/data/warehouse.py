"""DuckDB warehouse of XBRL facts, laid out as a company x fiscal-year x metric cube.

Numbers for answers come from here, never from text chunks: an exact, labelled value per company,
year and metric avoids the "wrong year, similar sentence" errors of text retrieval.

OLAP operations exposed:
    slice  - one company (or one year) across the other dimensions
    dice   - a sub-cube: chosen companies x years x metrics
    pivot  - companies as columns, years as rows, for one metric (feeds the dashboard)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from finintel.models import Fact

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (ticker VARCHAR PRIMARY KEY, name VARCHAR);
CREATE TABLE IF NOT EXISTS facts (
    ticker VARCHAR, fiscal_year INTEGER, metric VARCHAR, value DOUBLE, unit VARCHAR,
    tag VARCHAR, accession VARCHAR, filed VARCHAR, period_end VARCHAR,
    PRIMARY KEY (ticker, fiscal_year, metric)
);
"""


class Warehouse:
    def __init__(self, path: Path | str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    # ---------- loading ----------
    def add_company(self, ticker: str, name: str) -> None:
        self.con.execute("INSERT OR REPLACE INTO companies VALUES (?, ?)", [ticker.upper(), name])

    def load_facts(self, facts: list[Fact]) -> int:
        rows = [(f.ticker, f.fiscal_year, f.metric, f.value, f.unit, f.tag, f.accession, f.filed, f.period_end)
                for f in facts]
        if rows:
            self.con.executemany("INSERT OR REPLACE INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    # ---------- lookups ----------
    def companies(self) -> dict[str, str]:
        return dict(self.con.execute("SELECT ticker, name FROM companies ORDER BY ticker").fetchall())

    def years(self, ticker: str | None = None) -> list[int]:
        q = "SELECT DISTINCT fiscal_year FROM facts" + (" WHERE ticker = ?" if ticker else "") + " ORDER BY 1"
        return [r[0] for r in self.con.execute(q, [ticker.upper()] if ticker else []).fetchall()]

    def metrics(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT DISTINCT metric FROM facts ORDER BY 1").fetchall()]

    def get(self, ticker: str, metric: str, year: int) -> Fact | None:
        row = self.con.execute(
            "SELECT ticker, fiscal_year, metric, value, unit, tag, accession, filed, period_end FROM facts "
            "WHERE ticker = ? AND metric = ? AND fiscal_year = ?", [ticker.upper(), metric, year]).fetchone()
        return Fact(*row) if row else None

    # ---------- OLAP ----------
    def slice_company(self, ticker: str) -> pd.DataFrame:
        """All metrics x years for one company."""
        df = self.con.execute("SELECT fiscal_year, metric, value FROM facts WHERE ticker = ?",
                              [ticker.upper()]).df()
        return df.pivot(index="metric", columns="fiscal_year", values="value") if not df.empty else df

    def dice(self, tickers: list[str], years: list[int], metrics: list[str]) -> pd.DataFrame:
        if not (tickers and years and metrics):
            return pd.DataFrame(columns=["ticker", "fiscal_year", "metric", "value", "unit"])
        q = ("SELECT ticker, fiscal_year, metric, value, unit FROM facts WHERE ticker IN ({}) AND fiscal_year IN ({}) "
             "AND metric IN ({}) ORDER BY ticker, metric, fiscal_year").format(
            ",".join("?" * len(tickers)), ",".join("?" * len(years)), ",".join("?" * len(metrics)))
        return self.con.execute(q, [t.upper() for t in tickers] + list(years) + list(metrics)).df()

    def pivot(self, metric: str, tickers: list[str] | None = None) -> pd.DataFrame:
        """Years as rows, companies as columns, for one metric (dashboard view)."""
        q = "SELECT fiscal_year, ticker, value FROM facts WHERE metric = ?"
        params: list = [metric]
        if tickers:
            q += " AND ticker IN ({})".format(",".join("?" * len(tickers)))
            params += [t.upper() for t in tickers]
        df = self.con.execute(q, params).df()
        if df.empty:
            return df
        return df.pivot(index="fiscal_year", columns="ticker", values="value").sort_index()
