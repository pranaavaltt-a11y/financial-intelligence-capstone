"""Deterministic calculation tools.

The LLM never does arithmetic. Every figure in an answer is computed here from warehouse facts,
and each result records its formula and the facts it used, so the verifier can recompute it.
"""

from __future__ import annotations

from finintel.models import Calculation, Fact

# derived metric -> (numerator metric, denominator metric, unit, label)
DERIVED: dict[str, tuple[str, str, str, str]] = {
    "operating_margin": ("operating_income", "revenue", "%", "operating margin"),
    "net_margin": ("net_income", "revenue", "%", "net margin"),
    "gross_margin": ("gross_profit", "revenue", "%", "gross margin"),
    "rd_intensity": ("rd_expense", "revenue", "%", "R&D as % of revenue"),
    "debt_to_equity": ("total_liabilities", "stockholders_equity", "x", "liabilities-to-equity"),
}

LABELS = {
    "revenue": "revenue", "cost_of_revenue": "cost of revenue", "gross_profit": "gross profit",
    "operating_income": "operating income", "net_income": "net income", "rd_expense": "R&D expense",
    "operating_cash_flow": "operating cash flow", "eps_diluted": "diluted EPS", "total_assets": "total assets",
    "total_liabilities": "total liabilities", "stockholders_equity": "stockholders' equity", "cash": "cash",
    **{k: v[3] for k, v in DERIVED.items()},
}


class CalcError(ValueError):
    pass


def _n(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.4g}"


def ratio(name: str, num: Fact, den: Fact, ids: tuple[str, str]) -> Calculation:
    """Margins / ratios. % units are multiplied by 100."""
    if den.value == 0:
        raise CalcError(f"{den.metric} is zero for {den.ticker} FY{den.fiscal_year}")
    unit = DERIVED[name][2]
    val = num.value / den.value * (100 if unit == "%" else 1)
    return Calculation(name, f"{num.ticker} {LABELS[name]} FY{num.fiscal_year}", val, unit,
                       f"{_n(num.value)} / {_n(den.value)}" + (" x 100" if unit == "%" else ""), list(ids))


def growth(metric: str, old: Fact, new: Fact, ids: tuple[str, str]) -> Calculation:
    """Year-over-year (or multi-year) percentage change."""
    if old.value == 0:
        raise CalcError(f"{metric} is zero in FY{old.fiscal_year}; growth undefined")
    val = (new.value - old.value) / abs(old.value) * 100
    return Calculation(f"{metric}_growth", f"{new.ticker} {LABELS.get(metric, metric)} growth "
                       f"FY{old.fiscal_year}->FY{new.fiscal_year}", val, "%",
                       f"({_n(new.value)} - {_n(old.value)}) / {_n(abs(old.value))} x 100", list(ids))


def point_change(name: str, old: Calculation, new: Calculation) -> Calculation:
    """Change in a percentage metric, in percentage points (e.g. margin 30.1% -> 29.8% = -0.3 pp)."""
    return Calculation(f"{name}_change", f"{new.label.rsplit(' FY', 1)[0]} change", new.value - old.value, "pp",
                       f"{new.value:.2f} - {old.value:.2f}", old.inputs + new.inputs)


def cagr(metric: str, old: Fact, new: Fact, ids: tuple[str, str]) -> Calculation:
    years = new.fiscal_year - old.fiscal_year
    if years <= 0 or old.value <= 0 or new.value <= 0:
        raise CalcError("CAGR needs two positive values at least one year apart")
    val = ((new.value / old.value) ** (1 / years) - 1) * 100
    return Calculation(f"{metric}_cagr", f"{new.ticker} {LABELS.get(metric, metric)} CAGR "
                       f"FY{old.fiscal_year}->FY{new.fiscal_year}", val, "%",
                       f"({_n(new.value)} / {_n(old.value)})^(1/{years}) - 1", list(ids))
