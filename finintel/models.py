"""Shared data structures passed between layers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class Chunk:
    """A passage of a filing: either running text or one table."""

    chunk_id: str
    ticker: str
    fiscal_year: int
    section: str
    kind: str            # "text" | "table"
    text: str
    source: str = ""     # file name or URL of the filing

    def cite(self) -> str:
        return f"{self.ticker} 10-K FY{self.fiscal_year}, {self.section}"


@dataclass
class Fact:
    """One reported XBRL value, e.g. AAPL revenue for FY2023."""

    ticker: str
    fiscal_year: int
    metric: str
    value: float
    unit: str
    tag: str = ""
    accession: str = ""
    filed: str = ""
    period_end: str = ""

    def cite(self) -> str:
        acc = f", accession {self.accession}" if self.accession else ""
        return f"{self.ticker} XBRL {self.tag or self.metric} FY{self.fiscal_year}{acc}"


@dataclass
class Calculation:
    name: str            # e.g. "operating_margin"
    label: str           # e.g. "AAPL operating margin FY2023"
    value: float
    unit: str            # "%", "pp", "x", "USD"
    formula: str         # human-readable, e.g. "114,301 / 383,285"
    inputs: list[str] = field(default_factory=list)   # fact ids used


@dataclass
class Citation:
    ref: str             # "F1" for a fact, "S1" for a passage
    description: str
    snippet: str = ""


@dataclass
class VerificationReport:
    numbers_checked: int
    numbers_supported: int
    claims_checked: int
    claims_supported: int
    coverage: float                  # share of requested facts that were found
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.unsupported_numbers and not self.unsupported_claims and self.coverage > 0

    @property
    def confidence(self) -> float:
        num = self.numbers_supported / self.numbers_checked if self.numbers_checked else 1.0
        clm = self.claims_supported / self.claims_checked if self.claims_checked else 1.0
        score = 0.5 * num + 0.3 * clm + 0.2 * self.coverage
        if self.coverage == 0:      # nothing requested was found: never report high confidence
            score = min(score, 0.3)
        return round(score, 3)

    @property
    def confidence_label(self) -> str:
        c = self.confidence
        return "High" if c >= 0.8 else "Medium" if c >= 0.5 else "Low"


@dataclass
class Answer:
    question: str
    text: str
    calculations: list[Calculation]
    citations: list[Citation]
    verification: VerificationReport
    used_llm: bool = False
    attempts: int = 1
    notes: list[str] = field(default_factory=list)   # why the LLM answer was or wasn't used

    def to_markdown(self) -> str:
        lines = [self.text, ""]
        if self.calculations:
            lines.append("**Calculations**")
            lines += [f"- {c.label}: {fmt_value(c.value, c.unit)}  ({c.formula})" for c in self.calculations]
            lines.append("")
        lines.append("**Sources**")
        lines += [f"- [{c.ref}] {c.description}" for c in self.citations]
        v = self.verification
        lines += ["", f"**Confidence:** {v.confidence_label} ({v.confidence:.2f}) - "
                      f"{v.numbers_supported}/{v.numbers_checked} numbers and "
                      f"{v.claims_supported}/{v.claims_checked} claims verified - "
                      f"{'Gemini-written' if self.used_llm else 'template'} answer"]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def fmt_value(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "pp":
        return f"{value:+.1f} pp"
    if unit == "x":
        return f"{value:.2f}x"
    if unit in ("USD", "USD/shares"):
        if unit == "USD/shares":
            return f"${value:,.2f}"
        a = abs(value)
        if a >= 1e9:
            return f"${value / 1e9:,.2f} billion"
        if a >= 1e6:
            return f"${value / 1e6:,.1f} million"
        return f"${value:,.0f}"
    return f"{value:,.2f} {unit}".strip()


def write_jsonl(path: Path, items: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
