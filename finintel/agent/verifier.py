"""Code-based answer verification.

Two checks, neither of which asks an LLM to judge (the Multi-HyDE paper found LLM judges unreliable
on numbers):

1. Numbers - every figure in the answer must match a warehouse fact, a calculation result, or a number
   printed in a cited passage, within the rounding shown ("$383.29 billion" matches 383,285,000,000).
2. Claims - every sentence citing a passage [S#] must share enough content words with that passage.

The report feeds the confidence score shown with each answer and decides whether to retry.
"""

from __future__ import annotations

import re

from finintel.models import Calculation, Chunk, Fact, VerificationReport
from finintel.retrieval.text import content_words

SCALE = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
NUM_RE = re.compile(r"(?P<cur>\$)?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?"
                    r"(?P<scale>trillion|billion|million|thousand|bn|mn)?(?![a-z])\s?(?P<pct>%|percent|pp|percentage points?)?",
                    re.I)
REF_RE = re.compile(r"\[((?:[FCS]\d+)(?:\s*,\s*[FCS]\d+)*)\]")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")
QUOTE_RE = re.compile(r'"([^"]{20,})"')
CLAIM_THRESHOLD = 0.35


def _sentences(text: str) -> list[str]:
    """Split into sentences, never inside a quotation."""
    out, buf, inq = [], [], False
    for part in re.split(r'(")', text.strip()):
        if part == '"':
            inq = not inq
            buf.append(part)
            continue
        if inq:
            buf.append(part)
            continue
        pieces = SENT_RE.split(part)
        buf.append(pieces[0])
        for piece in pieces[1:]:
            out.append("".join(buf).strip())
            buf = [piece]
    if "".join(buf).strip():
        out.append("".join(buf).strip())
    return [s for s in out if s]


def _strip_refs(text: str) -> str:
    return REF_RE.sub(" ", text)


def extract_numbers(text: str) -> list[tuple[str, float, int, float]]:
    """Return (raw, value, decimals, scale) for each figure, skipping bare years like 2023 and FY2023."""
    out = []
    clean = re.sub(r"\bFY\s?\d{4}\b", " ", _strip_refs(text), flags=re.I)
    clean = re.sub(r"\b(?:10-K|10-Q|8-K|20-F)\b|\bItem\s+\d+[A-C]?\b", " ", clean, flags=re.I)  # form / item names
    for m in NUM_RE.finditer(clean):
        raw = m.group(0).strip()
        num = m.group("num").replace(",", "")
        val = float(num)
        scale = SCALE.get((m.group("scale") or "").lower(), 1.0)
        has_unit = bool(m.group("cur") or m.group("scale") or m.group("pct"))
        if not has_unit and "." not in num and 1900 <= val <= 2100:
            continue          # a year
        if not has_unit and val < 10 and "." not in num:
            continue          # small counts like "2 companies", list numbering
        decimals = len(num.split(".")[1]) if "." in num else 0
        out.append((raw, val, decimals, scale))
    return out


def _matches(val: float, decimals: int, scale: float, allowed: list[float]) -> bool:
    target = val * scale
    tol = max(0.5 * 10 ** (-decimals) * scale, abs(target) * 0.002)
    return any(abs(abs(target) - abs(a)) <= tol + 1e-9 for a in allowed)


def allowed_values(facts: list[Fact], calcs: list[Calculation], passages: list[Chunk]) -> list[float]:
    vals: list[float] = [f.value for f in facts] + [c.value for c in calcs]
    for p in passages:
        for _, v, _, s in extract_numbers(p.text):
            vals.append(v * s)
            vals.append(v * 1e6)   # filings state tables "in millions"
            vals.append(v * 1e3)   # ... or "in thousands"
    return vals


def verify(answer: str, facts: dict[str, Fact], calcs: dict[str, Calculation], passages: dict[str, Chunk],
           requested: int, found: int) -> VerificationReport:
    cited = {r.strip() for grp in REF_RE.findall(answer) for r in grp.split(",")}
    cited_passages = [p for ref, p in passages.items() if ref in cited]
    allowed = allowed_values(list(facts.values()), list(calcs.values()), cited_passages)

    nums = extract_numbers(answer)
    bad_nums = [raw for raw, v, d, s in nums if not _matches(v, d, s, allowed)]

    claims_checked = 0
    bad_claims: list[str] = []
    for sent in _sentences(answer):
        refs = {r.strip() for grp in REF_RE.findall(sent) for r in grp.split(",")}
        srefs = [r for r in refs if r.startswith("S")]
        if not srefs:
            continue
        claims_checked += 1
        quotes = QUOTE_RE.findall(sent)   # a quoted passage is checked on the quote, not the attribution
        words = content_words(" ".join(quotes) if quotes else _strip_refs(sent))
        evidence = set().union(*(content_words(passages[r].text) for r in srefs if r in passages))
        overlap = len(words & evidence) / len(words) if words else 1.0
        if overlap < CLAIM_THRESHOLD or any(r not in passages for r in srefs):
            bad_claims.append(sent.strip())

    coverage = found / requested if requested else (1.0 if passages else 0.0)
    return VerificationReport(len(nums), len(nums) - len(bad_nums), claims_checked,
                              claims_checked - len(bad_claims), round(coverage, 3), bad_nums, bad_claims)
