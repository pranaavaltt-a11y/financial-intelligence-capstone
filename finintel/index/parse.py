"""Parse a 10-K HTML filing into section-tagged text and table chunks.

* Hidden inline-XBRL header blocks are dropped.
* Each <table> becomes its own chunk, rows flattened to "cell | cell | cell" so numbers stay together.
* Text is split by 10-K item headings (Item 1A Risk Factors, Item 7 MD&A, ...) and packed into
  ~180-word chunks inside each section, so every chunk knows which section it came from.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from finintel.models import Chunk

SECTION_NAMES = {
    "1": "Item 1 Business", "1a": "Item 1A Risk Factors", "1b": "Item 1B Unresolved Staff Comments",
    "1c": "Item 1C Cybersecurity", "2": "Item 2 Properties", "3": "Item 3 Legal Proceedings",
    "4": "Item 4 Mine Safety", "5": "Item 5 Market for Equity", "6": "Item 6 Reserved",
    "7": "Item 7 MD&A", "7a": "Item 7A Market Risk", "8": "Item 8 Financial Statements",
    "9": "Item 9 Accountant Changes", "9a": "Item 9A Controls and Procedures",
}
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[.:\-–—]?\s*(.{0,80})$", re.IGNORECASE)
BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def _table_text(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [_clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in ("$", "%", ")")]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _section_for(line: str, current: str) -> str:
    m = ITEM_RE.match(line)
    if not m or len(line) > 120:
        return current
    key = m.group(1).lower()
    return SECTION_NAMES.get(key, f"Item {key.upper()}")


def parse_filing(html: str, ticker: str, fiscal_year: int, source: str = "",
                 words_per_chunk: int = 180, max_table_words: int = 450) -> list[Chunk]:
    soup = BeautifulSoup(html, "lxml")
    for hidden in soup.find_all(["ix:header", "script", "style"]):
        hidden.decompose()
    for el in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        el.decompose()

    body = soup.body or soup
    section = "Front Matter"
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_words = 0

    def new_id() -> str:
        return f"{ticker.upper()}-{fiscal_year}-{len(chunks):04d}"

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            chunks.append(Chunk(new_id(), ticker.upper(), fiscal_year, section, "text", " ".join(buf), source))
        buf, buf_words = [], 0

    # Walk tables and leaf-level text blocks in document order.
    for el in body.find_all(["table"] + BLOCK_TAGS):
        if el.name != "table" and (el.find_parent("table") or el.find(["table"] + BLOCK_TAGS)):
            continue  # nested block: its text is handled by the innermost block or the table
        if el.name == "table":
            ttext = _table_text(el)
            if len(ttext.split()) < 4:
                continue
            first = ttext.splitlines()[0]
            new_section = _section_for(first, section)
            if new_section != section:     # some filings put item headings inside layout tables
                flush()
                section = new_section
                continue
            flush()
            words = ttext.split(" ")
            chunks.append(Chunk(new_id(), ticker.upper(), fiscal_year, section, "table",
                                " ".join(words[:max_table_words]), source))
            continue
        line = _clean(el.get_text(" "))
        if not line:
            continue
        new_section = _section_for(line, section)
        if new_section != section:
            flush()
            section = new_section
            continue
        n = len(line.split())
        if buf_words + n > words_per_chunk and buf:
            flush()
        buf.append(line)
        buf_words += n
    flush()
    return chunks


def parse_file(path: Path, ticker: str, fiscal_year: int) -> list[Chunk]:
    return parse_filing(path.read_text(encoding="utf-8", errors="ignore"), ticker, fiscal_year, source=path.name)
