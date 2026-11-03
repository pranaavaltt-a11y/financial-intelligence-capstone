"""Tokenisation shared by BM25, the reranker and the verifier.

Numbers and years are kept as tokens: in financial filings "2023" vs "2022" is often the only thing
separating two near-identical passages (the Multi-HyDE paper's main argument for adding BM25).
"""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an and are as at be been but by for from had has have how in into is it its of on or our over that the
their them they this to was were what when which while who why will with would year years fiscal company
compared during than also such these those other based due including total net per
""".split())

TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    toks = TOKEN_RE.findall(text.lower().replace(",", ""))
    return [t for t in toks if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def content_words(text: str) -> set[str]:
    """Words only (no numbers) - used to judge whether a sentence is supported by a passage."""
    return {t for t in tokenize(text) if not t[0].isdigit() and len(t) > 2}
