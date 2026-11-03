from finintel.index.parse import parse_filing
from finintel.retrieval.dense import TfidfEmbedder
from finintel.retrieval.hybrid import HybridRetriever
from finintel.retrieval.text import tokenize


def test_sections_tables_and_hidden_header(filing_html):
    chunks = parse_filing(filing_html, "exmp", 2023)
    sections = {c.section for c in chunks}
    assert {"Item 1 Business", "Item 1A Risk Factors", "Item 7 MD&A", "Item 8 Financial Statements"} <= sections
    assert all("hidden xbrl header" not in c.text for c in chunks)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1 and tables[0].section == "Item 7 MD&A"
    assert "Revenue | 99,000 | 90,000" in tables[0].text
    assert all(c.ticker == "EXMP" and c.fiscal_year == 2023 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_small_chunks_split_by_word_budget(filing_html):
    assert len(parse_filing(filing_html, "EXMP", 2023, words_per_chunk=10)) > len(parse_filing(filing_html, "EXMP", 2023))


def test_tokenize_keeps_numbers_and_years():
    assert tokenize("Revenue rose 10% in 2023 to $99,000 million") == ["revenue", "rose", "10", "2023", "99000",
                                                                       "million"]


def test_hybrid_search_finds_explanation(filing_html):
    chunks = parse_filing(filing_html, "EXMP", 2023, words_per_chunk=25)
    r = HybridRetriever(chunks, TfidfEmbedder())
    hits = r.search("Why did operating margin decline?", k=3)
    assert hits and "research and development" in hits[0].chunk.text
    assert hits[0].chunk.section == "Item 7 MD&A"


def test_risk_question_prefers_risk_section(filing_html):
    chunks = parse_filing(filing_html, "EXMP", 2023, words_per_chunk=25)
    hits = HybridRetriever(chunks, TfidfEmbedder()).search("What competition risks does the company face?", k=2)
    assert hits[0].chunk.section == "Item 1A Risk Factors"


def test_metadata_filters(filing_html):
    chunks = parse_filing(filing_html, "EXMP", 2023)
    r = HybridRetriever(chunks, TfidfEmbedder())
    assert r.search("revenue", tickers={"OTHER"}) == []
    assert r.search("revenue", years={2019}) == []
    assert r.search("revenue", tickers={"EXMP"}, years={2023})
