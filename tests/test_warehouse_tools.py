import pytest

from finintel.agent import tools
from finintel.data.warehouse import Warehouse
from finintel.ingest.xbrl import extract_facts
from finintel.models import Fact, fmt_value


@pytest.fixture
def wh(exmp_facts_json, smpl_facts_json):
    w = Warehouse()
    w.add_company("EXMP", "Example Corp")
    w.add_company("SMPL", "Sample Industries Inc.")
    w.load_facts(extract_facts(exmp_facts_json, "EXMP") + extract_facts(smpl_facts_json, "SMPL"))
    yield w
    w.close()


def test_get_and_years(wh):
    assert wh.get("exmp", "revenue", 2022).value == 90e9
    assert wh.get("EXMP", "revenue", 1999) is None
    assert wh.years("SMPL") == [2022, 2023]
    assert wh.years() == [2021, 2022, 2023]


def test_slice_dice_pivot(wh):
    sl = wh.slice_company("EXMP")
    assert sl.loc["revenue", 2023] == 99e9
    d = wh.dice(["EXMP", "SMPL"], [2023], ["revenue", "net_income"])
    assert len(d) == 4
    pv = wh.pivot("revenue", ["EXMP", "SMPL"])
    assert list(pv.columns) == ["EXMP", "SMPL"] and pv.loc[2023, "SMPL"] == 50e9


def test_reload_replaces_not_duplicates(wh, exmp_facts_json):
    wh.load_facts(extract_facts(exmp_facts_json, "EXMP"))
    assert len(wh.dice(["EXMP"], [2023], ["revenue"])) == 1


def F(metric, year, value, t="EXMP"):
    return Fact(t, year, metric, value, "USD")


def test_ratio_margin():
    c = tools.ratio("operating_margin", F("operating_income", 2023, 23.76e9), F("revenue", 2023, 99e9), ("F1", "F2"))
    assert c.value == pytest.approx(24.0) and c.unit == "%" and c.inputs == ["F1", "F2"]
    assert c.label == "EXMP operating margin FY2023"


def test_growth_point_change_cagr():
    g = tools.growth("revenue", F("revenue", 2022, 90e9), F("revenue", 2023, 99e9), ("F1", "F2"))
    assert g.value == pytest.approx(10.0)
    m22 = tools.ratio("operating_margin", F("operating_income", 2022, 24e9), F("revenue", 2022, 90e9), ("a", "b"))
    m23 = tools.ratio("operating_margin", F("operating_income", 2023, 23.76e9), F("revenue", 2023, 99e9), ("c", "d"))
    pc = tools.point_change("operating_margin", m22, m23)
    assert pc.unit == "pp" and pc.value == pytest.approx(24.0 - 26.6667, abs=1e-3)
    cg = tools.cagr("revenue", F("revenue", 2021, 80e9), F("revenue", 2023, 99e9), ("x", "y"))
    assert cg.value == pytest.approx(((99 / 80) ** 0.5 - 1) * 100)


def test_calc_errors():
    with pytest.raises(tools.CalcError):
        tools.ratio("net_margin", F("net_income", 2023, 1.0), F("revenue", 2023, 0.0), ("a", "b"))
    with pytest.raises(tools.CalcError):
        tools.growth("revenue", F("revenue", 2022, 0.0), F("revenue", 2023, 1.0), ("a", "b"))


def test_fmt_value():
    assert fmt_value(99e9, "USD") == "$99.00 billion"
    assert fmt_value(12.345, "%") == "12.3%"
    assert fmt_value(-2.6, "pp") == "-2.6 pp"
    assert fmt_value(6.13, "USD/shares") == "$6.13"
