"""Load the FICTIONAL test companies into ./sample_data for a dry run of the app.

    python scripts/load_sample_data.py
    FININTEL_DATA_DIR=sample_data streamlit run finintel/app/streamlit_app.py

"Example Corp" and "Sample Industries" are invented, with made-up numbers. Use this only to check the
software works; present results from real SEC data (python -m finintel ingest ...).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from conftest import EXMP, FILING_HTML, SMPL, companyfacts  # noqa: E402

from finintel.config import Settings  # noqa: E402
from finintel.system import build  # noqa: E402


def main(out: str = "sample_data") -> None:
    raw = Path(out) / "raw"
    for t, name, facts in (("EXMP", "Example Corp (FICTIONAL)", companyfacts(EXMP)),
                           ("SMPL", "Sample Industries (FICTIONAL)", companyfacts(SMPL))):
        d = raw / t
        d.mkdir(parents=True, exist_ok=True)
        (d / "company.json").write_text(json.dumps({"ticker": t, "name": name}))
        (d / "companyfacts.json").write_text(json.dumps(facts))
    (raw / "EXMP" / "2023_10k.html").write_text(FILING_HTML)
    print(build(Settings(data_dir=Path(out), gemini_api_key=None), use_gemini=False))
    print(f"Loaded FICTIONAL sample data into {out}/ - for software checks only.")


if __name__ == "__main__":
    main(*sys.argv[1:])
