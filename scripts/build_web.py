"""Build the browser version of TaxRiskScan (Pyodide) into web/dist/.

The page runs the real ``taxriskscan`` package in the browser with Pyodide, so
the same code powers the CLI, the Streamlit app and the static site. The
output folder can be served by any static host (GitHub Pages, a claude.ai
artifact, Netlify, ...).

    python scripts/build_web.py
    python -m http.server -d web/dist 8000
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DIST = WEB / "dist"


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "py" / "taxriskscan" / "data").mkdir(parents=True)  # staging, removed below

    files: list[str] = []

    def add(src: Path, rel: str) -> None:
        shutil.copy2(src, DIST / "py" / rel)
        files.append(rel)

    for f in sorted((ROOT / "taxriskscan").glob("*.py")):
        add(f, f"taxriskscan/{f.name}")
    for f in sorted((ROOT / "taxriskscan" / "data").glob("*.csv")):
        add(f, f"taxriskscan/data/{f.name}")
    add(WEB / "bridge.py", "bridge.py")
    add(ROOT / "data" / "sample_cbcr.csv", "sample_cbcr.csv")
    add(ROOT / "data" / "sample_cbc.xml", "sample_cbc.xml")
    # One JSON bundle {relative path: text}: some static hosts refuse to serve .py files.
    bundle = {rel: (DIST / "py" / rel).read_text(encoding="utf-8") for rel in files}
    (DIST / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    shutil.rmtree(DIST / "py")

    # Pre-compute the sample result (shown instantly while Pyodide loads) and the CV AUC.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(WEB))
    from taxriskscan import model
    import bridge

    auc = round(model.train().cv_auc, 3)
    (DIST / "sample_result.json").write_text(
        bridge.run(str(ROOT / "data" / "sample_cbcr.csv"), 2026, "real", auc)
    )
    # Research brief on the public CbCR data, pre-computed (it needs cross-validation).
    from taxriskscan import research

    rr = research.run()
    (DIST / "research.json").write_text(json.dumps({
        "overview": rr.overview,
        "validation": rr.validation,
        "by_year": rr.by_year.to_dict("records"),
        "top_havens": rr.top_havens.head(10).to_dict("records"),
        "figures": {k: "data:image/png;base64," + v for k, v in rr.figures.items()},
        "brief_html": research.brief_html(rr),
    }), encoding="utf-8")
    html = (WEB / "index.html").read_text(encoding="utf-8")
    html = html.replace("null; /*CV_AUC*/", f"{auc}; /*CV_AUC*/")
    (DIST / "index.html").write_text(html, encoding="utf-8")
    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"Built {DIST} - {len(files)} package files, {size / 1e6:.1f} MB, CV AUC {auc}")


if __name__ == "__main__":
    main()
