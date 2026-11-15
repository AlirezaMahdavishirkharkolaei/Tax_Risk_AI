"""Outputs: risk heatmap, risk-score chart, summary CSV and a self-contained HTML audit report."""

from __future__ import annotations

import base64
import html
import io
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import jurisdictions, legal  # noqa: E402
from .pillar2 import group_totals  # noqa: E402
from .pipeline import DISCLAIMER, AnalysisResult  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# Single-hue sequential ramp (light -> dark) for magnitude.
BLUE_RAMP = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LEVEL_COLORS = {"High": "#d03b3b", "Medium": "#fab219", "Low": "#0ca30c"}
LEVEL_ICONS = {"High": "▲", "Medium": "●", "Low": "▼"}

HEATMAP_COLUMNS = {
    "Low tax\n(vs 15%)": "h_low_tax",
    "Profit per\nemployee": "h_ppe",
    "Profit vs\nsubstance": "h_gap",
    "Profit\nmargin": "h_margin",
    "Listed\njurisdiction": "h_listed",
    "ML\nprobability": "ml_probability",
    "Anomaly\n(k-means)": "anomaly_score",
    "Overall\nrisk": "h_overall",
}


def _label(row: pd.Series) -> str:
    label = row["jurisdiction"] if row["group"] == "Group" else f"{row['jurisdiction']}  ·  {row['group']}"
    if "year" in row.index and not pd.isna(row["year"]):
        label += f" {int(row['year'])}"
    return label


def heatmap_frame(res: AnalysisResult) -> pd.DataFrame:
    """Each indicator mapped to a 0-1 'risk intensity' scale, rows sorted by risk."""
    r = res.results.copy()
    t = res.config.thresholds
    profitable = r["profit_before_tax"] > 0
    r["h_low_tax"] = np.where(profitable, ((t.globe_minimum_rate - r["etr"]) / t.globe_minimum_rate).clip(0, 1), 0)
    # Profit with zero staff counts as maximum intensity; losses as none.
    ppe = r["profit_per_employee"].where(r["employees"] > 0, np.inf).where(profitable, 0)
    r["h_ppe"] = (ppe / t.very_high_profit_per_employee).clip(0, 1).fillna(0)
    r["h_gap"] = (r["profit_activity_gap"] / t.strong_profit_activity_mismatch).clip(0, 1)
    r["h_margin"] = (r["profit_margin"].fillna(0) / t.very_high_margin).clip(0, 1)
    r["h_listed"] = np.maximum(r["eu_non_cooperative"] * 1.0, r["zero_cit_jurisdiction"] * 0.5)
    r["h_overall"] = r["risk_score"] / 100
    r = r.sort_values("risk_score", ascending=False)
    frame = pd.DataFrame({k: r[v].astype(float).values for k, v in HEATMAP_COLUMNS.items()})
    frame.index = [_label(row) for _, row in r.iterrows()]
    return frame


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_2, length=0)


def plot_heatmap(res: AnalysisResult, path: str | Path | None = None):
    data = heatmap_frame(res)
    n = len(data)
    fig, ax = plt.subplots(figsize=(10, 1.4 + 0.36 * n), facecolor=SURFACE)
    _style(ax)
    cmap = LinearSegmentedColormap.from_list("risk_blue", BLUE_RAMP)
    im = ax.imshow(data.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    # 2px surface gaps between cells.
    ax.set_xticks(np.arange(-0.5, data.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(data.shape[1]), data.columns, fontsize=8.5, color=TEXT_2)
    ax.xaxis.tick_top()
    ax.set_yticks(range(n), data.index, fontsize=8.5, color=TEXT)
    # Only the overall column carries numbers.
    col = data.shape[1] - 1
    for i, v in enumerate(data.iloc[:, col]):
        ax.text(col, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8.5,
                color="white" if v > 0.55 else TEXT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    cbar.set_label("Risk intensity (0 = none, 1 = maximum)", color=TEXT_2, fontsize=8.5)
    ax.set_title("Risk heatmap by jurisdiction", loc="left", color=TEXT, fontsize=12, pad=46, fontweight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


def plot_scores(res: AnalysisResult, path: str | Path | None = None):
    r = res.results.sort_values("risk_score")
    labels = [_label(row) for _, row in r.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.34 * len(r)), facecolor=SURFACE)
    _style(ax)
    ax.barh(labels, r["risk_score"], color=[LEVEL_COLORS[lv] for lv in r["risk_level"]], height=0.62)
    for y, (score, level) in enumerate(zip(r["risk_score"], r["risk_level"])):
        ax.text(score + 1, y, f"{score:.0f}  {LEVEL_ICONS[level]} {level}", va="center", fontsize=8.5, color=TEXT_2)
    for x in (res.config.levels.medium, res.config.levels.high):
        ax.axvline(x, color=GRID, linewidth=1, zorder=0)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=8.5, labelcolor=TEXT)
    ax.tick_params(axis="x", labelsize=8, labelcolor=TEXT_2)
    ax.set_xlabel("Risk score (0-100)", color=TEXT_2, fontsize=8.5)
    ax.set_title("Overall risk score", loc="left", color=TEXT, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160, facecolor=SURFACE)
    return fig


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _pct(x) -> str:
    return "–" if pd.isna(x) else f"{x:.1%}"


def _money(x) -> str:
    return "–" if pd.isna(x) or np.isinf(x) else f"{x:,.0f}"


def html_report(res: AnalysisResult, title: str = "TaxRiskScan report") -> str:
    r = res.results.sort_values("risk_score", ascending=False)
    counts = res.level_counts()
    heat = _png_b64(plot_heatmap(res))
    bars = _png_b64(plot_scores(res))
    e = html.escape

    rows = []
    for _, x in r.iterrows():
        lvl = x["risk_level"]
        rows.append(
            f"<tr><td>{e(x['group'])}</td><td>{e(x['jurisdiction'])}</td>"
            f"<td class=num>{x['risk_score']:.0f}</td>"
            f"<td><span class='lvl {lvl.lower()}'>{LEVEL_ICONS[lvl]} {lvl}</span></td>"
            f"<td class=num>{_pct(x['etr'])}</td><td class=num>{_pct(x['oecd_etr_2016'])}</td>"
            f"<td class=num>{_money(x['profit_per_employee'])}</td>"
            f"<td class=num>{_pct(x['profit_share'])}</td><td class=num>{_pct(x['activity_share'])}</td>"
            f"<td class=num>{x['ml_probability']:.2f}</td><td class=num>{x['anomaly_score']:.2f}</td></tr>"
        )

    findings = []
    for _, x in r[r["risk_level"] != "Low"].iterrows():
        items = "".join(
            f"<li><b>{e(reason)}</b><br><span class=rec>→ {e(rec)}</span>{_legal_html(code)}</li>"
            for code, reason, rec in zip(x["flags"], x["reasons"], x["recommendations"])
        ) or "<li>No individual rule fired; the score comes from the ML signals. Review manually.</li>"
        findings.append(
            f"<section class=finding><h3>{e(x['jurisdiction'])} <small>{e(x['group'])} · score "
            f"{x['risk_score']:.0f} · {x['risk_level']}</small></h3>"
            f"<p class=meta>ML probability {x['ml_probability']:.2f} · anomaly {x['anomaly_score']:.2f}"
            + (f" · model drivers: {e(x['ml_drivers'])}" if x['ml_drivers'] else "") + "</p>"
            f"<ul>{items}</ul></section>"
        )
    findings_html = "".join(findings) or "<p>No medium or high-risk jurisdictions found.</p>"
    cv = f"{res.models.cv_auc:.2f}" if res.models.cv_auc is not None else "n/a"
    p2 = _pillar2_html(res)
    m = res.models

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{e(title)}</title>
<style>
:root{{--s:{SURFACE};--t:{TEXT};--t2:{TEXT_2};--g:{GRID}}}
body{{font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;color:var(--t);background:var(--s);margin:0}}
main{{max-width:1080px;margin:0 auto;padding:32px 20px 60px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 10px}} h3{{font-size:16px;margin:0 0 4px}}
small,.meta,.sub{{color:var(--t2);font-weight:400}}
.tiles{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}
.tile{{border:1px solid var(--g);border-radius:10px;padding:12px 16px;min-width:120px}}
.tile b{{display:block;font-size:26px}}
.disc{{border-left:3px solid var(--g);padding:8px 14px;color:var(--t2);font-size:13.5px}}
img{{max-width:100%;height:auto}}
.wrap{{overflow-x:auto;margin-bottom:18px}} table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--g);text-align:left;white-space:nowrap}}
th{{color:var(--t2);font-weight:600}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.lvl{{font-weight:600}} .lvl.high{{color:#b42f2f}} .lvl.medium{{color:#8a5d00}} .lvl.low{{color:#0a7d0a}}
.finding{{border:1px solid var(--g);border-radius:10px;padding:14px 18px;margin:12px 0}}
.finding ul{{margin:8px 0 0;padding-left:20px}} .law{{font-size:12.5px;color:var(--t2)}} .law a{{color:inherit}}
.note{{font-size:12.5px;color:var(--t2)}} .pass{{color:#0a7d0a;font-weight:600}} .fail{{color:#b42f2f;font-weight:600}} .und{{color:#8a5d00;font-weight:600}}
@media print{{.wrap{{overflow:visible}} th,td{{white-space:normal}} .finding{{break-inside:avoid}}}} .finding li{{margin:6px 0}} .rec{{color:var(--t2)}}
</style></head><body><main>
<h1>{e(title)}</h1>
<p class=sub>Generated {date.today().isoformat()} · {len(r)} entities in {r['group'].nunique()} group(s) ·
EU Annex I list as of {jurisdictions.EU_ANNEX_I_AS_OF}</p>
<div class=tiles>
<div class=tile><b>{counts['High']}</b>{LEVEL_ICONS['High']} High risk</div>
<div class=tile><b>{counts['Medium']}</b>{LEVEL_ICONS['Medium']} Medium risk</div>
<div class=tile><b>{counts['Low']}</b>{LEVEL_ICONS['Low']} Low risk</div>
</div>
<p class=disc>{e(DISCLAIMER)}</p>
<img alt="Heatmap of risk indicators per jurisdiction" src="data:image/png;base64,{heat}">
<img alt="Bar chart of risk score per jurisdiction" src="data:image/png;base64,{bars}">
<h2>Summary table</h2>
<div class=wrap><table><thead><tr><th>Group</th><th>Jurisdiction</th><th class=num>Score</th><th>Level</th>
<th class=num>ETR</th><th class=num>OECD ETR 2016*</th><th class=num>Profit / employee</th><th class=num>Profit share</th>
<th class=num>Activity share</th><th class=num>ML prob.</th><th class=num>Anomaly</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class=note>* Aggregate ETR of profitable multinationals in that jurisdiction, OECD CbCR statistics 2016 - context, not a test.</p>
<h2>Findings, review steps and legal basis</h2>{findings_html}
<h2>Pillar Two screening (fiscal year {res.fiscal_year})</h2>{p2}
<h2>Method</h2>
<p>Score = {res.config.weights.rules:.0%} rule-based indicators + {res.config.weights.classifier:.0%}
logistic-regression probability + {res.config.weights.clustering:.0%} k-means anomaly score.
Levels: High ≥ {res.config.levels.high:.0f}, Medium ≥ {res.config.levels.medium:.0f}.
The classifier was trained on <code>{e(m.training_source)}</code> ({m.n_training:,} jurisdiction rows,
{m.n_positive:,} positive). With the default real data (EU Tax Observatory Public CbCR database) it estimates how
closely an entity's financial profile resembles multinationals' tax-haven affiliates; jurisdiction names and lists
are not model inputs. Group-wise cross-validated ROC AUC: {cv}.</p>
</main></body></html>"""


def _legal_html(code: str) -> str:
    srcs = legal.basis_for(code)
    if not srcs:
        return ""
    parts = []
    for src in srcs:
        label = html.escape(src.label())
        parts.append(f'<a href="{html.escape(src.url)}">{label}</a>' if src.url else label)
    return "<br><span class=law>§ " + " · ".join(parts) + "</span>"


def _test(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "<span class=und>?</span>"
    return "<span class=pass>✓</span>" if v else "<span class=fail>✗</span>"


def _pillar2_html(res: AnalysisResult) -> str:
    p = res.pillar2
    e = html.escape
    totals = group_totals(p)
    t_rows = "".join(
        f"<tr><td>{e(r['group'])}</td><td class=num>{r['year']}</td><td class=num>{r['jurisdictions']}</td>"
        f"<td class=num>{r['safe_harbour_applies']}</td><td class=num>{_money(r['estimated_topup_tax'])}</td></tr>"
        for _, r in totals.iterrows()
    )
    rows = "".join(
        f"<tr><td>{e(r['group'])}</td><td>{e(r['jurisdiction'])}</td><td class=num>{_pct(r['simplified_etr'])}</td>"
        f"<td class=num>{_money(r['sbie'])}</td><td>{_test(r['test_de_minimis'])}</td><td>{_test(r['test_simplified_etr'])}</td>"
        f"<td>{_test(r['test_routine_profits'])}</td><td>{e(r['safe_harbour'])}</td>"
        f"<td class=num>{_money(r['estimated_topup_tax'])}{' ≤' if r['estimate_is_upper_bound'] else ''}</td></tr>"
        for _, r in p.sort_values("estimated_topup_tax", ascending=False).iterrows()
    )
    return f"""<div class=wrap><table><thead><tr><th>Group</th><th class=num>Year</th><th class=num>Jurisdictions</th>
<th class=num>Safe harbour applies</th><th class=num>Estimated top-up tax</th></tr></thead><tbody>{t_rows}</tbody></table></div>
<div class=wrap><table><thead><tr><th>Group</th><th>Jurisdiction</th><th class=num>Simplified ETR</th><th class=num>SBIE</th>
<th>De minimis</th><th>Simplified ETR</th><th>Routine profits</th><th>Transitional CbCR safe harbour</th>
<th class=num>Est. top-up tax</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class=note>Tests of the Transitional CbCR Safe Harbour (OECD, Dec. 2022, extended in Jan. 2026). Top-up = (15% − ETR) ×
(profit − SBIE), jurisdictionally blended. ? = needs payroll costs / tangible assets; ≤ = upper bound because payroll is
missing. Screening estimate from CbCR figures, not a GloBE computation; amounts assumed in EUR.</p>"""


def write_outputs(res: AnalysisResult, out_dir: str | Path, title: str = "TaxRiskScan report") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": out / "risk_summary.csv",
        "heatmap": out / "risk_heatmap.png",
        "scores": out / "risk_scores.png",
        "pillar2": out / "pillar2_summary.csv",
        "report": out / "report.html",
    }
    res.summary().to_csv(paths["summary"], index=False)
    res.pillar2.to_csv(paths["pillar2"], index=False)
    plt.close(plot_heatmap(res, paths["heatmap"]))
    plt.close(plot_scores(res, paths["scores"]))
    paths["report"].write_text(html_report(res, title), encoding="utf-8")
    return paths
