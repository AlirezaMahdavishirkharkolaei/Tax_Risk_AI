"""Research mode: dataset-level analysis of (public) country-by-country reports.

Runs on any dataset with a ``label`` column marking tax-haven jurisdictions -
by default the bundled EU Tax Observatory Public CbCR data. It answers the
questions a researcher, a university course or a policy unit typically asks:

1. What share of multinationals' profit is booked in tax havens, compared with
   their share of employees, revenue and tax - and how has it changed by year?
2. How do effective tax rates differ between havens and other jurisdictions?
3. Which havens attract the most profit, and the most profit per employee?
4. How exposed are these groups to the Pillar Two minimum tax (screening)?
5. How well does the TaxRiskScan score separate haven from non-haven rows
   when the classifier is evaluated out-of-sample (group-wise cross-validation)?

Public CbCRs are published voluntarily by a changing set of companies, so
year-to-year movements partly reflect who publishes. Every table carries the
number of reports behind it.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402

from . import metrics, model, pillar2, rules  # noqa: E402
from .config import DEFAULT_CONFIG, Config  # noqa: E402

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order

CITATION = ("Aliprandi, G., Borders, K., Gabriel, F., von Zedlitz, G. (2022): \"Public Country-by-Country "
            "Reports: a new database\", EU Tax Observatory.")


@dataclass
class ResearchResult:
    source: str
    overview: dict
    by_year: pd.DataFrame
    etr_by_year: pd.DataFrame
    top_havens: pd.DataFrame
    by_sector: pd.DataFrame
    pillar2_by_year: pd.DataFrame
    validation: dict
    figures: dict = field(default_factory=dict)

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "haven_shares_by_year": self.by_year,
            "etr_by_year": self.etr_by_year,
            "top_havens": self.top_havens,
            "haven_shares_by_sector": self.by_sector,
            "pillar2_by_year": self.pillar2_by_year,
        }


def _shares(d: pd.DataFrame) -> pd.Series:
    h = d["label"] == 1
    pos = d["profit_before_tax"].clip(lower=0)
    tax = d["_tax"].clip(lower=0)
    out = {
        "reports": d.groupby(["group", "year"]).ngroups,
        "groups": d["group"].nunique(),
        "haven_share_profit": pos[h].sum() / pos.sum() if pos.sum() else np.nan,
        "haven_share_employees": d.loc[h, "employees"].sum() / d["employees"].sum() if d["employees"].sum() else np.nan,
        "haven_share_revenue": d.loc[h, "revenue"].sum() / d["revenue"].sum() if d["revenue"].sum() else np.nan,
        "haven_share_tax": tax[h].sum() / tax.sum() if tax.sum() else np.nan,
    }
    out["profit_to_employee_ratio"] = (out["haven_share_profit"] / out["haven_share_employees"]
                                       if out["haven_share_employees"] else np.nan)
    return pd.Series(out)


def _etr(d: pd.DataFrame) -> pd.Series:
    p = d[d["profit_before_tax"] > 0]
    out = {}
    for name, part in (("haven", p[p["label"] == 1]), ("other", p[p["label"] == 0])):
        prof = part["profit_before_tax"].sum()
        out[f"{name}_weighted_etr"] = part["_tax"].sum() / prof if prof else np.nan
        out[f"{name}_median_etr"] = (part["_tax"] / part["profit_before_tax"]).median() if len(part) else np.nan
        out[f"{name}_rows"] = len(part)
    return pd.Series(out)


def _per(df: pd.DataFrame, key: str, fn) -> pd.DataFrame:
    """Apply ``fn`` to each group (keeping all columns) and stack the results."""
    rows = {k: fn(g) for k, g in df.groupby(key)}
    out = pd.DataFrame(rows).T
    out.index.name = key
    return out.reset_index()


def _out_of_sample_scores(df: pd.DataFrame, config: Config) -> dict:
    """Group-wise cross-validated ML probability, blended with the rules as in production."""
    ind = metrics.compute_indicators(df)
    X = metrics.feature_matrix(ind, config.thresholds.globe_minimum_rate)[metrics.FEATURES]
    y = df["label"].astype(int)
    prob = cross_val_predict(model._classifier(), X, y, groups=df["group"], cv=GroupKFold(5),
                             method="predict_proba")[:, 1]
    r = rules.evaluate(ind.assign(eu_non_cooperative=False, zero_cit_jurisdiction=False), config.thresholds)
    # Jurisdiction-list rules are switched off here: they would use the label's own information.
    w = config.weights
    score = 100 * (w.rules * r["rule_score"] + w.classifier * prob) / (w.rules + w.classifier)
    high = score >= config.levels.high
    flagged = score >= config.levels.medium
    return {
        "rows": int(len(y)),
        "ml_auc_out_of_sample": float(roc_auc_score(y, prob)),
        "score_auc_out_of_sample": float(roc_auc_score(y, score)),
        "high_share_havens": float(high[y == 1].mean()),
        "high_share_other": float(high[y == 0].mean()),
        "flagged_share_havens": float(flagged[y == 1].mean()),
        "flagged_share_other": float(flagged[y == 0].mean()),
        "precision_high": float(y[high].mean()) if high.any() else float("nan"),
        "base_rate": float(y.mean()),
        "note": "Scores combine out-of-fold ML probabilities with rule points; the anomaly term and the "
                "jurisdiction-list rules are excluded so the label is not used to predict itself.",
    }


def run(data=None, config: Config = DEFAULT_CONFIG) -> ResearchResult:
    src = model.resolve_training(data)
    df = metrics.load_dataset(src)
    if "label" not in df.columns:
        raise metrics.DataError("Research mode needs a 'label' column (1 = tax-haven jurisdiction).")
    if "year" not in df.columns:
        df["year"] = 0
    df["label"] = df["label"].astype(int)
    df["_tax"] = metrics.tax_for_etr(df)
    source = "DataFrame" if isinstance(src, pd.DataFrame) else Path(str(src)).name

    by_year = _per(df, "year", _shares)
    by_year = by_year[by_year["reports"] >= 5].reset_index(drop=True)
    etr_by_year = _per(df, "year", _etr)
    etr_by_year = etr_by_year[etr_by_year["year"].isin(by_year["year"])].reset_index(drop=True)

    h = df[df["label"] == 1]
    top = h.groupby("jurisdiction").agg(
        groups=("group", "nunique"),
        profit_booked=("profit_before_tax", lambda s: s.clip(lower=0).sum()),
        employees=("employees", "sum"),
        tax=("_tax", "sum"),
    )
    top["profit_per_employee"] = top["profit_booked"] / top["employees"].replace(0, np.nan)
    top["weighted_etr"] = top["tax"] / top["profit_booked"].replace(0, np.nan)
    top = top[top["groups"] >= 3].sort_values("profit_booked", ascending=False).head(15).reset_index()

    by_sector = pd.DataFrame()
    if "sector" in df.columns:
        by_sector = _per(df, "sector", _shares)
        by_sector = by_sector[by_sector["groups"] >= 3].sort_values("haven_share_profit", ascending=False)
        by_sector = by_sector.reset_index(drop=True)

    p2 = pillar2.assess(df.drop(columns=["label", "_tax"]))
    p2_year = p2.groupby("year").agg(
        jurisdictions=("jurisdiction", "size"),
        safe_harbour_applies=("safe_harbour", lambda s: (s == "Applies").mean()),
        safe_harbour_fails=("safe_harbour", lambda s: (s == "Fails").mean()),
        undetermined=("safe_harbour", lambda s: (s == "Undetermined").mean()),
        topup_upper_bound_eur=("estimated_topup_tax", "sum"),
    ).reset_index()
    p2_year = p2_year[p2_year["year"].isin(by_year["year"])].reset_index(drop=True)

    validation = _out_of_sample_scores(df.drop(columns=["_tax"]), config)

    pos = df["profit_before_tax"].clip(lower=0)
    overview = {
        "rows": len(df), "groups": int(df["group"].nunique()),
        "reports": int(df.groupby(["group", "year"]).ngroups),
        "years": f"{int(by_year['year'].min())}–{int(by_year['year'].max())}" if len(by_year) else "",
        "haven_rows": int(df["label"].sum()),
        "haven_share_profit": float(pos[df["label"] == 1].sum() / pos.sum()),
        "haven_share_employees": float(df.loc[df["label"] == 1, "employees"].sum() / df["employees"].sum()),
    }
    res = ResearchResult(source, overview, by_year, etr_by_year, top, by_sector, p2_year, validation)
    res.figures = _figures(res)
    return res


# ---- figures ---------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0, labelsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _figures(res: ResearchResult) -> dict[str, str]:
    figs = {}
    y = res.by_year
    if len(y):
        fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
        _style(ax)
        years = y["year"].astype(int)
        for i, (col, label) in enumerate([("haven_share_profit", "Profit"), ("haven_share_employees", "Employees"),
                                          ("haven_share_tax", "Tax")]):
            ax.plot(years, y[col] * 100, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
            ax.annotate(label, (years.iloc[-1], y[col].iloc[-1] * 100), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=TEXT_2)
        ax.set_ylim(0, max(5, float(y[["haven_share_profit", "haven_share_tax"]].max().max()) * 100 * 1.15))
        ax.set_xticks(years)
        ax.set_ylabel("Share booked in tax havens (%)", color=TEXT_2, fontsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=TEXT_2, ncols=3)
        ax.set_title("Share of profit, employees and tax in tax havens", loc="left", color=TEXT,
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        figs["shares"] = _fig_png(fig)

    e = res.etr_by_year
    if len(e):
        fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor=SURFACE)
        _style(ax)
        years = e["year"].astype(int)
        for i, (col, label) in enumerate([("other_weighted_etr", "Other jurisdictions"),
                                          ("haven_weighted_etr", "Tax havens")]):
            ax.plot(years, e[col] * 100, color=SERIES[i], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.axhline(15, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate("15% global minimum", (years.iloc[0], 15), xytext=(0, 5), textcoords="offset points",
                    fontsize=8, color=TEXT_2)
        ax.set_ylim(0, max(30, float(e[["other_weighted_etr", "haven_weighted_etr"]].max().max()) * 100 * 1.15))
        ax.set_xticks(years)
        ax.set_ylabel("Profit-weighted ETR (%)", color=TEXT_2, fontsize=8.5)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=TEXT_2, ncols=2)
        ax.set_title("Effective tax rate of profitable entities", loc="left", color=TEXT, fontsize=11,
                     fontweight="bold")
        fig.tight_layout()
        figs["etr"] = _fig_png(fig)

    t = res.top_havens.head(10).iloc[::-1]
    if len(t):
        fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.34 * len(t)), facecolor=SURFACE)
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        vals = t["profit_booked"] / 1e9
        ax.barh(t["jurisdiction"], vals, color=SERIES[0], height=0.62)
        for yy, (v, etr) in enumerate(zip(vals, t["weighted_etr"])):
            ax.text(v, yy, f"  €{v:,.0f}bn · ETR {etr:.0%}" if not pd.isna(etr) else f"  €{v:,.0f}bn",
                    va="center", fontsize=8, color=TEXT_2)
        ax.set_xlim(0, float(vals.max()) * 1.35)
        ax.tick_params(axis="y", labelcolor=TEXT)
        ax.set_xlabel("Profit booked, all years (EUR bn)", color=TEXT_2, fontsize=8.5)
        ax.set_title("Tax havens attracting the most profit", loc="left", color=TEXT, fontsize=11, fontweight="bold")
        fig.tight_layout()
        figs["top_havens"] = _fig_png(fig)
    return figs


# ---- outputs ------------------------------------------------------------------------

def _fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "x":
        return f"{v:.1f}×"
    if kind == "eur":
        return f"€{v / 1e9:,.1f}bn" if abs(v) >= 1e9 else f"€{v / 1e6:,.0f}m"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "year":
        return str(int(v))
    return html.escape(str(v))


def _table(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> str:
    head = "".join(f"<th class={'num' if k != 'text' else ''}>{html.escape(t)}</th>" for _, t, k in cols)
    body = "".join(
        "<tr>" + "".join(f"<td class={'num' if k != 'text' else ''}>{_fmt(row[c], k)}</td>" for c, _, k in cols) + "</tr>"
        for _, row in df.iterrows()
    )
    return f"<div class=wrap><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def brief_html(res: ResearchResult) -> str:
    o, v = res.overview, res.validation
    img = lambda k, alt: (f'<img alt="{alt}" src="data:image/png;base64,{res.figures[k]}">' if k in res.figures else "")  # noqa: E731
    sector = ""
    if len(res.by_sector):
        sector = "<h2>By sector</h2>" + _table(res.by_sector.head(12), [
            ("sector", "Sector", "text"), ("groups", "Groups", "int"), ("haven_share_profit", "Profit in havens", "pct"),
            ("haven_share_employees", "Employees in havens", "pct"), ("profit_to_employee_ratio", "Profit ÷ employee share", "x")])
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TaxRiskScan research brief</title>
<style>
body{{font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;color:{TEXT};background:{SURFACE};margin:0}}
main{{max-width:900px;margin:0 auto;padding:32px 20px 64px}} h1{{font-size:26px;margin:0 0 6px}}
h2{{font-size:19px;margin:34px 0 10px}} .sub,.note{{color:{TEXT_2}}} .note{{font-size:13px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}}
.kpi{{border:1px solid {GRID};border-radius:10px;padding:12px 14px}} .kpi b{{display:block;font-size:24px}}
img{{max-width:100%;height:auto;display:block;margin:10px 0}} .wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}} th,td{{padding:6px 8px;border-bottom:1px solid {GRID};text-align:left;white-space:nowrap}}
th{{color:{TEXT_2};font-weight:600}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
</style></head><body><main>
<h1>Profit shifting in public country-by-country reports</h1>
<p class=sub>TaxRiskScan research brief · {date.today().isoformat()} · data: <code>{html.escape(res.source)}</code></p>
<div class=kpis>
<div class=kpi><b>{o['groups']}</b>multinationals · {o['reports']} reports · {o['years']}</div>
<div class=kpi><b>{o['haven_share_profit']:.1%}</b>of profit booked in tax havens</div>
<div class=kpi><b>{o['haven_share_employees']:.1%}</b>of employees located in tax havens</div>
<div class=kpi><b>{v['score_auc_out_of_sample']:.2f}</b>out-of-sample AUC of the risk score</div>
</div>
<h2>1. Where profit is booked</h2>
{img('shares', 'Line chart of the share of profit, employees and tax booked in tax havens by year')}
{_table(res.by_year, [("year", "Year", "year"), ("reports", "Reports", "int"), ("haven_share_profit", "Profit", "pct"),
    ("haven_share_employees", "Employees", "pct"), ("haven_share_revenue", "Revenue", "pct"), ("haven_share_tax", "Tax", "pct"),
    ("profit_to_employee_ratio", "Profit ÷ employee share", "x")])}
<p class=note>Shares of the reporting groups' totals located in tax-haven jurisdictions. Profit uses positive profits only.</p>
<h2>2. Effective tax rates</h2>
{img('etr', 'Line chart of profit-weighted effective tax rates in tax havens and other jurisdictions by year')}
{_table(res.etr_by_year, [("year", "Year", "year"), ("haven_weighted_etr", "Havens, weighted", "pct"), ("haven_median_etr", "Havens, median", "pct"),
    ("other_weighted_etr", "Other, weighted", "pct"), ("other_median_etr", "Other, median", "pct")])}
<h2>3. Leading tax havens</h2>
{img('top_havens', 'Bar chart of profit booked in the leading tax havens')}
{_table(res.top_havens, [("jurisdiction", "Jurisdiction", "text"), ("groups", "Groups", "int"), ("profit_booked", "Profit booked", "eur"),
    ("employees", "Employees", "int"), ("profit_per_employee", "Profit / employee", "eur"), ("weighted_etr", "ETR", "pct")])}
<p class=note>Jurisdictions used by at least three groups, all years pooled.</p>
{sector}
<h2>4. Pillar Two exposure (screening)</h2>
{_table(res.pillar2_by_year, [("year", "Year", "year"), ("jurisdictions", "Jurisdictions", "int"),
    ("safe_harbour_applies", "Safe harbour applies", "pct"), ("safe_harbour_fails", "Fails", "pct"),
    ("undetermined", "Undetermined", "pct"), ("topup_upper_bound_eur", "Top-up (upper bound)", "eur")])}
<p class=note>Transitional CbCR Safe Harbour tests applied to each report as if the rules had applied that year. Public CbCRs
rarely disclose payroll, so the routine-profits test is often undetermined and the top-up figure is an upper bound.</p>
<h2>5. Does the risk score find havens it has not seen?</h2>
<p>With group-wise cross-validation (no multinational appears in both training and test folds), the classifier alone reaches an
ROC AUC of <b>{v['ml_auc_out_of_sample']:.2f}</b>; blended with the rule engine it reaches <b>{v['score_auc_out_of_sample']:.2f}</b>.
{v['high_share_havens']:.0%} of haven rows score <i>High</i>, against {v['high_share_other']:.0%} of other rows; among rows scored
<i>High</i>, {v['precision_high']:.0%} are havens (base rate {v['base_rate']:.0%}).</p>
<p class=note>{html.escape(v['note'])}</p>
<h2>Caveats</h2>
<p>Public CbCRs are published voluntarily, so the sample is not representative of all multinationals, and its composition changes
from year to year. "Tax haven" follows the classification in the source database. Shares are of the reporting groups' own totals.
These are descriptive statistics, not estimates of revenue loss.</p>
<h2>Data and citation</h2>
<p>{html.escape(CITATION)} Used under its terms (non-commercial use with citation). Analysis: TaxRiskScan
(https://github.com/AlirezaMahdavishirkharkolaei/Tax_Risk_AI).</p>
</main></body></html>"""


def write_outputs(res: ResearchResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in res.tables().items():
        if len(df):
            p = out / f"{name}.csv"
            df.to_csv(p, index=False)
            paths[name] = p
    for name, b64 in res.figures.items():
        p = out / f"{name}.png"
        p.write_bytes(base64.b64decode(b64))
        paths[f"fig_{name}"] = p
    p = out / "research_brief.html"
    p.write_text(brief_html(res), encoding="utf-8")
    paths["brief"] = p
    return paths
