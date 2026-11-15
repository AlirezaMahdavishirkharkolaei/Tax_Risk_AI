"""TaxRiskScan dashboard.  Run with:  streamlit run app.py"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from taxriskscan import DISCLAIMER, DataError, analyze, metrics, model
from taxriskscan.cli import TEMPLATE
from taxriskscan.jurisdictions import EU_ANNEX_I_AS_OF
from taxriskscan.legal import basis_for
from taxriskscan.pillar2 import group_totals
from taxriskscan.report import LEVEL_ICONS, html_report, plot_heatmap, plot_scores

ROOT = Path(__file__).parent
SAMPLE = ROOT / "data" / "sample_cbcr.csv"

st.set_page_config(page_title="TaxRiskScan", page_icon="⚖️", layout="wide")


@st.cache_resource
def get_models(source: str):
    return model.train(source)


st.title("TaxRiskScan")
st.caption("Tax-avoidance, profit-shifting and Pillar Two screening for Country-by-Country Reporting data · "
           f"EU Annex I list as of {EU_ANNEX_I_AS_OF}")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload CbCR data", type=["csv", "xml"],
                              help="CSV in the template layout, or an OECD CbC XML file (schema v2.0).")
    st.download_button("Download CSV template", TEMPLATE, "taxriskscan_template.csv", "text/csv")
    st.markdown(
        "**Required:** `jurisdiction`, `revenue`, `profit_before_tax`, `tax_paid` or `tax_accrued`, `employees`  \n"
        "**Optional:** `group`, `year`, `tangible_assets`, `payroll_costs`"
    )
    st.info("No file? The bundled sample (three fictional groups) is analysed.")
    st.header("Settings")
    training = st.radio(
        "Model trained on",
        ["real", "synthetic"],
        format_func=lambda s: {"real": "Real public CbCRs (EU Tax Observatory)", "synthetic": "Synthetic demo data"}[s],
    )
    fiscal_year = st.number_input("Fiscal year for Pillar Two (if no 'year' column)", 2023, 2035, 2026)

source = upload if upload is not None else SAMPLE
try:
    res = analyze(source, models=get_models(training), fiscal_year=int(fiscal_year))
except DataError as exc:
    st.error(str(exc))
    st.stop()

counts = res.level_counts()
p2 = res.pillar2
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{LEVEL_ICONS['High']} High risk", counts["High"])
c2.metric(f"{LEVEL_ICONS['Medium']} Medium risk", counts["Medium"])
c3.metric(f"{LEVEL_ICONS['Low']} Low risk", counts["Low"])
c4.metric("Safe harbour applies", f"{int((p2['safe_harbour'] == 'Applies').sum())} / {len(p2)}")
c5.metric("Est. top-up tax", f"{p2['estimated_topup_tax'].sum():,.0f}")
st.caption(DISCLAIMER)
for issue in res.issues:
    rows = f" (CSV rows {', '.join(map(str, issue.rows))})" if issue.rows else ""
    (st.warning if issue.level == "warning" else st.info)(issue.message + rows)

tab_heat, tab_table, tab_findings, tab_p2, tab_method = st.tabs(
    ["Heatmap", "Summary table", "Findings & legal basis", "Pillar Two", "Method & data"]
)

with tab_heat:
    st.pyplot(plot_heatmap(res), width="stretch")
    st.pyplot(plot_scores(res), width="stretch")

with tab_table:
    s = res.summary()
    st.dataframe(
        s, width="stretch", hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f"),
            "etr": st.column_config.NumberColumn("ETR", format="percent"),
            "oecd_etr_2016": st.column_config.NumberColumn(
                "OECD ETR 2016", format="percent",
                help="Aggregate ETR of profitable MNEs in the jurisdiction (OECD CbCR statistics)"),
            "profit_share": st.column_config.NumberColumn("Profit share", format="percent"),
            "activity_share": st.column_config.NumberColumn("Activity share", format="percent"),
            "profit_per_employee": st.column_config.NumberColumn("Profit / employee", format="localized"),
            "ml_probability": st.column_config.NumberColumn("ML probability", format="%.2f"),
            "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.2f"),
            "ml_drivers": st.column_config.TextColumn("Model drivers"),
        },
    )
    st.download_button("Download summary (CSV)", s.to_csv(index=False), "risk_summary.csv", "text/csv")

with tab_findings:
    st.download_button("Download full audit report (HTML)", html_report(res), "taxriskscan_report.html", "text/html")
    flagged = res.results[res.results["risk_level"] != "Low"].sort_values("risk_score", ascending=False)
    if flagged.empty:
        st.success("No medium or high-risk jurisdictions found.")
    for _, x in flagged.iterrows():
        with st.expander(f"{LEVEL_ICONS[x['risk_level']]} {x['jurisdiction']} · {x['group']} — "
                         f"score {x['risk_score']:.0f} ({x['risk_level']})", expanded=x["risk_level"] == "High"):
            for code, reason, rec in zip(x["flags"], x["reasons"], x["recommendations"]):
                law = " · ".join(f"[{src.label()}]({src.url})" if src.url else src.label() for src in basis_for(code))
                st.markdown(f"- **{reason}**  \n  → {rec}" + (f"  \n  § {law}" if law else ""))
            if not x["reasons"]:
                st.markdown("- No individual rule fired; the score comes from the ML signals.")
            drivers = f" · model drivers: {x['ml_drivers']}" if x["ml_drivers"] else ""
            st.caption(f"ML probability {x['ml_probability']:.2f} · anomaly score {x['anomaly_score']:.2f}{drivers}")

with tab_p2:
    st.markdown("Transitional CbCR Safe Harbour tests and an estimate of the top-up tax "
                "(15% − ETR) × (profit − substance-based income exclusion), per jurisdiction.")
    st.dataframe(group_totals(p2), width="stretch", hide_index=True)
    st.dataframe(
        p2, width="stretch", hide_index=True,
        column_config={
            "simplified_etr": st.column_config.NumberColumn("Simplified ETR", format="percent"),
            "transition_rate": st.column_config.NumberColumn("Transition rate", format="percent"),
            "topup_percentage": st.column_config.NumberColumn("Top-up %", format="percent"),
            "estimated_topup_tax": st.column_config.NumberColumn("Est. top-up tax", format="localized"),
            "sbie": st.column_config.NumberColumn("SBIE", format="localized"),
        },
    )
    st.caption("Screening estimate from CbCR figures, not a GloBE computation. Amounts assumed in EUR. "
               "Routine-profits test needs payroll costs and tangible assets; without payroll the estimate is an upper bound.")
    st.download_button("Download Pillar Two table (CSV)", p2.to_csv(index=False), "pillar2_summary.csv", "text/csv")

with tab_method:
    m = res.models
    w = res.config.weights
    cv = f"{m.cv_auc:.2f}" if m.cv_auc is not None else "n/a"
    st.markdown(
        f"""
**Risk score (0–100)** = {w.rules:.0%} rule-based indicators + {w.classifier:.0%} logistic-regression
probability + {w.clustering:.0%} k-means anomaly score. High ≥ {res.config.levels.high:.0f},
Medium ≥ {res.config.levels.medium:.0f}.

**Training data:** `{m.training_source}` — {m.n_training:,} jurisdiction rows, {m.n_positive:,} positive.
The default is the EU Tax Observatory's Public Country-by-Country Reports database (real CbCRs published by
~100 multinationals, 2015–2023); the label is whether the jurisdiction is a tax haven in that database.
Jurisdiction names and lists are **not** model inputs, so the model learns a *haven-like financial profile*.
Group-wise cross-validated ROC AUC: **{cv}**.

**Benchmarks:** OECD aggregated CbCR statistics, fiscal year 2016. See `DATA_SOURCES.md` for citations and terms.
"""
    )
    df_train = metrics.load_dataset(model.resolve_training(training))
    X_train = metrics.feature_matrix(metrics.compute_indicators(df_train))
    st.markdown("**Feature importance** (mean absolute effect on the probability, training data)")
    st.bar_chart(m.importance(X_train).rename(index=model.FEATURE_LABELS), horizontal=True)
