"""TaxRiskScan dashboard.  Run with:  streamlit run app.py"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from taxriskscan import DISCLAIMER, DataError, analyze, model
from taxriskscan.cli import TEMPLATE
from taxriskscan.jurisdictions import EU_ANNEX_I_AS_OF
from taxriskscan.report import LEVEL_ICONS, html_report, plot_heatmap, plot_scores

SAMPLE = Path(__file__).parent / "data" / "sample_cbcr.csv"

st.set_page_config(page_title="TaxRiskScan", page_icon="⚖️", layout="wide")


@st.cache_resource
def get_models():
    return model.train()


st.title("TaxRiskScan")
st.caption("Flag jurisdictions with low-tax, profit-shifting and substance risks in Country-by-Country "
           f"Reporting data · EU Annex I list as of {EU_ANNEX_I_AS_OF}")

with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload a CSV", type="csv")
    st.download_button("Download CSV template", TEMPLATE, "taxriskscan_template.csv", "text/csv")
    st.markdown(
        "**Required columns:** `jurisdiction`, `revenue`, `profit_before_tax`, `tax_paid`, `employees`  \n"
        "**Optional:** `group`, `tangible_assets`"
    )
    st.info("No file? The bundled sample (three fictional groups) is analysed.")

source = upload if upload is not None else SAMPLE
try:
    res = analyze(source, models=get_models())
except DataError as exc:
    st.error(str(exc))
    st.stop()

counts = res.level_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{LEVEL_ICONS['High']} High risk", counts["High"])
c2.metric(f"{LEVEL_ICONS['Medium']} Medium risk", counts["Medium"])
c3.metric(f"{LEVEL_ICONS['Low']} Low risk", counts["Low"])
c4.metric("Entities analysed", len(res.results))
st.caption(DISCLAIMER)

tab_heat, tab_table, tab_findings, tab_method = st.tabs(["Heatmap", "Summary table", "Findings", "Method"])

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
            "profit_share": st.column_config.NumberColumn("Profit share", format="percent"),
            "activity_share": st.column_config.NumberColumn("Activity share", format="percent"),
            "profit_per_employee": st.column_config.NumberColumn("Profit / employee", format="localized"),
            "ml_probability": st.column_config.NumberColumn("ML probability", format="%.2f"),
            "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.2f"),
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
            for reason, rec in zip(x["reasons"], x["recommendations"]):
                st.markdown(f"- **{reason}**  \n  → {rec}")
            if not x["reasons"]:
                st.markdown("- No individual rule fired; the score comes from the ML signals.")
            st.caption(f"ML probability {x['ml_probability']:.2f} · anomaly score {x['anomaly_score']:.2f}")

with tab_method:
    w = res.config.weights
    st.markdown(
        f"""
**Risk score (0–100)** = {w.rules:.0%} rule-based indicators + {w.classifier:.0%} logistic-regression
probability + {w.clustering:.0%} k-means anomaly score. High ≥ {res.config.levels.high:.0f},
Medium ≥ {res.config.levels.medium:.0f}.

- **Rules** – ETR below 10% / below the 15% Pillar Two minimum, profit per employee, profit share vs.
  share of staff and tangible assets inside the group, very high margins, EU Annex I listing and
  zero-tax jurisdictions. Loss-making entities are never penalised for paying no tax.
- **Logistic regression** – trained on a synthetic, labelled reference set
  (`scripts/generate_synthetic_data.py`). Group-wise cross-validated ROC AUC:
  {res.models.cv_auc:.2f}. Use `taxriskscan analyze --train your_labels.csv` to train on your own cases.
- **k-means** – clusters ordinary entities; the anomaly score is how far an entity sits from the
  nearest cluster, in the risky direction only.
"""
    )
    st.dataframe(res.models.coefficients().rename("coefficient").to_frame(), width="stretch")
