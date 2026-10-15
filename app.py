"""Streamlit app: find costly SQL in Snowflake and Databricks and get fixes."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from query_optimizer.costs import (  # noqa: E402
    add_costs, add_fingerprints, known_sizes, load_history, set_warehouse_sizes, summarise_by_fingerprint,
    unpriced_sizes, warehouses_without_size,
)
from query_optimizer.llm import check_rewrite, suggest_rewrite  # noqa: E402
from query_optimizer.rules import ParseError, analyze, score  # noqa: E402

SAMPLE_HISTORY = Path(__file__).parent / "data" / "sample_snowflake_query_history.csv"
SEVERITY_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}

EXAMPLE_SQL = """SELECT b.booking_id, c.email, b.amount
FROM analytics.bookings b
JOIN analytics.customers c
  ON b.customer_id = c.customer_id OR b.email = c.email
WHERE TO_DATE(b.booking_ts) = '2026-09-15'
  AND c.email LIKE '%@gmail.com'"""

st.set_page_config(page_title="LLM Cost & Query Optimizer", layout="wide")


@st.cache_data(show_spinner="Reading and grouping the query history...")
def read_history(data: bytes) -> pd.DataFrame:
    """Load and fingerprint an export once. Parsing every query is the slow step, so later
    clicks and price changes reuse this result instead of parsing again."""
    return add_fingerprints(load_history(pd.read_csv(io.BytesIO(data))))


@st.cache_data(show_spinner="Ranking query shapes...")
def priced_summary(history: pd.DataFrame, usd_per_credit: float, usd_per_dbu: float,
                   sizes: tuple[tuple[str, str], ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    priced = add_costs(set_warehouse_sizes(history, dict(sizes)), usd_per_credit, usd_per_dbu)
    return priced, summarise_by_fingerprint(priced)


def groq_key() -> str | None:
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("GROQ_API_KEY")


def show_findings(findings) -> None:
    if not findings:
        st.success("No anti-patterns found.")
        return
    for f in findings:
        with st.expander(f"{SEVERITY_LABEL[f.severity]}  ·  {f.title}", expanded=f.severity == "high"):
            st.markdown(f"**Why it costs more:** {f.detail}")
            st.markdown(f"**Fix:** {f.suggestion}")


def show_rewrite_check(check) -> None:
    if not check.parses:
        st.error(f"The rewrite is not valid SQL, so do not use it. {check.error}")
        return
    if check.score_before is not None:
        st.metric("Health score after rewrite", f"{check.score_after} / 100",
                  delta=check.score_after - check.score_before)
    if check.fixed:
        st.markdown("**Issues fixed**\n" + "\n".join(f"- {t}" for t in check.fixed))
    if check.remaining:
        st.warning("Still found in the rewrite: " + "; ".join(check.remaining))
    if check.new:
        st.warning("New issues in the rewrite: " + "; ".join(check.new))
    if check.same_columns is False:
        st.warning("The rewrite returns different output columns than the original query.")


def show_ai_rewrite(sql: str, platform: str, findings, metrics=None, key_suffix: str = "") -> None:
    key = groq_key()
    if not key:
        st.info("Add a GROQ_API_KEY (environment variable or .streamlit/secrets.toml) to get AI rewrite suggestions.")
        return
    if st.button("Suggest a rewrite with AI", key=f"ai_{key_suffix}"):
        with st.spinner("Asking the model for a cheaper version of this query..."):
            try:
                s = suggest_rewrite(sql, platform, findings, metrics, api_key=key)
            except Exception as exc:
                st.error(f"AI suggestion failed: {exc}")
                return
        st.markdown("**Rewritten query**")
        st.code(s.rewritten_sql, language="sql")
        show_rewrite_check(check_rewrite(sql, s.rewritten_sql, platform))
        if s.changes:
            st.markdown("**What changed**\n" + "\n".join(f"- {c}" for c in s.changes))
        if s.expected_impact:
            st.markdown(f"**Expected impact:** {s.expected_impact}")
        if s.caveats:
            st.markdown("**Check before using**\n" + "\n".join(f"- {c}" for c in s.caveats))
        st.caption("AI suggestions can be wrong. Compare results on a sample before replacing the original query.")


st.title("LLM Cost & Query Optimizer")
st.caption("Find the SQL that costs the most on Snowflake and Databricks, see why, and get a fix.")

with st.sidebar:
    st.header("Pricing")
    usd_per_credit = st.number_input("Snowflake USD per credit", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
    usd_per_dbu = st.number_input("Databricks USD per DBU", min_value=0.1, max_value=5.0, value=0.70, step=0.05)
    st.caption("Costs are estimates: run time x warehouse rate. Use them to rank queries, not to predict the bill.")
    st.header("AI suggestions")
    st.write("Connected" if groq_key() else "Not configured (rule checks still work)")

tab_history, tab_single = st.tabs(["Query history", "Analyze one query"])

with tab_history:
    st.subheader("Where is the money going?")
    st.write(
        "Upload a CSV export of `SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY` or Databricks `system.query.history`. "
        "Queries that differ only in literal values are grouped together."
    )
    upload = st.file_uploader("Query history CSV", type=["csv"])
    use_sample = st.toggle("Use the sample file (synthetic Snowflake data)", value=upload is None)

    data = None
    if upload is not None and not use_sample:
        data = upload.getvalue()
    elif use_sample:
        data = SAMPLE_HISTORY.read_bytes()

    if data is not None:
        try:
            history = read_history(data)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        sizes = {}
        blank = warehouses_without_size(history)
        if blank:
            size_options = known_sizes(history["platform"].iloc[0])
            with st.sidebar:
                st.header("Warehouse sizes")
                st.caption("The export has no size for these warehouses. Pick each one's size so its queries "
                           "can be priced. Until then they count as $0.")
                for warehouse in blank:
                    picked = st.selectbox(warehouse, size_options, index=None, placeholder="Pick a size",
                                          key=f"size_{warehouse}")
                    if picked:
                        sizes[warehouse] = picked

        history, summary = priced_summary(history, usd_per_credit, usd_per_dbu, tuple(sorted(sizes.items())))

        unpriced = unpriced_sizes(history)
        if unpriced:
            total = sum(unpriced.values())
            detail = ", ".join(f"{size}: {runs:,}" for size, runs in unpriced.items())
            st.warning(
                f"{total:,} {'run has' if total == 1 else 'runs have'} no known rate for the warehouse size and "
                f"{'is' if total == 1 else 'are'} counted as $0 (size {detail}). "
                "Their query shapes may rank lower than they should."
            )

        flagged = summary[summary["issues"] > 0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Queries", f"{len(history):,}")
        c2.metric("Distinct query shapes", f"{len(summary):,}")
        c3.metric("Estimated compute cost", f"${summary['total_cost_usd'].sum():,.2f}")
        c4.metric("Cost in flagged queries", f"{flagged['total_cost_usd'].sum() / max(summary['total_cost_usd'].sum(), 1e-9):.0%}")

        st.markdown("#### Most expensive query shapes")
        columns = ["runs", "total_cost_usd", "cost_share", "avg_seconds", "total_gb_scanned",
                   "avg_pruning_ratio", "health_score", "top_issue", "fingerprint"]
        if unpriced:
            columns.insert(1, "unpriced_runs")
        view = summary[columns]
        st.dataframe(
            view,
            width="stretch",
            hide_index=True,
            column_config={
                "runs": st.column_config.NumberColumn("Runs"),
                "unpriced_runs": st.column_config.NumberColumn(
                    "Unpriced runs", help="Runs counted as $0 because their warehouse size has no known rate."),
                "total_cost_usd": st.column_config.NumberColumn("Est. cost", format="$%.2f"),
                "cost_share": st.column_config.ProgressColumn("Share", format="percent", min_value=0, max_value=1),
                "avg_seconds": st.column_config.NumberColumn("Avg seconds", format="%.1f"),
                "total_gb_scanned": st.column_config.NumberColumn("GB scanned", format="%.1f"),
                "avg_pruning_ratio": st.column_config.NumberColumn(
                    "Partitions read", format="percent", help="Share of partitions scanned. Lower is better."),
                "health_score": st.column_config.NumberColumn("Health", help="100 = no issues found"),
                "top_issue": "Top issue",
                "fingerprint": st.column_config.TextColumn("Query shape", width="large"),
            },
        )

        st.markdown("#### Look at one query shape")
        options = {f"#{i + 1}  ${r.total_cost_usd:,.2f}  ·  {r.fingerprint[:90]}": i for i, r in summary.iterrows()}
        choice = st.selectbox("Query shape", list(options))
        row = summary.iloc[options[choice]]
        st.code(row["example_query"], language="sql")
        try:
            findings = analyze(row["example_query"], row["platform"])
        except ParseError as exc:
            st.warning(str(exc))
            findings = []
        show_findings(findings)
        metrics = {
            "runs": int(row["runs"]),
            "avg_seconds": round(float(row["avg_seconds"]), 1),
            "total_gb_scanned": round(float(row["total_gb_scanned"]), 1),
            "share_of_partitions_read": None if pd.isna(row["avg_pruning_ratio"]) else round(float(row["avg_pruning_ratio"]), 3),
        }
        show_ai_rewrite(row["example_query"], row["platform"], findings, metrics, key_suffix="history")

with tab_single:
    st.subheader("Check a query before you run it")
    platform = st.radio("Platform", ["snowflake", "databricks"], horizontal=True,
                        format_func=str.capitalize, key="platform_single")
    sql = st.text_area("SQL", value=EXAMPLE_SQL, height=200, key="sql_single")
    if sql.strip():
        try:
            findings = analyze(sql, platform)
        except ParseError as exc:
            st.error(str(exc))
        else:
            st.metric("Health score", f"{score(findings)} / 100", help="100 means no anti-patterns were found.")
            show_findings(findings)
            show_ai_rewrite(sql, platform, findings, key_suffix="single")
