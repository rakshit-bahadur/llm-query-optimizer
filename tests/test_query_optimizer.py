import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from query_optimizer.costs import (
    add_costs, add_fingerprints, detect_platform, load_history, set_warehouse_sizes, summarise_by_fingerprint,
    unpriced_sizes, warehouses_without_size,
)
from query_optimizer.llm import build_messages, check_rewrite, parse_response, suggest_rewrite
from query_optimizer.rules import ParseError, analyze, fingerprint, score

ROOT = Path(__file__).resolve().parents[1]


def ids(sql, dialect="snowflake"):
    return [f.rule_id for f in analyze(sql, dialect)]


# Rules

@pytest.mark.parametrize(
    "sql, expected",
    [
        ("select * from sales", {"FULL_SCAN", "SELECT_STAR"}),
        ("select id from sales where to_date(created_at) = '2026-01-01'", {"FUNCTION_ON_FILTER"}),
        ("select id from sales where upper(status) in ('A', 'B')", {"FUNCTION_ON_FILTER"}),
        ("select a.id from a join b on a.id = b.id or a.k = b.k where a.d > '2026-01-01'", {"OR_IN_JOIN"}),
        ("select a.id from a, b where a.d > '2026-01-01'", {"CARTESIAN_JOIN"}),
        ("select id from a where name like '%kumar' and d > '2026-01-01'", {"LEADING_WILDCARD"}),
        ("select id from a where d > '2026-01-01' and id not in (select id from b)", {"NOT_IN_SUBQUERY"}),
        ("select id from a where d > '2026-01-01' union select id from b where d > '2026-01-01'", {"UNION_DISTINCT"}),
        ("select region, count(distinct uid) from e where d > '2026-01-01' group by region", {"EXACT_COUNT_DISTINCT"}),
        ("select id, (select max(x) from p where p.uid = u.id) from u limit 5", {"SCALAR_SUBQUERY"}),
        ("select id from a where d > '2026-01-01' order by id", {"ORDER_WITHOUT_LIMIT"}),
    ],
)
def test_rules_detect_expected_issue(sql, expected):
    assert expected <= set(ids(sql))


def test_clean_query_has_no_findings():
    sql = "select id, amount from payments where created_date between '2026-01-01' and '2026-01-31'"
    assert ids(sql) == []
    assert score(analyze(sql)) == 100


def test_subquery_without_filter_is_not_reported_as_full_scan():
    sql = "select id from a where d > '2026-01-01' and id in (select id from b)"
    assert "FULL_SCAN" not in ids(sql)


def test_databricks_dialect_is_supported():
    assert "FUNCTION_ON_FILTER" in ids("select id from t where year(event_date) = 2026", "databricks")


def test_findings_are_sorted_by_severity_and_score_drops():
    findings = analyze("select * from a join b on a.x = b.x or a.y = b.y where lower(a.n) = 'x'")
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, key={"high": 0, "medium": 1, "low": 2}.get)
    assert score(findings) < 50


def test_parse_errors_are_clear():
    with pytest.raises(ParseError, match="empty"):
        analyze("   ")
    with pytest.raises(ParseError, match="one statement"):
        analyze("select 1; select 2")
    with pytest.raises(ValueError):
        analyze("select 1", "oracle")


def test_fingerprint_groups_queries_that_differ_only_in_literals():
    a = fingerprint("SELECT a FROM t WHERE x = 5 AND d >= '2026-01-01'")
    b = fingerprint("select a from t where x = 9 and d >= '2026-02-01'")
    c = fingerprint("select a from t where y = 9")
    assert a == b
    assert a != c


# Costs

def snowflake_rows():
    return pd.DataFrame({
        "QUERY_ID": ["q1", "q2", "q3"],
        "QUERY_TEXT": ["select * from big where to_date(ts) = '2026-01-01'",
                       "select * from big where to_date(ts) = '2026-01-02'",
                       "select id from small where d = '2026-01-01'"],
        "WAREHOUSE_SIZE": ["Large", "LARGE", "X-Small"],
        "TOTAL_ELAPSED_TIME": [3_600_000, 1_800_000, 60_000],
        "BYTES_SCANNED": [1e9, 1e9, 1e6],
        "PARTITIONS_SCANNED": [100, 100, 1],
        "PARTITIONS_TOTAL": [100, 100, 50],
        "USER_NAME": ["A", "B", "A"],
    })


def test_snowflake_costs_and_grouping():
    df = add_costs(load_history(snowflake_rows()), usd_per_credit=3.0)
    assert df.loc[0, "compute_units"] == pytest.approx(8.0)
    assert df.loc[0, "est_cost_usd"] == pytest.approx(24.0)
    assert df.loc[2, "pruning_ratio"] == pytest.approx(0.02)
    summary = summarise_by_fingerprint(df)
    assert len(summary) == 2
    top = summary.iloc[0]
    assert top["runs"] == 2
    assert top["total_cost_usd"] == pytest.approx(36.0)
    assert top["issues"] >= 2
    assert summary["cost_share"].sum() == pytest.approx(1.0)


def test_databricks_export_is_detected_and_priced():
    raw = pd.DataFrame({
        "statement_id": ["s1"],
        "statement_text": ["select * from events"],
        "total_duration_ms": [3_600_000],
        "warehouse_size": ["Small"],
        "read_bytes": [5e9],
        "read_files": [10],
        "pruned_files": [90],
    })
    assert detect_platform(raw) == "databricks"
    df = add_costs(load_history(raw), usd_per_dbu=0.7)
    assert df.loc[0, "compute_units"] == pytest.approx(12.0)
    assert df.loc[0, "est_cost_usd"] == pytest.approx(8.4)
    assert df.loc[0, "pruning_ratio"] == pytest.approx(0.1)


def test_unknown_warehouse_size_is_flagged_not_hidden():
    raw = snowflake_rows()
    raw.loc[2, "WAREHOUSE_SIZE"] = "Gigantic"
    df = add_costs(load_history(raw))
    assert not df.loc[2, "cost_known"]
    assert df.loc[2, "est_cost_usd"] == 0
    assert df.loc[0, "cost_known"]
    assert unpriced_sizes(df) == {"GIGANTIC": 1}
    summary = summarise_by_fingerprint(df)
    assert summary["unpriced_runs"].sum() == 1


def test_warehouse_size_spellings_are_unified():
    raw = snowflake_rows()
    raw["WAREHOUSE_SIZE"] = ["XXLarge", "2X-Large", "x_small"]
    df = add_costs(load_history(raw))
    assert list(df["warehouse_size"]) == ["2X-LARGE", "2X-LARGE", "X-SMALL"]
    assert df["cost_known"].all()


def databricks_rows(**extra):
    return pd.DataFrame({
        "statement_id": ["s1", "s2", "s3"],
        "statement_text": ["select id from events where d = '2026-01-01'"] * 3,
        "total_duration_ms": [3_600_000] * 3,
        "warehouse_id": ["wh-a", "wh-a", "wh-b"],
        **extra,
    })


def test_databricks_without_size_asks_instead_of_guessing():
    df = load_history(databricks_rows())
    assert df["warehouse_size"].isna().all()
    assert warehouses_without_size(df) == ["wh-a", "wh-b"]
    assert unpriced_sizes(add_costs(df)) == {"missing": 3}

    sized = add_costs(set_warehouse_sizes(df, {"wh-a": "Medium"}), usd_per_dbu=1.0)
    assert list(sized["compute_units"]) == pytest.approx([24.0, 24.0, 0.0])
    assert unpriced_sizes(sized) == {"missing": 1}


def test_set_warehouse_sizes_only_fills_blanks():
    df = load_history(databricks_rows(warehouse_size=["Large", None, "2X_Small"]))
    filled = set_warehouse_sizes(df, {"wh-a": "Small", "wh-b": "Small"})
    assert list(filled["warehouse_size"]) == ["LARGE", "SMALL", "2X-SMALL"]


def test_precomputed_fingerprints_give_the_same_summary():
    df = add_costs(load_history(snowflake_rows()))
    direct = summarise_by_fingerprint(df)
    cached = summarise_by_fingerprint(add_fingerprints(df))
    pd.testing.assert_frame_equal(direct, cached)


def test_unknown_export_is_rejected():
    with pytest.raises(ValueError, match="Unrecognised"):
        load_history(pd.DataFrame({"a": [1]}))


def test_bundled_sample_loads():
    df = add_costs(load_history(pd.read_csv(ROOT / "data" / "sample_snowflake_query_history.csv")))
    summary = summarise_by_fingerprint(df)
    assert len(df) > 100
    assert summary.iloc[0]["issues"] > 0


# LLM

def test_build_messages_includes_query_findings_and_metrics():
    findings = analyze("select * from t")
    messages = build_messages("select * from t", "snowflake", findings, {"runs": 3})
    assert "Snowflake" in messages[0]["content"]
    assert "select * from t" in messages[1]["content"]
    assert "SELECT * reads every column" in messages[1]["content"]
    assert '"runs": 3' in messages[1]["content"]


def test_parse_response_handles_code_fences():
    content = "```json\n" + json.dumps({"rewritten_sql": "select id from t", "changes": ["listed columns"],
                                        "expected_impact": "less IO", "caveats": []}) + "\n```"
    s = parse_response(content)
    assert s.rewritten_sql == "select id from t"
    assert s.changes == ["listed columns"]


def test_suggest_rewrite_calls_groq_and_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        suggest_rewrite("select 1", "snowflake", [])

    response = MagicMock(status_code=200)
    response.json.return_value = {"choices": [{"message": {"content": json.dumps(
        {"rewritten_sql": "select id from t", "changes": [], "expected_impact": "", "caveats": []})}}]}
    session = MagicMock()
    session.post.return_value = response
    s = suggest_rewrite("select * from t", "databricks", [], api_key="k", session=session)
    assert s.rewritten_sql == "select id from t"
    _, kwargs = session.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer k"
    assert kwargs["json"]["response_format"] == {"type": "json_object"}


# Rewrite check

def test_check_rewrite_reports_fixed_issues_and_score():
    check = check_rewrite("select * from t where to_date(ts) = '2026-01-01'",
                          "select id from t where ts >= '2026-01-01' and ts < '2026-01-02'", "snowflake")
    assert check.parses
    assert check.score_after == 100 and check.score_before < 100
    assert len(check.fixed) == 2
    assert check.remaining == [] and check.new == []


def test_check_rewrite_flags_sql_that_does_not_parse():
    check = check_rewrite("select id from t limit 5", "selec id frm t", "databricks")
    assert not check.parses
    assert "databricks" in check.error
    assert not check_rewrite("select id from t limit 5", "  ", "snowflake").parses


def test_check_rewrite_flags_new_issues_and_changed_columns():
    check = check_rewrite("select a, b from t where d > '2026-01-01' limit 5",
                          "select a from t where d > '2026-01-01' order by a", "snowflake")
    assert check.new == ["ORDER BY on the full result without LIMIT"]
    assert check.same_columns is False
    same = check_rewrite("select a as x from t limit 5", "select b as x from t limit 5", "snowflake")
    assert same.same_columns is True
