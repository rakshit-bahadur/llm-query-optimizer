"""Load query history exports and estimate compute cost per query.

Supported exports:
- Snowflake: SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
- Databricks: system.query.history (SQL warehouses)

Cost is an estimate: query run time x the warehouse's hourly rate. Warehouses
run several queries at once, so the sum over many queries is an upper bound on
the real bill. It is meant for ranking queries, not for invoicing.
"""

from __future__ import annotations

import pandas as pd

from .rules import ParseError, analyze, fingerprint, score

# Credits per hour by Snowflake warehouse size (standard warehouses).
SNOWFLAKE_CREDITS_PER_HOUR = {
    "X-SMALL": 1, "SMALL": 2, "MEDIUM": 4, "LARGE": 8, "X-LARGE": 16,
    "2X-LARGE": 32, "3X-LARGE": 64, "4X-LARGE": 128, "5X-LARGE": 256, "6X-LARGE": 512,
}

# Approximate DBUs per hour by Databricks SQL warehouse size.
DATABRICKS_DBU_PER_HOUR = {
    "2X-SMALL": 4, "X-SMALL": 6, "SMALL": 12, "MEDIUM": 24, "LARGE": 40,
    "X-LARGE": 80, "2X-LARGE": 144, "3X-LARGE": 272, "4X-LARGE": 528,
}

# Other spellings seen in exports and settings, mapped to the keys above.
SIZE_ALIASES = {
    "XSMALL": "X-SMALL", "XXSMALL": "2X-SMALL", "XLARGE": "X-LARGE", "XXLARGE": "2X-LARGE",
    "XXXLARGE": "3X-LARGE", "X2LARGE": "2X-LARGE", "X3LARGE": "3X-LARGE", "X4LARGE": "4X-LARGE",
    "X5LARGE": "5X-LARGE", "X6LARGE": "6X-LARGE",
}

COMMON_COLUMNS = [
    "query_id", "query_text", "user_name", "warehouse", "warehouse_size", "start_time",
    "elapsed_ms", "bytes_scanned", "partitions_scanned", "partitions_total",
]


def _require(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{source} export is missing columns: {', '.join(missing)}")


def normalise_size(sizes: pd.Series) -> pd.Series:
    """Upper-case warehouse sizes and unify spellings ('2X_Small', 'XXLarge'). Missing stays missing."""
    text = sizes.astype("string").str.upper().str.strip().str.replace("_", "-", regex=False)
    text = text.where(~text.isin(["", "NAN", "NONE", "NULL"]))
    compact = text.str.replace("-", "", regex=False)
    out = text.where(~compact.isin(SIZE_ALIASES.keys()), compact.map(SIZE_ALIASES)).astype(object)
    return out.where(out.notna(), None)


def normalise_snowflake(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.upper)
    _require(df, ["QUERY_ID", "QUERY_TEXT", "WAREHOUSE_SIZE", "TOTAL_ELAPSED_TIME"], "Snowflake")
    out = pd.DataFrame({
        "query_id": df["QUERY_ID"],
        "query_text": df["QUERY_TEXT"],
        "user_name": df.get("USER_NAME"),
        "warehouse": df.get("WAREHOUSE_NAME"),
        "warehouse_size": normalise_size(df["WAREHOUSE_SIZE"]),
        "start_time": pd.to_datetime(df.get("START_TIME"), errors="coerce"),
        "elapsed_ms": pd.to_numeric(df["TOTAL_ELAPSED_TIME"], errors="coerce").fillna(0),
        "bytes_scanned": pd.to_numeric(df.get("BYTES_SCANNED"), errors="coerce"),
        "partitions_scanned": pd.to_numeric(df.get("PARTITIONS_SCANNED"), errors="coerce"),
        "partitions_total": pd.to_numeric(df.get("PARTITIONS_TOTAL"), errors="coerce"),
    })
    out["platform"] = "snowflake"
    return out


def normalise_databricks(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=str.lower)
    _require(df, ["statement_id", "statement_text", "total_duration_ms"], "Databricks")
    # system.query.history has no size column. Without one the size stays missing, and the
    # app asks the user for each warehouse's size instead of guessing.
    size = df["warehouse_size"] if "warehouse_size" in df.columns else pd.Series([None] * len(df), index=df.index)
    out = pd.DataFrame({
        "query_id": df["statement_id"],
        "query_text": df["statement_text"],
        "user_name": df.get("executed_by"),
        "warehouse": df.get("warehouse_id"),
        "warehouse_size": normalise_size(size),
        "start_time": pd.to_datetime(df.get("start_time"), errors="coerce"),
        "elapsed_ms": pd.to_numeric(df["total_duration_ms"], errors="coerce").fillna(0),
        "bytes_scanned": pd.to_numeric(df.get("read_bytes"), errors="coerce"),
        "partitions_scanned": pd.to_numeric(df.get("read_files"), errors="coerce"),
        "partitions_total": pd.to_numeric(df.get("pruned_files"), errors="coerce")
        + pd.to_numeric(df.get("read_files"), errors="coerce"),
    })
    out["platform"] = "databricks"
    return out


def detect_platform(raw: pd.DataFrame) -> str:
    cols = {c.lower() for c in raw.columns}
    if {"query_id", "query_text", "total_elapsed_time"} <= cols:
        return "snowflake"
    if {"statement_id", "statement_text", "total_duration_ms"} <= cols:
        return "databricks"
    raise ValueError(
        "Unrecognised file. Export SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY or system.query.history as CSV."
    )


def load_history(raw: pd.DataFrame) -> pd.DataFrame:
    platform = detect_platform(raw)
    return normalise_snowflake(raw) if platform == "snowflake" else normalise_databricks(raw)


def known_sizes(platform: str) -> list[str]:
    return list(SNOWFLAKE_CREDITS_PER_HOUR if platform == "snowflake" else DATABRICKS_DBU_PER_HOUR)


def warehouses_without_size(df: pd.DataFrame) -> list[str]:
    """Warehouses that have runs with no warehouse size in the export."""
    missing = df.loc[df["warehouse_size"].isna(), "warehouse"].dropna().astype(str)
    return sorted(missing.unique())


def set_warehouse_sizes(df: pd.DataFrame, sizes: dict[str, str]) -> pd.DataFrame:
    """Fill in the size of warehouses the export left blank, from a {warehouse: size} mapping."""
    df = df.copy()
    given = normalise_size(df["warehouse"].astype(str).map(sizes))
    df["warehouse_size"] = df["warehouse_size"].where(df["warehouse_size"].notna(), given)
    return df


def unpriced_sizes(df: pd.DataFrame) -> dict[str, int]:
    """Runs that could not be priced, counted by their warehouse size ('missing' when blank)."""
    if "cost_known" not in df.columns:
        raise ValueError("Call add_costs first.")
    sizes = df.loc[~df["cost_known"], "warehouse_size"].fillna("missing")
    return {str(k): int(v) for k, v in sizes.value_counts().items()}


def add_costs(df: pd.DataFrame, usd_per_credit: float = 3.0, usd_per_dbu: float = 0.70) -> pd.DataFrame:
    """Add compute units and USD per run. Runs whose warehouse size has no known rate cost $0 and
    get cost_known = False, so callers can warn about them instead of hiding them."""
    df = df.copy()
    hours = df["elapsed_ms"] / 3_600_000
    sf_rate = df["warehouse_size"].map(SNOWFLAKE_CREDITS_PER_HOUR)
    db_rate = df["warehouse_size"].map(DATABRICKS_DBU_PER_HOUR)
    is_sf = df["platform"] == "snowflake"
    rate = sf_rate.where(is_sf, db_rate)
    df["cost_known"] = rate.notna()
    df["compute_units"] = (hours * rate).fillna(0)
    df["est_cost_usd"] = (df["compute_units"] * usd_per_credit).where(is_sf, df["compute_units"] * usd_per_dbu)
    df["pruning_ratio"] = (df["partitions_scanned"] / df["partitions_total"]).where(df["partitions_total"] > 0)
    return df


def add_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    """Add the literal-free query shape. This parses every query, so it is the slow step."""
    df = df.copy()
    df["fingerprint"] = [fingerprint(q, p) for q, p in zip(df["query_text"], df["platform"])]
    return df


def summarise_by_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    """Group repeated queries and rank them by total estimated cost."""
    if "fingerprint" not in df.columns:
        df = add_fingerprints(df)
    if "cost_known" not in df.columns:
        df = df.assign(cost_known=True)
    grouped = (
        df.groupby(["platform", "fingerprint"], as_index=False)
        .agg(
            runs=("query_id", "count"),
            total_cost_usd=("est_cost_usd", "sum"),
            avg_seconds=("elapsed_ms", lambda s: s.mean() / 1000),
            total_gb_scanned=("bytes_scanned", lambda s: s.sum() / 1e9),
            avg_pruning_ratio=("pruning_ratio", "mean"),
            example_query=("query_text", "first"),
            users=("user_name", "nunique"),
            unpriced_runs=("cost_known", lambda s: int((~s.astype(bool)).sum())),
        )
        .sort_values("total_cost_usd", ascending=False)
        .reset_index(drop=True)
    )
    issues, scores, top = [], [], []
    for sql, platform in zip(grouped["example_query"], grouped["platform"]):
        try:
            findings = analyze(sql, platform)
        except ParseError:
            findings = []
        issues.append(len(findings))
        scores.append(score(findings))
        top.append(findings[0].title if findings else "")
    grouped["issues"] = issues
    grouped["health_score"] = scores
    grouped["top_issue"] = top
    total = grouped["total_cost_usd"].sum()
    grouped["cost_share"] = grouped["total_cost_usd"] / total if total else 0.0
    return grouped
