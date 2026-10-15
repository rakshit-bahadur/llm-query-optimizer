"""Rule-based SQL anti-pattern detection.

Queries are parsed with sqlglot in the chosen dialect (snowflake or databricks).
Each rule inspects the syntax tree and returns findings with a severity, an
explanation and a concrete fix. Rules never execute SQL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterator

import sqlglot
from sqlglot import exp

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
DIALECTS = ("snowflake", "databricks")

# Functions that wrap a column and stop partition pruning / clustering from working.
PRUNING_BLOCKERS = (
    exp.Upper, exp.Lower, exp.Trim, exp.Cast, exp.TryCast, exp.Substring,
    exp.DateTrunc, exp.TimestampTrunc, exp.Year, exp.Month, exp.Day,
    exp.DateAdd, exp.DateSub, exp.Coalesce, exp.Concat, exp.ToChar,
    exp.TsOrDsToDate, exp.StrToTime, exp.StrToDate, exp.Anonymous,
)


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    title: str
    detail: str
    suggestion: str

    def to_dict(self) -> dict:
        return asdict(self)


class ParseError(ValueError):
    pass


def parse(sql: str, dialect: str = "snowflake") -> exp.Expression:
    if dialect not in DIALECTS:
        raise ValueError(f"dialect must be one of {DIALECTS}")
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        raise ParseError("The query is empty.")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise ParseError(f"Could not parse the query as {dialect} SQL: {exc}") from exc
    if len(statements) != 1:
        raise ParseError("Paste one statement at a time.")
    return statements[0]


def _selects(tree: exp.Expression) -> Iterator[exp.Select]:
    yield from tree.find_all(exp.Select)


def _has_table_source(select: exp.Select) -> bool:
    from_ = select.find(exp.From)
    return from_ is not None and from_.find(exp.Table) is not None


def _own_where(select: exp.Select) -> exp.Where | None:
    where = select.args.get("where")
    return where if isinstance(where, exp.Where) else None


def rule_select_star(tree: exp.Expression) -> list[Finding]:
    for select in _selects(tree):
        if any(isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star))
               for e in select.expressions) and _has_table_source(select):
            return [Finding(
                "SELECT_STAR", "medium", "SELECT * reads every column",
                "Both Snowflake and Databricks store data by column. SELECT * makes the engine read and "
                "decompress every column, even the ones you do not use.",
                "List only the columns you need.",
            )]
    return []


def _top_level_selects(tree: exp.Expression) -> Iterator[exp.Select]:
    """The SELECTs that produce the final result: the query itself or each UNION branch."""
    if isinstance(tree, exp.Select):
        yield tree
    elif isinstance(tree, exp.Union):
        yield from _top_level_selects(tree.left)
        yield from _top_level_selects(tree.right)
    elif isinstance(tree, exp.Subquery):
        yield from _top_level_selects(tree.this)


def rule_no_filter(tree: exp.Expression) -> list[Finding]:
    for select in _top_level_selects(tree):
        if (_has_table_source(select) and _own_where(select) is None and not select.args.get("limit")
                and not select.args.get("group") and not select.find(exp.AggFunc)
                and not select.args.get("joins")):
            table = select.find(exp.From).find(exp.Table).sql()
            return [Finding(
                "FULL_SCAN", "high", f"No filter on {table}",
                "The query has no WHERE clause, LIMIT or aggregation, so it scans and returns the whole table.",
                "Filter on the partition or clustering column (usually a date), or add a LIMIT while exploring.",
            )]
    return []


def rule_function_on_filter_column(tree: exp.Expression) -> list[Finding]:
    findings = []
    seen = set()
    for where in tree.find_all(exp.Where):
        for pred in where.find_all(exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between, exp.In):
            func = pred.this
            if isinstance(func, PRUNING_BLOCKERS) and func.find(exp.Column) is not None:
                col = func.find(exp.Column).sql()
                if col in seen:
                    continue
                seen.add(col)
                findings.append(Finding(
                    "FUNCTION_ON_FILTER", "high", f"Function applied to filter column {col}",
                    f"`{func.sql()}` wraps the column, so the engine cannot use partition pruning, "
                    "micro-partition metadata or data skipping on it and must evaluate every row.",
                    f"Compare the raw column with a range instead, for example "
                    f"`{col} >= '2026-01-01' AND {col} < '2026-02-01'` instead of a function on {col}.",
                ))
    return findings


def rule_leading_wildcard(tree: exp.Expression) -> list[Finding]:
    for like in tree.find_all(exp.Like, exp.ILike):
        pattern = like.expression
        if isinstance(pattern, exp.Literal) and pattern.is_string and pattern.this.startswith("%"):
            return [Finding(
                "LEADING_WILDCARD", "medium", "LIKE pattern starts with %",
                f"`{like.sql()}` cannot use clustering or search optimization, so every value is checked.",
                "Use a prefix pattern ('abc%') when possible, or Snowflake search optimization / a "
                "Databricks bloom filter index for substring lookups on large tables.",
            )]
    return []


def rule_cartesian_join(tree: exp.Expression) -> list[Finding]:
    for join in tree.find_all(exp.Join):
        kind = (join.args.get("kind") or "").upper()
        if kind == "CROSS" or (not join.args.get("on") and not join.args.get("using")
                                and not join.args.get("method") and kind not in ("SEMI", "ANTI")
                                and not isinstance(join.this, (exp.Lateral, exp.Unnest))):
            return [Finding(
                "CARTESIAN_JOIN", "high", f"Join without a condition: {join.this.sql()}",
                "A join with no ON or USING clause returns every combination of rows, which can explode "
                "the row count and the cost.",
                "Add the join condition, or keep CROSS JOIN only for small dimension tables you truly want to multiply.",
            )]
    return []


def rule_or_in_join(tree: exp.Expression) -> list[Finding]:
    for join in tree.find_all(exp.Join):
        on = join.args.get("on")
        if on is not None and on.find(exp.Or) is not None:
            return [Finding(
                "OR_IN_JOIN", "high", "OR inside a join condition",
                "Joins with OR cannot use a hash join, so the engine falls back to a nested loop that compares "
                "every pair of rows.",
                "Split the query into one join per condition and combine them with UNION ALL.",
            )]
    return []


def rule_not_in_subquery(tree: exp.Expression) -> list[Finding]:
    hit = False
    for node in tree.find_all(exp.Not):
        if isinstance(node.this, exp.In) and node.this.args.get("query") is not None:
            hit = True
    for node in tree.find_all(exp.NEQ):
        if isinstance(node.expression, exp.All) and node.expression.find(exp.Select):
            hit = True
    if hit:
        return [Finding(
            "NOT_IN_SUBQUERY", "medium", "NOT IN with a subquery",
            "If the subquery returns a single NULL, NOT IN returns no rows at all. It is also often planned "
            "less efficiently than an anti join.",
            "Use NOT EXISTS (correlated) or a LEFT JOIN ... WHERE right_key IS NULL.",
        )]
    return []


def rule_union_distinct(tree: exp.Expression) -> list[Finding]:
    for union in tree.find_all(exp.Union):
        if union.args.get("distinct"):
            return [Finding(
                "UNION_DISTINCT", "low", "UNION removes duplicates",
                "Plain UNION sorts or hashes the combined result to remove duplicates.",
                "Use UNION ALL if the parts cannot overlap or duplicates do not matter.",
            )]
    return []


def rule_order_without_limit(tree: exp.Expression) -> list[Finding]:
    order = tree.args.get("order")
    if isinstance(tree, (exp.Select, exp.Union)) and order is not None and not tree.args.get("limit"):
        return [Finding(
            "ORDER_WITHOUT_LIMIT", "low", "ORDER BY on the full result without LIMIT",
            "Sorting the whole result is expensive on large outputs and is usually done again by the BI tool.",
            "Add a LIMIT, or remove the ORDER BY if the consumer sorts the data itself.",
        )]
    return []


def rule_select_distinct(tree: exp.Expression) -> list[Finding]:
    for select in _selects(tree):
        if select.args.get("distinct") and select.args.get("joins"):
            return [Finding(
                "DISTINCT_AFTER_JOIN", "medium", "SELECT DISTINCT on a joined result",
                "DISTINCT after a join is often hiding duplicates created by a join on a non-unique key, and "
                "it adds a large shuffle or sort.",
                "Check the join keys. Deduplicate the input first, or use EXISTS if you only need to check a match.",
            )]
    return []


def rule_count_distinct(tree: exp.Expression) -> list[Finding]:
    for count in tree.find_all(exp.Count):
        if isinstance(count.this, exp.Distinct):
            return [Finding(
                "EXACT_COUNT_DISTINCT", "low", "Exact COUNT(DISTINCT ...)",
                "Exact distinct counts need memory for every distinct value and are slow on large tables.",
                "If an estimate within about 2% is fine, use APPROX_COUNT_DISTINCT (Snowflake and Databricks).",
            )]
    return []


def rule_correlated_scalar_subquery(tree: exp.Expression) -> list[Finding]:
    for select in _selects(tree):
        for projection in select.expressions:
            sub = projection.find(exp.Subquery)
            if sub is not None and sub.find(exp.Where) is not None:
                return [Finding(
                    "SCALAR_SUBQUERY", "medium", "Subquery inside the SELECT list",
                    "A correlated subquery in the SELECT list can run once per output row.",
                    "Rewrite it as a JOIN to a pre-aggregated CTE.",
                )]
    return []


RULES: list[Callable[[exp.Expression], list[Finding]]] = [
    rule_no_filter,
    rule_function_on_filter_column,
    rule_cartesian_join,
    rule_or_in_join,
    rule_select_star,
    rule_leading_wildcard,
    rule_not_in_subquery,
    rule_select_distinct,
    rule_correlated_scalar_subquery,
    rule_union_distinct,
    rule_order_without_limit,
    rule_count_distinct,
]


def analyze(sql: str, dialect: str = "snowflake") -> list[Finding]:
    tree = parse(sql, dialect)
    findings = [f for rule in RULES for f in rule(tree)]
    return sorted(findings, key=lambda f: SEVERITY_ORDER[f.severity])


def fingerprint(sql: str, dialect: str = "snowflake") -> str:
    """Normalise a query so runs that differ only in literal values group together."""
    try:
        tree = parse(sql, dialect)
    except ParseError:
        return " ".join((sql or "").split()).lower()

    def strip_literals(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Literal):
            return exp.Placeholder()
        return node

    return tree.transform(strip_literals).sql(dialect=dialect, normalize=True, pretty=False)


def score(findings: list[Finding]) -> int:
    """0-100 health score: 100 means no issues found."""
    penalty = {"high": 30, "medium": 12, "low": 5}
    return max(0, 100 - sum(penalty[f.severity] for f in findings))
