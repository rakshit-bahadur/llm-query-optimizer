"""Optional LLM rewrite suggestions through the Groq API (OpenAI-compatible).

The rule engine works without this module. When a GROQ_API_KEY is set, the app
sends the query, its dialect, the rule findings and any run metrics to the
model and asks for an explained rewrite. The response is advice to review, not
SQL to run blindly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests

from .rules import Finding

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a senior data engineer who tunes SQL for {platform}.
Rewrite the user's query to cost less and run faster while returning exactly the same result.
Rules:
- Keep the same output columns, row set and ordering semantics.
- Use only {platform} SQL syntax.
- Do not invent table or column names that are not in the query.
- If a change needs information you do not have (for example the clustering key), say so instead of guessing.
Reply with JSON only, in this shape:
{{"rewritten_sql": "...", "changes": ["short explanation of each change"], "expected_impact": "one sentence", "caveats": ["..."]}}"""


@dataclass
class Suggestion:
    rewritten_sql: str
    changes: list[str]
    expected_impact: str
    caveats: list[str]


def build_messages(sql: str, platform: str, findings: list[Finding], metrics: dict[str, Any] | None) -> list[dict]:
    issues = "\n".join(f"- [{f.severity}] {f.title}: {f.suggestion}" for f in findings) or "- none found"
    facts = json.dumps(metrics or {}, default=str)
    user = (
        f"Platform: {platform}\n\nQuery:\n```sql\n{sql.strip()}\n```\n\n"
        f"Issues found by static analysis:\n{issues}\n\nRun metrics (may be empty): {facts}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(platform=platform.capitalize())},
        {"role": "user", "content": user},
    ]


def parse_response(content: str) -> Suggestion:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The model did not return JSON.")
    data = json.loads(text[start:end + 1])
    return Suggestion(
        rewritten_sql=str(data.get("rewritten_sql", "")).strip(),
        changes=[str(c) for c in data.get("changes", [])],
        expected_impact=str(data.get("expected_impact", "")),
        caveats=[str(c) for c in data.get("caveats", [])],
    )


def suggest_rewrite(
    sql: str,
    platform: str,
    findings: list[Finding],
    metrics: dict[str, Any] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    session: requests.Session | None = None,
) -> Suggestion:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY to get AI rewrite suggestions.")
    http = session or requests.Session()
    resp = http.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": build_messages(sql, platform, findings, metrics),
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        timeout=60,
    )
    if resp.status_code == 401:
        raise RuntimeError("Groq rejected the API key.")
    resp.raise_for_status()
    return parse_response(resp.json()["choices"][0]["message"]["content"])
