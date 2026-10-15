"""Generate a synthetic Snowflake QUERY_HISTORY export for the demo.

The rows are invented and only shaped like ACCOUNT_USAGE.QUERY_HISTORY, so the
app can be tried without a Snowflake account.
"""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "sample_snowflake_query_history.csv"

TEMPLATES = [
    # (query template, warehouse, size, base seconds, GB scanned, scanned/total partitions, users, weight)
    ("SELECT * FROM analytics.bookings WHERE TO_DATE(booking_ts) = '{d}'",
     "BI_WH", "LARGE", 95, 180, (9800, 10000), ["tableau_svc"], 30),
    ("SELECT b.booking_id, c.email FROM analytics.bookings b JOIN analytics.customers c "
     "ON b.customer_id = c.customer_id OR b.email = c.email WHERE b.booking_date >= '{d}'",
     "ETL_WH", "MEDIUM", 240, 95, (4200, 10000), ["etl_svc"], 8),
    ("SELECT customer_id, booking_id, amount FROM analytics.bookings "
     "WHERE booking_date BETWEEN '{d}' AND '{d2}'",
     "BI_WH", "LARGE", 4, 6, (60, 10000), ["tableau_svc", "rakshit"], 60),
    ("SELECT region, COUNT(DISTINCT customer_id) FROM analytics.bookings "
     "WHERE booking_date >= '{d}' GROUP BY region",
     "BI_WH", "MEDIUM", 38, 40, (900, 10000), ["finance_user"], 20),
    ("SELECT DISTINCT t.trip_id, s.segment_id FROM travel.trips t, travel.segments s "
     "WHERE UPPER(t.status) = 'CONFIRMED'",
     "ADHOC_WH", "X-LARGE", 410, 520, (15000, 15000), ["analyst_1"], 3),
    ("SELECT invoice_id FROM finance.invoices WHERE invoice_id NOT IN (SELECT invoice_id FROM finance.payments) "
     "AND invoice_date >= '{d}'",
     "FIN_WH", "SMALL", 55, 22, (700, 2000), ["finance_user"], 12),
    ("SELECT * FROM travel.airport_codes",
     "ADHOC_WH", "X-SMALL", 1, 0.01, (1, 1), ["analyst_1", "analyst_2"], 25),
    ("SELECT agent_id, SUM(amount) AS total FROM finance.commissions WHERE paid_date >= '{d}' "
     "GROUP BY agent_id ORDER BY total DESC",
     "FIN_WH", "SMALL", 12, 3, (120, 2000), ["finance_user"], 18),
]


def main() -> None:
    rng = random.Random(11)
    start = datetime(2026, 9, 1)
    rows = []
    qid = 0
    for template, wh, size, secs, gb, (scanned, total), users, weight in TEMPLATES:
        for _ in range(weight):
            day = start + timedelta(days=rng.randint(0, 29), hours=rng.randint(0, 23), minutes=rng.randint(0, 59))
            d = (day - timedelta(days=rng.randint(1, 30))).date()
            qid += 1
            rows.append({
                "QUERY_ID": f"01b{qid:05d}-0000-sample",
                "QUERY_TEXT": template.format(d=d.isoformat(), d2=(d + timedelta(days=7)).isoformat()),
                "USER_NAME": rng.choice(users).upper(),
                "WAREHOUSE_NAME": wh,
                "WAREHOUSE_SIZE": size,
                "START_TIME": day.strftime("%Y-%m-%d %H:%M:%S"),
                "TOTAL_ELAPSED_TIME": int(secs * 1000 * rng.uniform(0.7, 1.4)),
                "BYTES_SCANNED": int(gb * 1e9 * rng.uniform(0.9, 1.1)),
                "PARTITIONS_SCANNED": min(total, int(scanned * rng.uniform(0.95, 1.05))),
                "PARTITIONS_TOTAL": total,
            })
    rows.sort(key=lambda r: r["START_TIME"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
