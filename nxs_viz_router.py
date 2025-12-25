# -*- coding: utf-8 -*-
"""
nxs_viz_router.py
-----------------
Lightweight router to fulfill chart/table requests directly from /chat.

Goals:
- Minimal, non-invasive patch: if message is a visualization request, we fetch data from Supabase,
  aggregate it, and return a response payload including Plotly JSON + Matplotlib fallback PNG + HTML table.
- Otherwise, return None and let the normal NXS brain handle the message.

Primary charting: Plotly (interactive)
Fallback: Matplotlib (PNG base64)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable
import re

import pandas as pd

from nxs_intents import classify_intent
from nxs_visual_engine import build_chart, VizOutput


# -----------------------------
# Detection & parsing utilities
# -----------------------------

_VIZ_KEYWORDS = [
    # Arabic
    "رسم", "رسوم", "بياني", "بيانية", "مخطط", "مخططات", "شارت", "chart",
    "عمودي", "خطي", "دائري", "خط", "bar", "line", "pie",
    "جدول", "table",
    "dashboard", "داشبورد",
]

_TABLE_HINTS = ["جدول", "table"]
_PIE_HINTS = ["دائري", "pie"]
_LINE_HINTS = ["خطي", "line"]
_BAR_HINTS = ["عمودي", "bar", "عمود"]


def is_viz_request(message: str) -> bool:
    m = (message or "").lower()
    return any(k in m for k in _VIZ_KEYWORDS)


def parse_chart_type(message: str) -> str:
    m = (message or "").lower()
    if any(k in m for k in _TABLE_HINTS):
        return "table"
    if any(k in m for k in _PIE_HINTS):
        return "pie"
    if any(k in m for k in _LINE_HINTS):
        return "line"
    if any(k in m for k in _BAR_HINTS):
        return "bar"
    # default chart type
    return "bar"


def choose_table(message: str, intent_info: Dict[str, Any]) -> str:
    m = (message or "").lower()

    # Explicit table hints by domain keywords
    if "shift" in m or "شفت" in m or "ورديه" in m or "وردية" in m or "on duty" in m or "no show" in m:
        return "shift_report"

    if "sgs" in m or "محطه" in m or "محطة" in m:
        return "sgs_flight_delay"

    if "dep_flight_delay" in m or "dep flight" in m or "tcc" in m or "مراقبه الحركه" in m or "مراقبة الحركة" in m:
        return "dep_flight_delay"

    # If user asks about flight delay generally, use station-level by default.
    if "delay" in m or "تأخير" in m:
        return "sgs_flight_delay"

    # default dashboard table
    return "shift_report"


def _metric_from_message_for_shift_report(message: str) -> str:
    m = (message or "").lower()

    # prioritized metrics
    if "on duty" in m or "اون ديوتي" in m or "حضور" in m or "على راس العمل" in m:
        return "On Duty"
    if "no show" in m or "نوشو" in m or "لم يحضر" in m:
        return "No Show"
    if "delayed departures" in m or "تأخير المغادرات" in m:
        return "Delayed Departures Domestic"
    if "departures" in m or "مغادر" in m or "مغادرة" in m:
        # pick a common numeric column
        return "Departures Domestic"
    if "arrivals" in m or "وصول" in m:
        return "Arrivals Domestic"
    if "cars" in m or "سيارات" in m:
        return "Cars In Service"
    if "wireless" in m or "لاسلكي" in m:
        return "Wireless Devices In Service"
    # safe default
    return "On Duty"


def _build_filters(intent_info: Dict[str, Any], table: str) -> Dict[str, str]:
    filters: Dict[str, str] = {}

    date_from = intent_info.get("date_from")
    date_to = intent_info.get("date_to")

    # Most tables use "Date" (as per your schema and DB DDL)
    if date_from:
        filters["Date"] = f"gte.{date_from}"
    if date_to:
        # If we already used Date gte, we add a second key by variants isn't supported in current helper,
        # so we use an 'and' style by letting supabase_select handle variants. We'll prefer lte on Date_to only when no gte.
        if "Date" not in filters:
            filters["Date"] = f"lte.{date_to}"
        else:
            # We'll encode an additional filter key that supabase_select will attempt with variants; we keep it explicit.
            filters["Date_to"] = f"lte.{date_to}"

    # Common filters if present
    dep = intent_info.get("department")
    if dep and table in ("shift_report", "dep_flight_delay", "employee_delay", "employee_absence", "operational_event"):
        filters["Department"] = f"eq.{dep}"

    airline = intent_info.get("airline")
    if airline and table in ("sgs_flight_delay", "dep_flight_delay"):
        # DB uses "Airlines"
        filters["Airlines"] = f"eq.{airline}"

    flight_number = intent_info.get("flight_number")
    if flight_number and table in ("sgs_flight_delay",):
        filters["Flight Number"] = f"eq.{flight_number}"

    # Employee ID filters
    emp = intent_info.get("employee_id")
    if emp and table in ("employee_delay", "employee_absence", "employee_sick_leave", "employee_overtime", "operational_event", "dep_flight_delay"):
        filters["Employee ID"] = f"eq.{emp}"

    return filters


def _aggregate_for_chart(df: pd.DataFrame, table: str, chart_type: str, message: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"table": table, "chart_type": chart_type}

    if df.empty:
        return df, meta

    # Defaults per table
    if table == "shift_report":
        x = "Department" if "Department" in df.columns else None
        y = _metric_from_message_for_shift_report(message)
        if y not in df.columns:
            # fallback to any numeric-ish column
            for cand in ["On Duty", "Departures Domestic", "Arrivals Domestic", "No Show"]:
                if cand in df.columns:
                    y = cand
                    break

        if x and y in df.columns:
            # group by department and sum the metric
            g = df.groupby(x, dropna=False)[y].sum(numeric_only=True).reset_index()
            meta.update({"x": x, "y": y, "title": f"{y} by {x}"})
            return g, meta

    if table in ("sgs_flight_delay", "dep_flight_delay"):
        # Common: count delays by Delay Code or by Airlines
        if chart_type == "pie":
            if "Delay Code" in df.columns:
                g = df.groupby("Delay Code").size().reset_index(name="Count")
                meta.update({"names": "Delay Code", "values": "Count", "title": "Delays by Delay Code"})
                return g, meta
            if "Airlines" in df.columns:
                g = df.groupby("Airlines").size().reset_index(name="Count")
                meta.update({"names": "Airlines", "values": "Count", "title": "Delays by Airline"})
                return g, meta

        # bar/line default: count by Date if exists, else by Delay Code
        if "Date" in df.columns:
            g = df.groupby("Date").size().reset_index(name="Count")
            meta.update({"x": "Date", "y": "Count", "title": "Delays over time"})
            return g, meta
        if "Delay Code" in df.columns:
            g = df.groupby("Delay Code").size().reset_index(name="Count")
            meta.update({"x": "Delay Code", "y": "Count", "title": "Delays by Delay Code"})
            return g, meta

    # If no special aggregation: return first 200 rows
    return df.head(200), meta


def maybe_handle_viz_chat(
    message: str,
    supabase_select_fn: Callable[[str, Optional[Dict[str, str]], int], List[Dict[str, Any]]],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    If message is a viz/table request:
      - fetch relevant data from Supabase
      - aggregate
      - render with Visual Engine
      - return (reply, meta_with_viz)
    Else:
      - return None
    """
    if not is_viz_request(message):
        return None

    intent_info = classify_intent(message)
    chart_type = parse_chart_type(message)
    table = choose_table(message, intent_info)

    filters = _build_filters(intent_info, table)

    # Attempt fetch (limit can be raised later; keep conservative for speed)
    rows = supabase_select_fn(table, filters=filters or None, limit=500)

    df = pd.DataFrame(rows) if isinstance(rows, list) else pd.DataFrame()

    df2, agg_meta = _aggregate_for_chart(df, table=table, chart_type=chart_type, message=message)

    # Decide plotting params based on agg_meta
    title = agg_meta.get("title")

    if chart_type == "pie":
        out: VizOutput = build_chart(
            df2,
            chart_type="pie",
            names=agg_meta.get("names"),
            values=agg_meta.get("values"),
            title=title,
        )
    elif chart_type in ("bar", "line"):
        out = build_chart(
            df2,
            chart_type=chart_type,
            x=agg_meta.get("x"),
            y=agg_meta.get("y"),
            title=title,
        )
    else:
        out = build_chart(df2, chart_type="table", title=title)

    # Reply text (short, UI-driven)
    reply = "✅ تم إنشاء الرسم/الجدول المطلوب وعرضه في الواجهة."

    meta: Dict[str, Any] = {
        "intent": "viz_request",
        "viz": {
            "ok": out.ok,
            "figure": out.figure,
            "png_base64": out.png_base64,
            "table_html": out.table_html,
            "meta": {"chart_type": chart_type, "table": table, "filters": filters, "agg": agg_meta, **(out.meta or {})},
            "error": out.error,
        },
    }

    return reply, meta
