# -*- coding: utf-8 -*-
"""
nxs_dashboard_engine.py
-----------------------
Dashboard engine for NXS (Ground Services / RUH).

Design goals:
- Patch-friendly: does not change existing NXS brain behavior.
- Works via Supabase REST (through a provided supabase_select function).
- Produces web-ready outputs:
    - Plotly JSON charts (primary)
    - Matplotlib PNG base64 fallback (via nxs_visual_engine)
    - HTML tables (via nxs_visual_engine)

Supports unified filters:
- Date / date range
- Shift
- Department (including TCC group mapping)
- Airlines (for delay tables)

This module intentionally keeps the "business logic" here, and leaves /chat routing to nxs_viz_router.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd

from nxs_visual_engine import build_chart


# =========================
# Department mapping (TCC)
# =========================

TCC_GROUP_DEPARTMENTS = ["TC", "TRC", "FIC Saudia", "FIC Nas", "LC Saudia", "LC Foreign"]

def resolve_department_filter(raw: Optional[str], message: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Returns:
      - single_department (eq) OR
      - department_list (in)
    """
    m = (message or "").lower()
    d = (raw or "").strip() if raw else ""

    # Detect TCC group by explicit keywords
    if any(k in m for k in ["tcc", "traffic control", "مراقبة الحركة", "مراقبه الحركه", "مركز مراقبة", "مركز مراقبه"]):
        return None, TCC_GROUP_DEPARTMENTS

    # If the user directly typed one of the sub-departments
    if d in TCC_GROUP_DEPARTMENTS:
        return d, None

    # If raw equals "TCC"
    if d.upper() == "TCC":
        return None, TCC_GROUP_DEPARTMENTS

    return (d or None), None


def _to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    try:
        return pd.DataFrame(rows or [])
    except Exception:
        return pd.DataFrame()


def _date_col_for(table: str) -> str:
    # As per your DB DDL and schema usage, most operational tables use "Date"
    if table in ("employee_master_db",):
        return "Record Date"
    if table in ("employee_overtime",):
        return "Assignment Date"
    return "Date"


def build_filters(table: str, intent_info: Dict[str, Any], message: str) -> Dict[str, str]:
    """
    Supabase REST filters (PostgREST).
    - eq.<value>
    - gte.<value>
    - lte.<value>
    - in.(a,b,c)
    """
    filters: Dict[str, str] = {}

    date_from = intent_info.get("date_from")
    date_to = intent_info.get("date_to")
    date_col = _date_col_for(table)

    if date_from:
        filters[date_col] = f"gte.{date_from}"
    if date_to:
        # if same key exists we use a second key (router supabase_select has variant logic)
        if date_col not in filters:
            filters[date_col] = f"lte.{date_to}"
        else:
            filters[f"{date_col}_to"] = f"lte.{date_to}"

    # Shift filter
    shift = intent_info.get("shift") or intent_info.get("Shift")
    if shift and table in ("shift_report", "employee_delay", "employee_absence", "operational_event", "dep_flight_delay"):
        filters["Shift"] = f"eq.{shift}"

    # Department filter (with TCC group support)
    dep_raw = intent_info.get("department") or intent_info.get("Department")
    dep_eq, dep_list = resolve_department_filter(dep_raw, message)
    if dep_list and table in ("shift_report", "employee_delay", "employee_absence", "operational_event", "dep_flight_delay"):
        # PostgREST IN format: in.(a,b,c) - values should not contain commas; our names are safe.
        inside = ",".join(dep_list)
        filters["Department"] = f"in.({inside})"
    elif dep_eq and table in ("shift_report", "employee_delay", "employee_absence", "operational_event", "dep_flight_delay"):
        filters["Department"] = f"eq.{dep_eq}"

    # Airline filter (delay tables)
    airline = intent_info.get("airline") or intent_info.get("Airlines")
    if airline and table in ("sgs_flight_delay", "dep_flight_delay"):
        filters["Airlines"] = f"eq.{airline}"

    return filters


def _unique_options(df: pd.DataFrame, col: str, max_items: int = 100) -> List[Any]:
    if col not in df.columns or df.empty:
        return []
    vals = [v for v in df[col].dropna().unique().tolist()]
    # keep stable order-ish
    return vals[:max_items]


def build_dashboard(
    message: str,
    intent_info: Dict[str, Any],
    supabase_select_fn: Callable[[str, Optional[Dict[str, str]], int], List[Dict[str, Any]]],
    limit: int = 5000,
) -> Tuple[str, Dict[str, Any]]:
    """
    Creates a dashboard bundle with multiple charts + table.
    Returns (reply_text, meta_payload).
    """

    m = (message or "").lower()

    # Choose dashboard domain
    if any(k in m for k in ["تأخير", "delay", "delays", "تأخيرات"]):
        table = "sgs_flight_delay" if "sgs" in m or "محطة" in m else "dep_flight_delay"
        domain = "flight_delays"
    elif any(k in m for k in ["شفت", "وردية", "shift", "on duty", "no show"]):
        table = "shift_report"
        domain = "shift_report"
    else:
        # default dashboard
        table = "shift_report"
        domain = "shift_report"

    filters = build_filters(table, intent_info, message)
    rows = supabase_select_fn(table, filters=filters, limit=limit)
    df = _to_df(rows)

    # Provide available filter options (derived from result set)
    available_filters = {
        "Department": _unique_options(df, "Department"),
        "Shift": _unique_options(df, "Shift"),
        "Airlines": _unique_options(df, "Airlines"),
    }

    charts: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []

    if df.empty:
        reply = "⚠️ لا توجد بيانات ضمن الفلاتر الحالية."
        meta = {
            "intent": "dashboard",
            "dashboard": {
                "ok": False,
                "domain": domain,
                "table": table,
                "filters": filters,
                "available_filters": available_filters,
                "charts": [],
                "tables": [],
                "error": "empty_result",
            },
        }
        return reply, meta

    # ---------- Dashboards ----------
    if domain == "flight_delays":
        # 1) Trend over Date (count)
        if "Date" in df.columns:
            trend = df.groupby("Date").size().reset_index(name="Delays")
            out1 = build_chart(trend, chart_type="line", x="Date", y="Delays", title="Delays Trend")
            charts.append({"id": "trend", "kind": "line", **_viz_to_dict(out1)})

        # 2) By Airline (bar)
        if "Airlines" in df.columns:
            by_air = df.groupby("Airlines").size().reset_index(name="Delays")
            out2 = build_chart(by_air, chart_type="bar", x="Airlines", y="Delays", title="Delays by Airline")
            charts.append({"id": "by_airline", "kind": "bar", **_viz_to_dict(out2)})

        # 3) By Delay Code (pie)
        if "Delay Code" in df.columns:
            by_code = df.groupby("Delay Code").size().reset_index(name="Count")
            out3 = build_chart(by_code, chart_type="pie", names="Delay Code", values="Count", title="Delay Codes Share")
            charts.append({"id": "by_code", "kind": "pie", **_viz_to_dict(out3)})

        # 4) Table (top rows)
        tables.append({
            "id": "rows",
            "title": "Sample Rows",
            "table_html": df.head(50).to_html(index=False, escape=True, border=0, classes="nxs-table"),
        })

        reply = "📊 تم إنشاء داشبورد التأخيرات حسب الفلاتر."

    else:
        # shift_report dashboard
        # Prefer common metrics if present
        metric_candidates = ["On Duty", "No Show", "Arrivals Domestic", "Departures Domestic",
                             "Arrivals International+Foreign", "Departures International+Foreign"]
        metric = next((c for c in metric_candidates if c in df.columns), None)

        # 1) Trend of metric over Date
        if metric and "Date" in df.columns:
            trend = df.groupby("Date")[metric].sum(numeric_only=True).reset_index()
            out1 = build_chart(trend, chart_type="line", x="Date", y=metric, title=f"{metric} Trend")
            charts.append({"id": "trend", "kind": "line", **_viz_to_dict(out1)})

        # 2) Department bar
        if metric and "Department" in df.columns:
            by_dep = df.groupby("Department")[metric].sum(numeric_only=True).reset_index()
            out2 = build_chart(by_dep, chart_type="bar", x="Department", y=metric, title=f"{metric} by Department")
            charts.append({"id": "by_department", "kind": "bar", **_viz_to_dict(out2)})

        # 3) Table
        tables.append({
            "id": "rows",
            "title": "Sample Rows",
            "table_html": df.head(50).to_html(index=False, escape=True, border=0, classes="nxs-table"),
        })

        reply = "📊 تم إنشاء داشبورد الشفت حسب الفلاتر."

    meta = {
        "intent": "dashboard",
        "dashboard": {
            "ok": True,
            "domain": domain,
            "table": table,
            "filters": filters,
            "available_filters": available_filters,
            "charts": charts,
            "tables": tables,
        },
    }
    return reply, meta


def _viz_to_dict(out) -> Dict[str, Any]:
    return {
        "ok": bool(getattr(out, "ok", False)),
        "figure": getattr(out, "figure", None),
        "png_base64": getattr(out, "png_base64", None),
        "table_html": getattr(out, "table_html", None),
        "meta": getattr(out, "meta", None),
        "error": getattr(out, "error", None),
    }
