# -*- coding: utf-8 -*-
"""
nxs_visual_engine.py
--------------------
Visual Engine for NXS:
- Primary: Plotly (interactive, web-ready)
- Fallback: Matplotlib (static PNG base64)
- Table rendering: HTML (pandas)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import base64
import io

import pandas as pd

# Plotly (primary)
import plotly.express as px

# Matplotlib (fallback)
import matplotlib.pyplot as plt


@dataclass
class VizOutput:
    ok: bool
    figure: Optional[Dict[str, Any]] = None          # Plotly figure JSON (dict)
    png_base64: Optional[str] = None                 # Matplotlib fallback (base64 PNG)
    table_html: Optional[str] = None                 # HTML table
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _to_df(rows: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any], None]) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, dict):
        if "rows" in rows and isinstance(rows["rows"], list):
            return pd.DataFrame(rows["rows"])
        return pd.DataFrame([rows])
    if isinstance(rows, list):
        return pd.DataFrame(rows)
    return pd.DataFrame()


def _render_table_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "<div style='opacity:.8'>لا توجد بيانات لعرضها.</div>"
    view = df.head(max_rows)
    return view.to_html(index=False, escape=True, border=0, classes="nxs-table")


def _mpl_png_base64(mpl_fig) -> str:
    buf = io.BytesIO()
    mpl_fig.savefig(buf, format="png", bbox_inches="tight", dpi=170)
    plt.close(mpl_fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_chart(
    rows: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any], None],
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    names: Optional[str] = None,
    values: Optional[str] = None,
    title: Optional[str] = None,
) -> VizOutput:
    """
    chart_type: bar | line | pie | table

    - bar/line يحتاجان: x, y
    - pie يحتاج: names, values
    - table: يرجع جدول فقط
    """
    try:
        df = _to_df(rows)
        table_html = _render_table_html(df)

        ct = (chart_type or "").strip().lower()
        if ct in {"table", ""}:
            return VizOutput(ok=True, table_html=table_html, meta={"mode": "table_only"})

        if df.empty:
            return VizOutput(ok=False, table_html=table_html, error="لا توجد بيانات كافية للرسم.")

        # Plotly primary
        if ct == "bar":
            if not x or not y:
                return VizOutput(ok=False, table_html=table_html, error="bar يحتاج x و y.")
            fig = px.bar(df, x=x, y=y, title=title)
        elif ct == "line":
            if not x or not y:
                return VizOutput(ok=False, table_html=table_html, error="line يحتاج x و y.")
            fig = px.line(df, x=x, y=y, title=title)
        elif ct == "pie":
            if not names or not values:
                return VizOutput(ok=False, table_html=table_html, error="pie يحتاج names و values.")
            fig = px.pie(df, names=names, values=values, title=title)
        else:
            return VizOutput(ok=False, table_html=table_html, error=f"نوع الرسم غير مدعوم: {chart_type}")

        figure_json = fig.to_dict()

        # Matplotlib fallback
        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)

        if ct == "bar":
            df.plot(kind="bar", x=x, y=y, ax=ax)
        elif ct == "line":
            df.plot(kind="line", x=x, y=y, ax=ax)
        elif ct == "pie":
            s = df.groupby(names)[values].sum() if names and values else df.iloc[:, 0]
            s.plot(kind="pie", ax=ax)
            ax.set_ylabel("")
        if title:
            ax.set_title(title)

        png64 = _mpl_png_base64(mpl_fig)

        return VizOutput(
            ok=True,
            figure=figure_json,
            png_base64=png64,
            table_html=table_html,
            meta={"mode": "plotly_primary_with_mpl_fallback", "rows": int(len(df))},
        )

    except Exception as e:
        return VizOutput(ok=False, error=str(e))
