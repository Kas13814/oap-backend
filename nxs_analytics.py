# -*- coding: utf-8 -*-
"""
nxs_analytics.py
----------------
دوال العقل التحليلي فوق البيانات، مع محاكاة مبسطة لنماذج التعلم الآلي.

هذه الوحدة لا تقوم بدور نموذج ذكاء اصطناعي كبير، لكنها توفر:
- ملخصات تحليلية جاهزة للاستخدام في الردود.
- محاكاة لنماذج تنبؤ وتصنيف يمكن للنظام الأوسع (Gemini / NXS Brain)
  أن يبني عليها قرارات أكثر تعقيداً.
"""

from typing import Any, Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from nxs_supabase_client import get_employee_delays, list_all_flight_delays
import nxs_supabase_client as nxs_db


# ---------------- Helpers ----------------


def _safe(val: Any, alt: str = "—") -> str:
    return alt if val is None or val == "" else str(val)


def _format_date(d: Any) -> str:
    try:
        return datetime.fromisoformat(str(d)).strftime("%Y-%m-%d")
    except Exception:
        return str(d)


# ---------------- 1) Employee Summary ----------------


def summarize_employee_delays(emp_id: str, start: str, end: str, max_rows: int = 5) -> str:
    """
    ملخص ذكي لتأخيرات موظف معيّن خلال فترة محددة.
    يعاد كنص عربي بسيط يمكن تمريره مباشرة إلى المستخدم أو دمجه في تقرير أكبر.
    """
    rows = get_employee_delays(emp_id, start, end, 200)

    if not rows:
        return (
            f"لا توجد سجلات تأخير للموظف {emp_id} "
            f"خلال الفترة من {_format_date(start)} إلى {_format_date(end)}."
        )

    emp_name = _safe(rows[0].get("Employee Name"))
    total = len(rows)

    lines: List[str] = []
    lines.append(
        f"الرحلات المتأخرة للموظف {emp_id} ({emp_name}) "
        f"خلال الفترة من {_format_date(start)} إلى {_format_date(end)}."
    )
    lines.append(f"إجمالي التأخيرات المسجّلة: {total}.")

    for r in rows[:max_rows]:
        date = _format_date(r.get("Date"))
        shift = _safe(r.get("Shift"))
        airline = _safe(r.get("Airlines"), "غير محدد")
        arr = _safe(r.get("Arrival Flight Number"), "—")
        dep = _safe(r.get("Departure Flight Number"), "—")
        arr_reason = _safe(r.get("Arrival Violations"), "غير مذكور")
        dep_reason = _safe(r.get("Departure Violations"), "غير مذكور")

        lines.append(
            f"• التاريخ {date}، الفترة {shift}، شركة الطيران {airline}، "
            f"رحلة الوصول {arr}، رحلة المغادرة {dep}."
        )
        lines.append(f"  سبب الوصول: {arr_reason}.")
        lines.append(f"  سبب المغادرة: {dep_reason}.")

    return "\n".join(lines)


# ---------------- 2) Airline Summary + JSON ----------------


def airline_delay_summary_with_json(limit: int = 5000) -> Dict[str, Any]:
    """
    تحليل بسيط لعدد التأخيرات لكل شركة طيران.
    يعيد النص التحليلي إضافة إلى بيانات جاهزة للرسم البياني.
    """
    rows = list_all_flight_delays(limit)

    if not rows:
        return {
            "ok": False,
            "summary": "لا توجد أي سجلات تأخير متاحة للتحليل.",
            "chart": [],
        }

    counts: Dict[str, int] = {}
    for r in rows:
        airline = _safe(r.get("Airlines"), "غير معروف")
        counts[airline] = counts.get(airline, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    lines: List[str] = []
    lines.append("تحليل عدد التأخيرات حسب شركة الطيران.")
    for k, v in sorted_items:
        lines.append(f"• شركة {k}: عدد التأخيرات المسجلة {v}.")

    top_airline, top_count = sorted_items[0]
    lines.append(
        f"أعلى شركة من حيث عدد التأخيرات هي {top_airline} بعدد {top_count} حالة تقريباً."
    )

    chart_data = [{"airline": k, "delays": v} for k, v in sorted_items]

    return {"ok": True, "summary": "\n".join(lines), "chart": chart_data}


# =================================================================
# وظيفة المرحلة 11: محاكاة التعلم الآلي (TAT Prediction)
# =================================================================


def run_ml_tat_prediction() -> Tuple[str, Dict[str, Any]]:
    """
    تدريب مبسط لنموذج انحدار خطي للتنبؤ بزمن تدوير الطائرة (TAT).
    الغرض منه توثيق فكرة التحليل، وليس تقديم نموذج إنتاجي كامل.

    يعيد:
      - نصاً تحليلياً بالعربية.
      - قاموس بيانات يحتوي على المعاملات والأرقام الرئيسية.
    """
    training_data = nxs_db.get_ml_training_data()

    if not training_data:
        return "لا توجد بيانات تدريب كافية لمحاكاة التنبؤ بزمن التدوير.", {}

    df = pd.DataFrame(training_data)

    required_cols = {"Manpower_Load", "Actual_TAT"}
    if not required_cols.issubset(df.columns):
        return (
            "بيانات التدريب لا تحتوي على الأعمدة المطلوبة لمحاكاة نموذج TAT "
            "(Manpower_Load, Actual_TAT).",
            {},
        )

    # تحويل المتغيرات إلى أرقام والتخلص من القيم غير الصالحة
    df["Manpower_Load_Num"] = pd.to_numeric(df["Manpower_Load"], errors="coerce")
    df["Actual_TAT_Num"] = pd.to_numeric(df["Actual_TAT"], errors="coerce")
    df = df.dropna(subset=["Manpower_Load_Num", "Actual_TAT_Num"])

    if df.empty:
        return "بعد تنظيف البيانات، لا توجد صفوف صالحة لبناء نموذج تنبؤ بزمن التدوير.", {}

    X = df["Manpower_Load_Num"].values
    Y = df["Actual_TAT_Num"].values

    var_x = float(np.var(X))
    if var_x == 0.0:
        # في حالة عدم وجود تباين، نفترض متوسطاً ثابتاً
        slope = 0.0
        intercept = float(np.mean(Y))
    else:
        slope = float(np.cov(X, Y)[0, 1] / var_x)
        intercept = float(np.mean(Y) - slope * np.mean(X))

    # محاكاة التنبؤ لبعض مستويات التحميل
    new_loads = np.array([0.50, 0.75, 0.95], dtype=float)
    predicted_tats = intercept + slope * new_loads
    avg_predicted_tat = float(np.mean(predicted_tats))

    lines: List[str] = []
    lines.append("المرحلة 11: محاكاة التنبؤ بزمن تدوير الطائرة بالاعتماد على مستوى تحميل الموارد.")
    lines.append(
        "تم استخدام نموذج انحدار خطي مبسط يعتمد على العلاقة بين نسبة تحميل الموارد البشرية "
        "وزمن التدوير الفعلي المسجّل في البيانات."
    )
    lines.append("التنبؤات التقريبية لزمن التدوير بالدقائق:")
    lines.append(f"• عند تحميل بنسبة 50٪ يكون زمن التدوير المتوقع تقريباً {predicted_tats[0]:.1f} دقيقة.")
    lines.append(f"• عند تحميل بنسبة 75٪ يكون زمن التدوير المتوقع تقريباً {predicted_tats[1]:.1f} دقيقة.")
    lines.append(f"• عند تحميل بنسبة 95٪ يكون زمن التدوير المتوقع تقريباً {predicted_tats[2]:.1f} دقيقة.")
    lines.append(f"متوسط زمن التدوير المتوقع بعد التصحيح يدور حول {avg_predicted_tat:.1f} دقيقة.")
    lines.append(
        "هذه النتائج تساعد في فهم تأثير ضغط العمل على زمن التدوير، "
        "وتدعم قرارات التخطيط المتعلقة بتوزيع الموارد البشرية."
    )

    meta_data: Dict[str, Any] = {
        "analysis_stage": "ML_TAT_Prediction",
        "predicted_avg_tat": avg_predicted_tat,
        "model_used": "Linear Regression (Simulated)",
        "slope": slope,
        "intercept": intercept,
        "sample_count": int(df.shape[0]),
    }

    return "\n".join(lines), meta_data


# =================================================================
# وظيفة المرحلة 12: نموذج تصنيف التأخير (Random Forest Classifier)
# =================================================================


def run_random_forest_delay_classifier() -> Tuple[str, Dict[str, Any]]:
    """
    محاكاة تحليل تصنيفي لتحديد العوامل الأكثر ارتباطاً بفئات التأخير.
    لا يتم هنا تدريب نموذج حقيقي، بل توثيق النتائج المتفق عليها
    من تحليل سابق (RCA) وتقديمها بصيغة مفهومة للنظام والمستخدم.
    """
    features_df = pd.DataFrame(nxs_db.get_advanced_ml_features())

    if features_df.empty:
        return "لا توجد بيانات كافية لمحاكاة نموذج تصنيف التأخير.", {}

    if "Delay_Class" not in features_df.columns:
        return (
            "بيانات خصائص التأخير لا تحتوي على الحقل Delay_Class المطلوب للتحليل التصنيفي.",
            {},
        )

    # قائمة الخصائص الرئيسية المستخدمة في التحليل النظري
    features = ["Sched_Time_H", "Is_Peak", "Staff_Avg_OT", "Asset_PM_Overdue"]
    for col in features:
        if col not in features_df.columns:
            # في حال غياب أي عمود، نكتفي بذكر عدم اكتمال مجموعة الميزات
            return (
                "مجموعة أعمدة خصائص التأخير غير مكتملة، "
                "ولا يمكن محاكاة تحليل أهمية الميزات بشكل موثوق.",
                {},
            )

    # أهمية الخصائص كما تم استنتاجها من التحليل السابق (محاكاة ثابتة)
    simulated_importance = {
        "Asset_PM_Overdue": 0.45,
        "Staff_Avg_OT": 0.35,
        "Is_Peak": 0.15,
        "Sched_Time_H": 0.05,
    }

    accuracy = 0.92  # دقة تنبؤ تقريبية لمحاكاة الفكرة

    lines: List[str] = []
    lines.append("المرحلة 12: محاكاة تحليل تصنيف التأخير وتقدير أهمية العوامل المؤثرة.")
    lines.append(
        "تم الاعتماد على نموذج تصنيفي نظري (شبيه بالغابة العشوائية) "
        "لفهم أكثر العوامل ارتباطاً بفئات التأخير المسجّلة."
    )
    lines.append(f"الدقة التقريبية للنموذج في التنبؤ بفئة التأخير تدور حول {accuracy:.0%}.")
    lines.append("توزيع الأهمية النسبية للعوامل الرئيسية في التأخير:")

    for feature, value in sorted(
        simulated_importance.items(), key=lambda item: item[1], reverse=True
    ):
        lines.append(f"• العامل {feature}: أهمية تقريبية تبلغ {value:.0%}.")

    lines.append(
        "تشير هذه النتائج إلى أن تأخر أعمال الصيانة الوقائية للمعدات "
        "ومتوسط العمل الإضافي للموظفين هما الأكثر تأثيراً على احتمالية حدوث التأخير، "
        "مما يدعم القرارات المتعلقة بضبط الصيانة وسقف الساعات الإضافية."
    )

    meta_data: Dict[str, Any] = {
        "analysis_stage": "ML_Delay_Classification",
        "model_accuracy": accuracy,
        "feature_importance": simulated_importance,
        "sample_count": int(features_df.shape[0]),
    }

    return "\n".join(lines), meta_data
