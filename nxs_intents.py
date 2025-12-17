# -*- coding: utf-8 -*-
"""
nxs_intents.py
-----------------------------
وحدة مسؤولة عن فهم نية السؤال (Intent) واستخراج الكيانات المهمة
(رقم الموظف، رقم الرحلة، الفترة الزمنية، شركة الطيران، القسم...)
بأسلوب ذكي وسريع يدعم العربية والإنجليزية، ويخدم ثلاثة مسارات رئيسية:

1) محلل عمليات وموارد بشرية فائق الذكاء (HR/Ops Analyst).
2) محامي TCC الاحترافي (TCC Advocate) لتحليل وتأطير المسؤولية.
3) دردشة حرة واقعية عالية الفهم (Free Chat).

هذه الوحدة لا تستدعي نماذج LLM، وإنما توفر طبقة
Rule‑Based سريعة يمكن لطبقة LLM أن تبني فوقها قرارات أعمق.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import re
from datetime import datetime, timedelta


# =====================
# 1. أنماط عامة للكيانات
# =====================

# رقم الموظف: نسمح من 6 إلى 10 خانات لتغطية معظم الأنظمة
EMP_ID_PATTERN = re.compile(r"\b(\d{6,10})\b")

# رقم الرحلة: حروف شركة الطيران + أرقام، مع السماح بمسافة اختيارية
FLIGHT_PATTERN = re.compile(r"\b([A-Z]{2}\s*\d{2,4})\b", re.IGNORECASE)

# كود تأخير (15I, 15F, PD, GL, 2R, 33A ...)
DELAY_CODE_PATTERN = re.compile(r"\b(1[0-9][A-Z]|[A-Z]{1,2}\d?|[0-9]{1,2}[A-Z])\b", re.IGNORECASE)

# مدى تاريخ بصيغة ISO (من 2024-01-01 إلى 2024-01-31)
DATE_RANGE_PATTERN = re.compile(
    r"(?:من|from)\s*(\d{4}-\d{2}-\d{2})\s*(?:إلى|الى|to)\s*(\d{4}-\d{2}-\d{2})"
)

# تاريخ منفرد بصيغة ISO
DATE_SINGLE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


# ================================
# 2. دوال مساعدة للغة والتواريخ
# ================================

def detect_language(text: str) -> str:
    """محاولة بسيطة لتحديد اللغة الغالبة في النص (ar / en)."""
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    latin_chars = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if arabic_chars > latin_chars:
        return "ar"
    if latin_chars > arabic_chars:
        return "en"
    return "mixed"


def _extract_employee_id(text: str) -> Optional[str]:
    m = EMP_ID_PATTERN.search(text)
    return m.group(1) if m else None


def _extract_flight_number(text: str) -> Optional[str]:
    # نعمل على النسخة الكبيرة لتفادي مشاكل الـ case
    m = FLIGHT_PATTERN.search(text.upper())
    if not m:
        return None
    return m.group(1).replace(" ", "").upper()


def _extract_delay_code(text: str) -> Optional[str]:
    m = DELAY_CODE_PATTERN.search(text.upper())
    if not m:
        return None
    return m.group(1).upper()


def _extract_date_range_iso(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    يحاول أولا إيجاد مدى تاريخ بصيغة ISO في النص.
    إن لم يجد، يبحث عن تاريخ منفرد.
    """
    m = DATE_RANGE_PATTERN.search(text)
    if m:
        start, end = m.group(1), m.group(2)
        return start, end

    m_single = DATE_SINGLE_PATTERN.search(text)
    if m_single:
        d = m_single.group(1)
        return d, d

    return None, None


def _extract_relative_date_range(text: str, lang: str) -> Tuple[Optional[str], Optional[str]]:
    """
    دعم بسيط لعبارات مثل:
    - اليوم، أمس، هذا الأسبوع، هذا الشهر، الشهر الماضي
    - today, yesterday, this week, this month, last month
    يعاد التاريخ بصيغة ISO (YYYY-MM-DD).
    """
    t = text.lower()
    today = datetime.utcnow().date()

    def iso(d):
        return d.isoformat()

    # اليوم / today
    if any(kw in t for kw in ["اليوم", "today"]):
        return iso(today), iso(today)

    # أمس / yesterday
    if any(kw in t for kw in ["أمس", "امس", "yesterday"]):
        d = today - timedelta(days=1)
        return iso(d), iso(d)

    # هذا الأسبوع / this week
    if any(kw in t for kw in ["هذا الاسبوع", "هذا الأسبوع", "this week"]):
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return iso(start), iso(end)

    # هذا الشهر / this month
    if any(kw in t for kw in ["هذا الشهر", "this month"]):
        start = today.replace(day=1)
        # بداية الشهر القادم - يوم واحد
        if start.month == 12:
            next_month = start.replace(year=start.year + 1, month=1)
        else:
            next_month = start.replace(month=start.month + 1)
        end = next_month - timedelta(days=1)
        return iso(start), iso(end)

    # الشهر الماضي / last month
    if any(kw in t for kw in ["الشهر الماضي", "last month"]):
        if today.month == 1:
            month = 12
            year = today.year - 1
        else:
            month = today.month - 1
            year = today.year
        start = today.replace(year=year, month=month, day=1)
        if month == 12:
            next_month = start.replace(year=year + 1, month=1)
        else:
            next_month = start.replace(month=month + 1)
        end = next_month - timedelta(days=1)
        return iso(start), iso(end)

    return None, None


def extract_date_range(text: str, lang: str) -> Tuple[Optional[str], Optional[str]]:
    """
    يحاول أولاً قراءة مدى ISO من النص، ثم العبارات النسبية.
    """
    start, end = _extract_date_range_iso(text)
    if start and end:
        return start, end

    return _extract_relative_date_range(text, lang)


def normalize(text: str) -> str:
    """
    تبسيط النص ليتناسب مع البحث عن الكلمات المفتاحية.
    """
    t = text.strip().lower()
    # توحيد بعض الحروف العربية
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي").replace("ة", "ه")
    return t


# ===================================================
# 3. تعريف النوايا (Intents) حسب المجالات الوظيفية
# ===================================================

def classify_intent(message: str) -> Dict[str, Any]:
    """
    يرجّع قاموس غني فيه:
      - intent      : نوع الطلب (مثلاً employee_profile, flight_analysis, tcc_defense, free_chat)
      - confidence  : درجة الثقة التقريبية (0–1)
      - language    : ar / en / mixed
      - employee_id : إن وُجد
      - flight_number
      - delay_code
      - airline
      - department
      - date_from / date_to
      - raw_text    : النص الأصلي
      - entities    : قاموس كامل بجميع الكيانات المستخرجة

    ملاحظة: هذا مصنف Rule‑Based سريع، مصمم ليكمّل عمل طبقة LLM
    وليس بديلاً عنها. يمكن للـ LLM أن يأخذ النتيجة كمدخل لتحسين التحليل.
    """

    text = message or ""
    raw_text = text
    lang = detect_language(text)
    norm = normalize(text)

    employee_id = _extract_employee_id(text)
    flight_number = _extract_flight_number(text)
    delay_code = _extract_delay_code(text)
    date_from, date_to = extract_date_range(text, lang)

    airline: Optional[str] = None
    department: Optional[str] = None

    # شركات الطيران الأساسية (يمكن توسيعها لاحقاً)
    if any(kw in norm for kw in ["السعوديه", "saudia", "الخطوط السعوديه", "sv "]):
        airline = "Saudia"
    elif any(kw in norm for kw in ["فلاي ناس", "flynas", "xy "]):
        airline = "Flynas"
    elif any(kw in norm for kw in ["فلاي اديل", "flyadeal", "fd "]):
        airline = "Flyadeal"

    # أقسام TCC
    if any(kw in norm for kw in ["tcc", "مراقبه الحركه"]):
        department = "TCC"
    elif "fic" in norm:
        department = "FIC"
    elif "load control" in norm or "lc " in norm:
        department = "LC"

    # ======================
    # منطق تحديد الـ Intent
    # ======================

    intent = "free_chat"
    confidence = 0.35

    # ---- 3.1 أسئلة تعريفية / مساعدة النظام ----
    if any(kw in norm for kw in ["من انت", "ما هي قدراتك", "وش تسوي", "how do you work", "what can you do"]):
        intent = "system_help"
        confidence = 0.95

    # ---- 3.2 نوايا الموظف (HR) ----
    if any(kw in norm for kw in ["من هو الموظف", "بطاقه الموظف", "profile", "ملف الموظف"]):
        intent = "employee_profile"
        confidence = 0.9
    elif any(kw in norm for kw in ["عمل اضافي", "overtime", "ساعات اضافه"]):
        intent = "employee_overtime"
        confidence = 0.9
    elif any(kw in norm for kw in ["تأخيرات الموظف", "سجل التأخير", "delay record"]) or (
        "تأخير" in norm and employee_id
    ):
        intent = "employee_delay_record"
        confidence = 0.9
    elif "غياب" in norm or "absence" in norm:
        intent = "employee_absence"
        confidence = 0.9
    elif "اجازه مرضيه" in norm or "sick leave" in norm:
        intent = "employee_sick_leave"
        confidence = 0.85
    elif any(kw in norm for kw in ["تحقيق", "investigation"]) and (employee_id or "id" in norm):
        intent = "employee_investigation"
        confidence = 0.9
    elif any(kw in norm for kw in ["تقييم الموظف", "اداء الموظف", "employee performance", "score"]):
        intent = "employee_performance_overview"
        confidence = 0.8

    # ---- 3.3 نوايا الرحلات / التأخيرات ----
    if "رحله" in norm or "flight" in norm or flight_number:
        # تفاصيل تأخير رحلة محددة
        if any(kw in norm for kw in ["سبب تأخير", "سبب تاخير", "why delayed", "delay reason"]):
            intent = "flight_delay_detail"
            confidence = 0.92
        # تحليل مسؤولية TCC عن الرحلة
        elif any(kw in norm for kw in ["هل tcc مسؤول", "مسؤوليه tcc", "دافع عن tcc", "defend tcc"]):
            intent = "tcc_defense_flight"
            confidence = 0.95
        # تحليل MGT / المناولة
        elif any(kw in norm for kw in ["mgt", "مناوله", "turnaround", "وقت التجهيز"]):
            intent = "flight_mgt_compliance"
            confidence = 0.9
        else:
            # تحليل عام للرحلة (حتى بدون تأخير)
            if confidence < 0.8:
                intent = "flight_analysis"
                confidence = max(confidence, 0.75)

    # ---- 3.4 إحصائيات التأخير / الشركات / الأقسام ----
    if any(kw in norm for kw in ["اكثر سبب للتأخير", "اكثر سبب للتاخير", "top delay reason", "top delay reasons"]):
        intent = "delay_root_cause_statistics"
        confidence = 0.93
    elif any(kw in norm for kw in ["اكثر شركه تاخير", "اكثر شركة تأخير", "most delayed airline"]):
        intent = "delay_by_airline_statistics"
        confidence = 0.92
    elif any(kw in norm for kw in ["تقرير التأخيرات", "احصائيات التأخير", "delay statistics"]):
        intent = "delay_overview_statistics"
        confidence = max(confidence, 0.85)

    # ---- 3.5 نوايا المناوبات / التقارير التشغيلية ----
    if any(kw in norm for kw in ["تقرير المناوبه", "تقرير المناوبة", "shift report", "تقرير الشفت"]):
        intent = "shift_report_summary"
        confidence = 0.9
    elif any(kw in norm for kw in ["اداء المناوبه", "اداء الشفت", "shift performance"]):
        intent = "shift_performance_analysis"
        confidence = 0.88
    elif any(kw in norm for kw in ["تقرير tcc", "اداء tcc", "tcc statistics"]):
        intent = "tcc_operations_overview"
        confidence = 0.9

    # ---- 3.6 تحليلات عامة / داشبورد ----
    if any(kw in norm for kw in ["تحليل", "dashboard", "لوحه تحكم", "dashbord"]):
        # إن كان هناك موظف أو رحلة، سيبقى intent المتخصص كما هو
        if intent == "free_chat" or confidence < 0.8:
            intent = "analytics_request"
            confidence = max(confidence, 0.8)

    # ---- 3.7 دردشة حرة واعية ----
    # إذا لم نتعرف على نمط واضح، نُبقي على free_chat بثقة متوسطة
    if intent == "free_chat" and confidence < 0.5:
        confidence = 0.5

    entities: Dict[str, Any] = {
        "employee_id": employee_id,
        "flight_number": flight_number,
        "delay_code": delay_code,
        "airline": airline,
        "department": department,
        "date_from": date_from,
        "date_to": date_to,
        "language": lang,
    }

    return {
        "intent": intent,
        "confidence": round(confidence, 3),
        "language": lang,
        "employee_id": employee_id,
        "flight_number": flight_number,
        "start_date": date_from,
        "end_date": date_to,
        "delay_code": delay_code,
        "airline": airline,
        "department": department,
        "raw_text": raw_text,
        "entities": entities,
    }


# ======================
# 4. اختبار يدوي بسيط
# ======================

if __name__ == "__main__":
    tests = [
        "من هو الموظف الذي رقمه الوظيفي 15013814؟",
        "اعرض تأخيرات الموظف 15013814 من 2024-12-31 إلى 2025-01-31",
        "ما سبب تأخير الرحلة SV123 أمس؟",
        "هل TCC مسؤول عن تأخير الرحلة SV 456؟ دافع عن TCC.",
        "أكثر سبب للتأخير خلال الشهر الماضي؟",
        "اعطني تقرير المناوبة لقسم مراقبة الحركة اليوم.",
        "حلل اداء TCC خلال هذا الشهر لشركة السعودية فقط.",
        "hi, what can you do for me?",
        "just talk with me about airport operations in general.",
    ]
    for t in tests:
        print("Q:", t)
        print("→", classify_intent(t))
        print("-" * 60)
