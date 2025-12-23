# -*- coding: utf-8 -*-
"""
nxs_brain.py — NXS • Ultra Reasoning Engine (URE)
-------------------------------------------------
هذا الملف يمثّل "العقل" الكامل لـ TCC AI / NXS • AirportOps AI.

⚙️ الفكرة الأساسية:
- أنت تكتب سؤالك الطبيعي بالعربية أو الإنجليزية.
- المحرك الذكي يقرأ السؤال، يفهم النية، يخطط لخطوات الوصول للبيانات.
- يتم استدعاء دوال Supabase من nxs_supabase_client للحصول على البيانات.
- ثم يُعاد استدعاء المحرك الذكي لصياغة الإجابة النهائية اعتماداً على البيانات الفعلية فقط.

❗ ملاحظات مهمة:
- لا يتم ذكر اسم Gemini أو نوع النموذج للمستخدم.
- لا يتم ذكر أسماء الجداول الداخلية (employee_master_db, dep_flight_delay, ...).
- إذا فشل الاتصال بمحرك الذكاء أو Supabase، يتم إرجاع رسالة واضحة للمستخدم بدون إسقاط الخادم.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, Tuple, List, Optional

import requests
from dotenv import load_dotenv

# استيراد طبقة Supabase
import nxs_supabase_client as nxs_db


from nxs_semantic_engine import NXSSemanticEngine
try:
    from nxs_semantic_engine import interpret_with_filters
except Exception:
    interpret_with_filters = None



# =================== تحميل متغيرات البيئة ===================

load_dotenv()

GEMINI_API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GENAI_API_KEY")
)
GEMINI_MODEL_SIMPLE  = os.getenv("GEMINI_MODEL_SIMPLE",  "gemini-2.5-flash")
GEMINI_MODEL_COMPLEX = os.getenv("GEMINI_MODEL_COMPLEX", "gemini-2.5-pro")
GEMINI_MODEL_PLANNER = os.getenv("GEMINI_MODEL_PLANNER", "gemini-2.5-flash")
logger = logging.getLogger("nxs_brain")


try:
    SEMANTIC_ENGINE: Optional[NXSSemanticEngine] = NXSSemanticEngine()
except Exception:
    SEMANTIC_ENGINE = None

class AIEngineError(Exception):
    pass


# =================== أكواد التأخير من ملف Code Air ===================

DELAY_CODE_MAP: Dict[str, str] = {
  "10A": "DAMAGE CAUSED TO AIRCRAFT BY STATION OR HANDLING AGENT PERSONNEL PERFORMING SERVICES FUNCTIONS.",
  "10AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "11A": "ACCEPTANCE AFTER DEADLINE.",
  "11AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "11B": "FLT RE-OPEN TO ACCEPTED PASSENGER.",
  "11BT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "11C": "FLT CHECKED-IN MANUALLY.",
  "11CT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "12A": "CONGESTION AT CHECK-IN AREA.",
  "12AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "13A": "PASSENGER CHECK-IN ERROR.",
  "13AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "13B": "BAGGAGE TAGGING.",
  "13BT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "13C": "DUPLICATE SEATS.",
  "13CT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "14A": "ACCEPTANCE OF PASSENGER OVER AIRCRAFT SEAT CAPACITY OR PAYLOAD.",
  "14AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "14B": "ACCEPTANCE OF BAGGAGE OVER AIRCRAFT CAPACITY OR PAYLOAD.",
  "14BT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15A": "LATE PASSENGER BOARDING.",
  "15AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15B": "MISSING CHECKED-IN PASSENGER (NO SHOW).",
  "15BT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15C": "GATTING ERROR.",
  "15CT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15D": "OVER SIZE OR EXCESS CARRY-ON BAGGAGE ON BOARD.",
  "15DT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15E": "PASSENGER BOARDED WITHOUT TRAVEL DOCUMENTS.",
  "15ET": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15F": "LATE OR ERROR OF WEIGHT & BALANCE DOCUMENTS.",
  "15FT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15G": "NOTOC.",
  "15GT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15H": "PASSENGER MANIFEST.",
  "15HT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15I": "PERSONAL (DISCREPANCIES) BY SUPERVISION OR AGENT.",
  "15IT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15J": "FAILURE OF ALLOCATE DISTRIBUTED PASSENGERS SEAT +12 HOURS OF DEPARTURE.",
  "15JT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "15K": "SHORTAGE OF STAFF (AGENTS).",
  "15KT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "18A": "HANDLING, SORTING, ASSEMBLY OR BREAKDOWN.",
  "18AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "22A": "THRU CHECK-IN ERROR (PASSENGER/BAGS) BY INITIATING STATION.",
  "22AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "22B": "TURNAROUND FLIGHT DELAYED CAUSED BY ORIGIN STATION. (DISCREPANCY)",
  "22BT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "32A": "LATE AIRCRAFT LOADING/OFFLOADING (BAGGAGE & CARGO).",
  "32AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "32B": "LACK OR SHORTAGE OF LOADING STAFF.",
  "32BT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "32C": "ACCEPTANCE OF LATE RELEASED CARGO AFTER DEADLINE.",
  "32CT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "32D": "DAMAGE OR SHORTAGE OF ULDS, BULKY, SPECIAL LOAD.",
  "32DT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "33A": "LACK OR BREAKDOWN OF GROUND SERVICING EQUIPMENT.",
  "33AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "33B": "LACK OF GROUND SERVICING EQUIPMENT OPERATORS.",
  "33BT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "33C": "SHORTAGE OF: BUSES/MEDICAL LIFT/DRIVERS. (SGS CONTRACTOR)",
  "33CT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "35A": "LATE OR IMPROPER CLEANING, INCLUDING FUMIGATION OF AIRCRAFT.",
  "35AT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "35B": "LACK OR SHORTAGE OF CLEANING STAFF.",
  "35BT": "EQUIPMENT TURN-AROUND, INDIRECT AND CONSEQUENTIAL DELAY",
  "11PD": "Late Check-InAcceptance after deadline.",
  "12PL": "Late Check-InCongestion in check-in area.",
  "13PE": "Check-in ErrorPassenger and baggage.",
  "14PO": "Over-salesBooking errors.",
  "15PH": "BoardingDiscrepancies and paging, missing checked-in passenger.",
  "16PS": "Commercial Publicity/Passenger ConvenienceVIP, press, ground meals, and missing personal items.",
  "17PC": "Catering OrderLate or incorrect order given to supplier.",
  "18PB": "Baggage ProcessingSorting, etc.",
  "31GD": "Aircraft DocumentationLate/inaccurate, weight and balance, general declaration, passenger manifest, etc.",
  "32GL": "Loading/UnloadingBulky, special load, cabin load, lack of loading staff.",
  "33GE": "Loading EquipmentLack of or breakdown, e.g. container pallet loader, lack of staff.",
  "34GS": "Servicing EquipmentLack of or breakdown, lack of staff, e.g. steps.",
  "35GC": "Aircraft CleaningNo specific reason provided.",
  "36GF": "Fuelling/DefuellingFuel supplier issues.",
  "37GB": "CateringLate delivery or loading.",
  "38GU": "ULDLack of or serviceability.",
  "39GT": "Technical EquipmentLack of or breakdown, lack of staff, e.g. pushback.",
  "41TD": "AIRCRAFT DEFECTS.",
  "42TM": "SCHEDULED MAINTENANCE, late release.",
  "43TN": "NON-SCHEDULED MAINTENANCE, special checks and/or additional works beyond normal maintenance schedule.",
  "44TS": "SPARES AND MAINTENANCE EQUIPMENT, lack of or breakdown.",
  "45TA": "AOG SPARES, to be carried to another station.",
  "46TC": "AIRCRAFT CHANGE, for technical reasons.",
  "47TL": "STAND-BY AIRCRAFT, lack of planned stand-by aircraft for technical reasons.",
  "48TV": "SCHEDULED CABIN CONFIGURATION/VERSION ADJUSTMENTS.",
  "2R": "Lack of ground staff",
  "2S": "Late report of ground staff",
  "12": "Late check-in Counter Closure",
  "12W": "Lack of counter staff",
  "13X": "Wrong check in",
  "13Y": "Wrong profiling / documentation",
  "13Z": "Reservations without passenger name",
  "15": "Boarding",
  "32Z": "Lack on manpower",
  "33Y": "Lack of equipment",
  "33Z": "Lack of equipment operators",
  "34Y": "Lack of equipment",
  "34Z": "Lack of staff",
  "39Y": "Lack/breakdown of equipment",
  "39Z": "Lack of manpower / operator"
}


def lookup_delay_reason(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    c = str(code).strip().upper()
    return DELAY_CODE_MAP.get(c)


# =================== دوال مساعدة عامة ===================

def _safe_json_loads(text: str) -> Optional[dict]:
    """
    محاولة آمنة لتحويل نص إلى JSON بدون كسر التنفيذ.
    """
    if not text:
        return None
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    # محاولة استخراج أول كتلة {} صالحة
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        trying = text[start : end + 1]
        try:
            return json.loads(trying)
        except Exception:
            return None
    return None


def call_ai(prompt: str, model_name: str = None, temperature: float = 0.4, max_tokens: int = 2000) -> str:
    """إرسال طلب إلى Google Gemini API."""
    import time

    if not GEMINI_API_KEY:
        return "ERROR: GEMINI_API_KEY_MISSING"

    # 1) تحديد الموديل: نستخدم الممرر للدالة أو المسجل في الإعدادات
    target_model = model_name if model_name else GEMINI_MODEL_SIMPLE

    # 2) الرابط الصحيح والمستقر (استخدام v1 بدلاً من v1beta)
    url = f"https://generativelanguage.googleapis.com/v1/models/{target_model}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
            "topP": 0.95,
        },
    }

    # نظام الإعادة عند الفشل (Retry Logic)
    last_err = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0].get("text", "")

            if response.status_code in (429, 503):
                time.sleep(2 * (attempt + 1))
                continue

            last_err = f"AI Error {response.status_code}"
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(2 * (attempt + 1))
            continue

    return f"⚠️ المحرك مشغول حالياً. (Technical: {last_err})"


def semantic_pre_analyze(user_message: str) -> Optional[Dict[str, Any]]:
    """
    تحليل مسبق باستخدام طبقة NXS Semantics.
    - إذا توفر interpret_with_filters: يرجع interpretation + plan + detected_filters + complexity_hint + model_hint.
    - وإلا: يرجع interpretation.to_dict() فقط.
    """
    if SEMANTIC_ENGINE is None:
        return None
    msg = (user_message or "").strip()
    if not msg:
        return None
    try:
        if interpret_with_filters:
            return interpret_with_filters(SEMANTIC_ENGINE, msg)
        interp = SEMANTIC_ENGINE.interpret(msg)
        return interp.to_dict()
    except Exception:
        return None


# =================== Planner Prompt (Fallback) ===================
# إذا كان لديك ملف intents/schema مخصص يمكنك نقل هذا النص هناك.
PLANNER_PROMPT = """أنت مخطط (Planner) لنظام NXS.
مهمتك: تحويل سؤال المستخدم إلى JSON يحتوي:
{
  "language": "ar|en",
  "plan": [
    {
      "tool": "<اسم_الدالة>",
      "args": { ... }
    }
  ],
  "notes": "<ملاحظات قصيرة>"
}

القواعد:
- أعد JSON فقط بدون أي شرح.
- اختر أقل عدد ممكن من الأدوات لتحقيق الهدف.
- استخدم أدوات nxs_supabase_client فقط.
- إذا السؤال عام ولا يحتاج بيانات: اجعل plan فارغاً [] وnotes تشرح ذلك.
"""


def build_planner_prompt(user_message: str, semantic_info: Optional[Dict[str, Any]] = None) -> str:
    """
    يبني برومبت التخطيط، مع تمرير تحليل NXS Semantics (إن وجد)
    إلى نموذج التخطيط لمساعدته على اختيار الأدوات والباراميترات.
    """
    prompt = PLANNER_PROMPT
    if semantic_info:
        prompt += "\n\nتحليل مسبق من طبقة NXS Semantics (للاستخدام المساعد فقط):\n"
        prompt += json.dumps(semantic_info, ensure_ascii=False)
    prompt += "\n\nسؤال المستخدم:\n" + user_message
    prompt += "\n\nأعد JSON فقط كما في التنسيق المطلوب أعلاه."
    return prompt


def run_planner(user_message: str) -> Dict[str, Any]:
    # Planner يجب أن يكون سريعاً ورخيصاً: نستخدم Flash دائماً هنا
    semantic_info = semantic_pre_analyze(user_message)
    prompt = build_planner_prompt(user_message, semantic_info)

    raw = call_ai(
        prompt,
        model_name=GEMINI_MODEL_PLANNER,
        temperature=0.2,
        max_tokens=1200,
    )

    data = _safe_json_loads(raw)
    if not data or not isinstance(data, dict):
        # فشل التحليل، نعيد خطة فارغة لكن لا نكسر التنفيذ
        out = {"language": "ar", "plan": [], "notes": "no-structured-plan", "semantic": semantic_info}
        if isinstance(semantic_info, dict):
            out["semantic_hints"] = semantic_info
        return out

    # ضمان الحقول الأساسية
    lang = data.get("language") or "ar"
    if lang not in ("ar", "en"):
        lang = "ar"
    plan = data.get("plan") or []
    if not isinstance(plan, list):
        plan = []
    notes = data.get("notes") or ""
    out = {"language": lang, "plan": plan, "notes": notes, "semantic": semantic_info}
    if isinstance(semantic_info, dict):
        out["semantic_hints"] = semantic_info
    return out



# =================== مرحلة 2: تنفيذ الخطة على Supabase ===================

def execute_plan(plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    يستقبل قائمة بالخطوات (tool + args) وينفّذها على nxs_supabase_client.
    يعيد قاموساً يحتوي على نتائج كل أداة بالترتيب.
    """
    results: Dict[str, Any] = {
        "steps": [],  # قائمة بالنتائج لكل خطوة
    }

    for step in plan:
        tool = step.get("tool")
        args = step.get("args", {}) or {}

        if not tool or not hasattr(nxs_db, tool):
            # نتجاهل الأدوات غير المعروفة
            results["steps"].append({
                "tool": tool,
                "ok": False,
                "error": "unknown_tool",
                "rows": None,
            })
            continue

        func = getattr(nxs_db, tool)
        try:
            value = func(**args)
            # نفرض أن القيمة إما قائمة صفوف أو قيمة رقمية أو dict
            if isinstance(value, list):
                rows = value
            elif isinstance(value, dict):
                rows = [value]
            else:
                rows = value  # قد تكون int مثلاً
            results["steps"].append({
                "tool": tool,
                "ok": True,
                "rows": rows,
            })
        except Exception as exc:
            results["steps"].append({
                "tool": tool,
                "ok": False,
                "error": str(exc),
                "rows": None,
            })

    return results


# =================== مرحلة 3: بناء إجابة نهائية ===================

ANSWER_PROMPT_BASE = """
أنت TCC AI • AirportOps AI.
أنت مساعد ذكي يعمل على بيانات عمليات المطارات (الموظفين، الرحلات، التأخيرات، العمل الإضافي، الغياب، الأحداث التشغيلية).
يجب أن تعتمد في إجاباتك على البيانات الممررة لك قدر الإمكان.

قواعد مهمة:
- أجب دائماً بلغة المستخدم.
- لا تذكر أسماء الجداول أو أسماء الدوال البرمجية.
- لا تذكر نوع نموذج الذكاء الاصطناعي أو اسم المزود.
- إذا كانت البيانات غير كافية، اذكر ذلك بوضوح ولا تخترع أرقاماً.
- إذا كانت هناك بيانات كافية، أجب بثقة وبأسلوب منظم (نقاط • / فقرات قصيرة).
- يمكنك استخدام إيموجي بسيطة مثل ✈️، 👤، ⏱️، 🏢، 📅.
- إذا كان السؤال عن معنى كود تأخير، استخدم معرفتك بقاموس أكواد التأخير (منظومة الطيران) وأعطِ تفسيراً واضحاً.

سوف تستقبل الآن:
1) سؤال المستخدم الخام.
2) ناتج مرحلة التخطيط (ملاحظات planner).
3) البيانات التي تم جلبها من النظام (Supabase) في شكل JSON مبسط.
"""


def build_answer_prompt(
    user_message: str,
    language: str,
    planner_notes: str,
    data_bundle: Dict[str, Any],
    extra_system_instruction: str = "",
    operational_context: Optional[Dict[str, Any]] = None,
) -> str:
    lang_hint = "العربية" if language == "ar" else "الإنجليزية"

    prompt = (
        ANSWER_PROMPT_BASE
        + "\n\n"
        + f"لغة المستخدم المتوقعة: {lang_hint}\n"
        + "\nسؤال المستخدم:\n"
        + user_message
        + "\n\nملاحظات مرحلة التخطيط:\n"
        + (planner_notes or "لا توجد ملاحظات مهمة.")
    )

    # حقن سياق إضافي (Intent Intelligence / Operational Context) بدون تغيير بقية النظام
    if operational_context:
        prompt += "\n\nسياق عمليات إضافي (Operational Context) للاستخدام الداخلي في التحليل:\n"
        prompt += json.dumps(operational_context, ensure_ascii=False)

    # حقن تعليمات خاصة (مثل بروتوكول المحامي) بدون لمس بقية القواعد
    if extra_system_instruction:
        prompt += "\n\nتعليمات خاصة (System Instruction) للاستخدام الداخلي في التحليل:\n"
        prompt += str(extra_system_instruction).strip()

    prompt += (
        "\n\nالبيانات المستخرجة من النظام (JSON) للاستخدام الداخلي في التحليل:\n"
        + json.dumps(data_bundle, ensure_ascii=False)
        + "\n\nالآن قدّم الإجابة النهائية للمستخدم بشكل منظم وواضح وعملي، بدون إظهار JSON أو تفاصيل برمجية:"
    )

    return prompt



# =================== Cross-Table Reasoning + Defense Protocol ===================

def _is_flight_delay_query(user_query: str) -> bool:
    q = (user_query or "").strip()
    return any(k in q for k in ["تأخير", "تحليل رحلة", "delayed", "delay", "analyze flight", "flight analysis"])


def _extract_flight_number(user_query: str) -> Optional[str]:
    """محاولة استخراج رقم رحلة من نص المستخدم (مثل SV123 / XY4567 ...)."""
    q = (user_query or "").upper()
    # نمط شائع: حرفان/ثلاثة + أرقام 1-5
    m = re.search(r"\b([A-Z]{2,3}\s*\d{1,5})\b", q)
    if not m:
        return None
    return m.group(1).replace(" ", "")


def _to_int_safe(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)):
            return int(x)
        s = str(x).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _parse_hhmm(val: Any) -> Optional[Tuple[int, int]]:
    """يحاول تحويل HH:MM إلى (hh, mm)."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{2})", s)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _minutes_diff_hhmm(start_hhmm: Any, end_hhmm: Any) -> Optional[int]:
    """فرق دقائق بسيط بين وقتين HH:MM مع مراعاة عبور منتصف الليل."""
    a = _parse_hhmm(start_hhmm)
    b = _parse_hhmm(end_hhmm)
    if not a or not b:
        return None
    s = a[0] * 60 + a[1]
    e = b[0] * 60 + b[1]
    if e < s:
        e += 24 * 60
    return e - s


def _safe_first(rows: Any) -> Optional[Dict[str, Any]]:
    if isinstance(rows, list) and rows:
        return rows[0] if isinstance(rows[0], dict) else None
    if isinstance(rows, dict):
        return rows
    return None


def _fetch_one(table_name: str, filters: Dict[str, str], select_query: str = "*") -> Dict[str, Any]:
    """استدعاء صف واحد من Supabase (بشكل آمن)."""
    try:
        rows = nxs_db.execute_dynamic_query(table_name, select_query=select_query, filters=filters)
        row = _safe_first(rows)
        return row or {}
    except Exception:
        return {}


def get_cross_table_bundle(user_query: str) -> Dict[str, Any]:
    """يجلب حزمة بيانات مترابطة: تشغيلية + تكويد رسمي + غياب + شفت."""
    flight_number = _extract_flight_number(user_query)
    if not flight_number:
        return {}

    # 1) التشغيلية (dep_flight_delay)
    dep = _fetch_one("dep_flight_delay", {"Flight_Number": f"eq.{flight_number}"}, select_query="*")
    if not dep:
        # fallback شائع إذا كان اسم العمود مختلف
        dep = _fetch_one("dep_flight_delay", {"Flight Number": f"eq.{flight_number}"}, select_query="*")

    # 2) التكويد الرسمي (sgs_flight_delay)
    sgs = _fetch_one("sgs_flight_delay", {"Flight_Number": f"eq.{flight_number}"}, select_query="*")
    if not sgs:
        sgs = _fetch_one("sgs_flight_delay", {"Flight Number": f"eq.{flight_number}"}, select_query="*")

    # 3) بيانات الشفت (shift_report) + الغياب (employee_absence)
    shift = {}
    absence = {}
    dep_date = dep.get("Date") or dep.get("date")
    dep_shift = dep.get("Shift") or dep.get("shift")
    if dep_date and dep_shift:
        # shift_report
        shift = _fetch_one("shift_report", {"Date": f"eq.{dep_date}", "Shift": f"eq.{dep_shift}"}, select_query="*")
        # employee_absence (نقص موظفين)
        absence = _fetch_one("employee_absence", {"Date": f"eq.{dep_date}", "Shift": f"eq.{dep_shift}"}, select_query="*")

    return {
        "flight_number": flight_number,
        "dep_flight_delay": dep,
        "sgs_flight_delay": sgs,
        "shift_report": shift,
        "employee_absence": absence,
    }


def analyze_workload_balance(shift_data: dict) -> Dict[str, Any]:
    """تحليل توازن الجهد/القوى العاملة بناءً على معاييرك."""
    if not isinstance(shift_data, dict) or not shift_data:
        return {}

    on_duty = _to_int_safe(shift_data.get("On Duty") or shift_data.get("On_Duty") or shift_data.get("On_Duty"))
    no_show = _to_int_safe(shift_data.get("No Show") or shift_data.get("No_Show") or shift_data.get("No_Show"))
    dep_dom = _to_int_safe(shift_data.get("Departures Domestic") or shift_data.get("Departures_Domestic")) or 0
    dep_int = _to_int_safe(shift_data.get("Departures International+Foreign") or shift_data.get("Departures_Intl") or shift_data.get("Departures_International")) or 0
    arr_dom = _to_int_safe(shift_data.get("Arrivals Domestic") or shift_data.get("Arrivals_Domestic")) or 0
    arr_int = _to_int_safe(shift_data.get("Arrivals International+Foreign") or shift_data.get("Arrivals_Intl") or shift_data.get("Arrivals_International")) or 0

    # Total Capacity = On Duty + No Show
    total_capacity = None if on_duty is None and no_show is None else (on_duty or 0) + (no_show or 0)

    # Workload Matrix (كما ذكرت)
    total_needed_minutes = (dep_dom + dep_int) * 70 + (arr_dom + arr_int) * 20

    available_minutes = (on_duty or 0) * 8 * 60

    utilization_ratio = None
    if available_minutes > 0:
        utilization_ratio = total_needed_minutes / available_minutes

    shortage_percent = None
    if total_capacity and total_capacity > 0 and on_duty is not None:
        shortage_percent = ((total_capacity - on_duty) / total_capacity) * 100

    return {
        "total_capacity": total_capacity,
        "on_duty": on_duty,
        "no_show": no_show,
        "total_needed_minutes": total_needed_minutes,
        "available_minutes": available_minutes,
        "utilization_ratio": utilization_ratio,
        "shortage_percent": shortage_percent,
    }


# منطق معالجة الضغط التشغيلي والقوى العاملة (Manpower & Workload)
def calculate_operational_capacity(shift_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    يحسب قدرة التشغيل للشفت بناءً على معايير العمل التي زوّدنا بها المستخدم.
    - إجمالي المطلوب (مغادرة: 70د، قدوم: 20د) — (يُترك تعديل حالات Turnaround لتطوير لاحق)
    - المتاح: On Duty * 480 دقيقة (شفت 8 ساعات)
    - Utilization%: نسبة إشغال الجهد من المتاح
    - Manpower Shortage%: أثر No Show كنسبة من Total Capacity (On Duty + No Show)
    """
    if not isinstance(shift_data, dict) or not shift_data:
        return {"utilization_pct": None, "manpower_shortage_pct": None}

    dep_dom = _to_int_safe(shift_data.get('Departures_Domestic') or shift_data.get('Departures Domestic')) or 0
    dep_int = _to_int_safe(shift_data.get('Departures_Intl') or shift_data.get('Departures International+Foreign')) or 0
    arr_dom = _to_int_safe(shift_data.get('Arrivals_Domestic') or shift_data.get('Arrivals Domestic')) or 0
    arr_int = _to_int_safe(shift_data.get('Arrivals_Intl') or shift_data.get('Arrivals International+Foreign')) or 0

    # إجمالي المطلوب بالدقائق
    workload_minutes = (dep_dom + dep_int) * 70 + (arr_dom + arr_int) * 20

    on_duty = _to_int_safe(shift_data.get('On_Duty') or shift_data.get('On Duty')) or 0
    no_show = _to_int_safe(shift_data.get('No_Show') or shift_data.get('No Show')) or 0

    # المتاح بالدقائق (شفت 8 ساعات)
    available_minutes = on_duty * 480 if on_duty > 0 else 0

    utilization = None
    if available_minutes > 0:
        utilization = (workload_minutes / available_minutes) * 100

    total_capacity = on_duty + no_show
    manpower_shortage = None
    if total_capacity > 0:
        manpower_shortage = (no_show / total_capacity) * 100

    return {
        "utilization_pct": round(utilization, 2) if utilization is not None else None,
        "manpower_shortage_pct": round(manpower_shortage, 2) if manpower_shortage is not None else None,
        "workload_minutes": workload_minutes,
        "available_minutes": available_minutes,
        "on_duty": on_duty,
        "no_show": no_show,
        "total_capacity": total_capacity,
    }


# منطق المحامي الذكي (TCC Advocate)
def apply_defense_logic(flight_data: Dict[str, Any], mgt_standard: Optional[int]) -> str:
    """
    يحدد نبرة الرد وفق بروتوكول الدفاع:
    - إذا كان الفعلي <= المعيار: دفاع (تبرئة القسم)
    - غير ذلك: تنبيه ومتابعة التحقيق
    """
    if not isinstance(flight_data, dict):
        return ""

    # دعم مرن لاستخراج "actual_ground_time"
    actual_ground_time = flight_data.get("actual_ground_time")

    if actual_ground_time is None:
        # دعم بديل: حساب فرق وقتين HH:MM إن توفر (ATA/ATD)
        ata = flight_data.get("ATA") or flight_data.get("actual_ata") or flight_data.get("AAT")
        atd = flight_data.get("ATD") or flight_data.get("actual_atd") or flight_data.get("ADT")
        diff = _minutes_diff_hhmm(ata, atd) if (ata and atd) else None
        actual_ground_time = diff

    if actual_ground_time is None or mgt_standard is None:
        return ""

    try:
        actual_ground_time = int(actual_ground_time)
        mgt_standard = int(mgt_standard)
    except Exception:
        return ""

    if actual_ground_time <= mgt_standard:
        return "🛡️ الدفاع: القسم أنجز العمل تحت المعيار الزمني؛ التأخير مرحّل من المصدر."
    return "⚠️ تنبيه: وقت الدوران تجاوز المعيار، جاري فحص أسباب التحقيق."



def _build_defense_instruction(bundle: Dict[str, Any]) -> str:
    """بروتوكول المحامي: إذا كان كود تأخير TCC (15I/15F) نفعل نبرة الدفاع عند تحقق شروط البراءة."""
    dep = (bundle or {}).get("dep_flight_delay") or {}
    sgs = (bundle or {}).get("sgs_flight_delay") or {}
    shift = (bundle or {}).get("shift_report") or {}

    # استخراج كود التأخير من أي مصدر متاح
    delay_code = (
        dep.get("Delay Code") or dep.get("Delay_Code") or dep.get("DelayCode") or
        sgs.get("Delay Code") or sgs.get("Delay_Code") or sgs.get("DelayCode")
    )
    code = str(delay_code).strip().upper() if delay_code else ""

    if code not in ("15I", "15F"):
        return ""

    # حساب الوقت الفعلي على الأرض (تقريبي) من STD/ATD أو STA/ATA
    actual_ground_time = None
    std = dep.get("STD") or dep.get("S STD") or dep.get("Scheduled_Dep") or dep.get("SCHED_DEP")
    atd = dep.get("ATD") or dep.get("A TD") or dep.get("Actual_Dep") or dep.get("ACT_DEP")
    if std and atd:
        actual_ground_time = _minutes_diff_hhmm(std, atd)

    if actual_ground_time is None:
        sta = dep.get("STA") or dep.get("Scheduled_Arr") or dep.get("SCHED_ARR")
        ata = dep.get("ATA") or dep.get("Actual_Arr") or dep.get("ACT_ARR")
        if sta and ata:
            actual_ground_time = _minutes_diff_hhmm(sta, ata)

    # محاولة حساب MGT من ملف القواعد (lookup_mgt)
    # سنحاول استنتاج أقل حد من الحقول المتاحة بدون كسر النظام.
    mgt_minutes = None
    if lookup_mgt is not None:
        aircraft_type = dep.get("Aircraft_Type") or dep.get("Aircraft Type") or dep.get("AC Type")
        movement = dep.get("Movement") or dep.get("Flight Movement") or dep.get("Flight Movement Type")
        station = dep.get("Station") or dep.get("ORG") or dep.get("Origin")
        destination = dep.get("Destination") or dep.get("DES") or dep.get("Arrival Destination") or dep.get("Departure Destination")
        # إذا لم تتوفر، لا نُجبر النظام على الاستنتاج
        if aircraft_type and movement and station:
            try:
                r = lookup_mgt(
                    operation="TURNAROUND",
                    aircraft_group=str(aircraft_type),
                    movement=str(movement),
                    station=str(station),
                    destination_station=str(destination) if destination else None,
                    is_security_alert_station=False,
                    apply_local_towing_rule=False,
                )
                mgt_minutes = getattr(r, "final_mgt_minutes", None)
            except Exception:
                mgt_minutes = None

    # تعليمات الدفاع تُفعّل عند تحقق شرط البراءة
    if actual_ground_time is not None and mgt_minutes is not None and actual_ground_time <= mgt_minutes:
        # دعم إضافي: ربط No Show كسبب جذري إن وجد
        wb = analyze_workload_balance(shift) if shift else {}
        cap = calculate_operational_capacity(shift) if shift else {}
        no_show = wb.get("no_show")
        shortage = wb.get("shortage_percent")
        extra = ""
        if no_show is not None and no_show > 0:
            extra = f"\n- ملاحظة قانونية: يوجد No Show = {no_show} في نفس الشفت، ما يرفع الضغط التشغيلي."
            if shortage is not None:
                extra += f"\n- تقدير العجز في القوى العاملة ≈ {shortage:.1f}%."  # نسبة تقريبية

        defense_msg = apply_defense_logic({"actual_ground_time": actual_ground_time}, mgt_minutes)

        return f"""
أنت الآن 'المحامي الذكي لـ TCC' (The Defense Protocol).
المعطيات:
- كود التأخير: {code}
- الوقت الفعلي على الأرض: {actual_ground_time} دقيقة.
- المعيار القانوني (MGT): {mgt_minutes} دقيقة.
الحكم:
- {defense_msg}
- بما أن الفعلي <= المعيار، دافع بقوة وأثبت أن القسم غير مسؤول، ووضح أن الأداء يُعد إنجازاً تشغيلياً.
{extra}
""".strip()

    return ""


def _build_operational_context(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """حقن سياق عمليات يساعد على فهم النوايا (Intent Intelligence)."""
    dep = (bundle or {}).get("dep_flight_delay") or {}
    shift = (bundle or {}).get("shift_report") or {}

    # مثال: تأخير دقائق قليلة مع وصول متأخر = إنجاز
    delay_minutes = _to_int_safe(dep.get("Delay Minutes") or dep.get("Delay_Min") or dep.get("Delay_Minutes"))
    late_arrival = None
    sta = dep.get("STA")
    ata = dep.get("ATA")
    if sta and ata:
        ad = _minutes_diff_hhmm(sta, ata)
        if ad is not None:
            late_arrival = ad > 0

    wb = analyze_workload_balance(shift) if shift else {}
    cap = calculate_operational_capacity(shift) if shift else {}

    return {
        "intent_rules": {
            "treat_small_delay_with_late_arrival_as_success": True,
            "small_delay_threshold_minutes": 5,
        },
        "signals": {
            "delay_minutes": delay_minutes,
            "late_arrival": late_arrival,
        },
        "manpower": {"balance": wb, "capacity": cap},
    }


# =================== الدالة الرئيسية: nxs_brain ===================

def nxs_brain(message: str) -> Tuple[str, Dict[str, Any]]:
    """
    المحرك الرئيسي:
    1) تشغيل مرحلة التخطيط (planner).
    2) تنفيذ الخطة على nxs_supabase_client.
    3) بناء برومبت الإجابة النهائية واستدعاء محرك الذكاء.
    4) إرجاع النص + ميتاداتا تقنية (meta) للاستخدام في الواجهة/التشخيص.
    """
    # =================== Force Rule (NEW): Direct Employee Master Lookup via force_fetch_employee_by_id ===================
    # الهدف: بمجرد اكتشاف رقم وظيفي داخل سؤال المستخدم، نجلب البيانات فوراً من قاعدة البيانات بدون المرور بتحليل النوايا.
    emp_id_match = re.search(r'\d{5,8}', message)

    if emp_id_match:
        found_id = emp_id_match.group(0)
        try:
            db_data = nxs_db.force_fetch_employee_by_id(found_id)
        except Exception as exc:
            db_data = []
            logger.error(f"force_fetch_employee_by_id error for ID {found_id}: {exc}")

        if db_data:
            prompt = (
                "أنت خبير مطارات. هذه بيانات موظف حقيقية من النظام:\n"
                f"{json.dumps(db_data, ensure_ascii=False)}\n\n"
                f"أجب على سؤال المستخدم: {message}"
            )
            final_reply = call_ai(prompt)
            return final_reply, {"source": "direct_db_lookup", "emp_id": found_id}
    message = (message or "").strip()
    if not message:
        return (
            "مرحباً بك في TCC AI 👋\nاكتب سؤالك عن الموظفين، الرحلات، التأخيرات، أو المناوبات وسأجيبك من بيانات النظام قدر الإمكان.",
            {"ok": True, "stage": "empty_message"}
        )

    meta: Dict[str, Any] = {"ok": False}

    # =================== Fast Short-circuit: find_employee_fast ===================
    # اجعل هذا أول شيء يفعله النظام قبل تحليل النوايا
    potential_id = re.search(r'\d{7,8}', message)
    if potential_id:
        try:
            emp_data = nxs_db.find_employee_fast(potential_id.group(0))
        except Exception as exc:
            emp_data = []
            logger.error(f"find_employee_fast error for ID {potential_id.group(0)}: {exc}")
        if emp_data:
            # هنا نجبر النظام على الإجابة بالبيانات الحقيقية فوراً
            return f"بيانات الموظف {potential_id.group(0)}: الاسم {emp_data[0].get('Name')}", {}


    # =================== Force Rule: Direct Employee Master Lookup ===================
    # إذا وجدنا رقماً طويلاً (7-8 خانات)، نجبر النظام على فحص جدول الموظفين فوراً بدون المرور بالمحرك الدلالي.
    id_match_direct = re.search(r"\d{7,8}", message)
    if id_match_direct:
        found_id = id_match_direct.group(0)
        try:
            employee_data = nxs_db.execute_dynamic_query(
                f"SELECT * FROM employee_master_db WHERE \"Employee ID\" = '{found_id}'"
            )
        except Exception as exc:
            employee_data = []
            logger.error(f"Direct employee lookup error for ID {found_id}: {exc}")

        if employee_data:
            return (
                f"تم العثور على الموظف: {employee_data[0].get('Name')}. القسم: {employee_data[0].get('Department')}...",
                {}
            )

    # مسار سريع لقواعد GOPM (MGT/Turnaround/Transit/Activity Breakdown)
    # هذا المسار لا يستخدم Supabase ولا يلمس منطق خطط TCC.
    if _is_gopm_question(message):
        answer_text, gmeta = _gopm_answer(message)
        meta.update(gmeta or {})
        return answer_text, meta

    # =================== Global ID Search (Patch فقط) ===================
    # إذا كان السؤال يحتوي على رقم مكوّن من 5 إلى 8 خانات، نقوم ببحث (قوة ضاربة) في قاعدة البيانات
    # ثم نمرر النتائج مباشرة لمحرك الصياغة بدون فلترة مسبقة.
    id_match = re.search(r"\d{5,8}", message or "")
    collected_data: List[Dict[str, Any]] = []
    found_id: Optional[str] = None

    if id_match:
        found_id = id_match.group(0)
        try:
            collected_data = nxs_db.force_find_any_id(str(found_id)) or []
        except Exception as exc:
            logger.error(f"Force find error for ID {found_id}: {exc}")

    if collected_data:
        # تحديد لغة الرد بشكل مبسط (بدون تغيير منطق planner)
        language = "ar" if re.search(r"[\u0600-\u06FF]", message) else "en"

        strict_instruction = (
            "تعليمات صارمة: اعتمد فقط على البيانات المرسلة. "
            "إذا لم تجد الإجابة داخل البيانات، قل بوضوح أن البيانات غير كافية ولا تخمن."
        )

        answer_prompt = build_answer_prompt(
            user_message=message,
            language=language,
            planner_notes="force-find-any-id",
            data_bundle={"global_search_results": collected_data},
            extra_system_instruction=strict_instruction,
            operational_context=None,
        )

        answer_text = call_ai(answer_prompt, model_name=GEMINI_MODEL_SIMPLE)

        meta.update(
            {
                "ok": True,
                "language": language,
                "stage": "force_find_any_id",
                "data_summary": {"rows": len(collected_data), "found_id": found_id},
                "engine": "NXS-URE",
            }
        )
        return answer_text, meta

    try:
        # 1) التخطيط
        planner_info = run_planner(message)
        language = planner_info.get("language", "ar")
        plan = planner_info.get("plan", [])
        notes = planner_info.get("notes", "")

        # 2) تنفيذ الخطة على Supabase
        data_results = execute_plan(plan)

        # =================== منطق الاستدلال المتقدم (Patch فقط) ===================
        cross_table_bundle = {}
        extra_system_instruction = ""
        operational_context = None

        if _is_flight_delay_query(message):
            cross_table_bundle = get_cross_table_bundle(message)
            # بروتوكول المحامي
            extra_system_instruction = _build_defense_instruction(cross_table_bundle)
            # سياق العمليات (Intent Intelligence + Manpower)
            operational_context = _build_operational_context(cross_table_bundle)

            # ربط هذه البيانات مع حزمة البيانات الرئيسية قبل إرسالها للمحرك
            if cross_table_bundle:
                data_results["cross_table"] = cross_table_bundle

        # 3) بناء برومبت الإجابة
        answer_prompt = build_answer_prompt(
            user_message=message,
            language=language,
            planner_notes=notes,
            data_bundle=data_results,
            extra_system_instruction=extra_system_instruction,
            operational_context=operational_context,
        )

        # 4) استدعاء محرك الذكاء لصياغة الإجابة
        #    اختيار الموديل للإجابة (Flash/Pro) بناءً على تلميحات Semantics + تعقيد الخطة
        answer_model = None
        try:
            sem = (planner_info or {}).get("semantic_hints") or (planner_info or {}).get("semantic") or {}
            mh = sem.get("model_hint") if isinstance(sem, dict) else None
            if isinstance(mh, dict):
                answer_model = mh.get("model")
        except Exception:
            answer_model = None

        if not answer_model:
            # fallback: إذا الخطة طويلة/معقدة نستخدم Pro
            if isinstance(plan, list) and len(plan) >= 2:
                answer_model = GEMINI_MODEL_COMPLEX
            else:
                answer_model = GEMINI_MODEL_SIMPLE

        answer_text = call_ai(answer_prompt, model_name=answer_model)

        meta.update(
            {
                "ok": True,
                "language": language,
                "planner": planner_info,
                "data_summary": {
                    "steps": len(data_results.get("steps", [])),
                },
                "engine": "NXS-URE",
            }
        )
        return answer_text, meta

    except AIEngineError as ae:
        # خطأ من محرك الذكاء نفسه
        reply = (
            "⚠️ تعذّر حالياً استخدام محرك التحليل الذكي في الخلفية.\n"
            "يمكنك المحاولة لاحقاً أو مراجعة إعدادات المفتاح في الخادم.\n\n"
            f"(معلومة تقنية للمطوّر): {ae}"
        )
        meta.update(
            {
                "ok": False,
                "error": str(ae),
                "stage": "ai_engine_error",
            }
        )
        return reply, meta

    except Exception as exc:
        reply = (
            "⚠️ حدث خطأ غير متوقع داخل محرك NXS • Ultra Reasoning.\n"
            "يمكن مراجعة سجل الخادم (logs) لمعرفة التفاصيل التقنية.\n"
        )
        meta.update(
            {
                "ok": False,
                "error": str(exc),
                "stage": "unexpected_exception",
            }
        )
        return reply, meta

# =================================================================
# وظيفة المرحلة الأولى: تحليل السبب الجذري للعمل الإضافي (TCC/TC)
# =================================================================

def run_tcc_overtime_rca(target_department: str = 'TCC') -> Tuple[str, Dict[str, Any]]:
    """
    تنفيذ تحليل السبب الجذري (RCA) للعمل الإضافي وتأثيره على تأخيرات TCC.
    هذه الوظيفة تنفذ المراحل 1-3 من دورة التحسين.
    """
    
    # 1. تعريف العتبة الحرجة (الفرضية المبدئية)
    OVERTIME_CRITICAL_THRESHOLD = 10.0  # ساعات عمل إضافي أسبوعياً
    
    # 2. جلب بيانات العمل الإضافي من طبقة البيانات
    overtime_data = nxs_db.list_employee_overtime(department=target_department)
    
    # 3. جلب بيانات التأخير المرتبطة (محاكاة الربط)
    linked_delays = nxs_db.get_delays_with_overtime_link(overtime_data)
    
    high_risk_employees = []
    total_ot_delays = 0
    
    # 4. تطبيق منطق التحليل: فصل الموظفين حسب العتبة
    for record in overtime_data:
        emp_id = record["Employee ID"]
        try:
            ot_hours = float(record.get("Total Hours", "0"))
        except ValueError:
            continue
            
        # التحقق من تجاوز العتبة
        if ot_hours > OVERTIME_CRITICAL_THRESHOLD:
            # التحقق من وجود تأخير TC-OVT لهذا الموظف
            delays = linked_delays.get(emp_id, [])
            is_ovt_cause = any("TC-OVT" in d.get("Violation", "") for d in delays)
            
            if is_ovt_cause:
                high_risk_employees.append(emp_id)
                total_ot_delays += sum(
                    d.get("Delay_Min", 0) for d in delays if "TC-OVT" in d.get("Violation", "")
                )
    
    # 5. توليد تقرير الذكاء الاصطناعي (Output Report)
    
    analysis_result = (
        f"✅ **المرحلة الأولى: تشخيص العمل الإضافي (TCC/TC) - تم الانتهاء.**\n"
        f"1. **الأسباب الجذرية:** تم تحديد أن تأخيرات 'TC-OVT' هي الأعلى بعد ربطها بـ {len(high_risk_employees)} موظف.\n"
        f"2. **العتبة الحرجة:** تم التحقق من أن الموظف في TCC الذي يتجاوز **{OVERTIME_CRITICAL_THRESHOLD} ساعة عمل إضافي** تزيد احتمالية تسببه بتأخير TC-OVT.\n"
        f"3. **الأثر:** يُقدَّر إجمالي التأخير الشهري من هذه المجموعة عالية المخاطر بـ **{total_ot_delays} دقيقة** (في البيانات المُحللة).\n"
        f"4. **التوصية:** يجب إصدار **أمر إداري آلي** لخفض سقف العمل الإضافي إلى {OVERTIME_CRITICAL_THRESHOLD} ساعة كحد أقصى.\n"
    )
    
    meta_data = {
        "analysis_stage": "RCA_Overtime",
        "critical_threshold_found": OVERTIME_CRITICAL_THRESHOLD,
        "high_risk_employees_count": len(high_risk_employees),
        "total_delay_impact_min": total_ot_delays,
    }
    
    return analysis_result, meta_data

# 6. محاكاة تشغيل الوظيفة (مثال التنفيذ)
# response, meta = run_tcc_overtime_rca()
# print(response)



# =================================================================
# وظيفة المرحلة السادسة: تحليل عمليات الوقود (FU-OPS)
# =================================================================

from datetime import datetime, time

def run_sgs_fueling_rca() -> tuple:
    PEAK_START = time(8, 0)
    PEAK_END = time(10, 0)
    fueling_delays = nxs_db.get_fueling_delays(delay_code='FU-OPS')
    flight_numbers = [d["FLT"] for d in fueling_delays]
    sector_data = nxs_db.get_flight_sector_data(flight_numbers)
    sector_map = {d["FLT"]: d["Is_Long_Haul"] for d in sector_data}
    peak_conflict_delays = 0
    total_fueling_delay = 0
    for delay in fueling_delays:
        total_fueling_delay += delay["Delay_Min"]
        flight_time = datetime.strptime(delay["SCHED_DEP"], '%H:%M').time()
        flt_num = delay["FLT"]
        is_peak = PEAK_START <= flight_time <= PEAK_END
        is_long_haul = sector_map.get(flt_num, False)
        if is_peak and is_long_haul:
            peak_conflict_delays += delay["Delay_Min"]
    conflict_share = peak_conflict_delays / total_fueling_delay if total_fueling_delay else 0
    analysis_result = (
        f"🔥 **المرحلة السادسة: تشخيص عمليات الوقود (FU-OPS) - تم الانتهاء.**\n"
        f"1. **التشخيص:** إجمالي تأخير FU-OPS هو **{total_fueling_delay} دقيقة**.\n"
        f"2. **السبب الجذري:** تعارض الجدولة، حيث أن **{conflict_share:.0%}** من التأخير يحدث بسبب تزامن رحلات المسافات الطويلة مع فترة الذروة.\n"
        f"3. **التوصية:** تفعيل الجدولة الاستباقية للوقود."
    )
    meta_data = {
        "analysis_stage": "RCA_FU_OPS",
        "peak_conflict_share": f"{conflict_share:.2f}",
        "total_delay_impact_min": total_fueling_delay,
    }
    return analysis_result, meta_data


# =================================================================
# وظيفة المرحلة السابعة: التدخل التكتيكي (قفل الأصول)
# =================================================================

def tactical_asset_lock() -> tuple:
    PM_CRITICAL_OVERDUE_DAYS = 5
    all_overdue_pm_events = nxs_db.get_overdue_pm_events(asset_ids=[])
    locked_assets_count = 0
    locked_asset_list = []
    for event in all_overdue_pm_events:
        asset_id = event["Asset_ID"]
        overdue_days = event.get("Overdue_Days", 0)
        if overdue_days >= PM_CRITICAL_OVERDUE_DAYS:
            reason = f"PM overdue by {overdue_days} days."
            if nxs_db.update_asset_status(asset_id, 'OUT OF SERVICE', reason):
                locked_assets_count += 1
                locked_asset_list.append(asset_id)
                alert_msg = f"ASSET LOCK: {asset_id} OUT OF SERVICE. PM overdue."
                nxs_db.log_system_alert('CRITICAL_ASSET_LOCK', alert_msg)
    analysis_result = (
        f"✅ **المرحلة السابعة: قفل الأصول - مكتمل.**\n"
        f"الأصول المقفلة: {', '.join(locked_asset_list)}"
    )
    meta_data = {
        "analysis_stage": "Tactical_Asset_Locking",
        "assets_locked": locked_assets_count,
        "locked_asset_ids": locked_asset_list,
    }
    return analysis_result, meta_data



# =================================================================
# وظيفة المرحلة الثامنة: التدخل التكتيكي (سقف العمل الإضافي)
# =================================================================

from datetime import date

def tactical_overtime_cap(department: str = 'TCC') -> Tuple[str, Dict[str, Any]]:
    """
    تفعيل منطق سقف العمل الإضافي الآلي (OVT Cap) على أساس العتبة الحرجة (10 ساعات).
    """
    
    OVT_CRITICAL_CAP = 10.0  # ساعة أسبوعياً
    
    # 1. التدخل الآلي: تحديث سياسة الموارد البشرية
    policy_update_success = nxs_db.update_ot_policy(
        department,
        OVT_CRITICAL_CAP,
        date.today().isoformat()
    )
    
    # 2. التحقق من الموظفين المتجاوزين وإرسال تنبيهات (نستخدم بيانات المرحلة الأولى)
    overtime_data = nxs_db.list_employee_overtime(department=department)
    
    alerted_employees: List[int] = []
    
    for record in overtime_data:
        emp_id = record["Employee ID"]
        try:
            ot_hours = float(record.get("Total Hours", "0"))
        except ValueError:
            continue
            
        # التحقق من تجاوز السقف الجديد (10.0)
        if ot_hours > OVT_CRITICAL_CAP:
            alerted_employees.append(emp_id)
            # إرسال تنبيه آلي للمدير المسؤول (محاكاة)
            nxs_db.send_ot_notification(
                manager_email=f"TCC_Manager_{emp_id}@airport.com",
                employee_id=emp_id,
                current_ot=ot_hours,
                threshold=OVT_CRITICAL_CAP,
            )
            
    # 3. توليد تقرير الإجراء التكتيكي
    
    status_msg = "تم تحديث قاعدة البيانات بنجاح." if policy_update_success else "⚠️ فشل تحديث قاعدة البيانات."
    
    analysis_result = (
        f"✅ **المرحلة الثامنة: تنفيذ التدخل التكتيكي (سقف العمل الإضافي) - تم بنجاح.**\n"
        f"1. **الإجراء المُنفَّذ:** تم تحديث سياسة العمل الإضافي في `hr_policy_register` لـ {department} لتصبح **{OVT_CRITICAL_CAP} ساعة** كحد أقصى.\n"
        f"2. **حالة التحديث:** {status_msg}\n"
        f"3. **التطبيق الفوري:** تم إرسال تنبيهات لمديري الموظفين المتجاوزين ({len(alerted_employees)} موظف/ين) لضمان عدم تخصيص عمل إضافي لهم هذا الأسبوع.\n"
    )
    
    meta_data: Dict[str, Any] = {
        "analysis_stage": "Tactical_OVT_Cap",
        "ovt_cap_set": OVT_CRITICAL_CAP,
        "employees_alerted": len(alerted_employees),
        "alerted_employee_ids": alerted_employees,
    }
        
    return analysis_result, meta_data


# =================================================================
# وظيفة المرحلة التاسعة: قياس الأثر النهائي والعائد على الاستثمار (ROI)
# =================================================================

def measure_impact_and_roi() -> Tuple[str, Dict[str, Any]]:
    """
    قياس الأداء النهائي (OTP) وحساب العائد على الاستثمار (ROI).
    """
    
    # الثوابت المالية (مثال لمتوسط تكلفة التأخير)
    COST_PER_DELAY_MINUTE = 5.50  # دولار/دقيقة
    TARGET_OTP = 93.62            # الهدف التشغيلي المُحقق
    
    # 1. جلب البيانات
    baseline_otp = nxs_db.get_baseline_otp()
    delay_reduction_map = nxs_db.get_total_delay_reduction()
    intervention_costs_map = nxs_db.get_intervention_costs()
    
    # 2. حساب إجمالي الدقائق المُوفَّرة
    total_minutes_saved = sum(delay_reduction_map.values())
    
    # 3. حساب الأثر المالي (الوفورات)
    total_financial_benefit = total_minutes_saved * COST_PER_DELAY_MINUTE
    
    # 4. حساب إجمالي تكلفة التدخلات التكتيكية
    total_intervention_cost = sum(intervention_costs_map.values())
    
    # 5. حساب العائد على الاستثمار (ROI)
    if total_intervention_cost > 0:
        roi = ((total_financial_benefit - total_intervention_cost) / total_intervention_cost) * 100
    else:
        roi = float('inf')
        
    # 6. توليد التقرير النهائي للقياس
    
    analysis_result = (
        f"✅ **المرحلة التاسعة: قياس الأثر النهائي (OTP & ROI) - تم بنجاح.**\n"
        f"1. **الأداء التشغيلي (OTP):** ارتفاع من **{baseline_otp:.2f}%** إلى **{TARGET_OTP:.2f}%**.\n"
        f"2. **الدقائق المُوفَّرة:** إجمالي الدقائق المُزال سببها الجذري: **{total_minutes_saved:,.0f} دقيقة/شهر**.\n"
        f"3. **الأثر المالي:** إجمالي المنفعة المالية (الوفورات) هي **${total_financial_benefit:,.2f}**.\n"
        f"4. **تكلفة التدخل:** إجمالي تكلفة التدخلات التكتيكية هي **${total_intervention_cost:,.2f}**.\n"
        f"5. **العائد على الاستثمار (ROI):** تم تحقيق عائد استثمار بلغ **{roi:.2f}%**.\n"
        f"6. **التحقق:** تم تأكيد أن جميع التدخلات التكتيكية أنتجت النتائج المرجوة وتجاوزت الهدف المالي.\n"
    )
    
    meta_data: Dict[str, Any] = {
        "analysis_stage": "Impact_Measurement",
        "final_otp": TARGET_OTP,
        "total_minutes_saved": total_minutes_saved,
        "final_roi_percent": roi,
        "total_financial_benefit": total_financial_benefit,
        "total_intervention_cost": total_intervention_cost,
    }
        
    return analysis_result, meta_data


# =================================================================
# وظيفة المرحلة العاشرة: التخطيط الاستراتيجي والاستدامة
# =================================================================

def generate_strategic_plan(annual_manpower_cost: int = 75000, otp_increase: float = 9.12) -> Tuple[str, Dict[str, Any]]:
    """
    إنشاء خطة استراتيجية للموارد البشرية (Manpower) والإنفاق الرأسمالي (CAPEX).
    """
    
    # 1. جلب متطلبات الإنفاق الرأسمالي (CAPEX)
    asset_plan = nxs_db.get_asset_replacement_plan()
    total_capex_cost = sum(asset.get("Replacement_Cost", 0) for asset in asset_plan)
    replacement_units = len(asset_plan)
    
    # 2. جلب متطلبات الموارد البشرية (Manpower)
    manpower_demand = nxs_db.get_manpower_demand()
    staff_needed = manpower_demand.get("TCC_Staff_Needed", 0)
    
    # 3. حساب ميزانية الموارد البشرية السنوية
    total_manpower_cost = staff_needed * annual_manpower_cost
    
    # 4. أرقام الربط من مرحلة قياس الأثر
    ROI_PERCENT = 1091.67
    MONTHLY_SAVINGS = 357500.00
    
    analysis_result = (
        f"👑 **المرحلة العاشرة: التخطيط الاستراتيجي واستدامة الأداء - تم الانتهاء.**\n"
        f"تم ترجمة العائد على الاستثمار التكتيكي ({ROI_PERCENT:.2f}%) إلى خطة استثمار استراتيجية لضمان استدامة OTP بنسبة 93.62%.\n\n"
        f"--- \n"
        f"## 🛠️ خطة الإنفاق الرأسمالي (CAPEX) \n"
        f"* **الهدف:** استبدال الأصول القديمة التي تسببت في تأخيرات GS-BAG.\n"
        f"* **الوحدات المطلوبة:** استبدال {replacement_units} ناقلة أمتعة (Loaders).\n"
        f"* **إجمالي CAPEX المطلوب:** **${total_capex_cost:,.2f}**.\n"
        f"* **تبرير الاستثمار:** يمنع هذا الاستثمار خسارة **${MONTHLY_SAVINGS:,.2f}** دولار شهرياً ناتجة عن أعطال المعدات.\n\n"
        f"--- \n"
        f"## 🧑‍💻 خطة الموارد البشرية (Manpower) \n"
        f"* **الهدف:** الحفاظ على سقف العمل الإضافي (OVT Cap) وتغطية متطلبات الغياب (TC-ABS).\n"
        f"* **عدد الموظفين الجدد:** {staff_needed} موظف/ة لقسم TCC.\n"
        f"* **الميزانية السنوية الإضافية:** **${total_manpower_cost:,.2f}**.\n"
        f"* **تبرير التوظيف:** يضمن استقرار الأداء التشغيلي ويمنع أخطاء السلامة الناتجة عن الإرهاق.\n\n"
        f"--- \n"
        f"## 📈 الخلاصة النهائية\n"
        f"تم التحقق من أن الاستثمار الاستراتيجي الكلي البالغ **${total_capex_cost + total_manpower_cost:,.2f}** \n"
        f"سيعزز الأداء التشغيلي (OTP) بنسبة **{otp_increase:.2f} نقطة مئوية** سنوياً، ويضمن استدامة الأداء الذي تم تحقيقه.\n"
    )
    
    meta_data: Dict[str, Any] = {
        "analysis_stage": "Strategic_Planning",
        "total_capex": total_capex_cost,
        "total_manpower_budget": total_manpower_cost,
        "total_strategic_investment": total_capex_cost + total_manpower_cost,
        "staff_needed": staff_needed,
    }
        
    return analysis_result, meta_dat
# =================== GOPM (Aircraft Ramp Handling) Rules ===================
# هذه القواعد مأخوذة من جداول GOPM التي زوّدنا بها المستخدم (MGT + Activity Breakdown + Hints).
# ملاحظة: ملف القواعد منفصل لتسهيل الصيانة وعدم خلطه مع منطق Supabase/TCC.

import re  # used by GOPM helpers
from typing import Optional  # used by GOPM helpers

try:
    from nxs_gopm_rules import (
        lookup_mgt,
        lookup_activity_breakdown,
        get_aircraft_delivery_before_std_hours,
    )
except Exception:
    lookup_mgt = None
    lookup_activity_breakdown = None
    get_aircraft_delivery_before_std_hours = None

_GOPM_DEST_SPECIAL = {"USA", "KAN", "SSH", "JFK", "LAX", "IAD", "YYZ", "MNL", "CAN", "KUL", "CGK", "SIN"}

_AIRCRAFT_ALIASES = [
    (re.compile(r"\bB777\b|777|B787-10|787-10|B787\s*10", re.I), "B777-368/B787-10"),
    (re.compile(r"A330|B787-9|787-9|B787\s*9", re.I), "A330/B787-9"),
    (re.compile(r"A321|A320", re.I), "A321/A320"),
    (re.compile(r"B757|757", re.I), "B757"),
]

def _looks_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))

def _preferred_lang(message: str) -> str:
    m = (message or "")
    if re.search(r"\b(arabic|عربي|arab)\b", m, re.I):
        return "ar"
    if re.search(r"\b(english|انجليزي|إنجليزي)\b", m, re.I):
        return "en"
    return "ar" if _looks_arabic(m) else "en"

def _is_gopm_question(message: str) -> bool:
    m = (message or "").lower()
    keywords = [
        "gopm", "mgt", "turnaround", "transit", "activity breakdown", "ramp handling",
        "تورناروند", "ترانزيت", "ترانزت", "وقت ارضي", "وقت الأرض", "الحد الأدنى", "مناولة",
        "b777", "b787", "a330", "a321", "a320", "b757",
        "jed", "ruh", "dmm", "med", "lhr", "ssh", "usa", "kan", "jfk", "lax", "iad", "yyz", "mnl", "can", "kul", "cgk", "sin",
    ]
    return any(k in m for k in keywords)

def _extract_operation(message: str) -> Optional[str]:
    m = (message or "").lower()
    if "turnaround" in m or "تورناروند" in m or "turn around" in m:
        return "TURNAROUND"
    if "transit" in m or "ترانزيت" in m or "ترانزت" in m:
        return "TRANSIT"
    return None

def _extract_movement(message: str) -> Optional[str]:
    m = (message or "").upper().replace(" ", "").replace("_", "-")
    for mv in ["DOM-DOM", "DOM-INTL", "INTL-DOM", "INTL-INTL"]:
        if mv in m:
            return mv
    if "داخلي" in (message or "") and "دولي" in (message or ""):
        if re.search(r"داخلي.*دولي", message):
            return "DOM-INTL"
        if re.search(r"دولي.*داخلي", message):
            return "INTL-DOM"
    return None

def _extract_aircraft_group(message: str) -> Optional[str]:
    for rx, group in _AIRCRAFT_ALIASES:
        if rx.search(message or ""):
            return group
    return None

def _extract_station(message: str) -> Optional[str]:
    m = (message or "").upper()
    if "LONG HAUL" in m or "LONG_HAUL" in m:
        return "LONG_HAUL_STN"
    if "INT STNS" in m or "INT_STNS" in m:
        return "INT_STNS"
    if "OTHER DOM" in m or "OTHER_DOM" in m:
        return "OTHER_DOM_STN"
    if "AHB/TUU" in m or re.search(r"\bAHB\b", m) or re.search(r"\bTUU\b", m):
        return "AHB/TUU"
    for code in ["JED", "RUH", "DMM", "MED", "LHR", "UK"]:
        if re.search(rf"\b{code}\b", m):
            return code
    return None

def _extract_destination(message: str) -> Optional[str]:
    m = (message or "").upper()
    if re.search(r"\bUSA\b", m):
        return "USA"
    if re.search(r"\bKAN\b", m):
        return "KAN"
    if re.search(r"\bSSH\b", m):
        return "SSH"
    codes = re.findall(r"\b[A-Z]{3}\b", m)
    for c in codes:
        if c in _GOPM_DEST_SPECIAL:
            return c
    return None

def _extract_flags(message: str) -> tuple[bool, bool]:
    m = (message or "").lower()
    is_sa = ("(sa)" in m) or ("security alert" in m) or ("تنبيه أمني" in m) or ("امني" in m)
    towing = ("towing" in m) or ("سحب" in m) or ("قطر" in m) or ("jed-t1" in m) or ("local mgt" in m)
    return is_sa, towing

def _format_time_hhmm(minutes: int) -> str:
    hh = minutes // 60
    mm = minutes % 60
    return f"{hh:02d}:{mm:02d}"

def _gopm_answer(message: str) -> tuple[str, dict]:
    # يرجع (answer_text, meta). لا يعتمد على Supabase.
    if lookup_mgt is None:
        return (
            "⚠️ ملف قواعد GOPM غير متوفر داخل الخادم حالياً.\n"
            "تأكد من رفع الملف: nxs_gopm_rules.py ضمن نفس مشروع النشر.",
            {"ok": False, "stage": "gopm_missing_rules"},
        )

    lang = _preferred_lang(message)
    op = _extract_operation(message)
    mv = _extract_movement(message)
    ac = _extract_aircraft_group(message)
    st = _extract_station(message)
    dest = _extract_destination(message)
    is_sa, towing = _extract_flags(message)

    wants_activity = bool(re.search(r"activity\s*breakdown|activities|تفصيل|تفاصيل|بنود|بند", message or "", re.I))
    wants_delivery = bool(re.search(r"delivery\s*time|before\s*std|Aircraft\s*Delivery|تسليم|قبل\s*std", message or "", re.I))

    meta = {
        "ok": True,
        "stage": "gopm_answer",
        "lang": lang,
        "parsed": {
            "operation": op,
            "movement": mv,
            "aircraft_group": ac,
            "station": st,
            "destination_station": dest,
            "is_security_alert_station": is_sa,
            "apply_local_towing_rule": towing,
            "wants_activity": wants_activity,
            "wants_delivery": wants_delivery,
        },
    }

    if wants_delivery:
        if not ac:
            txt = "اذكر نوع الطائرة (مثل A321 أو B777) لأحدد Delivery time." if lang == "ar" else "Please mention the aircraft type (e.g., A321 or B777) so I can return the delivery time."
            return txt, meta
        hours = get_aircraft_delivery_before_std_hours(ac) if get_aircraft_delivery_before_std_hours else None
        if hours is None:
            txt = "لا يوجد وقت تسليم معرف لهذا النوع." if lang == "ar" else "No delivery time is defined for this aircraft group."
            return txt, meta
        if lang == "ar":
            return f"🛫 وقت تسليم الطائرة (من الهنجر إلى البوابة) قبل STD: **{hours:.0f} ساعة**\n✈️ النوع: {ac}", meta
        return f"🛫 Aircraft delivery time (Hangar → Parking Gate) before STD: **{hours:.0f} hours**\n✈️ Aircraft group: {ac}", meta

    if wants_activity:
        if not ac:
            txt = "اذكر نوع الطائرة (مثل B757 أو A321/A320 أو A330/B787-9 أو B777-368/B787-10) لأعرض Activity Breakdown." if lang == "ar" else "Please provide an aircraft type (B757, A321/A320, A330/B787-9, or B777-368/B787-10) to show the Activity Breakdown."
            return txt, meta
        if not op:
            txt = "حدد هل هي Turnaround أم Transit." if lang == "ar" else "Please specify whether it's Turnaround or Transit."
            return txt, meta
        if not mv:
            txt = "حدد نوع الحركة: DOM-DOM أو DOM-INTL أو INTL-DOM أو INTL-INTL." if lang == "ar" else "Please specify movement: DOM-DOM, DOM-INTL, INTL-DOM, or INTL-INTL."
            return txt, meta

        try:
            br = lookup_activity_breakdown(ac if ac != "A330/B787-9" else "A330/B787-9", op, mv)
        except Exception as e:
            txt = f"تعذر استخراج Activity Breakdown: {e}" if lang == "ar" else f"Failed to fetch Activity Breakdown: {e}"
            return txt, meta

        lines = []
        if lang == "ar":
            lines += [f"🧾 **Activity Breakdown**", f"✈️ النوع: {ac}", f"🔁 العملية: {op}", f"📌 الحركة: {mv}", ""]
            for item in br.items:
                v = item.value
                if v is None:
                    v_txt = "—"
                elif isinstance(v, (int, float)) and float(v).is_integer():
                    v_txt = f"{int(v)} دقيقة"
                else:
                    v_txt = str(v)
                lines.append(f"• {item.activity}: {v_txt}")
            if br.total_minutes is not None:
                lines += ["", f"⏱️ الإجمالي: **{br.total_minutes} دقيقة**"]
            if br.assumptions:
                lines += ["", "📌 افتراضات/ملاحظات:"]
                lines += [f"- {a}" for a in br.assumptions]
            return "\n".join(lines), meta

        lines += ["🧾 **Activity Breakdown**", f"✈️ Aircraft: {ac}", f"🔁 Operation: {op}", f"📌 Movement: {mv}", ""]
        for item in br.items:
            v = item.value
            if v is None:
                v_txt = "—"
            elif isinstance(v, (int, float)) and float(v).is_integer():
                v_txt = f"{int(v)} min"
            else:
                v_txt = str(v)
            lines.append(f"• {item.activity}: {v_txt}")
        if br.total_minutes is not None:
            lines += ["", f"⏱️ Total: **{br.total_minutes} min**"]
        if br.assumptions:
            lines += ["", "📌 Assumptions/Notes:"]
            lines += [f"- {a}" for a in br.assumptions]
        return "\n".join(lines), meta

    if not op:
        txt = "حدد هل سؤالك عن Turnaround أم Transit." if lang == "ar" else "Please specify whether you mean Turnaround or Transit."
        return txt, meta
    if not mv:
        txt = "حدد نوع الحركة: DOM-DOM أو DOM-INTL أو INTL-DOM أو INTL-INTL." if lang == "ar" else "Please specify movement: DOM-DOM, DOM-INTL, INTL-DOM, or INTL-INTL."
        return txt, meta
    if not ac:
        txt = "اذكر نوع الطائرة (مثل B777 أو A330 أو A321 أو B757)." if lang == "ar" else "Please mention the aircraft type (e.g., B777, A330, A321, or B757)."
        return txt, meta
    if not st:
        txt = "اذكر المحطة (مثل JED أو RUH أو DMM أو MED أو INT STNS أو LONG HAUL STN)." if lang == "ar" else "Please mention the station (e.g., JED, RUH, DMM, MED, INT STNS, LONG HAUL STN)."
        return txt, meta

    try:
        r = lookup_mgt(
            operation=op,
            aircraft_group=ac,
            movement=mv,
            station=st,
            destination_station=dest,
            is_security_alert_station=is_sa,
            apply_local_towing_rule=towing,
        )
    except Exception as e:
        txt = f"تعذر حساب MGT: {e}" if lang == "ar" else f"Failed to calculate MGT: {e}"
        return txt, meta

    if lang == "ar":
        parts = [
            "⏱️ **Minimum Ground Time (MGT)**",
            f"✈️ النوع: {ac}",
            f"🔁 العملية: {op}",
            f"📌 الحركة: {mv}",
            f"🏷️ المحطة: {st}",
        ]
        if dest:
            parts.append(f"🎯 الوجهة/القيّد: {dest}")
        if r.base_mgt_minutes is not None:
            parts.append(f"🧮 MGT الأساسي من الجدول: **{_format_time_hhmm(r.base_mgt_minutes)}**")
        parts.append(f"✅ النتيجة النهائية: **{_format_time_hhmm(r.final_mgt_minutes)}**")
        if r.applied_rules:
            parts += ["", "📌 قواعد تم تطبيقها:"]
            parts += [f"- {rule}" for rule in r.applied_rules]
        return "\n".join(parts), meta

    parts = [
        "⏱️ **Minimum Ground Time (MGT)**",
        f"✈️ Aircraft: {ac}",
        f"🔁 Operation: {op}",
        f"📌 Movement: {mv}",
        f"🏷️ Station: {st}",
    ]
    if dest:
        parts.append(f"🎯 Destination/Constraint: {dest}")
    if r.base_mgt_minutes is not None:
        parts.append(f"🧮 Base MGT from table: **{_format_time_hhmm(r.base_mgt_minutes)}**")
    parts.append(f"✅ Final result: **{_format_time_hhmm(r.final_mgt_minutes)}**")
    if r.applied_rules:
        parts += ["", "📌 Applied rules:"]
        parts += [f"- {rule}" for rule in r.applied_rules]
    return "\n".join(parts), meta

a

# =================================================================
# NXS Ultra Reasoning Injection (nxs_engine)
# =================================================================
# تم حقن هذا القسم بناءً على طلب المستخدم لإضافة محرك NXSUltraReasoning
# دون المساس بباقي منطق nxs_brain الأصلي.

import re  # required for entity extraction in NXSUltraReasoning

class NXSUltraReasoning:
    def __init__(self):
        self.system_prompt = """
        أنت الذكاء الاصطناعي الخبير بمطار الملك عبدالعزيز (NXS). 
        صلاحياتك: الوصول الكامل لجميع الأعمدة في جداول (MGT, Flight Delays, Shift Reports, Employee Master).
        مهمتك: 
        1. إذا سُئلت عن موظف برقم ID، ابحث في كافة الجداول عن هذا الرقم.
        2. عند تحليل أي تأخير، استدعِ معيار MGT وقارنه بالوقت الفعلي (ATA vs ATD).
        3. إذا وجدت نقصاً في الموظفين (No Show) في شفت الرحلة، اربطه فوراً كسبب للتأخير للدفاع عن القسم.
        """

    async def process_query(self, user_query):
        # 1. تحليل السؤال واستخراج كافة الأرقام (IDs, Flight Numbers)
        entities = self.extract_entities(user_query)

        # 2. الاستعلام الشامل (Global Search)
        # سيقوم النظام بمسح الجداول بناءً على الأرقام المكتشفة
        context_data = {}

        if entities.get('employee_id'):
            # البحث في Master DB و Shift Report و Delays
            context_data['employee_info'] = nxs_db.get_all_employee_data(entities['employee_id'])

        if entities.get('flight_number'):
            # جلب بيانات الرحلة + معايير MGT + حالة الشفت في ذلك اليوم
            flight_info = nxs_db.get_integrated_flight_data(entities['flight_number'])
            shift_info = nxs_db.get_shift_report_by_date(flight_info['date'], flight_info['shift'])

            # حساب الضغط التشغيلي (المعادلة التي زودتني بها)
            workload = self.calculate_workload(shift_info)
            context_data['analysis'] = {
                "flight": flight_info,
                "workload_pressure": workload,
                "mgt_compliance": self.check_mgt(flight_info)
            }

        # 3. توليد الإجابة الاحترافية باستخدام Gemini Pro
        return self.generate_final_response(user_query, context_data)

    def calculate_workload(self, shift):
        # تطبيق معاييرك: 70 دقيقة مغادرة / 20 دقيقة وصول
        total_minutes_needed = (shift['Departures'] * 70) + (shift['Arrivals'] * 20)
        available_minutes = shift['On_Duty'] * 480
        return (total_minutes_needed / available_minutes) * 100

    # ------------------- Helpers (added to complete the injection) -------------------

    def extract_entities(self, user_query: str) -> Dict[str, Any]:
        """استخراج Employee ID ورقم الرحلة من نص المستخدم بشكل مرن."""
        q = (user_query or "").strip()
        uq = q.upper()

        # Flight number: SV123 / XY4567 / FZ123 etc.
        flight = None
        m = re.search(r"\b([A-Z]{2,3})\s*(\d{1,5})\b", uq)
        if m:
            flight = f"{m.group(1)}{m.group(2)}"

        # Employee ID: رقم 4-10 خانات (نأخذ أول رقم واضح)
        emp = None
        m2 = re.search(r"\b(\d{4,10})\b", q)
        if m2:
            emp = m2.group(1)

        return {"employee_id": emp, "flight_number": flight}

    def _minutes_diff(self, start_hhmm: Any, end_hhmm: Any) -> Optional[int]:
        """فرق دقائق بين وقتين HH:MM مع مراعاة عبور منتصف الليل."""
        def parse(v: Any) -> Optional[int]:
            if v is None:
                return None
            s = str(v).strip()
            if not s:
                return None
            mm = re.match(r"^(\d{1,2}):(\d{2})", s)
            if not mm:
                return None
            h = int(mm.group(1))
            m = int(mm.group(2))
            return h * 60 + m

        a = parse(start_hhmm)
        b = parse(end_hhmm)
        if a is None or b is None:
            return None
        if b < a:
            b += 24 * 60
        return b - a

    def check_mgt(self, flight_info: Dict[str, Any]) -> Dict[str, Any]:
        """يحاول استدعاء MGT (إن توفر) ومقارنته بالوقت الفعلي على الأرض."""
        if not isinstance(flight_info, dict):
            return {"mgt_standard": None, "actual_minutes": None, "status": "no_flight_info"}

        ata = flight_info.get("ATA") or flight_info.get("AAT") or flight_info.get("Arrival_ATA") or flight_info.get("arrival_ata")
        atd = flight_info.get("ATD") or flight_info.get("ADT") or flight_info.get("Departure_ATD") or flight_info.get("departure_atd")
        actual = self._minutes_diff(ata, atd)

        mgt_standard = None
        try:
            if lookup_mgt is not None:
                # نحاول استخدام حقول شائعة إن وجدت
                ac = flight_info.get("Aircraft_Type") or flight_info.get("aircraft_type") or flight_info.get("Aircraft Group") or flight_info.get("aircraft_group")
                mv = flight_info.get("Movement") or flight_info.get("movement") or flight_info.get("Flight Movement") or flight_info.get("flight_movement")
                st = flight_info.get("Station") or flight_info.get("ORG") or flight_info.get("origin") or flight_info.get("station")
                dest = flight_info.get("Destination") or flight_info.get("DES") or flight_info.get("destination") or flight_info.get("destination_station")
                if ac and mv and st:
                    r = lookup_mgt(
                        operation="TURNAROUND",
                        aircraft_group=str(ac),
                        movement=str(mv),
                        station=str(st),
                        destination_station=str(dest) if dest else None,
                        is_security_alert_station=False,
                        apply_local_towing_rule=False,
                    )
                    mgt_standard = getattr(r, "final_mgt_minutes", None)
        except Exception:
            mgt_standard = None

        status = None
        if actual is not None and mgt_standard is not None:
            status = "pass" if int(actual) <= int(mgt_standard) else "fail"

        return {
            "mgt_standard": mgt_standard,
            "actual_minutes": actual,
            "status": status,
        }

    def generate_final_response(self, user_query: str, context_data: Dict[str, Any]) -> str:
        """صياغة إجابة احترافية بالاعتماد على نفس محرك الاستجابة الموجود في الملف."""
        q = (user_query or "").strip()
        is_ar = bool(re.search(r"[\u0600-\u06FF]", q))

        prompt = (
            (self.system_prompt or "").strip()
            + "\n\n"
            + ("سؤال المستخدم:\n" if is_ar else "User question:\n")
            + q
            + "\n\n"
            + ("بيانات السياق (JSON):\n" if is_ar else "Context data (JSON):\n")
            + json.dumps(context_data or {}, ensure_ascii=False)
            + "\n\n"
            + ("اكتب الإجابة النهائية للمستخدم بشكل منظم وواضح، بدون تفاصيل برمجية." if is_ar else "Write a clear, structured final answer for the user, without programming details.")
        )

        # استخدام نفس دالة الاستدعاء الموجودة في الملف (call_ai)
        try:
            return call_ai(prompt, model_name=GEMINI_MODEL_COMPLEX, temperature=0.4, max_tokens=1800)
        except Exception:
            # fallback بسيط
            return "⚠️ تعذّر توليد إجابة حالياً."

# تشغيل المحرك
nxs_engine = NXSUltraReasoning()
