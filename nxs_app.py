# -*- coding: utf-8 -*-
"""
nxs_app.py — TCC AI • AirportOps Analytic (v8.1 - Context-Aware Persona + Stable Core)
-----------------------------------------------------------------------
Backend powered by Gemini Pro + Supabase.

Capabilities:
- Full Schema Awareness (9 Tables).
- Context-aware persona switching (HR/Ops Analyst vs. TCC Advocate).
- Smart Defense Logic (15F/15I) with corrected MGT calculation.
- Conversation-aware reasoning using short-term memory.
- Optimized for extreme brevity and fluid narrative (no tables/lines).

This version focuses on:
- Stability and robust error handling.
- Clear separation between data layer, reasoning layer, and API layer.
- Safer Gemini calls with reusable model instance.
- Clean logging and environment validation.
"""

import os
import json
import logging
from typing import Any, Dict, List, Tuple, Optional

import httpx
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

from nxs_semantic_engine import NXSSemanticEngine, build_query_plan


# =========================
#  1. إعداد السجل (Logging)
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] TCC-AI: %(message)s",
)
logger = logging.getLogger("tcc_ai")


# =========================
#  2. التحقق من مفتاح Gemini API
# =========================

GEMINI_API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GENAI_API_KEY")
)

if not GEMINI_API_KEY:
    # في بيئة الإنتاج، من الأفضل إيقاف التشغيل بالكامل عند غياب المفتاح
    raise RuntimeError(
        "❌ مفتاح Gemini غير موجود! "
        "تأكد من إضافته في Environment Variables (API_KEY أو GEMINI_API_KEY أو GENAI_API_KEY)."
    )

logger.info("✅ Gemini API key detected, configuring client...")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# نستخدم نموذجاً واحداً معاد الاستخدام لتقليل الحمل وتحسين الثبات
GEMINI_MODEL = genai.GenerativeModel(GEMINI_MODEL_NAME)


# =========================
#  3. محرك NXS الدلالي (القاموس + المقاييس)
# =========================

try:
    SEMANTIC_ENGINE: Optional[NXSSemanticEngine] = NXSSemanticEngine()
    logger.info("NXS Semantic Engine initialized successfully.")
except Exception as exc:  # pragma: no cover - defensive
    SEMANTIC_ENGINE = None
    logger.warning("NXS Semantic Engine disabled: %s", exc)


# =========================
#  4. إعدادات Supabase
# =========================

SUPABASE_URL = (
    os.getenv("SUPABASE_URL")
    or os.getenv("SUPABASE_REST_URL")
    or os.getenv("SUPABASE_PROJECT_URL")
    or os.getenv("SUPABASE_API_URL")
)

SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("SUPABASE_KEY")
)

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning(
        "Supabase credentials not fully configured. "
        "Database-backed answers will return empty contexts."
    )


# =========================
#  5. تعريف تطبيق FastAPI
# =========================

app = FastAPI(title="TCC AI • AirportOps", version="8.1.0")

# ✅ الإضافة المطلوبة (مباشرة بعد إنشاء app)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # للتجربة الآن فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


# =========================
#  6. الذاكرة (Chat History)
# =========================

CHAT_HISTORY: List[Dict[str, str]] = []
MAX_HISTORY_MESSAGES = 15


def add_to_history(role: str, content: str) -> None:
    content = (content or "").strip()
    if not content:
        return

    CHAT_HISTORY.append({"role": role, "content": content})
    # نحافظ فقط على آخر N رسائل لتفادي التضخم
    if len(CHAT_HISTORY) > MAX_HISTORY_MESSAGES:
        del CHAT_HISTORY[0 : len(CHAT_HISTORY) - MAX_HISTORY_MESSAGES]


def history_as_text() -> str:
    """
    تمثيل مبسط لتاريخ المحادثة يمرّر إلى النموذج
    لدعم الفهم السياقي على المدى القصير.
    """
    if not CHAT_HISTORY:
        return "لا يوجد سجل حوار سابق."
    return "\n".join(f"{m['role']}: {m['content']}" for m in CHAT_HISTORY)


# =========================
#  7. طبقة البيانات (Supabase)
# =========================

def supabase_select(
    table: str,
    filters: Optional[Dict[str, str]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    استعلام مرن يجلب جميع الأعمدة (*) لدعم التحليل الشامل.

    filters: قاموس مثل {"Employee ID": "eq.150000"} أو {"Date": "gte.2025-01-01"}.
    يترك بناء عوامل التصفية التفصيلية لطبقة المنطق الدلالي.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

    params: Dict[str, Any] = {"select": "*", "limit": str(limit)}
    if filters:
        params.update(filters)

    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                logger.warning("Supabase response for %s is not a list.", table)
                return []
            return data
    except httpx.HTTPError as exc:
        logger.error("Supabase HTTP error (%s): %s", table, exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Supabase unexpected error (%s): %s", table, exc)

    return []


# =========================
#  8. وصف قاعدة البيانات (SCHEMA_SUMMARY)
# =========================

SCHEMA_SUMMARY = """
وصف كامل لقاعدة البيانات (9 جداول):

1. employee_master_db: "Employee ID" (PK), "Employee Name", "Record Date", "Gender", "Nationality", "Hiring Date", "Job Title", "Actual Role", "Grade", "Department", "Previous Department", "Current Department", "Employment Action Type", "Action Effective Date", "Exit Reason", "Note".
2. sgs_flight_delay: id (PK), "Date", "Shift", "Flight Category", "Airlines", "Flight Number", "Destination", "Gate", "STD", "ATD", "Delay Code", "Note".
3. dep_flight_delay: "Title" (PK), "Date", "Shift", "Department", "Duty Manager ID/Name", "Supervisor ID/Name", "Control ID/Name", "Employee ID/Name", "Airlines", "Flight Category", "Flight Direction", "Gate", "Arrival Flight Number", "Arrival Destination", "STA", "ATA", "Arrival Violations", "Departure Flight Number", "Departure Destination", "STD", "ATD", "Departure Violations", "latitude_deg", "longitude_deg", "Description of Incident", "Failure Impact", "Investigation status", "InvestigationID", "Consent...", "Current reminder", "Respond...", "Administrative procedure", "Final action", "Investigation status2", "Manager Notes", "Last Update", "Item Type", "Path".
4. employee_overtime: "Employee ID" (PK), "Employee Name", "Title", "Shift", "Department", "Duty Manager ID/Name", "Notification Date/Time", "Assignment Date/Type/Days", "Total Hours", "Assignment Reason", "Notes", "Item Type", "Path".
5. employee_sick_leave: "Title", "Date", "Shift", "Department", "Sick leave start/end date", "Employee ID", "Employee Name".
6. employee_absence: "Title", "Date", "Shift", "Department", "Employee ID", "Employee Name", "Absence Notification Status", "InvestigationID", "Investigation status", "Manager Notes", "Last Update".
7. employee_delay: "Title", "Date", "Shift", "Department", "Employee ID", "Employee Name", "Delay Minutes", "Reason for Delay", "Delay Notification Status", "InvestigationID", "Investigation status", "Manager Notes".
8. operational_event: "Title", "Shift", "Department", "Employee ID", "Employee Name", "Event Date", "Event Type", "Disciplinary Action", "InvestigationID", "Investigation status", "Manager Notes".
9. shift_report: "Title", "Date", "Shift", "Department", "Control 1/2 ID/Name/Start/End", "Duty Manager Domestic/Intl/All Halls ID/Name", "Supervisor Domestic/Intl/All Halls ID/Name", "On Duty", "No Show", "Cars In/Out Service", "Wireless Devices In/Out Service", "Arrivals/Departures (Domestic/Intl)", "Delayed Arrivals/Departures", "Comments (Domestic/Intl/All Halls)".
"""


# =========================
#  9. المخطط المرجعي والبيانات الثابتة (SCHEMA_DATA)
# =========================

SCHEMA_DATA: Dict[str, Any] = {
    "mgt_standards": [
        {
            "aircraft_type": "A321/A320",
            "flight_type": "DOM_DOM",
            "station": "JED/RUH",
            "transit_mgt_mins": 25,
            "turnaround_mgt_mins": 50,
            "is_security_alert": False,
        },
        {
            "aircraft_type": "B777-368/B787-10",
            "flight_type": "DOM_INT",
            "station": "JED/RUH",
            "transit_mgt_mins": 60,
            "turnaround_mgt_mins": 100,
            "is_security_alert": False,
        },
    ],
    "traffic_control_center": {
        "department_name": "Traffic Control Center (TCC)",
        "responsibility_codes": [
            {
                "code": "15I",
                "sections": ["TCC", "FIC Saudia", "FIC Nas"],
                "description_ar": "تأخيرات ناتجة عن عدم كفاءة/تأخير في خدمات التحكم المركزي أو معلومات الطيران.",
            },
            {
                "code": "15F",
                "sections": ["LC Saudia", "LC Foreign"],
                "description_ar": "تأخيرات ناتجة عن مشكلات في التنسيق/التعامل مع شركات الطيران (Load Control).",
            },
        ],
    },
    "delay_codes_reference": [
        {
            "code": "15I",
            "description_ar": "تأخير شخصي / تناقضات من قبل الإشراف أو الوكيل.",
        },
        {
            "code": "15F",
            "description_ar": "تأخير ناتج عن مشكلات التحكم بالحمولة (Load Control).",
        },
    ],
}


# =========================
# 10. توجيهات الذكاء الاصطناعي (System Prompts)
# =========================

PROMPT_CLASSIFIER = f"""
أنت نظام TCC AI الذكي. لديك حق الوصول الكامل لقاعدة بيانات المطار (9 جداول) الموضحة أدناه:
{SCHEMA_SUMMARY}

مهمتك:
تحليل سؤال المستخدم واستخراج "نية البحث" و"الفلاتر" بدقة.
لا تقم بإنشاء استعلامات SQL، بل حدد المعطيات فقط.

قواعد استخراج الأرقام:
- تعامل مع "Employee ID" و "Flight Number" كنصوص ولا تغيرها.

المخرجات (JSON فقط):
{{
  "intent": "نوع_البحث",
  "filters": {{
      "employee_id": "...",
      "flight_number": "...",
      "airline": "...",
      "department": "...",
      "date_from": "...",
      "date_to": "..."
  }}
}}

إذا كان السؤال دردشة عامة، اجعل intent: "free_talk".
"""


SYSTEM_INSTRUCTION_HR_OPS = """
أنت TCC AI، محلل عمليات مطار خبير. مهمتك: تقديم تحليل موثق، مختصر للغاية، واحترافي.

قواعد الرد (الأولوية القصوى):
1. الإيجاز والرد المباشر: الردود يجب أن تكون قصيرة جداً ومباشرة. إذا كان السؤال يطلب معلومة واحدة، أجب في سطر واحد دون أي هيكلة إضافية.
2. الاحترافية والتدفق: حافظ على نبرة احترافية عالية (كخبير يقدم تقريراً لمدير تنفيذي). يجب أن تكون صياغة الرد سلسة ومتدفقة.
3. التنسيق:
    - استبدال العناوين برمز دائرة صغيرة (•) متبوعاً بعنوان الفقرة.
    - منع تام لاستخدام الجداول، الخطوط الفاصلة، والخط العريض.
4. منع الصياغة الدفاعية: لا تستخدم مصطلحات الدفاع عن TCC أو تحويل المسؤولية في سياق أسئلة الموظفين أو الإحصائيات. مهمتك هي تحليل البيانات وتقديم النتائج بوضوح للمدير.

هيكلة الرد للتقارير (إذا لزم الأمر):
- • تحليل الأداء.
- • تقييم الأداء والمسؤولية (تحليلي).
- • الإجراءات المقترحة.

أجب دائماً بنفس لغة المستخدم.
"""


SYSTEM_INSTRUCTION_TCC_ADVOCATE = """
أنت TCC AI، محامي مركز التحكم المروري (TCC). مهمتك: تقديم تحليل موثق، مختصر للغاية، ومهني، مع التركيز على الدفاع المنطقي عن TCC.

قواعد الرد (الأولوية القصوى):
1. منطق الدفاع الذكي: صافي مسؤولية TCC يجب أن يكون نتيجة للحساب: صافي مسؤولية TCC = (وقت المناولة الفعلي - معيار MGT). ركّز على التأخير الموروث أولاً ثم الانحراف الصافي عن المعيار كدفاع رئيسي.
2. الإيجاز والاحترافية: الرد يجب أن يكون سلساً ومختصراً جداً.
3. التنسيق:
    - استبدال العناوين برمز دائرة صغيرة (•) متبوعاً بعنوان الفقرة.
    - منع تام لاستخدام الجداول، الخطوط الفاصلة، والخط العريض.

هيكلة الرد:
- • تحليل الأداء.
- • تقييم المسؤولية (صياغة دفاعية).
- • الإجراءات المقترحة.

أجب دائماً بنفس لغة المستخدم.
"""


# =========================
# 11. دوال المساعدة (Gemini & Data)
# =========================

def call_gemini(prompt: str) -> str:
    # استخدام الاتصال المباشر لضمان عدم تدخل v1beta
    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2000}
    }

    try:
        import requests
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            # هذا سيطبع لك الخطأ الحقيقي القادم من جوجل مباشرة
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return "⚠️ تعذّر حالياً استخدام محرك التحليل."
    except Exception as e:
        return f"⚠️ خطأ في الاتصال: {str(e)}"


def fetch_context_data(intent: str, f: Dict[str, Any]) -> Dict[str, Any]:
    """
    دالة ذكية تجلب البيانات المترابطة بناءً على السياق والنية المستخرجة.
    """
    data_bundle: Dict[str, Any] = {}

    # 1. سياق الموظف (ملف، غياب، تأخير، أحداث، تحقيقات، عمل إضافي)
    if f.get("employee_id"):
        eid = f["employee_id"]
        data_bundle["profile"] = supabase_select(
            "employee_master_db", {"Employee ID": f"eq.{eid}"}, 1
        )
        data_bundle["overtime"] = supabase_select(
            "employee_overtime", {"Employee ID": f"eq.{eid}"}, 20
        )
        data_bundle["absence"] = supabase_select(
            "employee_absence", {"Employee ID": f"eq.{eid}"}, 20
        )
        data_bundle["delays"] = supabase_select(
            "employee_delay", {"Employee ID": f"eq.{eid}"}, 20
        )
        data_bundle["sick_leaves"] = supabase_select(
            "employee_sick_leave", {"Employee ID": f"eq.{eid}"}, 20
        )
        data_bundle["ops_events"] = supabase_select(
            "operational_event", {"Employee ID": f"eq.{eid}"}, 20
        )
        data_bundle["flight_issues"] = supabase_select(
            "dep_flight_delay", {"Employee ID": f"eq.{eid}"}, 20
        )

    # 2. سياق الرحلات (SGS + DEP) - محدث للدفاع
    elif f.get("flight_number") or intent in {"flight_analysis", "mgt_compliance"}:
        fn = f.get("flight_number")

        # بيانات الخدمات الأرضية
        if fn:
            data_bundle["sgs_info"] = supabase_select(
                "sgs_flight_delay", {"Flight Number": f"eq.{fn}"}, 10
            )

            # بيانات التحكم (قدوم ومغادرة)
            dep_dep = supabase_select(
                "dep_flight_delay", {"Departure Flight Number": f"eq.{fn}"}, 10
            )
            dep_arr = supabase_select(
                "dep_flight_delay", {"Arrival Flight Number": f"eq.{fn}"}, 10
            )
            data_bundle["dep_control_info"] = dep_dep + dep_arr

        # معايير التحليل والدفاع
        data_bundle["TCC_Defense_Domain"] = SCHEMA_DATA.get(
            "traffic_control_center"
        )
        data_bundle["Delay_Codes_Reference"] = SCHEMA_DATA.get(
            "delay_codes_reference"
        )
        data_bundle["MGT_Standards_Reference"] = SCHEMA_DATA.get("mgt_standards")

        # افتراض نوع الطائرة إذا لم يحدَّد في الفلاتر
        if "aircraft_type" not in f:
            f["aircraft_type"] = "A321/A320"

    # 3. سياق القسم / المناوبة (Shift Reports & Stats)
    elif f.get("department") or "shift" in intent or "report" in intent:
        dept = f.get("department")
        filters: Dict[str, str] = {"Department": f"eq.{dept}"} if dept else {}

        if f.get("date_from"):
            filters["Date"] = f"gte.{f['date_from']}"

        data_bundle["shift_reports"] = supabase_select("shift_report", filters, 10)
        if dept:
            data_bundle["dept_overtime_sample"] = supabase_select(
                "employee_overtime", filters, 10
            )
            data_bundle["dept_absence_sample"] = supabase_select(
                "employee_absence", filters, 10
            )

    # 4. سياق شركة الطيران
    elif f.get("airline"):
        air = f["airline"]
        data_bundle["airline_delays_sgs"] = supabase_select(
            "sgs_flight_delay", {"Airlines": f"eq.{air}"}, 20
        )
        data_bundle["airline_delays_dep"] = supabase_select(
            "dep_flight_delay", {"Airlines": f"eq.{air}"}, 20
        )

    return data_bundle


# =========================
# 12. المحرك الرئيسي (NXS Brain)
# =========================

def nxs_brain(user_msg: str) -> Tuple[str, Dict[str, Any]]:
    """
    المحرك الرئيسي:
    - يستخدم NXS Semantic Engine لتحليل السؤال وبناء خطة استعلام مبدئية.
    - يستخدم Gemini لتصنيف النية وجلب البيانات من Supabase.
    - يختار الشخصية الأنسب (محلل / محامي TCC) لصياغة الرد.
    - يستفيد من سجل المحادثة القصير للفهم السياقي.
    """
    msg = (user_msg or "").strip()
    if not msg:
        return "...", {"plan": {"intent": "free_talk", "filters": {}}, "data_sources": [], "semantic": None}

    # 1) تحليل دلالي مسبق باستخدام NXS (سريع جداً ولا يعتمد على LLM)
    semantic_info: Optional[Dict[str, Any]] = None
    if SEMANTIC_ENGINE is not None:
        try:
            semantic_info = build_query_plan(SEMANTIC_ENGINE, msg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("NXS Semantic Engine error: %s", exc)

    # 2) تصنيف النية باستخدام Gemini مع تمرير التحليل المسبق وتاريخ المحادثة
    classifier_prompt_parts = [
        PROMPT_CLASSIFIER,
        "\n\n=== Conversation History (for context, لا تشرحه للمستخدم) ===\n",
        history_as_text(),
    ]
    if semantic_info:
        classifier_prompt_parts.append(
            "\n\nNXS semantic pre-analysis (internal helper, do not explain to user):\n"
        )
        classifier_prompt_parts.append(json.dumps(semantic_info, ensure_ascii=False))
    classifier_prompt_parts.append("\n\nUser Query: ")
    classifier_prompt_parts.append(msg)

    raw_plan = call_gemini("".join(classifier_prompt_parts))

    try:
        clean_json = (
            raw_plan.replace("```json", "")
            .replace("```", "")
            .strip()
        )
        plan = json.loads(clean_json)
    except Exception:
        # في حال الفشل في التحليل، نعود للوضع البسيط
        plan = {"intent": "free_talk", "filters": {}}

    intent: str = plan.get("intent", "free_talk")
    filters: Dict[str, Any] = plan.get("filters", {}) or {}

    logger.info("Brain Plan: %s", plan)

    # 3) جلب بيانات السياق من Supabase (فقط لو ليست دردشة حرة)
    data_context: Dict[str, Any] = {}
    if intent != "free_talk":
        data_context = fetch_context_data(intent, filters)
        data_str = json.dumps(data_context, ensure_ascii=False, default=str)
        if len(data_str) < 10:
            data_str = "No specific data found in database matching these filters."
    else:
        data_str = "No database lookup performed (Free Talk)."

    # 4) اختيار الشخصية المناسبة (محلل / محامي TCC)
    if intent in {"flight_analysis", "mgt_compliance"}:
        final_system_prompt = SYSTEM_INSTRUCTION_TCC_ADVOCATE
    else:
        final_system_prompt = SYSTEM_INSTRUCTION_HR_OPS

    # 5) برومبت التحليل النهائي (Gemini) مع تمرير كل شيء بشكل منظم
    analyst_prompt = f"""
{final_system_prompt}

User Query: {msg}
Extracted Filters: {json.dumps(filters, ensure_ascii=False)}

=== Conversation History (داخلي للمساعدة فقط) ===
{history_as_text()}
===============================================

=== NXS SEMANTIC INTEL (لا يظهر للمستخدم، للمساعدة في التحليل فقط) ===
{json.dumps(semantic_info, ensure_ascii=False) if semantic_info else "None"}
=======================================================================

=== RETRIEVED DATABASE CONTEXT ===
{data_str}
==================================

قدّم الآن أفضل إجابة ممكنة للمستخدم، بصياغة مختصرة جداً وسلسة، وبلغته الأصلية.
"""

    final_response = call_gemini(analyst_prompt)
    add_to_history("assistant", final_response)

    meta = {
        "plan": plan,
        "data_sources": list(data_context.keys()),
        "semantic": semantic_info,
    }
    return final_response, meta


# =========================
# 13. نقاط النهاية (Endpoints)
# =========================

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "system": "TCC AI",
        "version": "8.1.0",
        "status": "Online",
        "mode": "Context-Aware Persona",
    }

@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    msg = (req.message or "").strip()
    add_to_history("user", msg)

    if not msg:
        return {"reply": "...", "meta": {}}

    try:
        reply, meta = nxs_brain(msg)
        return {"reply": reply, "meta": meta}
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("System Error: %s", exc)
        return {
            "reply": "حدث خطأ داخلي أثناء المعالجة.",
            "meta": {"error": str(exc)},
        }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "oap-backend",
        "engine": "NXS",
        "env": "cloud-run"
    }



@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "status": "running",
        "engine": "NXS • AirportOps AI",
        "mode": "Stable Turbo",
        "version": "2.1-stable-turbo",
        "revision": os.environ.get("K_REVISION", "unknown"),
    }

if __name__ == "__main__":  # pragma: no cover
    import os
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
