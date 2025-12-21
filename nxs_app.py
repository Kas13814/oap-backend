# -*- coding: utf-8 -*-
"""
nxs_app.py — OAP • AirportOps Analytic (v8.1 - Context-Aware Persona + Stable Core)
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


def _normalize_colname(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\s\-\/]+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_]+", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()

def _colname_variants(name: str) -> List[str]:
    base = name.strip()
    norm = _normalize_colname(base)
    variants: List[str] = []
    def add(x: str) -> None:
        x = x.strip()
        if x and x not in variants:
            variants.append(x)
    add(base); add(norm)
    if "_" in norm:
        add(norm.replace("_", " "))
        add(norm.replace("_", "-"))
        add(norm.replace("_", ""))
        add(" ".join([w.capitalize() for w in norm.split("_")]))
        add("_".join([w.capitalize() for w in norm.split("_")]))
    add(base.replace(" ", "_"))
    add(base.replace(" ", ""))
    add(base.lower())
    return variants

def _supabase_get(url: str, headers: Dict[str, str], params: Dict[str, str]) -> List[Dict[str, Any]]:
    with httpx.Client(timeout=45.0) as client:
        r = client.get(url, headers=headers, params=params)
        if r.status_code >= 400:
            return []
        try:
            data = r.json()
            return data if isinstance(data, list) else []
        except Exception:
            return []


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
    format="%(asctime)s [%(levelname)s] OAP-AI: %(message)s",
)
logger = logging.getLogger("oap_ai")


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
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
GEMINI_PRO_MODEL   = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro")
# default model (fast/cheap)
GEMINI_MODEL_NAME  = GEMINI_FLASH_MODEL
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

app = FastAPI(title="OAP • AirportOps", version="8.1.0")

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
    # Try strict params first, then robust column-name variants if the result is empty.
    # This fixes 'not found' when DB columns are snake_case but the planner produced Title Case (or vice versa).
    if not params:
        return _supabase_get(url, headers, params)

    candidates: List[Dict[str, str]] = [dict(params)]

    if len(params) == 1:
        k, v = next(iter(params.items()))
        for vk in _colname_variants(k):
            if vk == k:
                continue
            candidates.append({vk: v})
    else:
        norm_params: Dict[str, str] = {}
        changed = False
        for k, v in params.items():
            nk = _normalize_colname(k)
            norm_params[nk] = v
            changed = changed or (nk != k)
        if changed:
            candidates.append(norm_params)

    for cand in candidates:
        data = _supabase_get(url, headers, cand)
        if data:
            return data
    return []
def call_gemini(prompt: str, use_pro: bool = False) -> str:
    """
    Call Gemini via direct HTTP (stable v1).
    - use_pro=False => fast/cheap model (Flash)
    - use_pro=True  => stronger model (Pro)
    """
    if not GEMINI_API_KEY:
        return "⚠️ مفتاح GEMINI غير موجود على الخادم."

    target_model = GEMINI_PRO_MODEL if use_pro else GEMINI_FLASH_MODEL
    url = f"https://generativelanguage.googleapis.com/v1/models/{target_model}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
    }

    try:
        # httpx موجود لديك في المشروع بالفعل
        resp = httpx.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            # تسجيل الخطأ للمطور + إرجاع رسالة مختصرة للمستخدم
            logger.error(f"AI engine HTTP {resp.status_code}: {resp.text}")
            return f"⚠️ تعذّر حالياً استخدام محرك التحليل الذكي في الخلفية. (AI {resp.status_code})"
    except Exception as e:
        logger.exception("Gemini call failed")
        return f"⚠️ فشل الاتصال بمحرك الذكاء: {str(e)}"

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
        "system": "OAP - Operation Analytical Platform (KAS)",
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
