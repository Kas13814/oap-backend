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
import re
from typing import Any, Dict, List, Tuple, Optional


def _normalize_colname(name: str) -> str:
    # إلغاء التحويل القسري لـ snake_case لضمان مطابقة جداولك التسعة
    return name.strip()

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
import asyncio
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel

from nxs_visual_engine import build_chart

import time
import re

from nxs_semantic_engine import NXSSemanticEngine, build_query_plan
try:
    from nxs_semantic_engine import interpret_with_filters
except Exception:
    interpret_with_filters = None



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
GEMINI_FLASH_MODEL = os.getenv(
    "GEMINI_FLASH_MODEL",
    os.getenv("GEMINI_MODEL_SIMPLE", os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash"))
)
GEMINI_PRO_MODEL = os.getenv(
    "GEMINI_PRO_MODEL",
    os.getenv("GEMINI_MODEL_COMPLEX", "gemini-2.5-pro")
)
# default model (fast/cheap)
GEMINI_MODEL_NAME = GEMINI_FLASH_MODEL
# نستخدم نموذجاً واحداً معاد الاستخدام لتقليل الحمل وتحسين الثبات (اختياري)
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

SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

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




class VizRequest(BaseModel):
    rows: List[Dict[str, Any]] = []
    chart_type: str = "bar"   # bar | line | pie | table
    x: Optional[str] = None
    y: Optional[str] = None
    names: Optional[str] = None
    values: Optional[str] = None
    title: Optional[str] = None

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
        candidates: List[Dict[str, str]] = [dict(params)]

        # Identify filter keys (exclude control params)
        control_keys = {"select", "limit", "offset", "order"}
        filter_keys = [k for k in params.keys() if k not in control_keys]

        if len(filter_keys) == 1:
            k = filter_keys[0]
            v = str(params[k])
            for vk in _colname_variants(k):
                if vk == k:
                    continue
                cand = dict(params)
                cand.pop(k, None)
                cand[vk] = v
                candidates.append(cand)
        elif len(filter_keys) > 1:
            norm_params = dict(params)
            changed = False
            for k in list(filter_keys):
                nk = _normalize_colname(k)
                if nk != k:
                    norm_params[nk] = str(norm_params.pop(k))
                    changed = True
            if changed:
                candidates.append(norm_params)

        for cand in candidates:
            data = _supabase_get(url, headers, cand)
            if data:
                return data
        return []
    except Exception:
        return []


# =========================
#  8. Gemini Gateway (Anti-429 Guard)
# =========================

# إذا تم تجاوز الحصة (429) نُوقف محاولات Gemini مؤقتاً لتفادي تكرار الخطأ للمستخدم
_GEMINI_DISABLED_UNTIL_EPOCH: float = 0.0


def _set_gemini_cooldown(seconds: int) -> None:
    global _GEMINI_DISABLED_UNTIL_EPOCH
    now = time.time()
    _GEMINI_DISABLED_UNTIL_EPOCH = max(_GEMINI_DISABLED_UNTIL_EPOCH, now + max(0, int(seconds)))


def _gemini_is_disabled() -> bool:
    return time.time() < _GEMINI_DISABLED_UNTIL_EPOCH


async def call_gemini(prompt: str, use_pro: bool = False) -> str:
    """
    نسخة احترافية تستخدم httpx للتعامل مع الموديلات الهجينة (Pro/Flash)
    وتعمل بمفتاح Google AI Studio (بدون OAuth2).
    """
    if not GEMINI_API_KEY:
        return "⚠️ لم يتم ضبط GEMINI_API_KEY في الخادم."

    pro_model = os.getenv("GEMINI_PRO_MODEL", GEMINI_PRO_MODEL)
    flash_model = os.getenv("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL)
    target_model = pro_model if use_pro else flash_model

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
        },
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)

            if response.status_code == 200:
                result = response.json()
                return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

            if response.status_code == 429:
                logger.error("Quota exceeded / rate-limited: %s", response.text)
                return "⚠️ تم تجاوز حد الطلبات، حاول مرة أخرى بعد قليل."

            logger.error("Gemini API Error: %s - %s", response.status_code, response.text)
            return "⚠️ عذراً، واجه النظام صعوبة في التحليل."

        except Exception as e:
            logger.error("Connection Error: %s", str(e))
            return "⚠️ فشل الاتصال بمحرك الذكاء الاصطناعي."


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

def _fallback_answer(msg: str, plan: Dict[str, Any], data_context: Optional[Dict[str, Any]] = None) -> str:
    """Fallback ذكي بدون LLM لتجنّب إسقاط الخدمة عند أخطاء الحصة/الاتصال."""
    msg = (msg or "").strip()
    intent = (plan or {}).get("intent") or "free_talk"
    filters = (plan or {}).get("filters") or {}

    if intent == "free_talk":
        return "تمام. اكتب سؤالك بشكل مباشر (رحلة / موظف / قسم / تاريخ) وسأجيبك من بيانات النظام قدر الإمكان."

    # إذا عندنا سياق بيانات
    if data_context:
        keys = [k for k, v in data_context.items() if v]
        if keys:
            k0 = keys[0]
            sample = data_context.get(k0) or []
            n = len(sample) if isinstance(sample, list) else 1
            return f"تمت قراءة البيانات بنجاح. (سياق: {k0}، سجلات: {n}). حدّد رقم الرحلة/الموظف أو التاريخ للحصول على نتيجة أدق."

    if filters:
        return "لم أجد سجلات مطابقة حالياً لهذه المعايير. جرّب تحديد التاريخ أو رقم الرحلة/الموظف بشكل أوضح."
    return "حالياً لا توجد بيانات كافية للإجابة. اذكر التاريخ + رقم الرحلة أو رقم الموظف."


# =========================
# 12.A Model Router (Flash/Pro)
# =========================

def _choose_use_pro(
    semantic_info: Optional[Dict[str, Any]],
    user_msg: str,
    plan: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
    """يقرر هل نستخدم Pro (للأسئلة المعقدة) أم Flash (للأسئلة البسيطة).

    أولوية القرار:
    1) model_hint/complexity_hint القادم من interpret_with_filters (إن وجد).
    2) fallback خفيف: طول السؤال + عدد الفلاتر + نوع intent.
    """
    try:
        if isinstance(semantic_info, dict):
            mh = semantic_info.get("model_hint") if isinstance(semantic_info.get("model_hint"), dict) else None
            if mh and mh.get("tier") == "complex":
                return True, {"source": "semantic_model_hint", "hint": mh}
            if mh and mh.get("tier") == "simple":
                return False, {"source": "semantic_model_hint", "hint": mh}

            ch = semantic_info.get("complexity_hint") if isinstance(semantic_info.get("complexity_hint"), dict) else None
            if ch and ch.get("tier") == "complex":
                return True, {"source": "semantic_complexity_hint", "hint": ch}
            if ch and ch.get("tier") == "simple":
                return False, {"source": "semantic_complexity_hint", "hint": ch}
    except Exception:
        pass

    # Fallback heuristics (خفيفة وسريعة)
    msg = (user_msg or "").strip()
    words = msg.split()

    if plan and isinstance(plan.get("filters"), dict) and len(plan.get("filters") or {}) >= 2:
        return True, {"source": "fallback", "reason": "multiple_filters"}

    if plan and isinstance(plan.get("intent"), str) and plan.get("intent") in {"mgt_compliance", "flight_analysis"}:
        return True, {"source": "fallback", "reason": "analysis_intent"}

    if len(words) >= 18:
        return True, {"source": "fallback", "reason": "long_question"}

    return False, {"source": "fallback", "reason": "simple_question"}

async def nxs_brain(user_msg: str) -> Tuple[str, Dict[str, Any]]:
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
            # إذا كان interpret_with_filters متاحاً يرجع model_hint/complexity_hint أيضاً
            if interpret_with_filters:
                semantic_info = interpret_with_filters(SEMANTIC_ENGINE, msg)
            else:
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

    raw_plan = await call_gemini("".join(classifier_prompt_parts))

    # إذا Gemini غير متاح/تم تجاوز الحصة: استخدم خطة بديلة من التحليل الدلالي أو Free Talk
    if not raw_plan:
        if isinstance(semantic_info, dict) and semantic_info.get("intent"):
            plan = {"intent": semantic_info.get("intent", "free_talk"), "filters": semantic_info.get("filters", {}) or {}}
        else:
            plan = {"intent": "free_talk", "filters": {}}
    else:
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

    use_pro_final, routing_meta = _choose_use_pro(semantic_info, msg, plan)
    final_response = await call_gemini(analyst_prompt, use_pro=use_pro_final)
    if not final_response:
        # لا تُظهر أي خطأ للمستخدم — استخدم fallback محلي
        final_response = _fallback_answer(msg, plan, data_context)
    add_to_history("assistant", final_response)

    meta = {
        "plan": plan,
        "data_sources": list(data_context.keys()),
        "semantic": semantic_info,
        "model_routing": routing_meta,
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
async def chat(req: ChatRequest) -> Dict[str, Any]:
    msg = (req.message or "").strip()
    add_to_history("user", msg)

    if not msg:
        return {"reply": "...", "meta": {}}

    try:
        reply, meta = await nxs_brain(msg)
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

@app.post("/viz")
def viz(req: VizRequest) -> Dict[str, Any]:
    out = build_chart(
        rows=req.rows,
        chart_type=req.chart_type,
        x=req.x,
        y=req.y,
        names=req.names,
        values=req.values,
        title=req.title,
    )
    return {
        "ok": out.ok,
        "figure": out.figure,          # Plotly JSON (للواجهة)
        "png_base64": out.png_base64,  # fallback image
        "table_html": out.table_html,  # جدول HTML
        "meta": out.meta,
        "error": out.error,
    }


