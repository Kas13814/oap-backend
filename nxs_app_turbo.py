# -*- coding: utf-8 -*-
"""
nxs_app_turbo.py — NXS • AirportOps AI (Stable Turbo Edition)
-------------------------------------------------------------
هذه النسخة مصممة لتكون:
- سريعة ⚡
- مستقرة 🛡️ (لا ترجع 500 للمستخدم، بل رسالة مفهومة دائماً)
- متوافقة مع nxs_brain (أي نسخة حالية لديك)
- بدون أي ذكر مباشر لمحرك Gemini في الـ API

ملاحظة:
- إذا حدث أي خطأ داخل nxs_brain، سيتم التقاطه وإرجاع رسالة نصية للمستخدم مع meta توضح الخطأ.
- لا نستخدم HTTPException برمجية 500 حتى لا تظهر رسالة "Error: empty reply from server" في الواجهة.
"""

import time
import logging
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nxs_brain import nxs_brain

app = FastAPI(
    title="NXS • AirportOps AI",
    version="2.1-stable-turbo",
)

# ---------------- إعداد السجل ----------------
logger = logging.getLogger("nxs_app_turbo")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ------------- نماذج الطلب / الرد -------------
class ChatRequest(BaseModel):
    message: str
    lang: Optional[str] = "ar"  # احتياطي للمستقبل إذا أحببنا تمرير اللغة من الواجهة


class ChatResponse(BaseModel):
    reply: str
    meta: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None


# ------------- كاش بسيط لتسريع الأسئلة المكررة -------------
CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 20  # ثوانٍ قليلة لعمر الكاش (قصير حتى نبقى أقرب للبيانات الحية)


def cache_get(key: str) -> Optional[Dict[str, Any]]:
    item = CACHE.get(key)
    if not item:
        return None
    if time.time() - item["time"] > CACHE_TTL:
        return None
    return item["value"]


def cache_set(key: str, value: Dict[str, Any]) -> None:
    CACHE[key] = {"value": value, "time": time.time()}


# ------------- نقطة /chat الرئيسية -------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    نقطة المحادثة الرئيسية:
    - تقرأ message من المستخدم.
    - تنظّفها وتتأكد من عدم كونها فارغة.
    - تستدعي nxs_brain(message).
    - ترجع الرد + meta + زمن التنفيذ.
    - في حالة أي خطأ، ترجع رسالة نصية للمستخدم، وليس 500.
    """
    start = time.time()
    msg = (req.message or "").strip()

    if not msg:
        return ChatResponse(
            reply="الرسالة الواردة فارغة. يرجى كتابة سؤالك أو طلبك بشكل واضح.",
            meta={"ok": False, "reason": "empty_message"},
            latency_ms=round((time.time() - start) * 1000.0, 2),
        )

    # 1) فحص الكاش (إذا نفس السؤال تكرر خلال الفترة القصيرة)
    cached = cache_get(msg)
    if cached is not None:
        return ChatResponse(
            reply=cached["reply"],
            meta={
                **(cached.get("meta") or {}),
                "from_cache": True,
            },
            latency_ms=0.5,  # لأن الرد من الكاش شبه فوري
        )

    # 2) استدعاء nxs_brain مع حماية كاملة من الأخطاء
    try:
        reply, meta = nxs_brain(msg)
        # حفظ في الكاش
        cache_set(msg, {"reply": reply, "meta": meta})
        latency = round((time.time() - start) * 1000.0, 2)
        return ChatResponse(
            reply=reply,
            meta={
                **(meta or {}),
                "from_cache": False,
            },
            latency_ms=latency,
        )

    except Exception as e:  # pragma: no cover - حماية دفاعية
        # تسجيل الخطأ في السجل الداخلي
        logger.error("Unhandled error in /chat handler: %s", e, exc_info=True)

        fallback_reply = (
            "حدث خطأ داخلي أثناء معالجة الطلب داخل NXS • AirportOps AI. "
            "يمكن مراجعة سجل الخادم (logs) لمعرفة تفاصيل أكثر عن الخطأ."
        )
        latency = round((time.time() - start) * 1000.0, 2)
        return ChatResponse(
            reply=fallback_reply,
            meta={
                "ok": False,
                "error": str(e),
                "source": "nxs_app_turbo_chat_handler",
            },
            latency_ms=latency,
        )


# ------------- نقاط فحص الصحة / المعلومات العامة -------------
@app.get("/")
async def home() -> Dict[str, Any]:
    return {
        "status": "running",
        "engine": "NXS • AirportOps AI",
        "mode": "Stable Turbo",
        "version": "2.1-stable-turbo",
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "engine": "NXS • AirportOps AI",
        "uptime_mode": "Stable Turbo",
    }
