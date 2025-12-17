# -*- coding: utf-8 -*-
"""
nxs_brain_template.py
---------------------
قالب جاهز لبناء "عقل NXS" مخصص فوق:
- مصنف النوايا rule‑based من nxs_intents.classify_intent
- محرك الاستعلامات الديناميكية execute_dynamic_query من nxs_supabase_client
- نموذج LLM (Gemini أو غيره) عبر test_gemini_key.call_model_text

الفكرة:
1) نحدد النية والكيانات من السؤال.
2) نطلب من النموذج بناء خطة (PLAN) على شكل JSON فقط.
   - plan.mode : "chat_only" | "sql_and_answer"
   - plan.sql  : استعلام SQL جاهز لـ execute_dynamic_query إذا احتجنا بيانات.
3) إذا احتجنا بيانات، ننفّذ الاستعلام ثم نطلب من النموذج أن يجيب
   بالاعتماد على السؤال والـ PLAN والصفوف.
4) دائماً نعود بـ (answer, meta) حيث meta يحتوي على تفاصيل تقنية.
"""

from typing import Tuple, Dict, Any, List
import json
import textwrap
import traceback

from nxs_intents import classify_intent
from nxs_supabase_client import execute_dynamic_query
from test_gemini_key import call_model_text


# وصف مبسط للمخطط – يمكنك استبداله بالوصف الكامل عند الحاجة
DB_SCHEMA_DESCRIPTION = textwrap.dedent("""
هنا وصف مبسط للجداول الرئيسية (مثال):
- employee_master_db: بيانات أساسية لكل موظف.
- tcc_flight_delay / sgs_flight_delay: سجلات تأخيرات الرحلات حسب المصدر.
- flight_operations_event_record: حوادث تشغيلية وتحقيقات مرتبطة بالرحلات والموظفين.
- employee_overtime / employee_absence / employee_delay / employee_sick_leave: سجلات الموارد البشرية.
يمكنك استبدال هذا النص بالوصف الكامل للمخطط عند النشر.
""").strip()


# ----------------- أدوات مساعدة داخلية -----------------


def _extract_json_block(text: str) -> str:
    """
    يستخرج أول كتلة JSON صالحة من النص المرسل من النموذج.

    الموديلات أحياناً تعيد نصاً يحتوي على شرح قبل/بعد الـ JSON،
    لذا نحاول قص الجزء الواقع بين أول { وآخر }.
    """
    text = (text or "").strip()
    if not text:
        return "{}"

    # إذا النص كله JSON مباشر
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return "{}"


# ----------------- مرحلة التخطيط (PLAN) -----------------


def _plan_with_gemini(message: str, intent_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    يطلب من النموذج بناء خطة عالية المستوى على شكل JSON فقط.

    المخرجات المتوقعة (مثال):
    {
      "mode": "sql_and_answer" | "chat_only",
      "sql": "SELECT ... FROM ... WHERE ...",
      "reason": "شرح مختصر لماذا اخترت هذا المسار.",
      "preferred_language": "ar" | "en"
    }
    """
    helper = json.dumps(intent_info, ensure_ascii=False)
    prompt = f"""
أنت NXS • AirportOps AI.
مهمتك الآن هي مرحلة التخطيط فقط (PLAN)، وليس كتابة الإجابة النهائية.

المعلومات المتاحة لك:
- وصف المخطط (Schema):
{DB_SCHEMA_DESCRIPTION}

- تحليل النية والكيانات المستخرجة (Intent Info) من نظام داخلي سريع:
{helper}

سؤال المستخدم:
"""{message}"""

قواعد مهمة جداً:
1) أعد مخرجاتك على شكل JSON صالح فقط، بدون أي شرح خارجي، بدون أسطر زائدة.
2) إذا كان السؤال دردشة عامة لا تحتاج بيانات، اجعل:
   - mode = "chat_only"
   - sql  = ""
3) إذا احتجنا بيانات من قاعدة البيانات، اجعل:
   - mode = "sql_and_answer"
   - sql  = استعلام SQL آمن للقراءة فقط (SELECT) مبني على الجداول المتاحة في الوصف.
4) always use snake_case للمفاتيح.

أعد الآن كائن JSON واحد فقط يمثل خطة التنفيذ.
"""

    raw = call_model_text(prompt)
    try:
        plan_json = _extract_json_block(raw)
        plan = json.loads(plan_json)
    except Exception:
        # في حال الفشل، نرجع خطة بسيطة تعتمد على الدردشة فقط
        plan = {
            "mode": "chat_only",
            "sql": "",
            "reason": f"فشل تحليل JSON من النموذج. النص الخام المختصر: {str(raw)[:200]}",
            "preferred_language": "ar",
        }
    if not isinstance(plan, dict):
        plan = {
            "mode": "chat_only",
            "sql": "",
            "reason": "النموذج لم يرجع JSON منظم، تم التحويل إلى وضع الدردشة فقط.",
            "preferred_language": "ar",
        }
    # ضمان وجود الحقول الأساسية
    plan.setdefault("mode", "chat_only")
    plan.setdefault("sql", "")
    plan.setdefault("reason", "")
    plan.setdefault("preferred_language", intent_info.get("language", "ar"))
    return plan


# ----------------- مرحلة الإجابة مع البيانات -----------------


def _answer_with_data(message: str, plan: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    """
    يطلب من النموذج توليد الإجابة النهائية بالاعتماد على:
    - سؤال المستخدم
    - خطة التنفيذ plan
    - الصفوف rows المسترجعة من قاعدة البيانات

    ملاحظة شكل الرد:
    - عربي أو إنجليزي بحسب سياق السؤال (لا تخلط اللغتين دون سبب).
    - لا تستخدم جداول Markdown أو خطوط فاصلة، استخدم فقرات قصيرة ونقاط فقط إذا لزم.
    """
    lang = plan.get("preferred_language", "ar")
    rows_json = json.dumps(rows, ensure_ascii=False)

    helper = json.dumps(plan, ensure_ascii=False)
    prompt = f"""
أنت NXS • AirportOps AI، محلل عمليات مطار وموارد بشرية.

سؤال المستخدم:
"""{message}"""

خطة التنفيذ (PLAN) التي تم بناؤها مسبقاً:
{helper}

البيانات المسترجعة من قاعدة البيانات (rows):
{rows_json}

قواعد صياغة الرد:
1) أجب باللغة الأنسب بناءً على سؤال المستخدم، وإذا كان السؤال بالعربية فلتكن الإجابة بالعربية.
2) لا تستخدم أي جداول أو خطوط فاصلة، فقط فقرات وجمل ونقاط بسيطة عند الحاجة.
3) ركّز على توضيح الأرقام والاتجاهات والاستنتاجات العملية (ما الذي يجب على المدير أو المشرف فهمه؟).
4) إذا كانت البيانات قليلة أو فارغة، وضّح ذلك بهدوء واقترح ما يمكن جمعه لاحقاً.

قدّم الآن أفضل إجابة ممكنة.
"""

    answer = call_model_text(prompt)
    return answer


# ----------------- نقطة الدخول الرئيسية للعقل -----------------


def nxs_brain(message: str) -> Tuple[str, Dict[str, Any]]:
    """
    النسخة القالبية من العقل:
    - تحدد intent_info.
    - تبني plan باستخدام LLM.
    - بحسب mode:
        * chat_only     → استدعاء النموذج للإجابة مباشرة.
        * sql_and_answer → تنفيذ SQL ثم استدعاء النموذج مع البيانات.
    - دائماً ترجع (answer, meta) حيث meta مفيد للواجهات المتقدمة / الـ logging.
    """
    intent_info = classify_intent(message)
    try:
        plan = _plan_with_gemini(message, intent_info)
        mode = plan.get("mode", "chat_only")
        sql = (plan.get("sql") or "").strip()

        # 1) وضع الدردشة فقط (بدون استعلام)
        if mode == "chat_only" or not sql:
            helper = json.dumps(intent_info, ensure_ascii=False)
            prompt = f"""
أنت NXS • AirportOps AI.

سؤال المستخدم:
"""{message}"""

تحليل النية (من نظام داخلي):
{helper}

قواعد الرد:
1) أجب إجابة مختصرة وواضحة، مع إمكانية التعمق عند الحاجة، لكن بدون إسهاب لا داعي له.
2) لا تستخدم الجداول أو الخطوط الفاصلة، فقط فقرات ونقاط بسيطة.
3) إذا كان السؤال عاماً عن قدراتك، فسّر قدراتك بإيجاز شديد مع أمثلة عملية مرتبطة بالرحلات والموظفين.

قدّم الآن أفضل إجابة ممكنة.
"""

            answer = call_model_text(prompt)
            meta: Dict[str, Any] = {
                "mode": "chat_only",
                "intent_info": intent_info,
                "plan": plan,
            }
            return answer, meta

        # 2) وضع استعلام + إجابة
        rows: List[Dict[str, Any]] = execute_dynamic_query(sql)
        row_count = len(rows)
        answer = _answer_with_data(message, plan, rows)

        meta = {
            "mode": "sql_and_answer",
            "intent_info": intent_info,
            "plan": plan,
            "sql": sql,
            "row_count": row_count,
            "sample_rows": rows[:10],
        }
        return answer, meta

    except Exception as exc:
        tb = traceback.format_exc()
        err_txt = (
            "حدث خطأ داخلي داخل NXS أثناء استخدام الذكاء الاصطناعي. "
            "يمكن مراجعة السجل التفصيلي (traceback) لمعرفة السبب.
"
            f"نوع الخطأ: {type(exc).__name__}\n"
        )
        meta = {
            "error": str(exc),
            "traceback": tb,
            "intent_info": intent_info,
        }
        return err_txt, meta
