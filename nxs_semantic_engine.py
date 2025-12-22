# مراجعة أولية لطبقة NXS Semantic Engine
# تم الاحتفاظ بالمنطق كما هو بدون تغيير جوهري حتى لا تتأثر بقية أجزاء النظام.
# يمكنك استخدام هذا الملف مكان الملف الأصلي في حال رغبتك بتنظيم أفضل.

"""
nxs_semantic_engine.py

طبقة تفسير السؤال (Semantic Layer) لمشروع NXS:

- تقرأ:
    - nxs_dictionary.json  (تعريف الجداول والأعمدة)
    - metrics_catalog.json (تعريف المقاييس KPIs)

- تقوم بـ:
    - تحليل نص السؤال (عربي/إنجليزي).
    - استخراج الكلمات المفتاحية.
    - مطابقتها مع:
        - الأعمدة (columns) في الجداول/الـ Views.
        - المقاييس (metrics) في كتالوج المقاييس.
    - ترجيح النتائج وإرجاع:
        - أفضل الأعمدة (مع الجدول).
        - أفضل المقاييس.
        - نوع الكيان الرئيسي (موظف، قسم، رحلة، …).

يمكن استدعاؤها من nxs_brain.py أو من أي جزء آخر في النظام.
"""

from __future__ import annotations

# --- COLUMN_NAME_NORMALIZATION ---
# Prefer planning with snake_case columns when possible.
# The app layer will also try variants (spaces/caps) as fallback.
# --------------------------------

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import json
import re


# ============== Helpers ==============

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")


def normalize_text(text: str) -> str:
    """
    توحيد بسيط للنص:
    - تحويل إلى lowercase
    - إزالة التكرار في المسافات
    - إزالة أغلب العلامات
    """
    if not text:
        return ""
    text = text.strip().lower()
    # استبدال أي شيء غير حرف/رقم/مسافة بشرطة سفلية
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    تحويل النص إلى tokens بسيطة.
    لا نستخدم مكتبات خارجية لتسهيل النشر.
    """
    if not text:
        return []
    return text.split()


def guess_language(text: str) -> str:
    """تخمين سريع للغة (ar / en / mixed / unknown)."""
    if not text:
        return "unknown"
    has_ar = bool(ARABIC_RANGE.search(text))
    has_en = bool(re.search(r"[a-zA-Z]", text))
    if has_ar and has_en:
        return "mixed"
    if has_ar:
        return "ar"
    if has_en:
        return "en"
    return "unknown"


# ============== Dataclasses للنتائج ==============

@dataclass
class ColumnMatch:
    table: str
    table_entity_type: Optional[str]
    column_original: str
    column_canonical: str
    score: int


@dataclass
class MetricMatch:
    metric_id: str
    name_en: str
    name_ar: str
    entity: str
    source_type: str
    source_name: str   # table/view name
    source_column: str
    score: int


@dataclass
class InterpretationResult:
    query: str
    normalized_query: str
    language: str
    dominant_entity: Optional[str]
    top_columns: List[ColumnMatch]
    top_metrics: List[MetricMatch]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "language": self.language,
            "dominant_entity": self.dominant_entity,
            "top_columns": [c.__dict__ for c in self.top_columns],
            "top_metrics": [m.__dict__ for m in self.top_metrics],
        }


# ============== الكلاس الرئيسي ==============

class NXSSemanticEngine:
    """
    المحرك الدلالي لـ NXS:
    - يقوم بتحميل القاموس والمقاييس.
    - يبني Index داخلي للمصطلحات.
    - يفسر السؤال ويرجع أفضل الأعمدة/المقاييس.
    """

    def __init__(
        self,
        dictionary_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent

        if dictionary_path is None:
            dictionary_path = base_dir / "nxs_dictionary.json"
        else:
            dictionary_path = Path(dictionary_path)

        if metrics_path is None:
            metrics_path = base_dir / "metrics_catalog.json"
        else:
            metrics_path = Path(metrics_path)

        if not dictionary_path.exists():
            raise FileNotFoundError(f"nxs_dictionary.json not found at {dictionary_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics_catalog.json not found at {metrics_path}")

        self.dictionary_path = dictionary_path
        self.metrics_path = metrics_path

        # سيتم ملؤها في _load_all()
        self.tables: Dict[str, Dict[str, Any]] = {}
        self.views: Dict[str, Dict[str, Any]] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}

        # الفهارس الدلالية
        # term -> list of refs (kind, key)
        self.term_index: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)
        # phrase (multi-word) -> list of refs
        self.phrase_index: Dict[str, List[Tuple[str, Any]]] = defaultdict(list)

        self._load_all()

    # ---------- تحميل البيانات وبناء الفهرس ----------

    def _load_all(self) -> None:
        """تحميل ملفات JSON وبناء الفهارس."""
        with self.dictionary_path.open("r", encoding="utf-8") as f:
            dic = json.load(f)

        with self.metrics_path.open("r", encoding="utf-8") as f:
            met = json.load(f)

        # حفظ الجداول والـ Views في قواميس للوصول السريع
        self.tables = {t["name"]: t for t in dic.get("tables", [])}
        self.views = {v["name"]: v for v in dic.get("views", [])}
        self.metrics = {m["id"]: m for m in met.get("metrics", [])}

        # بناء الفهرس
        self.term_index.clear()
        self.phrase_index.clear()

        # 1) أعمدة الجداول
        for table_name, table_def in self.tables.items():
            entity_type = table_def.get("entity_type")
            for col in table_def.get("columns", []):
                self._index_column(
                    source_name=table_name,
                    entity_type=entity_type,
                    col=col,
                    kind="table",
                )

        # 2) أعمدة الـ Views
        for view_name, view_def in self.views.items():
            entity_type = view_def.get("entity_type")
            for col in view_def.get("columns", []):
                self._index_column(
                    source_name=view_name,
                    entity_type=entity_type,
                    col=col,
                    kind="view",
                )

        # 3) المقاييس
        for metric_id, metric_def in self.metrics.items():
            self._index_metric(metric_id, metric_def)

    def _index_column(
        self,
        source_name: str,
        entity_type: Optional[str],
        col: Dict[str, Any],
        kind: str,
    ) -> None:
        """
        إضافة عمود للفهرس.
        kind = "table" or "view"
        المفتاح: (kind, source_name, column_original_name)
        """
        original_name = col.get("original_name") or col.get("name") or ""
        canonical_name = col.get("canonical_name") or original_name
        key = (kind, source_name, original_name)

        # المصطلحات التي سنفهرسها:
        terms: List[str] = []

        terms.append(original_name)
        terms.append(canonical_name)

        syn_en = col.get("synonyms_en") or []
        syn_ar = col.get("synonyms_ar") or []
        terms.extend(syn_en)
        terms.extend(syn_ar)

        # بناء الفهارس (phrases + tokens)
        for term in terms:
            term = term or ""
            norm_phrase = normalize_text(term)
            if not norm_phrase:
                continue

            # فهرس العبارة كاملة (وزن أعلى)
            self.phrase_index[norm_phrase].append(("column", key))

            # فهرس الكلمات المفردة
            for tok in tokenize(norm_phrase):
                self.term_index[tok].append(("column", key))

    def _index_metric(self, metric_id: str, metric_def: Dict[str, Any]) -> None:
        """
        إضافة مقياس للفهرس.
        المفتاح: ("metric", metric_id)
        """
        key = metric_id
        terms: List[str] = []

        terms.append(metric_def.get("name_en", ""))
        terms.append(metric_def.get("name_ar", ""))

        syn_en = metric_def.get("synonyms_en") or []
        syn_ar = metric_def.get("synonyms_ar") or []
        terms.extend(syn_en)
        terms.extend(syn_ar)

        # بعض الكلمات من الـ id مفيدة
        terms.append(metric_id.replace("_", " "))

        for term in terms:
            term = term or ""
            norm_phrase = normalize_text(term)
            if not norm_phrase:
                continue

            self.phrase_index[norm_phrase].append(("metric", key))

            for tok in tokenize(norm_phrase):
                self.term_index[tok].append(("metric", key))

    # ---------- الدالة الرئيسية لتفسير السؤال ----------

    def interpret(
        self,
        query: str,
        top_k_columns: int = 5,
        top_k_metrics: int = 5,
    ) -> InterpretationResult:
        """
        تفسير سؤال المستخدم:
        - query: نص السؤال كما كتبه المستخدم.
        - يرجع InterpretationResult يحوي أفضل الأعمدة والمقاييس.
        """
        normalized = normalize_text(query)
        lang = guess_language(query)

        # لا يوجد سؤال فعلي
        if not normalized:
            return InterpretationResult(
                query=query,
                normalized_query=normalized,
                language=lang,
                dominant_entity=None,
                top_columns=[],
                top_metrics=[],
            )

        tokens = set(tokenize(normalized))

        # scores
        col_scores: Dict[Tuple[str, str, str], int] = defaultdict(int)
        metric_scores: Dict[str, int] = defaultdict(int)

        # 1) مطابقة بالtokens
        for tok in tokens:
            for kind, key in self.term_index.get(tok, []):
                if kind == "column":
                    k = key  # (kind, source_name, original_name)
                    col_scores[k] += 1
                elif kind == "metric":
                    metric_scores[key] += 1

        # 2) مطابقة بالعبارات الكاملة (وزن أعلى)
        for phrase, refs in self.phrase_index.items():
            if phrase and phrase in normalized:
                for kind, key in refs:
                    if kind == "column":
                        col_scores[key] += 2
                    elif kind == "metric":
                        metric_scores[key] += 3  # نعطي وزن أعلى للمقياس

        # تحويل النتائج إلى كائنات
        top_cols_sorted = sorted(
            col_scores.items(), key=lambda kv: kv[1], reverse=True
        )[:top_k_columns]

        column_matches: List[ColumnMatch] = []
        entity_counter: Dict[str, int] = defaultdict(int)

        for (kind, source_name, original_name), score in top_cols_sorted:
            # نحدد هل المصدر جدول أم View
            source_def = (
                self.tables.get(source_name)
                if kind == "table"
                else self.views.get(source_name)
            )
            if not source_def:
                continue

            entity_type = source_def.get("entity_type")
            if entity_type:
                entity_counter[entity_type] += score

            # نبحث عن تعريف العمود
            col_def = None
            for c in source_def.get("columns", []):
                if c.get("original_name") == original_name:
                    col_def = c
                    break
            canonical = (
                col_def.get("canonical_name")
                if col_def and col_def.get("canonical_name")
                else original_name
            )

            column_matches.append(
                ColumnMatch(
                    table=source_name,
                    table_entity_type=entity_type,
                    column_original=original_name,
                    column_canonical=canonical,
                    score=score,
                )
            )

        top_metrics_sorted = sorted(
            metric_scores.items(), key=lambda kv: kv[1], reverse=True
        )[:top_k_metrics]

        metric_matches: List[MetricMatch] = []
        for metric_id, score in top_metrics_sorted:
            m = self.metrics.get(metric_id)
            if not m:
                continue
            entity = m.get("entity")
            if entity:
                entity_counter[entity] += score * 2  # نرجّح الكيان من خلال المقاييس

            source = m.get("source", {})
            metric_matches.append(
                MetricMatch(
                    metric_id=metric_id,
                    name_en=m.get("name_en", ""),
                    name_ar=m.get("name_ar", ""),
                    entity=entity,
                    source_type=source.get("type", ""),
                    source_name=source.get("name", ""),
                    source_column=source.get("column", ""),
                    score=score,
                )
            )

        # اختيار الكيان المسيطر (dominant_entity)
        dominant_entity = None
        if entity_counter:
            dominant_entity = max(entity_counter.items(), key=lambda kv: kv[1])[0]

        return InterpretationResult(
            query=query,
            normalized_query=normalized,
            language=lang,
            dominant_entity=dominant_entity,
            top_columns=column_matches,
            top_metrics=metric_matches,
        )
# ============== طبقة تخطيط الاستعلام (Query Planning) ==============

def detect_gopm_intent(query: str) -> Dict[str, Any]:
    """
    يحدد إن كان السؤال متعلقًا بقواعد/جداول GOPM (MGT / Turnaround / Transit / Activity Breakdown).

    يرجع قاموسًا خفيفًا (لا يكسر أي طبقات أخرى) مثل:
    {
      "is_gopm": bool,
      "category": "mgt" | "activity" | "delivery" | None,
      "operation": "turnaround" | "transit" | "mixed" | None,
      "stations": [...],        # مثل RUH/JED/DMM/MED/AHB/TUU/LHR/UK
      "destinations": [...],    # مثل JFK/LAX/SSH/USA/KAN...
      "aircraft": [...],        # مثل B777-368/B787-10, A330/B787-9, A321/A320, B757 ...
      "flow": "DOM-DOM" | "DOM-INTL" | "INTL-DOM" | "INTL-INTL" | None,
    }

    ملاحظة: هذه الدالة لا تحسب الأوقات؛ فقط تصنيف/توجيه.
    """
    q = normalize_text(query)
    if not q:
        return {
            "is_gopm": False,
            "category": None,
            "operation": None,
            "stations": [],
            "destinations": [],
            "aircraft": [],
            "flow": None,
        }

    # كلمات مفتاحية (AR/EN) لـ GOPM
    gopm_markers = [
        "gopm", "mgt", "minimum ground time", "ground time",
        "turnaround", "transit", "activity breakdown", "ramp handling",
        "blocks in", "blocks out", "pushback", "door opening", "door clsd",
        "cabin cleaning", "galley", "catering", "pax", "disembarkation", "embarkation",
        "scheduled operations", "delivery before std", "ex-scheduled maintenance",
        # عربي
        "وقت الارض", "وقت الأرض", "الحد الادنى", "الحد الأدنى", "الزمن الارضى",
        "ترانزيت", "ترانزت", "ترانست", "ترانز", "ترانزت", "تيرن", "تيرن اراوند",
        "تيرناروند", "دوران", "تفصيل", "تفصيل الانشطة", "تفصيل الأنشطة",
        "تنظيف", "تموين", "كاترينج", "بوابة", "دفع", "سحب", "بوش باك",
    ]

    is_gopm = any(m in q for m in gopm_markers)

    # Operation
    operation = None
    if "turnaround" in q or "تيرن" in q or "دوران" in q:
        operation = "turnaround"
        is_gopm = True
    if "transit" in q or "تران" in q:
        # إذا وجد المصطلحين معاً نعتبره mixed
        if operation == "turnaround":
            operation = "mixed"
        else:
            operation = "transit"
        is_gopm = True
    if "mixed flight" in q or "mixed flights" in q or "رحلات مختلطة" in q or "مختلط" in q:
        operation = "mixed"
        is_gopm = True

    # Category (MGT جدول / Activity Breakdown / Delivery)
    category = None
    if any(x in q for x in ["mgt", "minimum ground time", "scheduled operations", "وقت الارض", "وقت الأرض", "الحد الادنى", "الحد الأدنى"]):
        category = "mgt"
    if any(x in q for x in ["activity breakdown", "تفصيل", "activities", "انشطة", "أنشطة"]):
        category = "activity"
    if any(x in q for x in ["delivery before std", "delivery", "hangar", "maintenance", "تسليم", "الصيانة"]):
        category = "delivery"

    # Stations (أكواد)
    station_codes = ["JED", "RUH", "DMM", "MED", "AHB", "TUU", "LHR", "UK"]
    stations = []
    for s in station_codes:
        if re.search(rf"\b{s.lower()}\b", q):
            stations.append(s)

    # Destination / special stations / groups
    # ملاحظة: USA/KAN/SSH/LONG-HAUL/INT STNS ليست ICAO/IATA دائماً لكنها قيود بالنص
    dest_codes = ["JFK", "LAX", "IAD", "YYZ", "CAN", "MNL", "KUL", "CGK", "SIN", "SSH", "KAN", "USA"]
    destinations = []
    for d in dest_codes:
        if re.search(rf"\b{d.lower()}\b", q):
            destinations.append(d)

    if "long haul" in q or "long-haul" in q or "رحلات بعيدة" in q or "بعيدة" in q:
        destinations.append("LONG_HAUL")
        is_gopm = True
    if "int stns" in q or "int stations" in q or "محطات دولية" in q:
        destinations.append("INT_STNS")
        is_gopm = True
    if "(sa)" in q or "security alert" in q or "تنبيه امني" in q or "أمني" in q:
        destinations.append("SA")
        is_gopm = True

    # Aircraft families (محاولة التقاط متسامحة)
    aircraft = []
    # B777-368 / B787-10
    if re.search(r"b?777[-\s]?368|787[-\s]?10|b787[-\s]?10|777[-\s]?368", q):
        aircraft.append("B777-368/B787-10")
        is_gopm = True
    # B777-268 / A330 (مذكورة في الجدول المختلط)
    if re.search(r"777[-\s]?268|b?777[-\s]?268", q) and ("a330" in q or "airbus 330" in q):
        aircraft.append("B777-268/A330")
        is_gopm = True
    # A330 / B787-9
    if "a330" in q or "787-9" in q or "b787-9" in q or "b787[-\s]?9" in q:
        aircraft.append("A330/B787-9")
        is_gopm = True
    # A321 / A320
    if "a321" in q or "a320" in q:
        aircraft.append("A321/A320")
        is_gopm = True
    # B757
    if "b757" in q or "757" in q:
        aircraft.append("B757")
        is_gopm = True

    # Flow DOM/INTL
    flow = None
    # english patterns
    if "dom-dom" in q or "dom / dom" in q:
        flow = "DOM-DOM"
    elif "dom-intl" in q or "dom-int" in q or "dom intl" in q:
        flow = "DOM-INTL"
    elif "intl-dom" in q or "int-dom" in q or "intl dom" in q:
        flow = "INTL-DOM"
    elif "intl-intl" in q or "int-int" in q:
        flow = "INTL-INTL"

    # arabic hints (تقريبي)
    if flow is None:
        if ("محلي" in q and "محلي" in q.replace("محلي", "", 1)):
            flow = "DOM-DOM"
        elif "محلي" in q and ("دولي" in q or "خارجي" in q):
            flow = "DOM-INTL"
        elif ("دولي" in q or "خارجي" in q) and "محلي" in q:
            flow = "INTL-DOM"
        elif ("دولي" in q or "خارجي" in q) and ("دولي" in q.replace("دولي", "", 1) or "خارجي" in q.replace("خارجي", "", 1)):
            flow = "INTL-INTL"

    # إن لم يتم تحديد category لكن GOPM markers موجودة
    if is_gopm and category is None:
        # الأفضل أن نميل لـ mgt إن وُجدت كودات محطات، وإلا activity إن وُجدت قائمة أنشطة
        if stations or destinations:
            category = "mgt"
        else:
            category = "activity"

    return {
        "is_gopm": bool(is_gopm),
        "category": category,
        "operation": operation,
        "stations": sorted(list(dict.fromkeys(stations))),  # preserve unique
        "destinations": sorted(list(dict.fromkeys(destinations))),
        "aircraft": sorted(list(dict.fromkeys(aircraft))),
        "flow": flow,
    }



def _extract_limit_from_text(text: str, default: int = 10) -> int:
    """
    يحاول استنتاج قيمة LIMIT من السؤال مثل:
    - top 5
    - أعلى 10
    - أفضل 3
    لو لم يجد رقم منطقي، يرجع القيمة الافتراضية.
    """
    if not text:
        return default
    nums = re.findall(r"\b([0-9]{1,3})\b", text)
    candidates = [int(n) for n in nums if 1 <= int(n) <= 100]
    if not candidates:
        return default
    return min(candidates)


def _default_group_by_for_entity(entity: Optional[str]) -> list:
    """
    يعيد أعمدة group by الافتراضية حسب نوع الكيان.
    لا يتحقق من وجود الأعمدة فعلياً في الـ View، هذه مسؤولية الطبقة الأعلى.
    """
    if not entity:
        return []
    entity = entity.lower()
    if entity == "employee":
        return ["employee_id", "employee_name"]
    if entity == "department":
        return ["Department"]
    if entity == "duty_manager":
        return ["Duty Manager Name"]
    if entity == "supervisor":
        return ["Supervisor Name"]
    if entity == "control":
        return ["Control Name"]
    if entity == "airline":
        return ["Airlines"]
    if entity == "shift":
        return ["Shift"]
    return []


def build_query_plan(engine: "NXSSemanticEngine", query: str) -> Dict[str, Any]:
    """
    يبني خطة استعلام مبسّطة فوق نتيجة interpret:
    - يختار أفضل مقياس (إن وجد).
    - يحدد الـ view/الجدول المناسب.
    - يقترح group by و limit.

    الناتج مناسب لتمريره للـ planner أو طبقة توليد SQL.
    """
    interp = engine.interpret(query)

    plan: Dict[str, Any] = {
        "entity": interp.dominant_entity,
        "metric": None,
        "table": None,
        "group_by": [],
        "limit": _extract_limit_from_text(interp.normalized_query),
    }

    if interp.top_metrics:
        m = interp.top_metrics[0]
        plan["metric"] = {
            "id": m.metric_id,
            "name_en": m.name_en,
            "name_ar": m.name_ar,
            "source_type": m.source_type,
            "source_name": m.source_name,
            "source_column": m.source_column,
        }
        plan["table"] = m.source_name
        if not plan["entity"] and m.entity:
            plan["entity"] = m.entity
    elif interp.top_columns:
        c = interp.top_columns[0]
        plan["table"] = c.table

    plan["group_by"] = _default_group_by_for_entity(plan["entity"])

    # GOPM auto-route (rules engine) + model recommendation
    gopm_intent = detect_gopm_intent(interp.normalized_query or query)
    if gopm_intent.get("is_gopm"):
        plan["route"] = "gopm"
        plan["gopm"] = gopm_intent

    complexity = estimate_complexity(query)
    plan["complexity"] = complexity
    plan["recommended_model"] = recommend_gemini_model(complexity["tier"])

    return {
        "interpretation": interp.to_dict(),
        "plan": plan,
    }
# ============== مثال تشغيل مباشر (للاختبار اليدوي) ==============

if __name__ == "__main__":
    engine = NXSSemanticEngine()

    tests = [
        "من أكثر مدير مناوب تسبب في تأخيرات الرحلات؟",
        "Top employees by total sick days",
        "تقرير عن تأخيرات الرحلات لكل قسم في TCC",
        "ما هو إجمالي دقائق التأخير الشخصية للموظف 15013814؟",
    ]

    for q in tests:
        print("\n==============================")
        print("Q:", q)
        res = engine.interpret(q)
        print("Language:", res.language)
        print("Dominant entity:", res.dominant_entity)

        print("\nTop metrics:")
        for m in res.top_metrics:
            print(
                f"  - {m.metric_id} (entity={m.entity}, score={m.score}, "
                f"source={m.source_name}.{m.source_column})"
            )

        print("\nTop columns:")
        for c in res.top_columns:
            print(
                f"  - {c.table}.{c.column_original} "
                f"(entity={c.table_entity_type}, score={c.score})"
            )

        plan = build_query_plan(engine, q)
        print("\nQuery plan:")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
# ============== طبقة تحليل إضافية للفلاتر (AGI Helper Layer) ==============

def extract_basic_filters_from_query(query: str) -> Dict[str, Any]:
    """
    دالة مساعدة لاستخراج بعض الفلاتر الأساسية من السؤال مباشرةً دون الرجوع لـ LLM.

    الهدف:
    - دعم أنماط AGI عبر إعطاء طبقة التخطيط معلومات أولية عن المعرفات الحساسة.
    - هذه الدالة لا تُستخدم لإصدار أحكام نهائية، بل كمدخل مساعد فقط.

    الفلاتر التي تحاول اكتشافها:
    - employee_id: أرقام من 6–10 خانات (مثل 15013814).
    - flight_number: نمط بسيط لحروف + أرقام (مثل SV123 أو SV 1234).
    - delay_code: أنماط مثل 15I أو 15F أو رموز حرفية قصيرة (PD, GL, 2R).
    - any_number: أول رقم يظهر في النص (للأسئلة العامة مثل "أعلى 5" إلخ).

    إن لم يُكتشف شيء، يرجع قاموساً فارغاً.
    """
    text_norm = normalize_text(query)
    filters: Dict[str, Any] = {}

    # employee_id: نفترض أنه رقم متصل من 6 إلى 10 خانات
    emp_match = re.search(r"\b(\d{6,10})\b", text_norm)
    if emp_match:
        filters["employee_id"] = emp_match.group(1)

    # flight_number: حروف + أرقام (SV123, SV 1234, XY987 إلخ)
    fn_match = re.search(r"\b([a-zA-Z]{2,3}\s*\d{2,4})\b", query)
    if fn_match:
        filters["flight_number"] = fn_match.group(1).replace(" ", "").upper()

    # delay_code: رموز مثل 15I أو 15F أو PD أو GL أو 2R
    delay_match = re.search(r"\b(1[0-9]{1}[A-Z]|[A-Z]{1,2}\d?|\d{1,2}[A-Z])\b", query)
    if delay_match:
        filters["delay_code"] = delay_match.group(1).upper()

    # any_number: رقم عام مفيد في بعض الأسئلة
    num_match = re.search(r"\b(\d{1,3})\b", text_norm)
    if num_match and "any_number" not in filters:
        filters["any_number"] = int(num_match.group(1))

    return filters

# ============== Model Routing (Fast vs Pro) ==============

def _count_numbers(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\d+\b", text))


def estimate_complexity(
    query: str,
    interpretation: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    detected_filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    تقدير "تعقيد" السؤال بسرعة (بدون LLM) بهدف اختيار الموديل المناسب:
    - simple  -> gemini-2.5-flash
    - complex -> gemini-2.5-pro

    المخرجات:
      {
        "tier": "simple" | "complex",
        "score": int (0..100),
        "reasons": [..]
      }

    ملاحظة:
    - هذه طبقة مساعدة فقط (Helper). القرار النهائي يمكن أن يتم في nxs_brain.py.
    """
    q_norm = normalize_text(query)
    toks = tokenize(q_norm)
    tok_len = len(toks)

    # كلمات تشير غالباً إلى تحليل عميق
    complex_markers_ar = {
        "حلل","تحليل","قارن","مقارنة","اتجاه","توجه","توقع","تنبؤ","تفسير","فسر","اشرح",
        "سبب","لماذا","جذر","جذري","استنتاج","استخلص","نمط","انماط","احصائية","إحصائية",
        "متوسط","نسبة","انحراف","معيار","ارتباط","تنبؤات","توصية","سيناريو","what-if"
    }
    complex_markers_en = {
        "analyze","analysis","compare","comparison","trend","forecast","predict","prediction",
        "explain","reason","root","cause","insight","pattern","correlation","variance",
        "recommend","scenario","what if","kpi","benchmark"
    }

    reasons = []
    score = 0

    # GOPM domain signal (MGT/Transit/Turnaround/Activity Breakdown)
    gopm_intent = detect_gopm_intent(q_norm)
    if gopm_intent.get("is_gopm"):
        reasons.append("GOPM: سؤال أوقات/قواعد من GOPM")

        # إذا كان السؤال يحتوي عدة عناصر (محطات/وجهات/طائرات) يصبح أقرب للتعقيد
        multi = 0
        multi += max(0, len(gopm_intent.get("stations", [])) - 1)
        multi += max(0, len(gopm_intent.get("destinations", [])) - 1)
        multi += max(0, len(gopm_intent.get("aircraft", [])) - 1)

        if gopm_intent.get("operation") == "mixed":
            multi += 1
        if multi >= 3:
            score += 25
            reasons.append("GOPM متعدد القيود (عدة محطات/وجهات/طائرات)")
        elif multi == 2:
            score += 15
            reasons.append("GOPM قيود متعددة (أكثر من عنصر)")
        elif multi == 1:
            score += 8
            reasons.append("GOPM مع عنصر إضافي")

        # أسئلة الشرح/السبب/التبرير
        if any(k in q_norm for k in ["why","explain","because","reason","justify","clarify",
                                     "ليش","لماذا","اشرح","فسر","سبب","تبرير","وضح"]):
            score += 10
            reasons.append("طلب تفسير/سبب")
    # طول السؤال
    if tok_len >= 18:
        score += 20
        reasons.append("سؤال طويل (عدد كلمات كبير)")
    elif tok_len >= 10:
        score += 10
        reasons.append("سؤال متوسط الطول")

    # وجود كلمات تحليل/مقارنة
    joined = " ".join(toks)
    hit_complex = 0
    for w in complex_markers_ar:
        if w in joined:
            hit_complex += 1
    for w in complex_markers_en:
        if w in joined:
            hit_complex += 1

    if hit_complex >= 2:
        score += 30
        reasons.append("يحتوي كلمات تحليل/مقارنة/تفسير")
    elif hit_complex == 1:
        score += 15
        reasons.append("يحتوي مؤشر واحد للتحليل")

    # أرقام كثيرة -> غالباً تحليل
    n_numbers = _count_numbers(query)
    if n_numbers >= 3:
        score += 15
        reasons.append("يحتوي عدة أرقام/قيم")
    elif n_numbers == 2:
        score += 8
        reasons.append("يحتوي رقمين")

    # عدد الفلاتر المكتشفة
    if detected_filters:
        df = {k: v for k, v in detected_filters.items() if v not in (None, "", [])}
        if len(df) >= 2:
            score += 10
            reasons.append("اكتشاف أكثر من فلتر (موظف/رحلة/كود...)")
        elif len(df) == 1:
            score += 4
            reasons.append("اكتشاف فلتر واحد")

    # تعقيد الخطة (إن وجدت)
    if plan:
        gb = plan.get("group_by") or []
        metric = plan.get("metric")
        if metric and gb:
            score += 15
            reasons.append("الخطة تتضمن Metric + Group By")
        elif metric:
            score += 8
            reasons.append("الخطة تتضمن Metric")
        if plan.get("limit", 0) and plan.get("limit", 0) >= 25:
            score += 5
            reasons.append("طلب عدد نتائج كبير")

    # ضمان الحدود
    score = max(0, min(100, score))

    tier = "complex" if score >= 55 else "simple"
    if not reasons:
        reasons.append("سؤال مباشر/بسيط")

    return {"tier": tier, "score": score, "reasons": reasons}


def recommend_gemini_model(
    complexity_tier: str,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    يحدد اسم الموديل المقترح بناءً على التعقيد.
    - يدعم override عبر Environment Variables (اختياري):
        GEMINI_MODEL_SIMPLE
        GEMINI_MODEL_COMPLEX
    """
    import os
    env = env or os.environ

    simple_model = env.get("GEMINI_MODEL_SIMPLE", "gemini-2.5-flash")
    complex_model = env.get("GEMINI_MODEL_COMPLEX", "gemini-2.5-pro")

    if complexity_tier == "complex":
        return {"tier": "complex", "model": complex_model}
    return {"tier": "simple", "model": simple_model}


def interpret_with_filters(engine: "NXSSemanticEngine", query: str) -> Dict[str, Any]:
    """
    واجهة مريحة تجمع بين:
    - التفسير الدلالي (interpret).
    - واستخراج فلاتر أساسية من النص مباشرةً.
    - تقدير تعقيد السؤال (Model Routing Hint) لاختيار (Flash/Pro).

    تعيد:
    {
      "interpretation": {...},   # نفس هيكل InterpretationResult.to_dict()
      "plan": {...},             # ناتج build_query_plan
      "detected_filters": {...}, # ناتج extract_basic_filters_from_query
      "complexity_hint": {...},  # tier/score/reasons
      "model_hint": {...},       # tier/model
    }
    """
    interp_and_plan = build_query_plan(engine, query)
    detected_filters = extract_basic_filters_from_query(query)

    complexity_hint = estimate_complexity(
        query=query,
        interpretation=interp_and_plan.get("interpretation"),
        plan=interp_and_plan.get("plan"),
        detected_filters=detected_filters,
    )
    model_hint = recommend_gemini_model(complexity_hint["tier"])

    return {
        "interpretation": interp_and_plan["interpretation"],
        "plan": interp_and_plan["plan"],
        "detected_filters": detected_filters,
        "complexity_hint": complexity_hint,
        "model_hint": model_hint,
    }
