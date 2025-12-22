# nxs_supabase_client.py
# استخدام REST API لـ Supabase بدون supabase-py

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

import requests

# dotenv اختياري للتطوير المحلي فقط (Production يعتمد على Environment Variables)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore

if load_dotenv:
    # إذا لم يوجد ملف .env فلن يحدث شيء، وهذا آمن على السيرفر
    load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# ملاحظة احترافية:
# - لا نُسقِط السيرفر عند غياب بيانات Supabase (للاستقرار أثناء النشر)
# - لكن نُرجّع [] من طبقة البيانات مع تحذير واضح في السجل
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_ANON_KEY)

# عنوان REST الأساسي
REST_BASE_URL = (SUPABASE_URL.rstrip("/") + "/rest/v1") if SUPABASE_URL else ""

# الرؤوس المشتركة لكل الطلبات
COMMON_HEADERS = (
    {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if SUPABASE_ANON_KEY
    else {}
)

logger = logging.getLogger("tcc_supabase")
if not SUPABASE_ENABLED:
    logger.warning(
        "Supabase is not configured (SUPABASE_URL/SUPABASE_ANON_KEY missing). "
        "Database calls will return empty results."
    )



# ============================
# دالة GET عامة
# ============================
# ============================
# دالة GET عامة (Supabase PostgREST)
# ============================
# ندعم تمرير params كـ dict أو قائمة tuples حتى نسمح بتكرار نفس المفتاح (مثل Date=gte.. و Date=lte..)
ParamsType = Any  # keep lightweight (requests handles dict or list[tuple])

def _as_params(params: ParamsType) -> List[Tuple[str, str]]:
    if params is None:
        return []
    if isinstance(params, dict):
        out: List[Tuple[str, str]] = []
        for k, v in params.items():
            if isinstance(v, (list, tuple)):
                for vv in v:
                    out.append((str(k), str(vv)))
            else:
                out.append((str(k), str(v)))
        return out
    # assume iterable of (k,v)
    return [(str(k), str(v)) for (k, v) in params]

def _get(table: str, params: ParamsType) -> List[Dict[str, Any]]:
    """
    تنفيذ طلب GET عام على جدول معيّن في Supabase.
    table: اسم الجدول مثل 'employee_master_db'
    params: باراميترات PostgREST (فلترة، limit، order، ...)

    مبدأ الأداء:
    - لا نجلب 10000 صف ثم نفلتر محلياً إلا عند الضرورة.
    - اجعل limit صغيراً قدر الإمكان وادفع الفلاتر إلى Supabase.
    """
    if not SUPABASE_ENABLED:
        return []

    url = f"{REST_BASE_URL}/{table}"

    p = _as_params(params)

    # limit افتراضي + حماية من القيم الضخمة
    has_limit = any(k == "limit" for k, _ in p)
    if not has_limit:
        p.append(("limit", "1000"))
    else:
        # cap
        new_p: List[Tuple[str, str]] = []
        for k, v in p:
            if k == "limit":
                try:
                    n = int(str(v))
                    if n > 5000:
                        v = "5000"
                except Exception:
                    pass
            new_p.append((k, v))
        p = new_p

    try:
        resp = requests.get(url, headers=COMMON_HEADERS, params=p, timeout=20)
        resp.raise_for_status()  # 4xx/5xx
    except requests.exceptions.RequestException as e:
        logger.warning("Supabase GET request failed for table %s: %s", table, e)
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    if isinstance(data, dict) and data.get("message") == "Not Found":
        return []
    if isinstance(data, dict):
        return [data]
    return data


# ============================
# دوال مساعدة عامة (Reusable Helpers)
# ============================
def _extract_employee_id(row: Dict[str, Any]) -> Optional[str]:
    """
    محاولة ذكية لاستخراج Employee ID من صف.
    """
    candidate_keys = [
        "Employee ID",
        "employee_id",
        "EmployeeID",
        "Employee_Id",
        "Employee Id",
    ]
    for key in candidate_keys:
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    return None


def _in_date_range(
    row: Dict[str, Any],
    date_keys: List[str],
    start_date: str,
    end_date: str,
) -> bool:
    """
    التحقق من أن الصف يقع ضمن الفترة [start_date, end_date]
    بناءً على أول عمود تاريخ يجد قيمة صالحة.
    """
    for key in date_keys:
        if key in row and row[key] not in (None, ""):
            val = str(row[key])
            date_str = val[:10]  # غالباً 'YYYY-MM-DD'
            if start_date <= date_str <= end_date:
                return True
    return False


def _normalize_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[str, str]:
    """
    جعل الفترة الزمنية آمنة:
    - إذا لم يُحدّد نستخدم نطاق واسع جداً.
    """
    s = (start_date or "0001-01-01")[:10]
    e = (end_date or "9999-12-31")[:10]
    if s > e:
        s, e = e, s
    return s, e


def _filter_employee_range(
    rows: List[Dict[str, Any]],
    employee_id: Optional[str],
    date_keys: List[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict[str, Any]]:
    """
    ترشيح قائمة صفوف حسب Employee ID (اختياري) + فترة زمنية (اختيارية).
    """
    s, e = _normalize_date_range(start_date, end_date)
    target = str(employee_id).strip() if employee_id is not None else None
    out: List[Dict[str, Any]] = []

    for r in rows:
        # فلترة حسب الموظف
        if target is not None:
            emp = _extract_employee_id(r)
            if emp != target:
                continue
        
        # فلترة حسب التاريخ
        if date_keys and not _in_date_range(r, date_keys, s, e):
            continue
            
        out.append(r)

    return out


# ============================
# دوال جلب البيانات
# ============================


def global_search_by_id(search_id: str) -> List[Dict[str, Any]]:
    """
    البحث الشامل عن أي رقم (ID) في الجداول الرئيسية لضمان عدم ضياع البيانات.
    """
    results = []
    # الجداول التي سيتم مسحها فوراً عند وجود رقم
    target_tables = [
        {"table": "employee_master_db", "col": "Employee ID"},
        {"table": "shift_report", "col": "Title"},
        {"table": "dep_flight_delay", "col": "Employee ID"}
    ]

    for entry in target_tables:
        try:
            url = f"{REST_BASE_URL}/{entry['table']}"
            params = {entry['col']: f"eq.{search_id}", "select": "*"}
            resp = requests.get(url, headers=COMMON_HEADERS, params=params)
            if resp.status_code == 200 and resp.json():
                data = resp.json()
                # إضافة اسم الجدول للبيانات ليعرف Gemini مصدرها
                for row in data:
                    row["_source_table"] = entry['table']
                results.extend(data)
        except Exception as e:
            logger.error(f"Error searching {entry['table']}: {e}")

    return results

def search_everywhere(value: str) -> Optional[List[Dict[str, Any]]]:
    """
    بحث 'ماسح' عن قيمة (غالباً رقم وظيفي) عبر عدة جداول وأعمدة محتملة.

    يبحث حالياً في:
    1) employee_master_db
    2) dep_flight_delay
    3) shift_report

    ملاحظة PostgREST:
    - نستخدم باراميتر or=(...) للبحث عبر أعمدة متعددة.
    - الأعمدة التي تحتوي مسافات تُكتب بين علامتي اقتباس مزدوجة.
    """
    v = (value or "").strip()
    if not v:
        return None

    # أعمدة محتملة لكل جدول (يمكن توسيعها لاحقاً حسب الحاجة)
    table_columns: Dict[str, List[str]] = {
        "employee_master_db": ['"Employee ID"', "employee_id", "EmployeeID", "id", '"Title"'],
        "dep_flight_delay": ['"Employee ID"', "employee_id", "EmployeeID", "id", '"Title"', '"Duty Manager ID"', '"Supervisor ID"', '"Control ID"'],
        "shift_report": [
            '"Employee ID"', "employee_id", "EmployeeID", "id", '"Title"',
            '"Control 1 ID"', '"Control 2 ID"',
            '"Duty Manager Domestic ID"', '"Duty Manager International+Foreign ID"', '"Duty Manager All Halls ID"',
            '"Supervisor Domestic ID"', '"Supervisor International+Foreign ID"', '"Supervisor All Halls ID"',
        ],
    }

    tables = ["employee_master_db", "dep_flight_delay", "shift_report"]

    for table in tables:
        cols = table_columns.get(table, ['"Employee ID"', "id", '"Title"'])
        # or=("Employee ID".eq.123,id.eq.123,"Title".eq.123)
        parts = [f"{c}.eq.{v}" for c in cols]
        or_clause = "(" + ",".join(parts) + ")"

        rows = _get(table, [("select", "*"), ("limit", "200"), ("or", or_clause)])
        if rows:
            return rows

    return None
def get_employee_info(emp_id: str) -> Optional[Dict[str, Any]]:
    """جلب معلومات موظف واحد عبر بحث ماسح في الجداول (search_everywhere)."""
    rows = search_everywhere(str(emp_id))
    if rows and isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return None

    target = str(emp_id).strip()
    for r in rows:
        emp = _extract_employee_id(r)
        if emp == target:
            return r
    return None


def list_all_flight_delays(limit: int = 5000) -> List[Dict[str, Any]]:
    """جلب قائمة عامة بتأخيرات الرحلات من جدول sgs_flight_delay."""
    params = {"select": "*", "limit": limit, "order": "Date.desc"}
    return _get("sgs_flight_delay", params)


def list_dep_flight_delays(limit: int = 1000) -> List[Dict[str, Any]]:
    """جلب تأخيرات القسم (TCC / FIC / LC ...) من جدول dep_flight_delay بدون فلترة حسب موظف."""
    table_name = "dep_flight_delay"
    params = {"select": "*", "limit": limit, "order": "Date.asc"}
    return _get(table_name, params)


def get_employee_delays(
    employee_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب تأخيرات الرحلات لموظف معيّن من dep_flight_delay."""
    emp = (employee_id or "").strip()
    if not emp:
        return []

    s, e = _normalize_date_range(start_date, end_date)
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.asc"),
        ("Employee ID", f"eq.{emp}"),
        ("Date", f"gte.{s}"),
        ("Date", f"lte.{e}"),
    ]
    return _get("dep_flight_delay", params)

def list_employee_absence(
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """جلب سجلات الغياب (employee_absence)."""
    table_name = "employee_absence"
    params = {"select": "*", "limit": limit, "order": "Date.asc"}
    return _get(table_name, params)


def get_employee_absence(
    employee_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب الغياب لموظف معيّن خلال فترة."""
    emp = (employee_id or "").strip()
    if not emp:
        return []
    s, e = _normalize_date_range(start_date, end_date)
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.asc"),
        ("Employee ID", f"eq.{emp}"),
        ("Date", f"gte.{s}"),
        ("Date", f"lte.{e}"),
    ]
    return _get("employee_absence", params)

def list_employee_delay_log(
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """جلب سجلات تأخير الحضور عن الدوام (employee_delay)."""
    table_name = "employee_delay"
    params = {"select": "*", "limit": limit, "order": "Date.asc"}
    return _get(table_name, params)


def get_employee_delay_log(
    employee_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب سجلات تأخير الحضور لموظف معيّن خلال فترة."""
    emp = (employee_id or "").strip()
    if not emp:
        return []
    s, e = _normalize_date_range(start_date, end_date)
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.asc"),
        ("Employee ID", f"eq.{emp}"),
        ("Date", f"gte.{s}"),
        ("Date", f"lte.{e}"),
    ]
    return _get("employee_delay", params)

def list_employee_overtime(
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """جلب سجلات العمل الإضافي (employee_overtime)."""
    table_name = "employee_overtime"
    params = {"select": "*", "limit": limit, "order": '"Assignment Date".asc'}
    return _get(table_name, params)


def get_employee_overtime(
    employee_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب سجلات العمل الإضافي لموظف معيّن خلال فترة."""
    emp = (employee_id or "").strip()
    if not emp:
        return []
    s, e = _normalize_date_range(start_date, end_date)
    # ملاحظة: جدول employee_overtime يعتمد غالباً على "Assignment Date" كحقل تاريخ
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", '"Assignment Date".asc'),
        ("Employee ID", f"eq.{emp}"),
        ("Assignment Date", f"gte.{s}"),
        ("Assignment Date", f"lte.{e}"),
    ]
    return _get("employee_overtime", params)

def list_employee_sick_leave(
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """جلب سجلات الإجازات المرضية (employee_sick_leave)."""
    table_name = "employee_sick_leave"
    params = {"select": "*", "limit": limit, "order": "Date.asc"}
    return _get(table_name, params)


def get_employee_sick_leave(
    employee_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب سجلات الإجازات المرضية لموظف معيّن خلال فترة."""
    emp = (employee_id or "").strip()
    if not emp:
        return []
    s, e = _normalize_date_range(start_date, end_date)
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.asc"),
        ("Employee ID", f"eq.{emp}"),
        ("Date", f"gte.{s}"),
        ("Date", f"lte.{e}"),
    ]
    return _get("employee_sick_leave", params)

def list_operational_events(
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """جلب سجلات الأحداث التشغيلية (operational_event)."""
    table_name = "operational_event"
    params = {"select": "*", "limit": limit, "order": '"Event Date".asc'}
    return _get(table_name, params)


def get_employee_operational_events(
    employee_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب الأحداث التشغيلية لموظف معيّن خلال فترة."""
    emp = (employee_id or "").strip()
    if not emp:
        return []
    s, e = _normalize_date_range(start_date, end_date)
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", '"Event Date".asc'),
        ("Employee ID", f"eq.{emp}"),
        ("Event Date", f"gte.{s}"),
        ("Event Date", f"lte.{e}"),
    ]
    return _get("operational_event", params)

def get_employee_count_by_department(department: str) -> int:
    """الحصول على عدد الموظفين في قسم معيّن (من employee_master_db)."""
    table_name = "employee_master_db"
    # نستخدم فلترة POSTGREST مباشرة لـ Count
    params = {
        "select": "count()",
        "Department": f"eq.{department}",
    }
    
    url = f"{REST_BASE_URL}/{table_name}"
    headers = COMMON_HEADERS.copy()
    headers["Prefer"] = "count=exact" 
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        content_range = resp.headers.get("Content-Range")
        if content_range:
            # مثال: Content-Range: 0-0/1500
            return int(content_range.split("/")[1])
        return 0
    except requests.exceptions.RequestException:
        return 0


# ============================
# توحيد مسميات شركات الطيران
# ============================

AIRLINE_SYNONYMS: Dict[str, List[str]] = {
    "saudia": [
        "saudia",
        "saudi airlines",
        "saudi arabian airlines",
        "sv",
        "sv ",
        " saudia ",
        "الخطوط السعودية",
    ],
    "flynas": [
        "flynas",
        "fly nas",
        "xy",
        "طيران ناس",
    ],
    "flyadeal": [
        "flyadeal",
        "fly adeal",
        "fd",
        "طيران أديل",
        "طيران اديل",
    ],
    "riyadh airlines": [
        "riyadh airlines",
        "riyadh air",
        "rx",
        "الرياض للطيران",
    ],
}

def _normalize_airline_name(raw: Optional[str]) -> str:
    """إرجاع اسم قياسي لشركة الطيران اعتماداً على المرادفات المعروفة."""
    if not raw:
        return ""
    s = str(raw).strip().lower()
    # مطابقة مباشرة
    for canonical, variants in AIRLINE_SYNONYMS.items():
        for v in variants:
            if v.strip().lower() == s:
                return canonical
    # محاولة المطابقة كجزء من النص
    for canonical, variants in AIRLINE_SYNONYMS.items():
        for v in variants:
            if v.strip().lower() in s:
                return canonical
    return s


# ============================================
# البحث عن رحلة معيّنة برقم الرحلة
# ============================================

def get_dep_flight_events_by_flight_number(
    flight_number: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """جلب سجلات dep_flight_delay المطابقة لرقم رحلة (قدوم أو مغادرة)."""
    fn = (flight_number or "").strip().upper()
    if not fn:
        return []

    # نجلب من السيرفر مرّتين (Arrival/Departure) لتفادي تعقيد OR مع أعمدة فيها مسافات
    params_dep: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.desc"),
        ("Departure Flight Number", f"eq.{fn}"),
    ]
    params_arr: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.desc"),
        ("Arrival Flight Number", f"eq.{fn}"),
    ]

    dep = _get("dep_flight_delay", params_dep)
    arr = _get("dep_flight_delay", params_arr)

    # دمج بدون تكرار
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in dep + arr:
        key = (
            str(r.get("Date") or ""),
            str(r.get("Departure Flight Number") or ""),
            str(r.get("Arrival Flight Number") or ""),
            str(r.get("Airlines") or ""),
            str(r.get("Gate") or r.get("GATE") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= limit:
            break
    return out

def get_sgs_flight_events_by_flight_number(
    flight_number: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """جلب سجلات sgs_flight_delay المطابقة لرقم رحلة."""
    fn = (flight_number or "").strip().upper()
    if not fn:
        return []
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.desc"),
        ("Flight Number", f"eq.{fn}"),
    ]
    return _get("sgs_flight_delay", params)

def normalize_flight_number(raw: Optional[str]) -> Tuple[str, str]:
    """
    يحوّل رقم الرحلة لصيغة موحّدة للمقارنة.
    يعيد (full_normalized, digits_only).
    مثال: " sv 0485 " => ("SV485", "485")
    """
    if raw is None:
        return "", ""
    s = str(raw).strip().upper()
    if not s:
        return "", ""
    # أزل المسافات
    s_no_space = "".join(ch for ch in s if not ch.isspace())
    # استخرج الحروف والأرقام
    letters = "".join(ch for ch in s_no_space if ch.isalpha())
    digits = "".join(ch for ch in s_no_space if ch.isdigit())
    if digits:
        try:
            # إزالة الأصفار على اليسار
            num = str(int(digits))
        except ValueError:
            num = digits
    else:
        num = ""
    full_norm = (letters + num) if (letters or num) else s_no_space
    return full_norm, digits


def flight_number_matches(
    query_flight: str,
    row_flight: str,
    row_airline: Optional[str] = None,
    query_airline: Optional[str] = None,
) -> bool:
    """
    مقارنة ذكية بين رقم الرحلة في السؤال ورقم الرحلة في الصف.
    - تطبيع رقم الرحلة (حروف + أرقام بدون مسافات وأصفار زائدة).
    - السماح بمطابقة على مستوى الأرقام فقط (مثلاً 485 مع SV485).
    - استعمال شركة الطيران كعامل مساعد فقط عند الحاجة، لكن عدم منع التطابق لمجرد اختلاف التكويد.
    """
    q_full, q_digits = normalize_flight_number(query_flight)
    r_full, r_digits = normalize_flight_number(row_flight)

    if not (q_full or q_digits) or not (r_full or r_digits):
        return False

    # تطابق كامل (حروف + أرقام)
    if q_full and r_full and q_full == r_full:
        return True

    # تطابق على مستوى الأرقام فقط (مثلاً 485 == SV485)
    if q_digits and r_digits and q_digits == r_digits:
        return True

    return False

# ============================================
# أدوات متقدمة لتحليل تأخيرات الرحلات / الأقسام / الأكواد
# ============================================

def get_flight_delays_by_airline(
    airline: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب جميع سجلات تأخيرات الرحلات لشركة طيران معيّنة من جدول sgs_flight_delay."""
    airline_norm = _normalize_airline_name(airline)
    if not airline_norm:
        return []
    s, e = _normalize_date_range(start_date, end_date)
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.desc"),
        ("Airlines", f"ilike.*{airline_norm}*"),
        ("Date", f"gte.{s}"),
        ("Date", f"lte.{e}"),
    ]
    return _get("sgs_flight_delay", params)

def get_dep_delays_by_airline(
    airline: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب سجلات التأخيرات لشركة طيران معيّنة من جدول dep_flight_delay."""
    airline_norm = _normalize_airline_name(airline)
    if not airline_norm:
        return []
    s, e = _normalize_date_range(start_date, end_date)
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.desc"),
        ("Airlines", f"ilike.*{airline_norm}*"),
        ("Date", f"gte.{s}"),
        ("Date", f"lte.{e}"),
    ]
    return _get("dep_flight_delay", params)

def get_dep_delays_by_department(
    department: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب سجلات التأخيرات المسجَّلة على قسم معيّن من dep_flight_delay."""
    dept_norm = (department or "").strip()
    if not dept_norm:
        return []
    s, e = _normalize_date_range(start_date, end_date)
    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.desc"),
        ("Department", f"ilike.*{dept_norm}*"),
        ("Date", f"gte.{s}"),
        ("Date", f"lte.{e}"),
    ]
    return _get("dep_flight_delay", params)

def get_flight_delays_by_delay_code(
    delay_code: str,
    airline: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب سجلات تأخيرات الرحلات من جدول sgs_flight_delay حسب كود التأخير."""
    code_norm = (delay_code or "").strip().upper()
    if not code_norm:
        return []
    airline_norm = _normalize_airline_name(airline) if airline else ""
    s, e = _normalize_date_range(start_date, end_date)

    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.desc"),
        ("Delay Code", f"eq.{code_norm}"),
        ("Date", f"gte.{s}"),
        ("Date", f"lte.{e}"),
    ]
    if airline_norm:
        params.append(("Airlines", f"ilike.*{airline_norm}*"))

    return _get("sgs_flight_delay", params)

def get_dep_delays_by_delay_code(
    delay_code: str,
    airline: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب سجلات التأخيرات من جدول dep_flight_delay حسب كود التأخير."""
    code_norm = (delay_code or "").strip().upper()
    if not code_norm:
        return []
    airline_norm = _normalize_airline_name(airline) if airline else ""
    s, e = _normalize_date_range(start_date, end_date)

    params: List[Tuple[str, str]] = [
        ("select", "*"),
        ("limit", str(limit)),
        ("order", "Date.desc"),
        # في dep_flight_delay الكود غالباً داخل Departure Violations كنص، لذلك نستخدم ilike
        ("Departure Violations", f"ilike.*{code_norm}*"),
        ("Date", f"gte.{s}"),
        ("Date", f"lte.{e}"),
    ]
    if airline_norm:
        params.append(("Airlines", f"ilike.*{airline_norm}*"))

    return _get("dep_flight_delay", params)

def list_employee_overtime_simulated(department: Optional[str] = 'TCC', limit: int = 1000) -> List[Dict[str, Any]]:
    """
    دالة وهمية (Simulation) لجلب سجلات العمل الإضافي لاستخدامها في مرحلة
    التشخيص وتحليل الأسباب الجذرية (Root Cause Analysis) بدون الاعتماد على Supabase.

    ملاحظة:
    - هذه الدالة لا تعتمد على قاعدة البيانات الفعلية، بل تعيد بيانات ثابتة (simulated_data).
    - الهدف منها اختبار منطق التحليل في nxs_brain.py أو أي طبقة تحليل أخرى.
    """
    # محاكاة بيانات العمل الإضافي لموظفي TCC كما في الوصف:
    simulated_data = [
        {"Employee ID": 101, "Total Hours": "6",   "Assignment Date": "2025-11-01"},
        {"Employee ID": 102, "Total Hours": "11.5","Assignment Date": "2025-11-01"},  # فوق العتبة
        {"Employee ID": 103, "Total Hours": "9",   "Assignment Date": "2025-11-02"},
        {"Employee ID": 104, "Total Hours": "10.2","Assignment Date": "2025-11-03"},  # فوق العتبة
        {"Employee ID": 105, "Total Hours": "5",   "Assignment Date": "2025-11-03"},
        # ... المزيد من البيانات عند الحاجة ...
    ]

    # محاكاة فلترة بسيطة على القسم:
    if department == 'TCC':
        return simulated_data[:limit]
    return []


def get_delays_with_overtime_link(overtime_records: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """
    دالة مساعدة تربط التأخيرات (dep_flight_delay) بالموظفين الذين عملوا عملاً إضافياً.

    ⚠️ ملاحظة مهمة:
    - هذه دالة محاكاة (Sandbox) كما هو موضّح في النص الأصلي.
    - في التنفيذ الحقيقي يمكن استبدال منطق المحاكاة هنا بربط فعلي باستخدام بيانات dep_flight_delay.
    - حالياً تعيد قاموساً ثابتاً يوضح أن الموظفين 102 و 104 (الذين تجاوزوا 10 ساعات)
      لديهم تأخيرات/مخالفات مرتبطة بالعمل الإضافي.
    """
    # محاكاة نتيجة الربط (Employee ID: [Delay Records])
    simulated_link: Dict[int, List[Dict[str, Any]]] = {
        101: [{"Violation": "TC-OTH", "Delay_Min": 5}],
        102: [{"Violation": "TC-OVT", "Delay_Min": 15}, {"Violation": "TC-GTC", "Delay_Min": 8}],
        103: [{"Violation": "TC-OTH", "Delay_Min": 4}],
        104: [{"Violation": "TC-OVT", "Delay_Min": 20}, {"Violation": "TC-OVT", "Delay_Min": 12}],
        105: [],
    }
    # يمكن لاحقاً استخدام overtime_records لعمل منطق أكثر واقعية
    return simulated_link



# ============================
# محاكاة تحليل الغياب المفاجئ (TC-ABS) - Root Cause Sandbox
# ============================

from typing import List, Dict, Any, Optional

def list_employee_absences(department: Optional[str] = 'TCC', start_date: str = '2025-11-01') -> List[Dict[str, Any]]:
    simulated_data = [
        {"Title": 301, "Date": "2025-11-05", "Department": "TCC", "Employee ID": 201},
        {"Title": 302, "Date": "2025-11-05", "Department": "TCC", "Employee ID": 202},
        {"Title": 303, "Date": "2025-11-05", "Department": "TCC", "Employee ID": 203},
        {"Title": 304, "Date": "2025-11-06", "Department": "TCC", "Employee ID": 204},
    ]
    return [d for d in simulated_data if d["Department"] == department and d["Date"] >= start_date]

def get_shift_report_by_date_and_dept(target_date: str, department: str = 'TCC') -> Optional[Dict[str, Any]]:
    simulated_reports = {
        "2025-11-05": {"Title": 401, "Department": "TCC", "On Duty": 30, "No Show": 3, "Delayed Departures Domestic": 10},
        "2025-11-06": {"Title": 402, "Department": "TCC", "On Duty": 30, "No Show": 1, "Delayed Departures Domestic": 2},
    }
    return simulated_reports.get(target_date)

def get_delays_linked_to_shift_report(shift_title: int) -> List[Dict[str, Any]]:
    if shift_title == 401:
        return [
            {"Flight Number": "SV101", "Violation": "TC-ABS", "Delay_Min": 15},
            {"Flight Number": "NAS303", "Violation": "TC-ABS", "Delay_Min": 10},
            {"Flight Number": "TC202", "Violation": "TC-OTH", "Delay_Min": 5},
        ]
    return []



# ============================
# محاكاة تحليل تنسيق البوابات (TC-GTC) - Root Cause Sandbox
# ============================

def get_gate_coordination_delays(delay_code: str = 'TC-GTC', limit: int = 50) -> List[Dict[str, Any]]:
    """
    جلب سجلات التأخيرات من جدول dep_flight_delay بكود TC-GTC (محاكاة).
    """
    simulated_data = [
        {"Flight Number": "SQ501", "Violation": "TC-GTC", "Delay_Min": 25, "GATE": "A3"},
        {"Flight Number": "EK205", "Violation": "TC-GTC", "Delay_Min": 15, "GATE": "A4"},
        {"Flight Number": "SV320", "Violation": "TC-GTC", "Delay_Min": 10, "GATE": "B7"},
        {"Flight Number": "QR110", "Violation": "TC-GTC", "Delay_Min": 30, "GATE": "A5"},
        {"Flight Number": "LH441", "Violation": "TC-GTC", "Delay_Min": 8,  "GATE": "C2"},
    ]
    return [d for d in simulated_data if d["Violation"] == delay_code][:limit]

def get_flight_log_gate_changes(flight_numbers: List[str]) -> List[Dict[str, Any]]:
    """
    جلب سجلات تغيير البوابات من flight_log للرحلات المتأخرة (محاكاة).
    """
    return [
        {"FLT": "SQ501", "Initial_GATE": "A10", "Final_GATE": "A3", "Change_Time": 25},
        {"FLT": "EK205", "Initial_GATE": "A8",  "Final_GATE": "A4", "Change_Time": 15},
        {"FLT": "QR110", "Initial_GATE": "A1",  "Final_GATE": "A5", "Change_Time": 30},
    ]




# ============================
# محاكاة تحليل مناولة الأمتعة GS-BAG - Root Cause Sandbox
# ============================

def get_sgs_delays_by_delay_code(delay_code: str = 'GS-BAG', limit: int = 200):
    simulated_data = [
        {"FLT": "BA110", "Violation": "GS-BAG", "Delay_Min": 20, "Asset_ID": "LDR-12"}, 
        {"FLT": "KL405", "Violation": "GS-BAG", "Delay_Min": 15, "Asset_ID": "LDR-05"},
        {"FLT": "AF330", "Violation": "GS-BAG", "Delay_Min": 25, "Asset_ID": "LDR-12"},
        {"FLT": "LH888", "Violation": "GS-BAG", "Delay_Min": 10, "Asset_ID": "TUG-01"},
        {"FLT": "SV999", "Violation": "GS-BAG", "Delay_Min": 35, "Asset_ID": "LDR-05"},
    ]
    return [d for d in simulated_data if d["Violation"] == delay_code][:limit]

def get_asset_breakdown_events(asset_type: str = 'Loader', start_date: str = '2025-10-01'):
    return [
        {"Event_ID": 501, "Asset_ID": "LDR-12", "Type": "Loader", "Breakdown_Type": "Major", "Asset_Age_Yrs": 7},
        {"Event_ID": 502, "Asset_ID": "LDR-05", "Type": "Loader", "Breakdown_Type": "Major", "Asset_Age_Yrs": 8},
        {"Event_ID": 503, "Asset_ID": "LDR-05", "Type": "Loader", "Breakdown_Type": "Minor", "Asset_Age_Yrs": 8},
        {"Event_ID": 504, "Asset_ID": "LDR-12", "Type": "Loader", "Breakdown_Type": "Minor", "Asset_Age_Yrs": 7},
        {"Event_ID": 505, "Asset_ID": "TUG-01", "Type": "Pushback", "Breakdown_Type": "Major", "Asset_Age_Yrs": 2},
    ]

# ============================
# محاكاة تحليل الصيانة الوقائية MT-SP - Root Cause Sandbox
# ============================

def get_maintenance_delays(delay_code: str = 'MT-SP', limit: int = 150):
    simulated_data = [
        {"FLT": "AA701", "Violation": "MT-SP", "Delay_Min": 30, "Asset_ID": "TUG-08"}, 
        {"FLT": "DL555", "Violation": "MT-SP", "Delay_Min": 25, "Asset_ID": "GPU-14"},
        {"FLT": "UA100", "Violation": "MT-SP", "Delay_Min": 15, "Asset_ID": "TUG-08"},
        {"FLT": "EK001", "Violation": "MT-SP", "Delay_Min": 40, "Asset_ID": "GPU-14"},
    ]
    return [d for d in simulated_data if d["Violation"] == delay_code][:limit]


def get_overdue_pm_events(asset_ids: list):
    return [
        {"Event_ID": 601, "Asset_ID": "TUG-08", "Event_Type": "PM Overdue", "Overdue_Days": 10},
        {"Event_ID": 602, "Asset_ID": "GPU-14", "Event_Type": "PM Overdue", "Overdue_Days": 6},
        {"Event_ID": 603, "Asset_ID": "TUG-05", "Event_Type": "PM Overdue", "Overdue_Days": 2},
    ]




# ============================
# محاكاة تحليل عمليات الوقود FU-OPS - Root Cause Sandbox
# ============================

def get_fueling_delays(delay_code: str = 'FU-OPS', limit: int = 100) -> List[Dict[str, Any]]:
    """جلب سجلات التأخيرات من جدول sgs_flight_delay بكود FU-OPS (محاكاة)."""
    simulated_data = [
        {"FLT": "AA701", "Violation": "FU-OPS", "Delay_Min": 10, "SCHED_DEP": "08:15"},
        {"FLT": "BA050", "Violation": "FU-OPS", "Delay_Min": 8,  "SCHED_DEP": "09:30"},
        {"FLT": "LH900", "Violation": "FU-OPS", "Delay_Min": 5,  "SCHED_DEP": "11:30"},
        {"FLT": "EK001", "Violation": "FU-OPS", "Delay_Min": 12, "SCHED_DEP": "09:00"},
    ]
    return [d for d in simulated_data if d["Violation"] == delay_code][:limit]


def get_flight_sector_data(flight_numbers: List[str]) -> List[Dict[str, Any]]:
    """جلب بيانات القطاع من flight_log لتحديد ما إذا كانت الرحلة طويلة المسافة (محاكاة)."""
    return [
        {"FLT": "AA701", "Destination": "JFK", "Is_Long_Haul": True},
        {"FLT": "BA050", "Destination": "LHR", "Is_Long_Haul": True},
        {"FLT": "LH900", "Destination": "FRA", "Is_Long_Haul": False},
        {"FLT": "EK001", "Destination": "DXB", "Is_Long_Haul": True},
    ]


# ============================
# محاكاة منطق قفل الأصول الآلي - Asset Locking Sandbox
# ============================

def update_asset_status(asset_id: str, new_status: str, reason: str) -> bool:
    """تحديث حالة الأصل في جدول asset_register (تنفيذ القفل الآلي - محاكاة)."""
    # في التنفيذ الحقيقي يتم استدعاء Supabase UPDATE API
    if asset_id in ["TUG-08", "GPU-14"]:
        # مثال لسجل تحديث في النظام الحقيقي:
        # print(f"DB Update: Asset {asset_id} status changed to '{new_status}'. Reason: {reason}")
        return True
    return False


def log_system_alert(alert_type: str, message: str) -> bool:
    """تسجيل تنبيه آلي لفرق الصيانة في جدول alerts_log (محاكاة)."""
    # في التنفيذ الحقيقي يتم استدعاء Supabase POST API
    # print(f"ALERT LOGGED: Type={alert_type}, Message='{message}'")
    return True




# ============================
# المرحلة الثامنة: سياسات العمل الإضافي وقياس الأثر النهائي - Policy & ROI Sandbox
# ============================

def update_ot_policy(department: str, max_hours: float, policy_date: str) -> bool:
    """تحديث سقف العمل الإضافي في جدول hr_policy_register (محاكاة)."""
    # في التنفيذ الحقيقي: يتم استدعاء Supabase UPDATE API
    # هنا نضمن أن سياسة TCC تطبق السقف الحرج 10 ساعات
    return department == 'TCC' and max_hours == 10.0


def send_ot_notification(manager_email: str, employee_id: int, current_ot: float, threshold: float) -> bool:
    """إرسال تنبيه آلي للمدير حول موظف يتجاوز سقف العمل الإضافي (محاكاة)."""
    # في التنفيذ الحقيقي: يتم استدعاء خدمة إشعارات (Email / Teams / SMS)
    return True


def get_baseline_otp() -> float:
    """جلب الأداء الأولي (Baseline OTP) قبل التدخلات (محاكاة)."""
    return 84.50


def get_total_delay_reduction() -> Dict[str, float]:
    """جلب إجمالي دقائق التأخير المُزالة (الموفَّرة) شهرياً لكل كود تأخير مستهدف (محاكاة)."""
    return {
        "TC-OVT": 15000.0,
        "TC-ABS": 8000.0,
        "TC-GTC": 7500.0,
        "GS-BAG": 12000.0,
        "MT-SP": 18000.0,
        "FU-OPS": 4500.0,
    }


def get_intervention_costs() -> Dict[str, float]:
    """جلب إجمالي التكاليف المباشرة للتدخلات التكتيكية (بدون CAPEX) (محاكاة)."""
    return {
        "Software_Deployment": 5000.0,
        "Minor_Asset_Repair": 15000.0,
        "Training_and_Policy_Change": 10000.0,
    }


def get_asset_replacement_plan() -> List[Dict[str, Any]]:
    """جلب القائمة النهائية للأصول القديمة التي تحتاج إلى استبدال (CAPEX) (محاكاة)."""
    return [
        {"Asset_ID": "LDR-05", "Type": "Loader", "Replacement_Cost": 80000.0},
        {"Asset_ID": "LDR-12", "Type": "Loader", "Replacement_Cost": 80000.0},
        {"Asset_ID": "LDR-21", "Type": "Loader", "Replacement_Cost": 80000.0},
        {"Asset_ID": "LDR-35", "Type": "Loader", "Replacement_Cost": 80000.0},
    ]


def get_manpower_demand(department: str = 'TCC') -> Dict[str, int]:
    """جلب عدد الموظفين الإضافيين المطلوبين للحفاظ على سقف العمل الإضافي (محاكاة)."""
    if department == 'TCC':
        return {"TCC_Staff_Needed": 12}
    return {f"{department}_Staff_Needed": 0}


def get_advanced_ml_features() -> List[Dict[str, Any]]:
    """
    جلب مجموعة بيانات شاملة (Features) لنموذج Random Forest لتصنيف التأخير.
    """
    # ⚠️ في التنفيذ الحقيقي: يتم استدعاء Supabase Views لدمج عدة جداول عملياتية (5+ جداول)
    
    # محاكاة لبيانات مُركّبة (تشمل حالات سابقة تسببت في تأخيرات)
    return [
        {"FLT": "AA300", "Sched_Time_H": 8,  "Is_Peak": 1, "Staff_Avg_OT": 12.0, "Asset_PM_Overdue": 6, "Delay_Class": "ASSET"},     # مشكلة في الأصول
        {"FLT": "BA400", "Sched_Time_H": 15, "Is_Peak": 0, "Staff_Avg_OT": 8.5,  "Asset_PM_Overdue": 1, "Delay_Class": "NO_DELAY"},
        {"FLT": "EK600", "Sched_Time_H": 9,  "Is_Peak": 1, "Staff_Avg_OT": 10.5, "Asset_PM_Overdue": 2, "Delay_Class": "MANPOWER"}, # مشكلة في العمالة
        {"FLT": "LH800", "Sched_Time_H": 18, "Is_Peak": 0, "Staff_Avg_OT": 15.0, "Asset_PM_Overdue": 1, "Delay_Class": "MANPOWER"}, # مشكلة في العمالة
        # ... 1000+ صف من البيانات في التطبيق الفعلي ...
    ]

