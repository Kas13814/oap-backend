# nxs_supabase_client.py
# استخدام REST API لـ Supabase بدون supabase-py

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

import requests
from dotenv import load_dotenv

# ============================
# تحميل متغيرات البيئة من ملف .env
# ============================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL:
    # ❌ خطأ: يرجى وضع قيمة SUPABASE_URL في ملف .env
    raise RuntimeError("❌ SUPABASE_URL غير موجود في ملف .env")

if not SUPABASE_ANON_KEY:
    # ❌ خطأ: يرجى وضع قيمة SUPABASE_ANON_KEY في ملف .env
    raise RuntimeError("❌ SUPABASE_ANON_KEY غير موجود في ملف .env")

# عنوان REST الأساسي
REST_BASE_URL = SUPABASE_URL.rstrip("/") + "/rest/v1"

# الرؤوس المشتركة لكل الطلبات
COMMON_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


logger = logging.getLogger("tcc_supabase")


# ============================
# دالة GET عامة
# ============================
def _get(table: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    تنفيذ طلب GET عام على جدول معيّن في Supabase.
    table: اسم الجدول مثل 'employee_master_db'
    params: باراميترات PostgREST (فلترة، limit، إلخ)
    """
    url = f"{REST_BASE_URL}/{table}"
    
    # تحديد limit افتراضي إذا لم يتم تحديده
    if "limit" not in params:
        params["limit"] = 10000

    try:
        resp = requests.get(url, headers=COMMON_HEADERS, params=params, timeout=20)
        resp.raise_for_status() # رفع استثناء في حال وجود أخطاء 4xx أو 5xx
    except requests.exceptions.RequestException as e:
        logger.warning("Supabase GET request failed for table %s: %s", table, e)
        return []

    data = resp.json()
    if isinstance(data, dict) and 'message' in data and data['message'] == 'Not Found':
        # حالة خطأ من Supabase
        return []
    if isinstance(data, dict):
        # في حال تم جلب صف واحد
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
def get_employee_info(emp_id: str) -> Optional[Dict[str, Any]]:
    """جلب معلومات موظف واحد من جدول employee_master_db."""
    table_name = "employee_master_db"
    # يتم الجلب بدون فلترة في URL للمرونة
    rows = _get(table_name, {"select": "*", "limit": 10000})

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
    rows = list_dep_flight_delays(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)
    filtered = _filter_employee_range(
        rows, employee_id, ["Date"], s, e
    )
    return filtered[:limit]


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
    rows = list_employee_absence(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)
    filtered = _filter_employee_range(
        rows, employee_id, ["Date"], s, e
    )
    return filtered[:limit]


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
    rows = list_employee_delay_log(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)
    filtered = _filter_employee_range(
        rows, employee_id, ["Date"], s, e
    )
    return filtered[:limit]


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
    rows = list_employee_overtime(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)
    filtered = _filter_employee_range(
        rows, employee_id, ["Assignment Date"], s, e
    )
    return filtered[:limit]


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
    rows = list_employee_sick_leave(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)
    filtered = _filter_employee_range(
        rows, employee_id, ["Date"], s, e
    )
    return filtered[:limit]


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
    rows = list_operational_events(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)
    filtered = _filter_employee_range(
        rows, employee_id, ["Event Date"], s, e
    )
    return filtered[:limit]


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

    rows = list_dep_flight_delays(limit=10000)
    result: List[Dict[str, Any]] = []
    for r in rows:
        arr = str(r.get("Arrival Flight Number") or "").strip().upper()
        dep = str(r.get("Departure Flight Number") or "").strip().upper()
        if fn == arr or fn == dep:
            result.append(r)
            if len(result) >= limit:
                break
    return result


def get_sgs_flight_events_by_flight_number(
    flight_number: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """جلب سجلات sgs_flight_delay المطابقة لرقم رحلة."""
    fn = (flight_number or "").strip().upper()
    if not fn:
        return []

    rows = list_all_flight_delays(limit=10000)
    result: List[Dict[str, Any]] = []
    for r in rows:
        num = str(r.get("Flight Number") or "").strip().upper()
        if fn == num:
            result.append(r)
            if len(result) >= limit:
                break
    return result



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

    rows = list_all_flight_delays(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)

    result: List[Dict[str, Any]] = []
    for r in rows:
        a_norm = _normalize_airline_name(r.get("Airlines"))
        if airline_norm and a_norm and airline_norm != a_norm:
            continue
        if not _in_date_range(r, ["Date"], s, e):
            continue
        result.append(r)
        if len(result) >= limit:
            break
    return result


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

    rows = list_dep_flight_delays(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)

    result: List[Dict[str, Any]] = []
    for r in rows:
        a_norm = _normalize_airline_name(r.get("Airlines"))
        if airline_norm and a_norm and airline_norm != a_norm:
            continue
        if not _in_date_range(r, ["Date"], s, e):
            continue
        result.append(r)
        if len(result) >= limit:
            break
    return result


def get_dep_delays_by_department(
    department: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """جلب سجلات التأخيرات المسجَّلة على قسم معيّن من dep_flight_delay."""
    dept_norm = (department or "").strip().lower()
    if not dept_norm:
        return []

    rows = list_dep_flight_delays(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)

    result: List[Dict[str, Any]] = []
    for r in rows:
        d = str(r.get("Department") or "").strip().lower()
        if dept_norm not in d:
            continue
        if not _in_date_range(r, ["Date"], s, e):
            continue
        result.append(r)
        if len(result) >= limit:
            break
    return result


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
    rows = list_all_flight_delays(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)

    result: List[Dict[str, Any]] = []
    for r in rows:
        c = str(r.get("Delay Code") or "").strip().upper()
        if code_norm not in c:
            continue

        if airline_norm:
            a_norm = _normalize_airline_name(r.get("Airlines"))
            if airline_norm and a_norm and airline_norm != a_norm:
                continue

        if not _in_date_range(r, ["Date"], s, e):
            continue

        result.append(r)
        if len(result) >= limit:
            break
    return result


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
    rows = list_dep_flight_delays(limit=10000)
    s, e = _normalize_date_range(start_date, end_date)

    result: List[Dict[str, Any]] = []
    for r in rows:
        viol = str(r.get("Departure Violations") or "").strip().upper()
        if code_norm not in viol:
            continue

        if airline_norm:
            a_norm = _normalize_airline_name(r.get("Airlines"))
            if airline_norm and a_norm and airline_norm != a_norm:
                continue

        if not _in_date_range(r, ["Date"], s, e):
            continue

        result.append(r)
        if len(result) >= limit:
            break
    return result

# ============================================
# محاكاة تحليل العمل الإضافي وربطه بالتأخيرات (Root Cause Sandbox)
# ============================================

from typing import List, Dict, Any, Optional  # استيراد للاستخدام في الدوال أدناه (مكرر لا يسبب مشكلة)

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

