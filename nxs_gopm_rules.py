# nxs_gopm_rules.py
# GOPM — Ground Operations Procedures Manual rules/tables (Aircraft Ramp Handling)
#
# This module encodes:
# - 13.16.3.1 Scheduled operations TURNAROUND (Aircraft Minimum Ground Time - MGT)
# - 13.16.3.2 Scheduled operations TRANSIT (Aircraft MGT)
# - 13.16.3.3 Scheduled operations Mixed Flights (MGT)
# - Activity breakdown tables (minutes) for:
#   * B777-368 / B787-10 (Turnaround + Transit)
#   * A330 / B787-9 (Transit)
#   * A321 / A320 (Turnaround + Transit)
#   * B757 (Turnaround + Transit)
#   * Combined / Mixed flights (minutes)
# - 13.16.13.1 Aircraft Delivery before S.T.D (hours)
#
# Notes:
# - Values like "15.00/20*" mean: 15 minutes with 20 resources (per footnote "*").
#   The part before "/" is minutes. The part after "/" is the resource count (porters/trucks),
#   depending on the activity's footnote.
# - (SA) means: security alert stations (use the SA variant when is_security_alert_station=True).
# - Local towing rule: for (DOM-INTL / INTL-DOM) at (JED-T1) & (RUH), Local MGT = Aircraft MGT + 20 minutes.
# - Destination constraints apply to TURNAROUND only (per the provided hint list).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, List, Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_minutes_from_hhmm(hhmm: str) -> int:
    """Convert 'HH:MM' -> total minutes."""
    hhmm = hhmm.strip()
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)

def _to_hhmm_from_minutes(total_minutes: int) -> str:
    """Convert total minutes -> 'HH:MM' (zero-padded)."""
    if total_minutes < 0:
        total_minutes = 0
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh:02d}:{mm:02d}"

# A single MGT cell may have a normal and an (SA) variant.
MGTCell = Union[str, Tuple[str, str]]  # 'HH:MM' or ('HH:MM', 'HH:MM_SA')

def _pick_mgt_cell(cell: MGTCell, is_security_alert_station: bool) -> str:
    if isinstance(cell, tuple):
        return cell[1] if is_security_alert_station else cell[0]
    return cell

# ---------------------------------------------------------------------------
# 13.16.3 — Aircraft Minimum Ground Time (MGT)
# ---------------------------------------------------------------------------

# Columns used in the GOPM tables.
# Turnaround columns:
#   JED, RUH, DMM, MED, OTHER_DOM_STN, INT_STNS, LONG_HAUL_STN, UK
# Transit columns:
#   JED, RUH, DMM, MED, AHB_TUU, OTHER_DOM_STN, INT_STNS, LHR

TURNAROUND_MGT: Dict[str, Dict[str, Dict[str, MGTCell]]] = {
    # B777-368 / B787-10
    "B777-368/B787-10": {
        "DOM-DOM":     {"JED": "01:10", "RUH": "01:10", "DMM": "01:05", "MED": "01:05", "OTHER_DOM_STN": "01:00"},
        "DOM-INTL":    {"JED": "01:20", "RUH": "01:20", "DMM": "01:15", "MED": "01:15"},
        "INTL-DOM":    {"JED": "01:30", "RUH": "01:30", "DMM": "01:25", "MED": "01:25"},
        # INTL-INTL has station categories for INT_STNS / LONG_HAUL_STN / UK (with SA variants where shown).
        "INTL-INTL":   {
            "JED": "01:35", "RUH": "01:35", "DMM": "01:30", "MED": "01:30",
            "INT_STNS": ("01:25", "01:40"),          # (SA)
            "LONG_HAUL_STN": ("01:30", "01:45"),     # (SA)
            "UK": "01:40",
        },
    },

    # A330 / B787-9
    "A330/B787-9": {
        "DOM-DOM":     {"JED": "01:05", "RUH": "01:05", "DMM": "00:55", "MED": "00:55", "OTHER_DOM_STN": "00:50"},
        "DOM-INTL":    {"JED": "01:10", "RUH": "01:10", "DMM": "01:00", "MED": "01:00"},
        "INTL-DOM":    {"JED": "01:20", "RUH": "01:20", "DMM": "01:10", "MED": "01:10"},
        "INTL-INTL":   {
            "JED": "01:25", "RUH": "01:25", "DMM": "01:15", "MED": "01:15",
            "INT_STNS": ("01:15", "01:30"),          # (SA)
            "LONG_HAUL_STN": ("01:20", "01:35"),     # (SA)
            "UK": "01:30",
        },
    },

    # A321 / A320
    "A321/A320": {
        "DOM-DOM":     {"JED": "00:45", "RUH": "00:45", "DMM": "00:40", "MED": "00:40", "OTHER_DOM_STN": "00:40"},
        "DOM-INTL":    {"JED": "00:50", "RUH": "00:50", "DMM": "00:45", "MED": "00:45", "OTHER_DOM_STN": "00:45"},
        "INTL-DOM":    {"JED": "00:55", "RUH": "00:55", "DMM": "00:50", "MED": "00:50", "OTHER_DOM_STN": "00:50"},
        "INTL-INTL":   {
            "JED": "01:00", "RUH": "01:00", "DMM": "00:55", "MED": "00:55",
            "INT_STNS": ("00:55", "01:05"),          # (SA)
        },
    },
}

TRANSIT_MGT: Dict[str, Dict[str, Dict[str, MGTCell]]] = {
    "B777-368/B787-10": {
        "DOM-DOM":   {"JED": "01:05", "RUH": "01:05", "DMM": "01:00", "MED": "01:00", "AHB_TUU": "00:55", "OTHER_DOM_STN": "00:50"},
        "DOM-INTL":  {"JED": "01:15", "RUH": "01:15", "DMM": "01:00", "MED": "01:00"},
        "INTL-DOM":  {"JED": "00:55", "RUH": "00:55", "DMM": "00:40", "MED": "00:40"},
        "INTL-INTL": {"INT_STNS": ("01:00", "01:15"), "LHR": "01:00"},  # (SA) variant on INT_STNS
    },
    "A330/B787-9": {
        "DOM-DOM":   {"JED": "01:00", "RUH": "01:00", "DMM": "00:45", "MED": "00:45", "AHB_TUU": "00:55", "OTHER_DOM_STN": "00:50"},
        "DOM-INTL":  {"JED": "01:10", "RUH": "01:10", "DMM": "00:45", "MED": "00:45"},
        "INTL-DOM":  {"JED": "00:45", "RUH": "00:45", "DMM": "00:35", "MED": "00:35"},
        "INTL-INTL": {"INT_STNS": ("01:00", "01:15"), "LHR": "01:00"},  # (SA)
    },
    "A321/A320": {
        "DOM-DOM":   {"JED": "00:40", "RUH": "00:40", "DMM": "00:30", "MED": "00:30", "AHB_TUU": "00:35", "OTHER_DOM_STN": "00:35"},
        "DOM-INTL":  {"JED": "00:40", "RUH": "00:40", "DMM": "00:30", "MED": "00:30"},
        "INTL-DOM":  {"JED": "00:30", "RUH": "00:30", "DMM": "00:25", "MED": "00:25"},
        "INTL-INTL": {"INT_STNS": "00:30"},
    },
}

# 13.16.3.3 Scheduled Operations Mixed Flights (MGT)
MIXED_FLIGHTS_MGT: Dict[str, Dict[str, str]] = {
    # INT-DOM (Inbound) JED/RUH/DMM/MED
    "INTL-DOM_INBOUND_JED_RUH_DMM_MED": {
        "B777-368/B787-10": "01:15",
        "A330/B787-9": "01:00",
    },
    # DOM-INT (Outbound) JED
    "DOM-INTL_OUTBOUND_JED": {
        "B777-368/B787-10": "01:15 / 01:45 USA",
        "A330/B787-9": "01:10 / 01:30 USA",
    },
    # DOM-INT (Outbound) RUH/DMM/MED
    "DOM-INTL_OUTBOUND_RUH_DMM_MED": {
        "B777-368/B787-10": "01:00 / 01:45 USA",
        "A330/B787-9": "00:50 / 01:30 USA",
    },
}

# Hint constants (from the 13.16.3.2 image)
# (SA) = security alert stations.
LOCAL_TOWING_ADD_MINUTES = 20  # towing time

TURNAROUND_MIN_TO_USA_MINUTES = 170  # 02:50
TURNAROUND_MIN_TO_KAN_MINUTES = 120  # 02:00
TURNAROUND_MIN_TO_ALL_USA_STATIONS_MINUTES = 120  # 02:00
TURNAROUND_MIN_TO_SSH_MINUTES = 75   # 01:15

# Long-haul stations list (as provided)
LONG_HAUL_STATIONS = {"MNL", "CAN", "YYZ", "IAD", "LAX", "JFK", "KUL", "CGK", "SIN"}

# "USA stations" mentioned in the hint list (keep only those explicitly shown in images).
USA_STATIONS_IN_HINT_LIST = {"YYZ", "IAD", "LAX", "JFK"}

# ---------------------------------------------------------------------------
# Public MGT lookup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MGTLookupResult:
    operation: str
    aircraft_group: str
    movement: str
    station_key: str
    base_mgt: str
    adjusted_mgt: str
    applied_rules: List[str]

def lookup_mgt(
    operation: str,
    aircraft_group: str,
    movement: str,
    station: str,
    *,
    destination_station: Optional[str] = None,
    is_security_alert_station: bool = False,
    apply_local_towing_rule: bool = False,
) -> MGTLookupResult:
    """
    Look up MGT for TURNAROUND or TRANSIT.

    operation: "TURNAROUND" | "TRANSIT"
    aircraft_group: e.g. "B777-368/B787-10", "A330/B787-9", "A321/A320"
    movement: "DOM-DOM" | "DOM-INTL" | "INTL-DOM" | "INTL-INTL"
    station: for scheduled ops tables: "JED","RUH","DMM","MED","OTHER_DOM_STN","AHB_TUU","INT_STNS","LONG_HAUL_STN","UK","LHR"
      - If a real destination is known, you can pass destination_station and station may be "INT_STNS"/"LONG_HAUL_STN".
    destination_station: e.g. "JFK", "SSH", "KAN", "USA" ...
    is_security_alert_station: if True, use the (SA) variant where provided.
    apply_local_towing_rule: if True, add towing time to JED-T1/RUH mixed movements (DOM-INTL / INTL-DOM).

    Returns base and adjusted MGT + applied rules list.
    """
    op = operation.strip().upper()
    movement = movement.strip().upper()
    station_key = station.strip().upper()
    dest = destination_station.strip().upper() if destination_station else None

    if op not in {"TURNAROUND", "TRANSIT"}:
        raise ValueError(f"Unsupported operation: {operation}")

    if op == "TURNAROUND":
        table = TURNAROUND_MGT
    else:
        table = TRANSIT_MGT

    if aircraft_group not in table:
        raise KeyError(f"Unknown aircraft_group: {aircraft_group}")

    if movement not in table[aircraft_group]:
        raise KeyError(f"Unknown movement '{movement}' for aircraft_group '{aircraft_group}'")

    row = table[aircraft_group][movement]
    if station_key not in row:
        raise KeyError(f"Unknown station '{station_key}' for op={op}, aircraft_group={aircraft_group}, movement={movement}")

    base_cell = row[station_key]
    base_hhmm = _pick_mgt_cell(base_cell, is_security_alert_station=is_security_alert_station)

    applied: List[str] = []
    adjusted_minutes = _to_minutes_from_hhmm(base_hhmm)

    # Rule: Local towing (JED-T1) & (RUH) for mixed movements DOM-INTL / INTL-DOM add 20 minutes.
    if apply_local_towing_rule and movement in {"DOM-INTL", "INTL-DOM"} and station_key in {"JED", "RUH", "JED-T1"}:
        adjusted_minutes += LOCAL_TOWING_ADD_MINUTES
        applied.append("Local towing rule: +20 minutes at JED-T1/RUH for DOM-INTL or INTL-DOM")

    # Rule: TURNAROUND destination constraints (minimums).
    if op == "TURNAROUND" and dest:
        # 3) Minimum ground time (turnaround) to USA is 02:50 HRS.
        if dest == "USA":
            min_required = TURNAROUND_MIN_TO_USA_MINUTES
            if adjusted_minutes < min_required:
                adjusted_minutes = min_required
                applied.append("Turnaround destination minimum: USA => 02:50")
        # 5) Minimum ground time (turnaround) for all USA stations is 02:00 HRS.
        elif dest in USA_STATIONS_IN_HINT_LIST:
            min_required = TURNAROUND_MIN_TO_ALL_USA_STATIONS_MINUTES
            if adjusted_minutes < min_required:
                adjusted_minutes = min_required
                applied.append("Turnaround destination minimum: USA stations (JFK/LAX/IAD/YYZ) => 02:00")
        # 4) Minimum ground time (turnaround) to KAN is 02:00 HRS.
        elif dest == "KAN":
            min_required = TURNAROUND_MIN_TO_KAN_MINUTES
            if adjusted_minutes < min_required:
                adjusted_minutes = min_required
                applied.append("Turnaround destination minimum: KAN => 02:00")
        # 6) Minimum ground time (turnaround) for SSH station is 01:15 (Security/Baggage Identification).
        elif dest == "SSH":
            min_required = TURNAROUND_MIN_TO_SSH_MINUTES
            if adjusted_minutes < min_required:
                adjusted_minutes = min_required
                applied.append("Turnaround destination minimum: SSH => 01:15 (Security/Baggage Identification)")

        # Long haul station category (definition + list are in the hint). If the caller provided a long-haul destination,
        # and the table row has a LONG_HAUL_STN category value, prefer that category (with SA variant if applicable).
        if dest in LONG_HAUL_STATIONS and "LONG_HAUL_STN" in row:
            lh_cell = row["LONG_HAUL_STN"]
            lh_hhmm = _pick_mgt_cell(lh_cell, is_security_alert_station=is_security_alert_station)
            lh_minutes = _to_minutes_from_hhmm(lh_hhmm)
            # Use the larger of computed and the long-haul category (safe).
            if adjusted_minutes < lh_minutes:
                adjusted_minutes = lh_minutes
                applied.append("Long-haul station category applied (LONG_HAUL_STN)")

    adjusted_hhmm = _to_hhmm_from_minutes(adjusted_minutes)

    # Mixed flights note (13.16.3.3): turnaround/transit times are the same as SV type of aircraft.
    # This is informational; the mixed flights table is exposed separately via MIXED_FLIGHTS_MGT.
    if op in {"TURNAROUND", "TRANSIT"}:
        applied.append("Note: Mixed flights MGT uses SV aircraft type (see MIXED_FLIGHTS_MGT)")

    return MGTLookupResult(
        operation=op,
        aircraft_group=aircraft_group,
        movement=movement,
        station_key=station_key,
        base_mgt=base_hhmm,
        adjusted_mgt=adjusted_hhmm,
        applied_rules=applied,
    )

# ---------------------------------------------------------------------------
# Activity Breakdown tables (minutes)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StarValue:
    minutes: float
    resource_count: int
    footnote: str  # "*" or "**"

def _sv(minutes: float, resource_count: int, footnote: str) -> StarValue:
    return StarValue(minutes=minutes, resource_count=resource_count, footnote=footnote)

@dataclass(frozen=True)
class ActivityBreakdownResult:
    aircraft_group: str
    operation: str  # TURNAROUND / TRANSIT / COMBINED_MIXED
    movement: str
    activities: Dict[str, Union[float, str, StarValue]]  # float minutes, "-" or StarValue
    total_or_min_ground_time: Union[float, str]
    assumptions: List[str]
    notes: List[str]

# B777-368 / B787-10 — TURNAROUND (100% LF)
B777_TURNAROUND = {
    "operation": "TURNAROUND",
    "assumptions": ["100% Load Factor"],
    "notes": [
        "Cabin cleaning and galley services are done simultaneously (apply for all aircraft types)."
    ],
    "movements": {
        "DOM-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 12.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": _sv(12.00, 20, "*"),
            "Galley Service": _sv(17.00, 2, "**"),
            "Cabin Security": 5.00,
            "Passenger Enplaning": 26.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 65.00,
        },
        "DOM-INTL": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 12.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": 20.00,
            "Galley Service": 25.00,
            "Cabin Security": 5.00,
            "Passenger Enplaning": 28.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 75.00,
        },
        "INTL-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 12.00,
            "Custom Clearance": 15.00,
            "Cabin Cleaning": 20.00,
            "Galley Service": 20.00,
            "Cabin Security": 5.00,
            "Passenger Enplaning": 28.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 85.00,
        },
        "INTL-INTL": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 12.00,
            "Custom Clearance": 15.00,
            "Cabin Cleaning": 25.00,
            "Galley Service": 25.00,
            "Cabin Security": 5.00,
            "Passenger Enplaning": 28.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 90.00,
        },
    },
}

# B777-368 / B787-10 — TRANSIT (50% LF)
B777_TRANSIT = {
    "operation": "TRANSIT",
    "assumptions": ["50% Load Factor"],
    "notes": [],
    "movements": {
        "DOM-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 10.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": _sv(15.00, 20, "*"),
            "Galley Service": _sv(20.00, 2, "**"),
            "Cabin Security": "-",
            "Passenger Enplaning": 25.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 60.00,
        },
        "DOM-INTL": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": "-",
            "Custom Clearance": "-",
            "Cabin Cleaning": 20.00,
            "Galley Service": 30.00,
            "Cabin Security": "-",
            "Passenger Enplaning": 25.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 60.00,
        },
        "INTL-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 10.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": 20.00,
            "Galley Service": 25.00,
            "Cabin Security": "-",
            "Passenger Enplaning": "-",
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 40.00,
        },
        "INTL-INTL": {
            "Blocks IN / Door Opening": "-",
            "PAX deplaning": "-",
            "Custom Clearance": "-",
            "Cabin Cleaning": "-",
            "Galley Service": "-",
            "Cabin Security": "-",
            "Passenger Enplaning": "-",
            "FNLZTN / Door CLSD": "-",
            "BLOCKS OUT": "-",
            "Total Ground Time": "-",
        },
    },
}

# A330 / B787-9 — TRANSIT (50% LF)
A330_TRANSIT = {
    "operation": "TRANSIT",
    "assumptions": ["50% Load Factor"],
    "notes": [],
    "movements": {
        "DOM-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 5.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": _sv(15.00, 15, "*"),
            "Galley Service": _sv(18.00, 2, "**"),
            "Cabin Security": "-",
            "Passenger Enplaning": 17.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 45.00,
        },
        "DOM-INTL": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": "-",
            "Custom Clearance": "-",
            "Cabin Cleaning": 18.00,
            "Galley Service": 23.00,
            "Cabin Security": "-",
            "Passenger Enplaning": 17.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 45.00,
        },
        "INTL-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 5.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": 18.00,
            "Galley Service": 25.00,
            "Cabin Security": "-",
            "Passenger Enplaning": "-",
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 35.00,
        },
        "INTL-INTL": {
            "Blocks IN / Door Opening": "-",
            "PAX deplaning": "-",
            "Custom Clearance": "-",
            "Cabin Cleaning": "-",
            "Galley Service": "-",
            "Cabin Security": "-",
            "Passenger Enplaning": "-",
            "FNLZTN / Door CLSD": "-",
            "BLOCKS OUT": "-",
            "Total Ground Time": "-",
        },
    },
}

# A321/A320 — TURNAROUND (100% LF)
A321_TURNAROUND = {
    "operation": "TURNAROUND",
    "assumptions": ["100% Load Factor"],
    "notes": [],
    "movements": {
        "DOM-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 8.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": _sv(10.00, 8, "*"),
            "Galley Services": _sv(10.00, 1, "**"),
            "Cabin Security": 3.00,
            "Passenger Enplaning": 14.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 40.00,
        },
        "DOM-INTL": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 8.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": 15.00,
            "Galley Services": 15.00,
            "Cabin Security": 3.00,
            "Passenger Enplaning": 14.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 45.00,
        },
        "INTL-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 8.00,
            "Custom Clearance": 5.00,
            "Cabin Cleaning": 15.00,
            "Galley Services": 15.00,
            "Cabin Security": 3.00,
            "Passenger Enplaning": 14.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 50.00,
        },
        "INTL-INTL": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 8.00,
            "Custom Clearance": 5.00,
            "Cabin Cleaning": 20.00,
            "Galley Services": 20.00,
            "Cabin Security": 3.00,
            "Passenger Enplaning": 14.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 55.00,
        },
    },
}

# A321/A320 — TRANSIT (50% LF)
A321_TRANSIT = {
    "operation": "TRANSIT",
    "assumptions": ["50% Load Factor"],
    "notes": [],
    "movements": {
        "DOM-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 4.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": _sv(10.00, 8, "*"),
            "Galley Service": _sv(14.00, 1, "**"),
            "Cabin Security": "-",
            "Passenger Enplaning": 7.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 30.00,
        },
        "DOM-INTL": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": "-",
            "Custom Clearance": "-",
            "Cabin Cleaning": 13.00,
            "Galley Service": 18.00,
            "Cabin Security": "-",
            "Passenger Enplaning": 7.00,
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 30.00,
        },
        "INTL-DOM": {
            "Blocks IN / Door Opening": 1.00,
            "PAX deplaning": 4.00,
            "Custom Clearance": "-",
            "Cabin Cleaning": 16.00,
            "Galley Service": 16.00,
            "Cabin Security": "-",
            "Passenger Enplaning": "-",
            "FNLZTN / Door CLSD": 3.00,
            "BLOCKS OUT": 1.00,
            "Total Ground Time": 25.00,
        },
        "INTL-INTL": {
            "Blocks IN / Door Opening": "-",
            "PAX deplaning": "-",
            "Custom Clearance": "-",
            "Cabin Cleaning": "-",
            "Galley Service": "-",
            "Cabin Security": "-",
            "Passenger Enplaning": "-",
            "FNLZTN / Door CLSD": "-",
            "BLOCKS OUT": "-",
            "Total Ground Time": "-",
        },
    },
}

# B757 — TURNAROUND (100% LF)
B757_TURNAROUND = {
    "operation": "TURNAROUND",
    "assumptions": ["100% Load Factor"],
    "notes": [],
    "movements": {
        "DOM-DOM": {
            "Dock Jetty/ open Door": 1.00,
            "PAX Disembarkation": 7.00,
            "ACFT Customs checks": "-",
            "ACFT Cleaning": _sv(12.00, 8, "*"),
            "Catering Services": 15.00,
            "Cabin security": 3.00,
            "PAX Embarkation": 15.00,
            "FNLZTN / Door CLSD": 3.00,
            "Pushback": 1.00,
            "Min Ground Time": 45.00,
        },
        "DOM-INTL": {
            "Dock Jetty/ open Door": 1.00,
            "PAX Disembarkation": 7.00,
            "ACFT Customs checks": "-",
            "ACFT Cleaning": _sv(12.00, 8, "*"),
            "Catering Services": 15.00,
            "Cabin security": 3.00,
            "PAX Embarkation": 15.00,
            "FNLZTN / Door CLSD": 3.00,
            "Pushback": 1.00,
            "Min Ground Time": 45.00,
        },
        "INTL-DOM": {
            "Dock Jetty/ open Door": 1.00,
            "PAX Disembarkation": 7.00,
            "ACFT Customs checks": 5.00,
            "ACFT Cleaning": _sv(12.00, 8, "*"),
            "Catering Services": 20.00,
            "Cabin security": 3.00,
            "PAX Embarkation": 15.00,
            "FNLZTN / Door CLSD": 3.00,
            "Pushback": 1.00,
            "Min Ground Time": 55.00,
        },
        "INTL-INTL": {
            "Dock Jetty/ open Door": 1.00,
            "PAX Disembarkation": 7.00,
            "ACFT Customs checks": 5.00,
            "ACFT Cleaning": _sv(12.00, 8, "*"),
            "Catering Services": 25.00,
            "Cabin security": 3.00,
            "PAX Embarkation": 15.00,
            "FNLZTN / Door CLSD": 3.00,
            "Pushback": 1.00,
            "Min Ground Time": 60.00,
        },
    },
}

# B757 — TRANSIT (50% LF)
B757_TRANSIT = {
    "operation": "TRANSIT",
    "assumptions": ["50% Load Factor"],
    "notes": [],
    "movements": {
        "DOM-DOM": {
            "Dock Jetty/ open Door": 1.00,
            "PAX Disembarkation": 7.00,
            "ACFT Customs checks": "-",
            "ACFT Cleaning": _sv(5.00, 5, "*"),
            "Catering Services": 12.00,
            "Cabin security": "-",
            "PAX Embarkation": 15.00,
            "FNLZTN / Door CLSD": 3.00,
            "Pushback": 1.00,
            "Min Ground Time": 39.00,
        },
        "DOM-INTL": {
            "Dock Jetty/ open Door": 1.00,
            "PAX Disembarkation": "-",
            "ACFT Customs checks": "-",
            "ACFT Cleaning": _sv(5.00, 5, "*"),
            "Catering Services": 12.00,
            "Cabin security": "-",
            "PAX Embarkation": 15.00,
            "FNLZTN / Door CLSD": 3.00,
            "Pushback": 1.00,
            "Min Ground Time": 32.00,
        },
        "INTL-DOM": {
            "Dock Jetty/ open Door": 1.00,
            "PAX Disembarkation": 7.00,
            "ACFT Customs checks": "-",
            "ACFT Cleaning": _sv(5.00, 5, "*"),
            "Catering Services": 15.00,
            "Cabin security": "-",
            "PAX Embarkation": "-",
            "FNLZTN / Door CLSD": 3.00,
            "Pushback": 1.00,
            "Min Ground Time": 27.00,
        },
        "INTL-INTL": {
            "Dock Jetty/ open Door": "-",
            "PAX Disembarkation": "-",
            "ACFT Customs checks": "-",
            "ACFT Cleaning": "-",
            "Catering Services": "-",
            "Cabin security": "-",
            "PAX Embarkation": "-",
            "FNLZTN / Door CLSD": "-",
            "Pushback": "-",
            "Min Ground Time": "-",
        },
    },
}

# Combined / Mixed flights (minutes) — 13.16.10
COMBINED_MIXED_FLIGHTS = {
    "operation": "COMBINED_MIXED",
    "assumptions": ["100% Load Factor"],
    "notes": [],
    "movements": {
        "N/A": {
            # Activities table columns:
            # - B777-368/B787-10
            # - B777-268/A330
            # - A321/A320 => N/A in the image
            "B777-368/B787-10": {
                "Dock Jetty/ Open Door": 1.00,
                "PAX Disembarkation": 13.00,
                "ACFT Customs Checks": 7.00,
                "ACFT Cleaning": 30.00,
                "Catering Services": 30.00,
                "Cabin security": 5.00,
                "PAX Embarkation": 25.00,
                "FNLZTN / Door CLSD": 3.00,
                "Pushback": 1.00,
                "Min Ground Time": 85.00,
            },
            "B777-268/A330": {
                "Dock Jetty/ Open Door": 1.00,
                "PAX Disembarkation": 10.00,
                "ACFT Customs Checks": 7.00,
                "ACFT Cleaning": 28.00,
                "Catering Services": 28.00,
                "Cabin security": 5.00,
                "PAX Embarkation": 20.00,
                "FNLZTN / Door CLSD": 3.00,
                "Pushback": 1.00,
                "Min Ground Time": 75.00,
            },
            "A321/A320": "N/A",
        }
    },
}

ACTIVITY_BREAKDOWNS: Dict[str, Dict[str, Any]] = {
    "B777-368/B787-10_TURNAROUND": B777_TURNAROUND,
    "B777-368/B787-10_TRANSIT": B777_TRANSIT,
    "A330/B787-9_TRANSIT": A330_TRANSIT,
    "A321/A320_TURNAROUND": A321_TURNAROUND,
    "A321/A320_TRANSIT": A321_TRANSIT,
    "B757_TURNAROUND": B757_TURNAROUND,
    "B757_TRANSIT": B757_TRANSIT,
    "COMBINED_MIXED": COMBINED_MIXED_FLIGHTS,
}

def lookup_activity_breakdown(
    aircraft_group: str,
    operation: str,
    movement: str,
) -> ActivityBreakdownResult:
    """
    Look up activity breakdown for a given aircraft_group + operation + movement.
    """
    op = operation.strip().upper()
    movement = movement.strip().upper()
    key = f"{aircraft_group}_{op}"

    if key not in ACTIVITY_BREAKDOWNS:
        raise KeyError(f"No activity breakdown found for {aircraft_group} / {op}")

    blob = ACTIVITY_BREAKDOWNS[key]
    if movement not in blob["movements"]:
        raise KeyError(f"No movement '{movement}' for {aircraft_group} / {op}")

    row = blob["movements"][movement]

    # Total/min ground time field name differs by table. Normalize it.
    total_field_candidates = ["Total Ground Time", "Min Ground Time"]
    total_val: Union[float, str] = "-"
    for tf in total_field_candidates:
        if tf in row:
            total_val = row[tf]
            break

    return ActivityBreakdownResult(
        aircraft_group=aircraft_group,
        operation=op,
        movement=movement,
        activities=dict(row),
        total_or_min_ground_time=total_val,
        assumptions=list(blob.get("assumptions", [])),
        notes=list(blob.get("notes", [])),
    )

# ---------------------------------------------------------------------------
# 13.16.13.1 — Aircraft Delivery before S.T.D (hours)
# ---------------------------------------------------------------------------

AIRCRAFT_DELIVERY_BEFORE_STD_HOURS: Dict[str, float] = {
    "B777-368/B787-10": 2.00,
    "A300/B787-9": 2.00,
    "A321/A320": 1.00,
}

def lookup_delivery_before_std_hours(aircraft_group: str) -> float:
    """
    Return delivery before STD (hours), from hangar to parking gate — Aircraft Ex-Scheduled Maintenance.
    """
    key = aircraft_group.strip()
    if key in AIRCRAFT_DELIVERY_BEFORE_STD_HOURS:
        return AIRCRAFT_DELIVERY_BEFORE_STD_HOURS[key]
    # Accept the project-wide naming "A330/B787-9" as equivalent to the GOPM row "A300/B787-9".
    if key == "A330/B787-9":
        return AIRCRAFT_DELIVERY_BEFORE_STD_HOURS["A300/B787-9"]
    raise KeyError(f"No delivery time found for aircraft_group '{aircraft_group}'")
