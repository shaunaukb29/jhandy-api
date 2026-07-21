"""Knowledge retrieval from NHTSA complaints, recalls, and OBD ontology DB.

Three query paths:
1. complaints_fts  – NHTSA field complaints matched by symptom keywords
2. recalls_fts     – NHTSA recall defect summaries
3. obd_components  – OBD code → component name mapping
4. obd_ontology    – OBD code → subsystem category mapping
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "knowledge_base.db"

# NHTSA subsystem labels → internal subsystem keys
_NHTSA_TO_SUBSYSTEM: dict[str, str] = {
    "ENGINE": "engine_internal",
    "POWER TRAIN": "drivetrain",
    "POWER TRAIN:AUTOMATIC TRANSMISSION:TORQUE CONVERTER": "drivetrain",
    "SERVICE BRAKES": "brakes",
    "SUSPENSION": "suspension",
    "STEERING": "steering",
    "WHEELS": "suspension",
    "ELECTRICAL SYSTEM": "charging_starting",
    "FUEL/PROPULSION SYSTEM": "fuel_ignition",
    "FUEL SYSTEM, GASOLINE": "fuel_ignition",
    "ENGINE AND ENGINE COOLING:EXHAUST SYSTEM": "engine_internal",
    "ELECTRONIC STABILITY CONTROL (ESC)": "brakes",
}


def _nhtsa_subsystem(raw: str) -> str:
    raw_up = raw.upper().strip()
    if raw_up in _NHTSA_TO_SUBSYSTEM:
        return _NHTSA_TO_SUBSYSTEM[raw_up]
    # Prefix match
    for key, val in _NHTSA_TO_SUBSYSTEM.items():
        if raw_up.startswith(key):
            return val
    return raw.lower().replace(" ", "_")


def _make_fts_query(description: str) -> str:
    """Build an FTS OR-query from the most informative words in the description."""
    # Strip short stop-words, keep diagnostic terms
    words = re.findall(r'\b[a-z]{4,}\b', description.lower())
    # Prioritise mechanically specific terms
    priority = {"knock", "rattle", "grind", "squeal", "vibrat", "clunk", "bearing",
                 "axle", "wheel", "brake", "engine", "transmission", "steering",
                 "exhaust", "belt", "chain", "timing", "piston", "valve"}
    scored = sorted(words, key=lambda w: (any(p in w for p in priority), len(w)), reverse=True)
    unique = list(dict.fromkeys(scored))[:12]  # deduplicated, top 12
    return " OR ".join(f'"{w}"' for w in unique)


def retrieve_similar(description: str, top_k: int = 15) -> list[dict]:
    """Query NHTSA complaints + recalls by symptom description. Returns Observation-ready hits."""
    if not DB_PATH.exists() or not description:
        return []

    query = _make_fts_query(description)
    if not query:
        return []

    hits: list[dict] = []

    try:
        conn = sqlite3.connect(DB_PATH, timeout=3)
        cursor = conn.cursor()

        # 1. NHTSA complaints: field complaints carry real driver language
        try:
            cursor.execute(
                """SELECT subsystem, description FROM complaints_fts
                   WHERE complaints_fts MATCH ? LIMIT ?""",
                (query, top_k),
            )
            for subsystem_raw, desc_text in cursor.fetchall():
                hits.append({
                    "subsystem": _nhtsa_subsystem(subsystem_raw),
                    "component": "",
                    "weight": 0.35,     # higher weight: real-world complaint evidence
                    "label": f"NHTSA complaint: {desc_text[:120]}",
                    "source_table": "complaints",
                })
        except Exception:
            pass

        # 2. NHTSA recalls
        try:
            cursor.execute(
                """SELECT category, defect_summary FROM recalls_fts
                   WHERE recalls_fts MATCH ? LIMIT ?""",
                (query, top_k),
            )
            for cat, defect in cursor.fetchall():
                hits.append({
                    "subsystem": _nhtsa_subsystem(cat),
                    "component": "",
                    "weight": 0.3,
                    "label": f"NHTSA recall: {defect[:120]}",
                    "source_table": "recalls",
                })
        except Exception:
            pass

    except Exception:
        pass
    finally:
        if "conn" in locals():
            conn.close()

    return hits


def retrieve_obd_components(code: str) -> list[dict]:
    """Return components listed in the DB for a given OBD code."""
    if not DB_PATH.exists() or not code:
        return []
    code = code.strip().upper()
    results = []
    try:
        conn = sqlite3.connect(DB_PATH, timeout=3)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT component_name FROM obd_components WHERE code = ?",
            (code,),
        )
        for (name,) in cursor.fetchall():
            results.append({"component": name, "code": code})
        conn.close()
    except Exception:
        pass
    return results


def retrieve_obd_subsystem(code: str) -> str | None:
    """Return the subsystem category for an OBD code from the DB ontology."""
    if not DB_PATH.exists() or not code:
        return None
    code = code.strip().upper()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=3)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT subsystem FROM obd_ontology WHERE code = ?",
            (code,),
        )
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            return row[0].strip().lower().replace(" ", "_")
    except Exception:
        pass
    return None
