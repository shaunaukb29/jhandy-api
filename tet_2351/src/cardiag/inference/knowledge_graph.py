import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "knowledge_base.db"

@dataclass
class ComponentDef:
    name: str
    prior: float
    boosting_codes: list[str]
    boosting_symptoms: list[str]
    tests: list[str]
    subsystem: str = ""
    parent_group: str = ""
    severity: str = "moderate"
    driveability: str = "monitor"
    acoustic: bool = True
    typical_sounds: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    related_components: list[str] = field(default_factory=list)


def get_all_candidate_components(description: str, active_codes: list[str] = None) -> list[ComponentDef]:
    """Retrieve all static components + dynamically load OBD components by keyword or active code."""
    import copy

    from cardiag.inference.components import SUBSYSTEM_COMPONENTS

    components = {}

    # 1. Load ALL static components (no subsystem gating).
    # IMPORTANT: deep-copy each ComponentDef. SUBSYSTEM_COMPONENTS is a
    # module-level dict shared across every request; without copying, the
    # prior += 0.1 recall boost below (and the boosting_codes.append in step 3)
    # would mutate the *same shared object* forever, so priors accumulate
    # across unrelated cars/users/sessions instead of resetting per request.
    for sub, cdefs in SUBSYSTEM_COMPONENTS.items():
        for cdef in cdefs:
            local = copy.deepcopy(cdef)
            local.subsystem = sub
            components[local.name] = local

    if not DB_PATH.exists():
        return list(components.values())

    try:
        conn = sqlite3.connect(DB_PATH, timeout=3)
        cursor = conn.cursor()

        # 2. Boost priors based on NHTSA recalls matching the description
        if description:
            keywords = [w for w in description.replace(',', ' ').replace('.', ' ').split() if len(w) > 3]
            if keywords:
                query = " OR ".join(f'"{kw}"' for kw in keywords)
                try:
                    cursor.execute('''
                        SELECT category, defect_summary FROM recalls_fts 
                        WHERE recalls_fts MATCH ? LIMIT 10
                    ''', (query,))
                    
                    for cat, defect in cursor.fetchall():
                        for name, cdef in components.items():
                            if name.lower() in defect.lower():
                                cdef.prior += 0.1
                except Exception:
                    pass

        # 3. Load dynamic components for any active OBD codes
        if active_codes:
            for code in active_codes:
                try:
                    cursor.execute('''
                        SELECT c.component_name, o.subsystem 
                        FROM obd_components c
                        JOIN obd_ontology o ON o.code = c.code
                        WHERE c.code = ?
                    ''', (code.upper().strip(),))
                    
                    for name, subsystem in cursor.fetchall():
                        if name not in components:
                            components[name] = ComponentDef(
                                name=name,
                                prior=0.1,
                                boosting_codes=[code.upper().strip()],
                                boosting_symptoms=[],
                                tests=["Check component using OBD scan tool.", "Visual inspection."],
                                subsystem=subsystem.lower().replace(" ", "_") if subsystem else "",
                                acoustic=False
                            )
                        else:
                            if code.upper().strip() not in components[name].boosting_codes:
                                components[name].boosting_codes.append(code.upper().strip())
                except Exception:
                    pass

    except Exception:
        pass
    finally:
        if "conn" in locals():
            conn.close()

    return list(components.values())