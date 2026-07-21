"""OBD-II Multimodal Reasoning Layer.

Fuses active DTCs, a free-text symptom description, and audio classifier predictions
to produce a component-level, mechanic-grade diagnosis with ranked hypotheses,
evidence items, recommended tests, and an interactive follow-up question.

Design goals
------------
* **Lean**: Only audio-relevant codes are loaded (~985 of 3,711 powertrain codes).
* **Component-level**: Goes past subsystem labels (e.g. "engine_internal") to specific
  parts (VVT actuator, timing chain tensioner, …) ranked by probability.
* **Evidence-driven**: Every hypothesis shows the specific evidence supporting it and
  the first concrete inspection step.
* **Offline-first**: Template explanations work without any network calls. Ollama
  enriches them when available.
* **Interactive**: A targeted follow-up question is selected from a per-subsystem pool
  using a simple information-gain heuristic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cardiag.inference.evidence import EvidenceGraph, Hypothesis, EvidenceLink, Observation
from cardiag.inference.formatter import DiagnosisFormatter
import uuid

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parents[2] / "data"

# ------------------------------------------------------------------ audio-relevant keyword filter
_AUDIO_KEYWORDS = (
    "misfire", "knock", "ping", "rattle", "tick", "click", "tap",
    "squeal", "squeak", "grind", "rumble", "whine", "hiss", "whistle",
    "camshaft", "crankshaft", "timing", "cylinder deactivat",
    "oil pressure", "oil level", "low oil",
    "belt", "pulley", "tensioner",
    "turbo", "supercharg", "boost",
    "exhaust", "leak", "backfire",
    "wheel speed", "abs", "brake",
    "bearing", "gear", "transmission slip",
    "alternator", "charging", "voltage low",
    "steering", "power steering",
    "injector", "fuel pressure", "fuel pump",
    "coolant temp", "overh",
)

# ------------------------------------------------------------------ description → cause-class map
_DESC_TO_CAUSE: list[tuple[tuple[str, ...], str]] = [
    (("misfire", "ignit", "spark", "cylinder", "injector",
      "fuel pressure", "fuel pump"),                          "fuel_ignition"),
    (("oil pressure", "oil level", "low oil"),               "low_oil"),
    (("camshaft", "crankshaft", "timing", "knock", "ping",
      "rattle", "tick", "tap", "bearing internal"),           "engine_internal"),
    (("belt", "pulley", "tensioner", "serpentine"),          "belt"),
    (("alternator", "charging", "voltage"),                  "accessories"),
    (("turbo", "supercharg", "boost"),                       "accessories"),
    (("steering", "power steering"),                         "power_steering"),
    (("exhaust", "backfire", "leak"),                        "exhaust"),
    (("wheel speed", "abs", "brake", "squeal", "grind"),    "brakes"),
    (("transmission slip", "gear"),                          "transmission"),
    (("coolant", "overh", "water pump", "radiator", "hose",
      "thermostat", "head gasket"),                           "cooling"),
    (("cv joint", "drive shaft", "propshaft", "u-joint",
      "differential", "transfer case"),                       "drivetrain"),
    (("strut", "shock", "bushing", "control arm",
      "ball joint", "sway bar"),                              "suspension"),
]

_COHERENCE_BOOST = 0.35

_LOC_TO_SUBSYSTEMS = {
    "engine_bay": {"engine_internal", "fuel_ignition", "belt", "low_oil", "accessories", "cooling"},
    "accessories": {"belt", "accessories", "cooling", "power_steering"},
    "exhaust": {"exhaust"},
    "wheel_area": {"brakes", "suspension", "drivetrain"},
    "transmission": {"transmission", "drivetrain"},
    "steering": {"power_steering", "suspension"},
    "brakes": {"brakes"},
}

# ------------------------------------------------------------------ human-readable OBD interpretation
# For common codes, override the raw database description with a mechanic's explanation.
_CODE_EXPLANATIONS: dict[str, str] = {
    "P0010": "P0010 means the ECU cannot control Bank 1 intake cam phasing — actuator or oil flow problem in the VVT system.",
    "P0011": "P0011 means the Bank 1 intake camshaft is more advanced than commanded — points to a VVT actuator, cam phasing solenoid, or oil flow fault.",
    "P0012": "P0012 means the Bank 1 intake cam is over-retarded — the VVT system cannot hold or advance timing as commanded.",
    "P0013": "P0013 means the Bank 1 exhaust cam actuator circuit has a fault — likely a solenoid, wiring, or oil pressure issue.",
    "P0014": "P0014 means the Bank 1 exhaust cam is over-advanced — VVT actuator or solenoid is stuck or restricted.",
    "P0016": "P0016 detects a cam-to-crank timing mismatch at Bank 1 — often caused by a stretched timing chain or jumped timing.",
    "P0017": "P0017 is a Bank 1 exhaust cam-to-crank correlation error — similar to P0016 but on the exhaust cam.",
    "P0087": "P0087 means rail fuel pressure is below the target — the fuel pump may be weak, the filter restricted, or the regulator leaking.",
    "P0300": "P0300 is a random/multiple cylinder misfire — the engine is misfiring across more than one cylinder, suggesting a fuel delivery, ignition, or compression fault.",
    "P0301": "P0301 is a cylinder 1 misfire — a spark plug, coil, injector, or compression fault on that specific cylinder.",
    "P0302": "P0302 is a cylinder 2 misfire.",
    "P0303": "P0303 is a cylinder 3 misfire.",
    "P0304": "P0304 is a cylinder 4 misfire.",
    "P0420": "P0420 means the Bank 1 catalytic converter efficiency is below the threshold — converter may be worn, poisoned, or there is an exhaust leak upstream.",
    "P0430": "P0430 means Bank 2 catalytic converter efficiency is low.",
    "P0520": "P0520 is an oil pressure sensor circuit fault — verify actual oil pressure with a mechanical gauge before assuming a sensor problem.",
    "P0521": "P0521 means the oil pressure sensor reading is out of the expected range — confirm actual pressure with a mechanical gauge.",
    "P0700": "P0700 is a general transmission control fault — read TCM-specific codes for the root cause.",
    "P0740": "P0740 indicates the torque converter clutch solenoid circuit is faulty — shudder at 40–55 mph under light throttle is a common symptom.",
    "P0741": "P0741 means the torque converter clutch is slipping — converter or clutch pack wear.",
    "C0035": "C0035 is a front-left wheel speed sensor fault — can cause ABS activation, traction control faults, or a grinding-like noise on hard braking.",
    "C0040": "C0040 is a front-right wheel speed sensor fault.",
    "C0045": "C0045 is a rear-left wheel speed sensor fault.",
    "C0050": "C0050 is a rear-right wheel speed sensor fault.",
}


def _classify(desc: str) -> str | None:
    """Map a DTC description or user text to a cardiag cause class.

    Scores every cause class by how many of its keywords appear (not just the
    first one that matches), so a description with one incidental engine word
    and three brake words still resolves to brakes. List order in
    _DESC_TO_CAUSE must not affect the result — only the strength of match.
    """
    d = desc.lower()
    best_cause, best_score = None, 0
    for keywords, cause in _DESC_TO_CAUSE:
        score = sum(1 for kw in keywords if kw in d)
        if score > best_score:
            best_cause, best_score = cause, score
    return best_cause


class ReasoningEngine:
    """Multimodal reasoning: fuses audio predictions, OBD codes, and description."""

    def __init__(self, db_path: str | Path | None = None):
        self._log = logging.getLogger("cardiag.reasoning")

    # ---------------------------------------------------------------- public API

    def lookup(self, code: str) -> tuple[str, str | None]:
        """Return (description, cause_class) for a DTC code."""
        from cardiag.inference.knowledge import lookupDTC
        info = lookupDTC(code)
        if info:
            desc = info["description"]
            cause = _classify(desc)
            return desc, cause
        return f"OBD-II code {code} — not in database", None

    def reason(self, diag_dict: dict, active_codes: list[str],
               description: str = "",
               backend: str = "offline",
               asked_question_ids: list[str] | None = None) -> dict:
        """Fuse audio predictions, OBD codes, and description into a rich diagnosis.

        Returns a payload the frontend renders directly. All three evidence sources
        are optional; whatever is present is used. The existing ``causes`` list in
        *diag_dict* is mutated in-place with coherence-boosted probabilities.
        """
        description = (description or "").strip()

        if not active_codes and not description and not diag_dict.get("verdict"):
            return {"parsed_codes": [], "coherence_matches": [],
                    "explanation": "", "components": [], "assessment": ""}

        from cardiag.inference.description_parser import interpret as _interp
        from cardiag.inference.perception import (
            observe_audio,
            observe_description,
            observe_obd,
            observe_retrieval,
        )
        from cardiag.inference.retrieval import retrieve_similar

        observations = []
        observations.extend(observe_audio(diag_dict))
        sym_features: dict = {}
        if description:
            sym_features = _interp(description)
            observations.extend(observe_description(sym_features))

        parsed: list[dict] = []
        for raw in active_codes:
            code = raw.strip().upper()
            if not code: continue
            desc, cat = self.lookup(code)
            parsed.append({"code": code, "description": desc, "category": cat, "indicates_subsystem": cat})
        observations.extend(observe_obd(parsed))

        if description:
            hits = retrieve_similar(description)
            observations.extend(observe_retrieval(hits))

        # Perform CLAP audio similarity search
        audio_path = diag_dict.get("file")
        audio_hits = []
        import os
        import numpy as np
        if audio_path and os.path.exists(audio_path):
            try:
                from cardiag.audio.embed import model_vectors
                from cardiag.audio.vector_db import find_similar_audio
                ev = model_vectors(audio_path)
                if ev.vectors.shape[0] > 0:
                    mean_vec = np.mean(ev.vectors, axis=0)
                    mean_vec = mean_vec / (np.linalg.norm(mean_vec) or 1.0)
                    audio_hits = find_similar_audio(mean_vec, top_k=3)
                    
                    for hit in audio_hits:
                        raw_cause = hit.get("cause") or hit.get("l1") or hit.get("kind") or "Reference Case"
                        cause_name = raw_cause.replace("_", " ").title()
                        observations.append(Observation(
                            id=f"sim_{uuid.uuid4().hex[:8]}",
                            source="audio_similarity",
                            category="audio_similarity",
                            component=hit.get("cause", ""),
                            subsystem="",
                            confidence=hit.get("similarity", 0.5),
                            label=f"Acoustically matches reference case of {cause_name} ({hit['similarity']:.0%} similarity).",
                            features={"clip_id": hit["clip_id"], "video": hit["video"]}
                        ))
            except Exception as e:
                logging.warning(f"Failed to run audio similarity search: {e}")

        active_subsystems = set()
        primary_subsystem = None
        for obs in observations:
            if obs.subsystem:
                active_subsystems.add(obs.subsystem)
                if not primary_subsystem and obs.category in ("audio", "description"):
                    primary_subsystem = obs.subsystem

        desc_cat = _classify(description) if description else None
        if not primary_subsystem and desc_cat:
            primary_subsystem = desc_cat
            active_subsystems.add(desc_cat)

        if not primary_subsystem and parsed and parsed[0].get("category"):
            primary_subsystem = parsed[0]["category"]
            active_subsystems.add(primary_subsystem)

        if sym_features.get("location"):
            for loc in sym_features["location"]:
                if loc in _LOC_TO_SUBSYSTEMS:
                    implied = _LOC_TO_SUBSYSTEMS[loc]
                    active_subsystems.update(implied)

        from cardiag.inference.knowledge_graph import get_all_candidate_components
        candidates = get_all_candidate_components(description, active_codes)

        # The candidate list above is generated purely from text + DTCs, so an
        # audio-only finding (no matching description keywords or active code)
        # would otherwise never get a candidate to attach to and gets thrown
        # away before scoring even starts, regardless of how confident the
        # audio model is. Backfill any audio cause missing from `candidates`
        # so it can compete on equal footing in _fuse_evidence instead of
        # being silently starved out.
        candidates = self._merge_audio_candidates(candidates, diag_dict)

        components = self._fuse_evidence(candidates, observations, primary_subsystem, description, sym_features)
        
        # Overwrite primary_subsystem with the actual winning component's subsystem 
        # so follow-up questions and assessment texts are properly aligned!
        if components:
            primary_subsystem = components[0].subsystem or primary_subsystem

        confidence = self._calibrate_confidence(components, observations, diag_dict)

        matches = [obs.subsystem for obs in observations if obs.category == "audio" and any(o.subsystem == obs.subsystem for o in observations if o.category != "audio")]

        # 7. OBD interpretations
        obd_interps = [
            {"code": c["code"],
             "interpretation": self._interpret_obd_code(c["code"], c["description"])}
            for c in parsed
        ]

        # 8. Assessment paragraph
        assessment = self._build_assessment(
            diag_dict, primary_subsystem, components, parsed,
            sym_features.get("narrative", ""),
        )

        # 9. Follow-up question (Fix F)
        followup = self._select_followup(
            primary_subsystem, components, asked_ids=asked_question_ids or [], sym_features=sym_features
        )

        # 10. Backward-compat one-liner explanation
        explanation = self._explain(diag_dict, parsed, matches, backend,
                                    description=description)

        graph = EvidenceGraph(
            raw_predictions={"fault_probability": diag_dict.get("fault_probability")},
            observations=observations,
            hypotheses=components
        )
        
        formatter = DiagnosisFormatter()
        return formatter.format(
            graph=graph,
            diag_dict=diag_dict,
            parsed_codes=parsed,
            matches=matches,
            sym_features=sym_features,
            primary_subsystem=primary_subsystem,
            followup=followup,
            obd_interps=obd_interps,
            confidence=confidence,
            assessment=assessment,
            explanation=explanation,
            similar_cases=audio_hits
        )

    # ---------------------------------------------------------------- candidate backfill

    def _merge_audio_candidates(self, candidates: list, diag_dict: dict) -> list:
        """Add any audio-flagged cause missing from the text/DTC candidate list.

        `get_all_candidate_components` only looks at the description and active
        codes, so a component the audio model is confident about — but that the
        driver didn't mention and has no matching DTC — would never be eligible
        for scoring in `_fuse_evidence`. That silently discards accurate audio
        evidence rather than letting it compete. We fix that here instead of
        biasing the scoring weights themselves.
        """
        import types

        existing = {c.name.lower() for c in candidates}
        merged = list(candidates)
        for cause in diag_dict.get("causes", []):
            part = cause.get("part")
            if not part or part.lower() in existing:
                continue
            merged.append(types.SimpleNamespace(
                name=part,
                subsystem=cause.get("subsystem", "") or "",
                prior=max(0.001, min(0.999, float(cause.get("p", cause.get("probability", 0.05))))),
                boosting_codes=cause.get("boosting_codes", ()) or (),
                boosting_symptoms=cause.get("boosting_symptoms", ()) or (),
                tests=cause.get("tests", []) or [],
                severity=cause.get("severity", "moderate"),
                driveability=cause.get("driveability", "monitor"),
                acoustic=cause.get("acoustic", True),
            ))
            existing.add(part.lower())
        return merged

    # ---------------------------------------------------------------- component scoring

    def _fuse_evidence(self, candidates: list, observations: list[Observation], primary_subsystem: str | None, raw_desc: str = "", sym_features: dict | None = None) -> list[Hypothesis]:
        import math
        scored = []
        desc_lower = raw_desc.lower()

        for comp in candidates:
            # log-odds
            prior = max(0.001, min(0.999, comp.prior))
            log_odds = math.log(prior / (1 - prior))
            links = []

            for obs in observations:
                # Basic matching - this should ideally use hierarchy
                matches_comp = obs.matches(comp.name, comp.subsystem)
                # Or if the observation has a code in boosting_codes
                if obs.category == "obd" and obs.features.get("code", "").upper() in comp.boosting_codes:
                    matches_comp = True

                # Description text matching: search raw description, not obs label
                if obs.category == "description":
                    kw_hits = sum(1 for kw in comp.boosting_symptoms if kw in desc_lower)
                    if kw_hits > 0:
                        # Cap the text boost so it acts as corroboration, not an override
                        contribution = min(1.0, obs.confidence * (0.3 + 0.2 * kw_hits))
                        log_odds += contribution
                        links.append(EvidenceLink(
                            observation_id=obs.id,
                            relationship="supports",
                            weight=contribution,
                            explanation=obs.label
                        ))
                        continue

            # Group retrieval hits to prevent 20 identical NHTSA complaints from pushing probability to 1.0 automatically.
            retrieval_hits = 0
            retrieval_confidence_sum = 0.0
            
            # Did this component match ANY audio observation?
            audio_matched = any(obs.category == "audio" and obs.matches(comp.name, comp.subsystem) for obs in observations)

            for obs in observations:
                if obs.matches(comp.name, comp.subsystem):
                    # Boost
                    if obs.category == "audio":
                        if not getattr(comp, "acoustic", True):
                            continue # non-acoustic parts cannot be diagnosed by audio
                        boost = obs.confidence * 2.0
                        log_odds += boost
                        links.append(EvidenceLink(observation_id=obs.id, relationship="supports", weight=boost, explanation=obs.label))
                    elif obs.category == "retrieval":
                        retrieval_hits += 1
                        retrieval_confidence_sum += obs.confidence
                    elif obs.category == "obd":
                        if obs.features.get("code", "").upper() in comp.boosting_codes:
                            boost = obs.confidence * 1.5
                            log_odds += boost
                            links.append(EvidenceLink(observation_id=obs.id, relationship="supports", weight=boost, explanation=obs.label))
                        elif obs.subsystem and obs.subsystem.lower() in comp.subsystem.lower():
                            boost = obs.confidence * 0.5
                            log_odds += boost
                            links.append(EvidenceLink(observation_id=obs.id, relationship="supports", weight=boost, explanation=obs.label))
                    elif obs.category == "audio_similarity":
                        # Stronger boost for high similarity matching failure recording
                        boost = obs.confidence * 2.0
                        log_odds += boost
                        links.append(EvidenceLink(observation_id=obs.id, relationship="supports", weight=boost, explanation=obs.label))
                    elif obs.category not in ("description", "audio", "retrieval", "obd", "audio_similarity"):
                        boost = obs.confidence
                        log_odds += boost
                        links.append(EvidenceLink(observation_id=obs.id, relationship="supports", weight=boost, explanation=obs.label))
                    
                elif obs.category == "audio" and not audio_matched and obs.subsystem and obs.subsystem != comp.subsystem:
                    # Audio evidence that contradicts this component's subsystem penalizes it,
                    # but only if this component didn't match ANY of the audio predictions.
                    penalty = -(obs.confidence * 0.5)
                    log_odds += penalty
                    links.append(EvidenceLink(observation_id=obs.id, relationship="refutes", weight=penalty, explanation="Contradicts primary audio signature."))
                    
            if retrieval_hits > 0:
                # Add a single capped boost for all retrieval hits combined
                boost = min(0.75, retrieval_confidence_sum / math.sqrt(retrieval_hits))
                log_odds += boost
                links.append(EvidenceLink(observation_id="retrieval_agg", relationship="supports", weight=boost, explanation=f"Corroborated by {retrieval_hits} similar NHTSA owner complaints or recalls."))

            # ---- Mechanical Constraints & Contradiction Penalties -----------------
            text_locations = sym_features.get("location", []) if sym_features else []
            if text_locations and comp.subsystem:
                valid_subsystems = set()
                for loc in text_locations:
                    valid_subsystems.update(_LOC_TO_SUBSYSTEMS.get(loc, set()))
                
                # If this component's subsystem is physically impossible given the stated location
                if comp.subsystem not in valid_subsystems:
                    audio_conf = sum(obs.confidence for obs in observations if obs.category == "audio" and obs.matches(comp.name, comp.subsystem))
                    text_conf = sum(obs.confidence for obs in observations if obs.category == "description") or 1.0
                    
                    # Base mechanical constraint penalty
                    penalty = -1.5
                    
                    # Confidence-weighted contradiction penalty (Audio says X, Text says Y)
                    if audio_conf > 0:
                        penalty -= (audio_conf * text_conf * 2.0)
                        
                    log_odds += penalty
                    links.append(EvidenceLink(observation_id="mech_constraint", relationship="refutes", weight=penalty, explanation="Location mechanically contradicts user description."))

            # ---- Structured feature gates -----------------------------------------
            # Use the parsed SymptomProfile to strongly adjust scores based on
            # temporal pattern, not just keyword co-occurrence.
            import math as _math
            timing = set(sym_features.get("timing", [])) if sym_features else set()
            rpm_mods = set(sym_features.get("rpm_modifiers", [])) if sym_features else set()
            comp_lower = comp.name.lower()

            # Components that are cold-start-only (resolve when warm)
            _cold_only = ("vvt", "variable valve timing", "timing chain tensioner",
                          "cam phaser", "hydraulic valve lifter", "piston slap")
            # Components that persist warm and worsen under load
            _persistent_knock = ("connecting rod bearing", "main bearing", "wrist pin")

            if "persists_warm" in timing:
                # Noise that stays when warm → penalise cold-start components heavily
                if any(k in comp_lower for k in _cold_only):
                    log_odds -= 1.8
                # Boost persistent-knock components
                if any(k in comp_lower for k in _persistent_knock):
                    log_odds += 1.2

            if "cold_start" in timing and "persists_warm" not in timing:
                # Noise that is cold-start only → boost cold-only components
                if any(k in comp_lower for k in _cold_only):
                    log_odds += 0.9

            if "acceleration" in timing or "worse_higher_rpm" in rpm_mods:
                # Worsens under load/RPM → strongly boost bottom-end knock components
                if any(k in comp_lower for k in _persistent_knock):
                    log_odds += 1.4
                # Cold-start-only components don't worsen with RPM load
                if any(k in comp_lower for k in _cold_only):
                    log_odds -= 1.0

            if "warm_up" in timing and "persists_warm" not in timing:
                # Resolves as it warms → boost cold-only; penalise persistent components
                if any(k in comp_lower for k in _cold_only):
                    log_odds += 1.0
                if any(k in comp_lower for k in _persistent_knock):
                    log_odds -= 1.5
            # -----------------------------------------------------------------------

            probability = 1 / (1 + _math.exp(-log_odds))

            if probability >= 0.05:
                scored.append((log_odds, Hypothesis(
                    id=f"hypo_{uuid.uuid4().hex[:8]}",
                    name=comp.name,
                    subsystem=comp.subsystem,
                    probability=round(probability, 3),
                    evidence_links=links,
                    recommended_tests=comp.tests,
                    severity=getattr(comp, "severity", "moderate"),
                    driveability=getattr(comp, "driveability", "monitor")
                )))

        # Sort by raw log odds to perfectly break ties at the extremes
        scored.sort(key=lambda x: -x[0])
        
        return [h[1] for h in scored[:5]]

    def _calibrate_confidence(self, components: list[Hypothesis], observations: list[Any], diag_dict: dict) -> float:
        fault_p = float(diag_dict.get("fault_probability", 0.0))
        audio_quality = fault_p

        # Source agreement
        sources_present = set(obs.source for obs in observations)
        subsystems_supported = set(obs.subsystem for obs in observations if obs.subsystem)
        agreement = 1.0 if len(subsystems_supported) == 1 and len(sources_present) > 1 else 0.5

        separation = 0.5
        if len(components) > 1:
            separation = components[0].probability - components[1].probability
        elif len(components) == 1:
            separation = 1.0

        conf = (0.4 * audio_quality) + (0.35 * agreement) + (0.25 * min(separation * 3, 1.0))
        return round(min(1.0, max(0.0, conf)), 2)

    # ---------------------------------------------------------------- OBD interpretation

    def _interpret_obd_code(self, code: str, raw_desc: str) -> str:
        code = code.upper()
        if code in _CODE_EXPLANATIONS:
            return _CODE_EXPLANATIONS[code]
        # Generic interpretation from raw description
        if raw_desc and "not in database" not in raw_desc:
            prefix = raw_desc.split("(")[0].strip().rstrip("-—").strip()
            return (
                f"{code} indicates: {prefix}. "
                f"Check the relevant sensor, actuator, or wiring for this system."
            )
        return f"{code} is an active fault — consult a workshop manual for this code."

    # ---------------------------------------------------------------- assessment paragraph

    def _build_assessment(self, diag: dict, subsystem: str | None,
                           components: list[Hypothesis], parsed_codes: list[dict],
                           desc_narrative: str) -> str:
        parts: list[str] = []
        fault_p = float(diag.get("fault_probability", 0))
        sub_label = (subsystem or "").replace("_", " ")

        top_probs = [c.probability for c in components[:2]]
        
        # Check for uncertainty mode
        if len(top_probs) >= 2:
            gap = top_probs[0] - top_probs[1]
            # If the leading candidate is not extremely strong (< 65%) and the gap is small (< 15%)
            if top_probs[0] < 0.65 and gap < 0.15:
                top_names = [c.name for c in components[:3]]
                c_list = ", ".join(top_names)
                return (f"The diagnostic evidence is currently weak or conflicting. "
                        f"I cannot confidently distinguish between: {c_list}. "
                        "The audio signature does not strongly match a single known failure mode, "
                        "and text/OBD evidence does not decisively isolate a specific component. "
                        "Try physical tests such as: removing the serpentine belt briefly, checking pulleys for play, "
                        "using a mechanic's stethoscope, or scanning for pending OBD codes to isolate the fault.")


        # Audio finding
        if fault_p >= 0.5:
            parts.append(
                f"The audio analysis identifies a fault in the {sub_label} system "
                f"with {fault_p:.0%} confidence."
            )
        elif diag.get("model_loaded") is False:
            parts.append("No audio model is loaded — assessment is based on OBD codes and symptom description only.")
        else:
            parts.append(f"Audio analysis suggests a possible {sub_label} fault.")

        # OBD evidence - conditional on matching the primary audio/descriptor subsystem
        if parsed_codes:
            corroborating = [c for c in parsed_codes if c["category"] == subsystem]
            conflicts = [c for c in parsed_codes if c["category"] != subsystem]

            if corroborating:
                codes_str = ", ".join(c["code"] for c in corroborating)
                if len(corroborating) == 1:
                    parts.append(f"Active code {codes_str} corroborates this finding.")
                else:
                    parts.append(f"Active codes ({codes_str}) corroborate this finding.")

            for c in conflicts:
                code_sub = (c["category"] or "unknown").replace("_", " ")
                parts.append(
                    f"Note: Active code {c['code']} indicates a potential {code_sub} issue, "
                    f"which differs from the primary audio subsystem assessment."
                )

        # Description narrative (as an interpreted addendum, not the raw text)
        if desc_narrative:
            parts.append(desc_narrative)

        # Conclusion
        if components:
            top = components[0]
            second = components[1] if len(components) > 1 else None
            gap = top.probability - (second.probability if second else 0)
            if fault_p > 0 and fault_p < 0.60:
                parts.append("The audio signal is weak or ambiguous. Possible fault, but cannot distinguish specific component with high certainty.")
                # We still output the categories in the UI, but the assessment text warns about false precision.
            elif top.probability > 0.45 and gap > 0.12:
                parts.append(f"The leading suspect is {top.name}.")
            elif second:
                parts.append(
                    f"The strongest candidates are {top.name} and {second.name}."
                )

        return " ".join(parts)

    # ---------------------------------------------------------------- follow-up selection

    def _select_followup(self, subsystem: str | None, components: list[Hypothesis],
                          asked_ids: list[str], sym_features: dict | None = None) -> dict | None:
        if not subsystem or not components:
            return None

        from cardiag.inference.components import FOLLOWUP_QUESTIONS

        questions = FOLLOWUP_QUESTIONS.get(subsystem, [])
        if not questions:
            return None

        top_names = {c.name for c in components[:3]}
        top_probs = [c.probability for c in components[:2]]

        # Don't ask if already confident: top > 60% and gap > 25 pp
        if (len(top_probs) >= 2 and
                top_probs[0] > 0.60 and (top_probs[0] - top_probs[1]) > 0.25):
            return None

        # Redundant question mapping based on extracted features (Fix F)
        redundant_map = {
            "ei_cold_only": {"cold_start", "warm_up", "persists_warm"},
            "ei_rpm_dependent": {"disappears_higher_rpm", "worse_higher_rpm"},
            "ex_cold_louder": {"cold_start", "warm_up", "persists_warm"},
            "brk_only_braking": {"braking"},
            "ps_turning_only": {"turning"},
            "acc_tracks_rpm": {"worse_higher_rpm", "disappears_higher_rpm"},
            "dt_turning_click": {"turning"},
            "sp_bumps_clunk": {"turning"},
        }

        # Check features
        features_extracted = set()
        if sym_features:
            features_extracted.update(sym_features.get("timing", []))
            features_extracted.update(sym_features.get("rpm_modifiers", []))

        best_q = None
        best_score = -1.0

        for q in questions:
            if q.id in asked_ids:
                continue
            # If the features required to answer this question are already present in the description, skip it!
            if q.id in redundant_map and (redundant_map[q.id] & features_extracted):
                continue
            
            # Information gain proxy: variance in multipliers among the top 3 components
            score = 0.0
            if len(components) >= 2:
                c1 = components[0].name
                c2 = components[1].name
                yes_diff = abs(q.yes_multipliers.get(c1, 1.0) - q.yes_multipliers.get(c2, 1.0))
                no_diff = abs(q.no_multipliers.get(c1, 1.0) - q.no_multipliers.get(c2, 1.0))
                score = yes_diff + no_diff
                
                if len(components) >= 3:
                    c3 = components[2].name
                    score += abs(q.yes_multipliers.get(c1, 1.0) - q.yes_multipliers.get(c3, 1.0))
                    score += abs(q.no_multipliers.get(c1, 1.0) - q.no_multipliers.get(c3, 1.0))
            else:
                score = 1.0 if (q.target_components and components[0].name in q.target_components) else 0.5
                
            if score > best_score:
                best_score = score
                best_q = q

        # Only return a question if it actually provides some separation (score > 0)
        return best_q if best_score > 0 else None

    # ---------------------------------------------------------------- explanation (compat)

    def _explain(self, diag: dict, codes: list[dict], matches: list[str],
                 backend: str = "huggingface", description: str = "") -> str:
        if not codes and not description:
            return ""
        if backend == "ollama" or backend == "huggingface":
            result = self._hf_explain(diag, codes, matches, description=description)
            if result:
                return result
        return self._template(diag, codes, matches, description=description)

    def _hf_explain(self, diag: dict, codes: list[dict], matches: list[str],
                        description: str = "") -> str:
        from cardiag.pipeline.llm import _hf_one
        code_str = "; ".join(f"{c['code']} ({c['description']})" for c in codes)
        verdict = diag.get("verdict", "uncertain")
        top_causes = ", ".join(c["part"] for c in diag.get("causes", [])[:2])
        prompt = (
            "You are an expert mechanic. A user submitted an audio clip of their car "
            f"sounding like: '{description}'. "
            f"The vehicle threw OBD-II codes: {code_str}. "
            f"Our acoustic model returned a verdict of '{verdict}' and suggested "
            f"{top_causes} as the top components. "
            "In 2 sentences, explain how the audio and OBD codes connect to point to "
            "this failure. Be confident, direct, and do not use lists."
        )
        if matches:
            match_str = ", ".join(matches)
            prompt += f" Subsystems that agree: {match_str}."
        try:
            return _hf_one(prompt).strip()
        except Exception:
            return ""

    @staticmethod
    def _template(diag: dict, codes: list[dict], matches: list[str],
                  description: str = "") -> str:
        top_causes = ", ".join(
            c["part"].replace("_", " ") for c in diag.get("causes", [])[:2]
        )
        parts: list[str] = []
        if codes:
            top_code = codes[0]
            if matches:
                matched_str = ", ".join(m.replace("_", " ") for m in matches)
                parts.append(
                    f"The audio model and OBD-II code {top_code['code']} "
                    f"both point at {matched_str} — inspect there first."
                )
            else:
                parts.append(
                    f"Audio analysis suggests a fault in {top_causes or 'an unknown area'}, "
                    f"while {top_code['code']} points to a different subsystem. "
                    f"Check both independently."
                )
        else:
            parts.append(
                f"Audio analysis suggests a fault in {top_causes or 'an unknown area'}."
            )
        return " ".join(parts)


# ---------------------------------------------------------------- follow-up update (used by /api/followup)

def apply_followup_answer(components: list, question_id: str,
                           answer: str, subsystem: str,
                           asked_ids: list[str]) -> dict:
    """Apply a yes/no follow-up answer and return updated components + next question.

    Called by the web layer; does not touch audio models or the database.
    Components may arrive as dicts (from the UI) or Hypothesis objects.
    """
    import copy
    from cardiag.inference.components import get_followup_question

    # Normalize: if components are dicts (from the UI), wrap them into Hypothesis objects
    normalized = []
    for comp in components:
        if isinstance(comp, dict):
            normalized.append(Hypothesis(
                id=comp.get("id", f"hypo_{uuid.uuid4().hex[:8]}"),
                name=comp.get("name", ""),
                subsystem=comp.get("subsystem", ""),
                probability=float(comp.get("probability", 0)),
                evidence_links=[],
                recommended_tests=comp.get("tests", []),
                severity=comp.get("severity", "moderate"),
                driveability=comp.get("driveability", "monitor"),
            ))
        else:
            normalized.append(comp)

    q = get_followup_question(question_id)
    if q is None or answer == "skip":
        return {"components": [c.to_legacy_dict() for c in normalized],
                "followup_question": None,
                "followup_id": None, "followup_options": None}

    multipliers = q.yes_multipliers if answer == "yes" else q.no_multipliers

    updated = []
    for comp in normalized:
        m = multipliers.get(comp.name, 1.0)
        new_comp = copy.deepcopy(comp)
        new_comp.probability = comp.probability * m
        updated.append(new_comp)

    total = sum(c.probability for c in updated) or 1.0
    for c in updated:
        c.probability = round(c.probability / total, 3)
    updated.sort(key=lambda x: -x.probability)

    # Select next question
    new_asked = asked_ids + [question_id]
    engine = ReasoningEngine()
    next_q = engine._select_followup(subsystem, updated, asked_ids=new_asked)

    result: dict = {"components": [c.to_legacy_dict() for c in updated], "asked_ids": new_asked}
    if next_q:
        result["followup_question"] = next_q.text
        result["followup_id"]       = next_q.id
        result["followup_options"]  = next_q.options
        result["followup_yes_multipliers"] = next_q.yes_multipliers
        result["followup_no_multipliers"]  = next_q.no_multipliers
    else:
        result["followup_question"] = None

    return result


def _ollama_up() -> bool:
    import socket
    try:
        with socket.create_connection(("localhost", 11434), timeout=0.5):
            return True
    except OSError:
        return False
