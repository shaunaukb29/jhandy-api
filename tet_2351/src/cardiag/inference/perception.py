import uuid
from cardiag.inference.evidence import Observation


def observe_audio(diag_dict: dict) -> list[Observation]:
    """Extract observations from audio diagnosis output."""
    obs = []
    fault_prob = diag_dict.get("fault_probability", 0.0)
    causes = diag_dict.get("causes", [])

    if fault_prob >= 0.4:
        for cause in causes:
            c_name = cause.get("part", "")
            c_prob = cause.get("p", 0.0)
            weight = fault_prob * c_prob
            obs.append(Observation(
                id=f"audio_{uuid.uuid4().hex[:8]}",
                source="clap_audio_model",
                category="audio",
                component="",
                subsystem=c_name, # Audio causes map roughly to subsystems / high-level groups
                confidence=weight,
                label=f"Audio indicates {c_name} issue (probability: {c_prob:.0%})",
                features={"probability": c_prob, "fault_probability": fault_prob}
            ))
    return obs


def observe_description(sym_features: dict) -> list[Observation]:
    """Extract observations from description parser output."""
    obs = []
    profile = sym_features.get("profile")
    if not profile:
        return obs

    obs.append(Observation(
        id=f"desc_{uuid.uuid4().hex[:8]}",
        source="description_parser",
        category="description",
        component="",
        subsystem="",
        confidence=1.0,
        label=sym_features.get("narrative", ""),
        features={
            "location": profile.location,
            "sound": profile.sound,
            "frequency": profile.frequency,
            "warm_state": profile.warm_state,
            "load": profile.load,
            "cold_only": profile.cold_only,
            "acceleration": profile.acceleration,
            "braking": profile.braking,
            "turning": profile.turning,
            "speed_dependent": profile.speed_dependent
        }
    ))
    return obs


def observe_obd(parsed_codes: list[dict]) -> list[Observation]:
    """Extract observations from OBD codes."""
    obs = []
    for code_info in parsed_codes:
        subsystem = code_info.get("indicates_subsystem", "")
        desc = code_info.get("description", "")
        code = code_info.get("code", "")
        obs.append(Observation(
            id=f"obd_{code.lower()}_{uuid.uuid4().hex[:4]}",
            source="user_obd_input",
            category="obd",
            component="",
            subsystem=subsystem,
            confidence=1.0,
            label=f"OBD code {code} present: {desc}",
            features={"code": code}
        ))
    return obs


def observe_followup(answers: dict) -> list[Observation]:
    """Extract observations from followup answers."""
    # To be refined based on exact implementation of followup integration
    obs = []
    return obs


def observe_retrieval(hits: list[dict]) -> list[Observation]:
    """Extract observations from retrieval RAG."""
    obs = []
    for hit in hits:
        obs.append(Observation(
            id=f"rag_{uuid.uuid4().hex[:8]}",
            source="retrieval_db",
            category="retrieval",
            component=hit.get("component", ""),
            subsystem=hit.get("subsystem", ""),
            confidence=hit.get("weight", 0.5),
            label=hit.get("label", ""),
            features={}
        ))
    return obs
