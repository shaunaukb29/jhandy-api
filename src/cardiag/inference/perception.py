from dataclasses import dataclass


@dataclass
class Observation:
    """A single piece of extracted evidence."""
    source: str          # "audio", "description", "obd", "followup", "retrieval"
    component: str       # target component name (or "" for subsystem-level)
    subsystem: str       # target subsystem (or "" if general)
    weight: float        # contribution strength (0.0–1.0)
    label: str           # human-readable evidence sentence
    features: dict       # structured features (character, timing, location, etc.)

    def matches(self, comp_name: str, comp_subsystem: str) -> bool:
        """Check if this observation applies to the given component or subsystem.

        Normalises underscores → spaces so that audio-model class names
        (e.g. ``wheel_bearing``, ``water_pump``) match component names
        (``Wheel bearing``) and subsystem keys (``brakes``).
        """
        def _norm(s: str | None) -> str:
            """Normalize optional evidence fields without rejecting an unknown DTC.

            The OBD knowledge base does not classify every valid code into one
            of the app's mechanical subsystems. An unclassified code is still
            useful to display, but it must not stop audio reasoning.
            """
            return (s or "").lower().replace("_", " ")

        obs_comp = _norm(self.component)
        obs_sub = _norm(self.subsystem)
        cn = _norm(comp_name)
        cs = _norm(comp_subsystem)

        if obs_comp and obs_comp in cn:
            return True
        if obs_sub and (obs_sub in cs or obs_sub in cn):
            return True
        if not self.component and not self.subsystem:
            return True
        return False


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
                source="audio",
                component="",
                subsystem=c_name, # Audio causes map roughly to subsystems / high-level groups
                weight=weight,
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
        source="description",
        component="",
        subsystem="",
        weight=1.0,
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
            source="obd",
            component="",
            subsystem=subsystem,
            weight=1.0,
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
            source="retrieval",
            component=hit.get("component", ""),
            subsystem=hit.get("subsystem", ""),
            weight=hit.get("weight", 0.5),
            label=hit.get("label", ""),
            features={}
        ))
    return obs
