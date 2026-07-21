from dataclasses import dataclass, field
from typing import Any

@dataclass
class Observation:
    """A single piece of extracted evidence from any modality."""
    id: str
    source: str          # e.g., "audio_model_v1", "user_obd", "description_parser"
    category: str        # e.g., "audio", "obd", "description", "retrieval", "followup"
    component: str       # target component name (or "" for subsystem-level)
    subsystem: str       # target subsystem (or "" if general)
    confidence: float    # 0.0-1.0
    label: str           # human-readable explanation
    features: dict       # structured data (e.g., {"code": "P0301"})

    def matches(self, comp_name: str, comp_subsystem: str) -> bool:
        """Check if this observation applies to the given component or subsystem."""
        def _norm(s: str | None) -> str:
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


@dataclass
class EvidenceLink:
    """A connection between an Observation and a Hypothesis."""
    observation_id: str
    relationship: str    # "supports", "refutes", "contradicts"
    weight: float        # impact on the log-odds or score
    explanation: str     # human-readable why


@dataclass
class Hypothesis:
    """A candidate explanation for the diagnosis."""
    id: str
    name: str
    subsystem: str
    probability: float
    evidence_links: list[EvidenceLink] = field(default_factory=list)
    recommended_tests: list[str] = field(default_factory=list)
    severity: str = "moderate"
    driveability: str = "monitor"

    def to_legacy_dict(self) -> dict:
        """Helper to convert to the legacy dict format for the UI."""
        # The UI expects a list of evidence strings
        evidence_strings = []
        for link in self.evidence_links:
            if link.explanation and link.explanation not in evidence_strings:
                evidence_strings.append(link.explanation)

        return {
            "name": self.name,
            "subsystem": self.subsystem,
            "probability": round(self.probability, 3),
            "evidence": evidence_strings,
            "evidence_breakdown": {link.observation_id: link.weight for link in self.evidence_links},
            "tests": self.recommended_tests,
            "severity": self.severity,
            "driveability": self.driveability
        }


@dataclass
class EvidenceGraph:
    """The full graph of raw inputs -> observations -> hypotheses."""
    raw_predictions: dict = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
