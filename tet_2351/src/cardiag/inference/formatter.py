from typing import Any
from cardiag.inference.evidence import EvidenceGraph

class DiagnosisFormatter:
    """Transforms an EvidenceGraph into the legacy dictionary payload for the UI."""
    
    def __init__(self):
        pass

    def format(self, 
               graph: EvidenceGraph, 
               diag_dict: dict, 
               parsed_codes: list[dict], 
               matches: list[str], 
               sym_features: dict, 
               primary_subsystem: str, 
               followup: Any, 
               obd_interps: list[dict], 
               confidence: float, 
               assessment: str, 
               explanation: str,
               similar_cases: list[dict] = None) -> dict:
        
        # Convert hypotheses back to legacy dict format
        components = [h.to_legacy_dict() for h in graph.hypotheses]

        result = {
            "parsed_codes":           parsed_codes,
            "coherence_matches":      matches,
            "explanation":            explanation,
            "assessment":             assessment,
            "confidence":             confidence,
            "description_interpretation": sym_features.get("narrative", "") if sym_features else "",
            "components":             components,
            "obd_interps":            obd_interps,
            "obd_interpretations":    obd_interps,
            "primary_subsystem":      primary_subsystem or "",
            "similar_cases":          similar_cases or [],
            "evidence_graph":         {
                "observations": [vars(o) for o in graph.observations],
                "hypotheses": [{"id": h.id, "name": h.name, "probability": h.probability} for h in graph.hypotheses]
            }
        }
        
        if followup:
            result["followup_question"] = followup.text
            result["followup_id"]       = followup.id
            result["followup_options"]  = followup.options
            result["followup_yes_multipliers"] = followup.yes_multipliers
            result["followup_no_multipliers"]  = followup.no_multipliers
            
        return result
