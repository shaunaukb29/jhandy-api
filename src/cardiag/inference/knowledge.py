"""Automotive Knowledge Abstraction Layer.

This module provides a clean public interface for the AI and reasoning layers
to retrieve structured automotive knowledge for DTCs without querying ontology
files directly.

It uses a provider-based architecture: multiple knowledge providers can be
registered, and their results are merged. This allows future integration of
service manuals, TSBs, VIN decoding, etc., without changing the public interface.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("cardiag.knowledge")

# ------------------------------------------------------------------ Provider base interface

class KnowledgeProvider:
    """Base class for all automotive knowledge providers."""
    def lookup_dtc(self, code: str) -> dict[str, Any] | None:
        """Retrieve knowledge for a specific DTC code.

        Should return a dictionary matching the schema:
        {
            "code": str,
            "description": str,
            "category": str,
            "code_type": str,
            "fault_condition": str,
            "indicates_subsystem": str,
            "indicates_vehicle_parts": list[str],
            "suspect_components": list[dict[str, Any]],
            "symptoms": list[str],
            "possible_causes": list[str],
            "suggested_confirmation_tests": list[str],
            "repair_recommendations": list[str],
            "confidence": float
        }
        or None if the code is not handled by this provider.
        """
        raise NotImplementedError


# ------------------------------------------------------------------ JSON/Ontology provider

class CompiledOntologyProvider(KnowledgeProvider):
    """Loads and queries DTC knowledge from the compiled JSON knowledge base."""

    def __init__(self, json_path: Path | None = None):
        if json_path is None:
            # Default to data/obd_knowledge_base.json relative to package root
            json_path = Path(__file__).resolve().parents[3] / "data" / "obd_knowledge_base.json"

        self.json_path = json_path
        self._db: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            if self.json_path.exists():
                with open(self.json_path, encoding="utf-8") as f:
                    self._db = json.load(f)
                _LOG.info("Loaded compiled OBD database with %d codes.", len(self._db))
            else:
                _LOG.warning(
                    "Compiled OBD database not found at %s. Empty fallback used.",
                    self.json_path
                )
        except Exception as e:
            _LOG.error("Failed to load compiled OBD database: %s", e)
        self._loaded = True

    def lookup_dtc(self, code: str) -> dict[str, Any] | None:
        self._ensure_loaded()
        c = code.strip().upper()
        return self._db.get(c)


class CsvDtcProvider(KnowledgeProvider):
    """Small offline fallback for deployments that omit the large ontology file."""

    def __init__(self, csv_path: Path | None = None):
        if csv_path is None:
            csv_path = Path(__file__).resolve().parents[3] / "data" / "obd_codes.csv"
        self.csv_path = csv_path
        self._db: dict[str, dict[str, str]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as handle:
                self._db = {
                    row["Code"].strip().upper(): row
                    for row in csv.DictReader(handle)
                    if row.get("Code")
                }
        except FileNotFoundError:
            _LOG.warning("OBD CSV fallback not found at %s.", self.csv_path)
        self._loaded = True

    def lookup_dtc(self, code: str) -> dict[str, Any] | None:
        self._ensure_loaded()
        row = self._db.get(code.strip().upper())
        if not row:
            return None
        return {
            "code": row["Code"],
            "description": row.get("Condition Description", ""),
            "code_type": row.get("Trouble Code System", ""),
            "confidence": 0.5,
        }


# ------------------------------------------------------------------ Registry

class KnowledgeRegistry:
    """Manages registered knowledge providers and merges their outputs."""

    def __init__(self):
        self._providers: list[KnowledgeProvider] = []

    def register_provider(self, provider: KnowledgeProvider) -> None:
        """Add a provider to the query chain."""
        self._providers.append(provider)

    def lookup_dtc(self, code: str) -> dict[str, Any] | None:
        """Queries all providers in order and merges their findings."""
        code = code.strip().upper()
        merged: dict[str, Any] = {}

        for provider in self._providers:
            res = provider.lookup_dtc(code)
            if not res:
                continue

            # If it's the first provider to return something, initialize the dict
            if not merged:
                merged = dict(res)
                continue

            # Otherwise, merge lists and use highest confidence
            for key, val in res.items():
                if isinstance(val, list):
                    # Union lists preserving order
                    current = merged.get(key, [])
                    if isinstance(current, list):
                        # Simple element merge for list of strings or dicts
                        for item in val:
                            if item not in current:
                                current.append(item)
                        merged[key] = current
                elif key == "confidence":
                    merged[key] = max(merged.get(key, 0.0), val)
                elif key == "description" and not merged.get(key):
                    merged[key] = val
                elif key == "indicates_subsystem" and not merged.get(key):
                    merged[key] = val

        return merged if merged else None


# Initialize the global registry and register default providers
_REGISTRY = KnowledgeRegistry()
_REGISTRY.register_provider(CompiledOntologyProvider())
_REGISTRY.register_provider(CsvDtcProvider())

def register_provider(provider: KnowledgeProvider) -> None:
    """Public interface to register additional knowledge sources (e.g. manuals, TSBs)."""
    _REGISTRY.register_provider(provider)


# ------------------------------------------------------------------ Public Interface

def lookupDTC(code: str) -> dict[str, Any] | None:
    """Retrieve all structured knowledge for a single DTC code."""
    return _REGISTRY.lookup_dtc(code)

def lookupMultipleDTCs(codes: list[str]) -> dict[str, dict[str, Any]]:
    """Retrieve structured knowledge for multiple DTC codes."""
    res = {}
    for code in codes:
        code = code.strip().upper()
        if not code:
            continue
        info = lookupDTC(code)
        if info:
            res[code] = info
    return res

def getPossibleCauses(code: str) -> list[str]:
    """Retrieve potential physical or electrical causes for a DTC."""
    info = lookupDTC(code)
    return info.get("possible_causes", []) if info else []

def getSuggestedTests(code: str) -> list[str]:
    """Retrieve suggested confirmation tests for a DTC."""
    info = lookupDTC(code)
    return info.get("suggested_confirmation_tests", []) if info else []

def getRelatedSymptoms(code: str) -> list[str]:
    """Retrieve related symptoms associated with a DTC."""
    info = lookupDTC(code)
    return info.get("symptoms", []) if info else []
