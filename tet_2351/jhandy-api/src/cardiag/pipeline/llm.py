"""LLM interface stub — no external model dependencies.

The HuggingFace InferenceClient and Ollama backends have been removed to keep
the deployment lightweight. All explanation callers fall through to the
template-based engine in reasoning.py, which requires no network calls.
"""
import json


def _hf_one(prompt: str) -> str:
    """Stub: returns empty string so callers use template fallback."""
    return ""


def run_batch(items, backend="template", workers=1):
    """items: list of (id, prompt). Returns {id: completion}."""
    return {i: "" for i, _ in items}


def parse_json(text):
    """Best-effort extract the first JSON object/array from a completion."""
    for a, b in (("{", "}"), ("[", "]")):
        i, j = text.find(a), text.rfind(b)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                pass
    return None
