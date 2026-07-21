"""Free-text symptom description interpreter.

Extracts structured symptom fingerprints from a driver's natural-language description
and produces an interpreted narrative — not an echo of what the user wrote.

Example
-------
    >>> from cardiag.inference.description_parser import interpret
    >>> r = interpret("loud knocking from engine bay, worse when cold, goes away after 2 min")
    >>> r["narrative"]
    'Description indicates a heavy knocking originating from the engine bay, present on
     cold starts and resolving as the engine warms — consistent with a temperature- or
     pressure-sensitive component.'
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ------------------------------------------------------------------ keyword tables

_TIMING: dict[str, list[str]] = {
    "cold_start":  ["cold start", "when cold", "cold engine", "after sitting",
                    "first start", "morning start", "just started", "starts cold",
                    "startup", "on startup", "initial start"],
    "warm_up":     ["warm up", "warms up", "goes away", "disappears after",
                    "when warm", "after a few minutes", "engine warms",
                    "gets warm", "temperature rises", "once warm", "warmed up"],
    "idle":        ["at idle", "idling", "when stopped", "in park", "stationary",
                    "neutral", "not moving", "sitting still"],
    "acceleration":["acceleration", "accelerating", "hard acceleration", "full throttle",
                    "stepping on gas", "pressing throttle", "open throttle"],
    "braking":     ["when braking", "during braking", "press the brake", "applying brakes",
                    "slowing down", "deceleration"],
    "turning":     ["when turning", "during turns", "turning left", "turning right",
                    "cornering", "steering wheel"],
    "constant":    ["always", "constant", "continuously", "all the time",
                    "non-stop", "persistent", "never stops"],
    "highway":     ["highway", "motorway", "freeway", "at speed", "high speed",
                    "70 mph", "60 mph", "80 mph"],
}

_CHARACTER: dict[str, list[str]] = {
    "knock":   ["knock", "knocking", "bang", "banging", "thud"],
    "rattle":  ["rattle", "rattling", "chain rattle", "loose", "metallic rattle"],
    "tick":    ["tick", "ticking", "click", "clicking", "tappet"],
    "squeal":  ["squeal", "squealing", "squeak", "screech", "high pitched", "shriek"],
    "grind":   ["grind", "grinding", "metal on metal", "scraping", "grating"],
    "whine":   ["whine", "whining", "hum", "humming", "drone", "droning"],
    "hiss":    ["hiss", "hissing", "whistle", "whistling", "air leak", "escaping air"],
    "clunk":   ["clunk", "clunking", "thump", "bang over bumps", "single bang"],
    "rumble":  ["rumble", "rumbling", "growl", "growling", "low frequency"],
}

_LOCATION: dict[str, list[str]] = {
    "engine_bay":    ["engine bay", "under the hood", "hood",
                      "top of engine", "motor", "valve cover"],
    "accessories":   ["belt", "belt area", "front accessory drive", "front of engine", 
                      "timing cover", "idler", "tensioner", "alternator", "power steering", 
                      "a/c compressor", "ac compressor", "water pump", "pulley"],
    "exhaust":       ["exhaust", "rear", "tailpipe", "underneath",
                      "muffler", "back of car", "behind"],
    "wheel_area":    ["wheel", "tire", "corner", "suspension", "front left",
                      "front right", "rear left", "rear right", "wheel arch", "axle"],
    "transmission":  ["transmission", "gearbox", "when shifting", "gear", "gearstick"],
    "steering":      ["steering", "steering wheel", "when turning", "column"],
    "brakes":        ["brake", "braking", "pedal"],
}

_RPM_MODIFIERS: dict[str, list[str]] = {
    "disappears_higher_rpm": ["disappears above", "goes away above", "quieter at higher",
                              "stops at high rpm", "disappears at speed",
                              "above 1500", "above 2000", "higher rpm",
                              "higher revs"],
    "worse_higher_rpm":      ["worse at higher", "louder at higher", "increases with rpm",
                              "worse revving", "gets louder with rpm", "changes with rpm",
                              "rev the engine"],
    "speed_dependent":       ["speed dependent", "faster i go", "vehicle speed",
                              "changes with speed", "increases with speed"],
}

# ------------------------------------------------------------------ human-readable labels

_CHAR_LABELS: dict[str, str] = {
    "knock":  "heavy knocking",
    "rattle": "metallic rattling",
    "tick":   "rhythmic ticking",
    "squeal": "high-pitched squealing",
    "grind":  "grinding or metal-on-metal noise",
    "whine":  "mechanical whining",
    "hiss":   "hissing or air-escaping sound",
    "clunk":  "single clunking impact",
    "rumble": "low-frequency rumble",
}

_LOC_LABELS: dict[str, str] = {
    "engine_bay":   "originating from the engine bay",
    "accessories":  "originating from the front accessory drive or belts",
    "exhaust":      "coming from the exhaust / underside",
    "wheel_area":   "localised near a wheel or corner",
    "transmission": "associated with the gearbox or drivetrain",
    "steering":     "linked to the steering system",
    "brakes":       "occurring during braking",
}

_TIMING_LABELS: dict[str, str] = {
    "cold_start":   "present on cold starts",
    "warm_up":      "resolving as the engine warms",
    "persists_warm": "remaining present after the engine warms up",
    "idle":         "most prominent at idle",
    "acceleration": "occurring under acceleration",
    "braking":      "triggered by braking",
    "turning":      "appearing when the steering wheel is turned",
    "constant":     "persistent regardless of conditions",
    "highway":      "noticeable at highway speeds",
}

_TIMING_IMPLICATIONS: dict[tuple[str, ...], str] = {
    ("cold_start", "warm_up"): (
        "— consistent with a component that depends on oil temperature, viscosity, or pressure"
    ),
    ("cold_start",): (
        "— note: noise that disappears after warmup typically points to a thermally sensitive component"
    ),
    ("persists_warm",): (
        "— note: noise that persists when warm points to a component subject to continuous mechanical wear rather than thermal expansion"
    ),
    ("idle",): (
        "— components that load at idle (alternator, power steering pump, AC) are worth inspecting"
    ),
    ("acceleration",): (
        "— load-dependent symptoms often involve fuel delivery, torque transfer, or driveline"
    ),
    ("braking",): (
        "— braking-only symptoms point strongly to the brake system or wheel bearings"
    ),
    ("turning",): (
        "— turning-triggered symptoms are characteristic of steering and drivetrain faults"
    ),
    ("highway",): (
        "— speed-dependent noise is typical of wheel bearings, driveline, or tyre issues"
    ),
}


# ------------------------------------------------------------------ main function

@dataclass
class SymptomProfile:
    location: str | None = None
    sound: str | None = None
    frequency: str | None = None
    warm_state: str | None = None
    load: bool | None = None
    cold_only: bool | None = None
    acceleration: bool | None = None
    braking: bool | None = None
    turning: bool | None = None
    speed_dependent: bool | None = None

def interpret(text: str) -> dict:
    """Extract structured symptom features from a natural-language description.

    Returns a dict with:
    - ``character``: list of noise type keys found
    - ``timing``: list of timing pattern keys found
    - ``location``: list of location keys found
    - ``rpm_modifiers``: list of RPM/speed modifier keys found
    - ``narrative``: interpreted one-sentence summary (NOT an echo of the input)
    - ``profile``: SymptomProfile object with structured boolean/string features
    """
    if not text or not text.strip():
        return {}

    t = text.lower()

    character = _match(t, _CHARACTER)
    timing     = _match(t, _TIMING)
    location   = _match(t, _LOCATION)
    rpm_mods   = _match(t, _RPM_MODIFIERS)

    # Negation handling is now built into _match

    # Negation and Exclusion Handling for Timing
    persists_warm = False
    persists_patterns = [
        r"remains\s+(present|noticeable|there)\s+(after|when|once|even after)\s+(the\s+)?(engine\s+)?warm(s|ed)?",
        r"does(n't|\s+not)\s+(go\s+away|disappear)\s+when\s+warm",
        r"persists\s+(when|after)\s+warm",
        r"still\s+(present|there)\s+when\s+warm"
    ]
    if any(re.search(pat, t) for pat in persists_patterns):
        persists_warm = True

    if persists_warm:
        if "warm_up" in timing:
            timing.remove("warm_up")
        timing.append("persists_warm")

    narrative = _build_narrative(character, timing, location, rpm_mods)

    profile = SymptomProfile(
        location=location[0] if location else None,
        sound=character[0] if character else None,
        frequency="rpm_sync" if "worse_higher_rpm" in rpm_mods else "speed_sync" if "speed_dependent" in rpm_mods else "constant" if "constant" in timing else None,
        warm_state="persists_warm" if "persists_warm" in timing else "cold_only" if "cold_start" in timing and "warm_up" in timing else "warm_only" if "warm_up" not in timing and "cold_start" not in timing and "persists_warm" in timing else None,
        load="acceleration" in timing,
        cold_only="cold_start" in timing and "warm_up" in timing,
        acceleration="acceleration" in timing,
        braking="braking" in timing,
        turning="turning" in timing,
        speed_dependent="speed_dependent" in rpm_mods
    )

    return {
        "character":     character,
        "timing":        timing,
        "location":      location,
        "rpm_modifiers": rpm_mods,
        "narrative":     narrative,
        "profile":       profile,
    }


def _match(text: str, table: dict[str, list[str]]) -> list[str]:
    """Return the keys from *table* whose keyword lists have at least one match in *text*."""
    found = []

    # Simple negation prefixes
    neg_pattern = r"\b(rather than|instead of|not|no|never|not coming from|without)\s+(the\s+)?"

    for key, patterns in table.items():
        for p in patterns:
            if p in text:
                # Check if it's negated
                is_negated = False
                escaped_p = re.escape(p)
                if re.search(neg_pattern + escaped_p + r"\b", text):
                    is_negated = True
                if not is_negated:
                    found.append(key)
                    break
    return found


def _build_narrative(character: list[str], timing: list[str],
                     location: list[str], rpm_mods: list[str]) -> str:
    """Compose an interpreted narrative from extracted symptom features."""
    parts: list[str] = []

    # Noise character
    if character:
        char_phrases = [_CHAR_LABELS.get(c, c) for c in character[:2]]
        parts.append("Description indicates " + _join(char_phrases))
    else:
        parts.append("Description indicates a reported noise or symptom")

    # Location
    if location:
        loc_phrases = [_LOC_LABELS.get(l, l) for l in location[:2]]
        parts[-1] += " " + _join(loc_phrases)

    # Timing
    timing_phrases = [_TIMING_LABELS[t] for t in timing if t in _TIMING_LABELS]
    if timing_phrases:
        parts.append(_join(timing_phrases, connector=", ").capitalize())

    # RPM/speed modifiers (override or extend timing)
    if "disappears_higher_rpm" in rpm_mods:
        parts.append("noise disappears at higher RPM")
    if "worse_higher_rpm" in rpm_mods:
        parts.append("noise worsens at higher RPM")
    if "speed_dependent" in rpm_mods:
        parts.append("intensity tracks vehicle speed")

    # Implication suffix from timing combination
    implication = ""
    for combo, text in _TIMING_IMPLICATIONS.items():
        if all(t in timing for t in combo):
            implication = text
            break

    result = ". ".join(p for p in parts if p)
    if implication:
        result += " " + implication

    return result + "."


def _join(items: list[str], connector: str = " and ") -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return connector.join(items)
