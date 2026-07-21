"""
Carithm Audio Diagnoser — Streamlit UI
Acoustic fault triage + OBD-II reasoning
"""

import sys
import os

# MUST be set before numba/librosa/torch are imported anywhere in the process.
# numba's default OpenMP threading layer can collide with PyTorch's own bundled
# OpenMP runtime in the same process, causing native heap corruption
# ("malloc(): unaligned tcache chunk detected") that aborts the whole worker
# with no Python-level traceback. "workqueue" avoids OpenMP entirely.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import tempfile
from html import escape
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from cardiag import Classifier

# ─────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Carithm — Car Sound Diagnosis",
    page_icon="🔧",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #09090b;
    --surface: #111113;
    --surface2: #18181b;
    --surface3: #27272a;
    --border: rgba(255,255,255,0.07);
    --border2: rgba(255,255,255,0.13);
    --text: #fafafa;
    --muted: #a1a8b3;
    --muted2: #71717a;
    --accent: #2f81f7;
    --accent-dim: rgba(47,129,247,0.1);
    --green: #22c55e;
    --green-dim: rgba(34,197,94,0.12);
    --amber: #f59e0b;
    --amber-dim: rgba(245,158,11,0.12);
    --red: #ef4444;
    --red-dim: rgba(239,68,68,0.12);
}

html, body, .stApp {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: radial-gradient(900px 500px at 50% -5%, rgba(47,129,247,0.14), transparent 60%), var(--bg) !important;
    color: var(--text) !important;
}

.block-container { max-width: 880px; padding-top: 3.5rem; padding-bottom: 5rem; }

#MainMenu, footer, header { visibility: hidden; }

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(47,129,247,0.15) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 10px !important;
    padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 24px !important;
    width: 100% !important;
    font-size: 15px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Labels */
label, .stTextInput label, .stTextArea label, .stFileUploader label {
    color: var(--muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
}

/* Cards */
.cd-logo {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    margin-bottom: 2px;
}
.cd-logo span { color: var(--accent); }

.cd-tagline {
    color: var(--muted2);
    font-size: 13px;
    margin-bottom: 24px;
}

.cd-section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted2);
    margin-bottom: 14px;
}

.verdict-badge {
    display: inline-block;
    padding: 7px 16px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.verdict-fault    { background: var(--red-dim);   color: var(--red);   border: 1px solid var(--red); }
.verdict-normal   { background: var(--green-dim); color: var(--green); border: 1px solid var(--green); }
.verdict-uncertain{ background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber); }

.cd-card {
    background: linear-gradient(145deg, rgba(24,24,27,.94), rgba(17,17,19,.98));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    margin: 18px 0;
    box-shadow: 0 12px 32px rgba(0,0,0,.15);
}
.cd-card h4 {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted2);
    margin: 0 0 14px 0;
}

.bar-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 5px;
}
.bar-track {
    height: 7px;
    background: var(--surface3);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 12px;
}
.bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #2f81f7, #60a5fa);
    border-radius: 99px;
}

.cause-name { font-size: 14px; color: var(--text); font-weight: 500; }
.cause-note { font-size: 12px; color: var(--muted2); font-style: italic; margin-top: 2px; margin-bottom: 10px; }

.obd-tag {
    display: inline-block;
    padding: 3px 10px;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid rgba(47,129,247,0.25);
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px 3px;
    font-family: monospace;
}

.tip-box {
    background: var(--accent-dim);
    border: 1px solid rgba(47,129,247,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 20px;
}
.tip-box b { color: var(--text); }

.assessment-card {
    border-color: rgba(47,129,247,.42) !important;
    box-shadow: inset 3px 0 0 var(--accent), 0 12px 32px rgba(0,0,0,.15);
}
.assessment-copy { color: var(--text); font-size: 15px; line-height: 1.72; margin: 0; }
.cause-item { padding: 18px 0; border-bottom: 1px solid var(--border); }
.cause-item:last-child { border-bottom: 0; padding-bottom: 0; }
.cause-heading { display: flex; align-items: center; gap: 10px; }
.cause-rank { color: var(--accent); font-size: 11px; font-weight: 700; letter-spacing: .06em; }
.cause-detail { font-size: 12px; color: var(--muted); margin-top: 12px; }
.cause-detail b { display: block; color: var(--text); font-size: 11px; letter-spacing: .05em; text-transform: uppercase; margin-bottom: 5px; }
.followup-card { border-color: rgba(47,129,247,.34) !important; }
.followup-question { color: var(--text); font-size: 16px; line-height: 1.55; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────
st.markdown("<div class='cd-logo'>Carithm</div>", unsafe_allow_html=True)
st.markdown("<div class='cd-tagline'>Upload engine audio for an instant acoustic fault diagnosis.</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading acoustic model (downloading ~2GB weights on first run, this may take a few minutes)...")
def load_model():
    # Force the ~2GB CLAP model to download and load now at boot time
    # rather than lazily during the first diagnosis click to prevent timeouts.
    try:
        from huggingface_hub import snapshot_download
        snapshot_download("laion/clap-htsat-unfused", local_files_only=False)
    except Exception:
        pass
    from cardiag.audio.clap import Clap
    Clap()
    # Pre-warm Silero VAD so it doesn't try to reach the network during diagnosis
    try:
        from silero_vad import load_silero_vad
        load_silero_vad()
    except Exception:
        pass
    return Classifier.load()

clf = load_model()

def sanitize_and_check_audio(uploaded_file) -> str | None:
    """Security sanitization and validation on uploaded/recorded audio files.
    
    1. Checks file size (max 10MB).
    2. Validates type/extension.
    3. Loads with soundfile to verify integrity (preventing shell/execution exploits).
    4. Truncates/limits duration to 15 seconds maximum.
    """
    if not uploaded_file:
        return None
        
    # 1. Size check
    if uploaded_file.size > 10 * 1024 * 1024:
        st.error("File is too large (maximum 10MB).")
        return None
        
    # 2. Extension check
    raw_name = getattr(uploaded_file, "name", "recorded_audio.wav") or "recorded_audio.wav"
    suffix = Path(raw_name).suffix.lower() or ".wav"
    if suffix not in (".wav", ".mp3", ".ogg", ".m4a", ".flac", ".aac"):
        st.error("Invalid audio format.")
        return None

    # Write to a secure temp file (tempfile handles secure unique name generation)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp_path = tmp.name

    # 3 & 4. Integrity check & duration limit (using soundfile/librosa)
    try:
        import soundfile as sf
        import librosa
        
        info = sf.info(tmp_path)
        duration = info.duration
        
        # Enforce 15 seconds limit
        if duration > 15.0:
            st.warning("Recording exceeds 15 seconds limit. Truncating to the first 15 seconds.")
            y, sr = librosa.load(tmp_path, sr=None)
            y_trimmed = y[:int(15.0 * sr)]
            # IMPORTANT: soundfile cannot encode MP3 (or several other input
            # formats) — writing truncated audio back into a file with the
            # ORIGINAL suffix (e.g. .mp3) silently produces an invalid/garbage
            # file, which downstream reads as "no usable audio" every time.
            # Always re-save as .wav, which soundfile can always write.
            os.unlink(tmp_path)
            tmp_path = tmp_path + ".trimmed.wav"
            sf.write(tmp_path, y_trimmed, sr)
            
    except Exception as e:
        st.error("Uploaded file is corrupt or not a valid audio recording.")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None

    return tmp_path

# ─────────────────────────────────────────────────────────────────────────
# Recording tip
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="tip-box">
    <b>Recording tip:</b> Hold your phone microphone 12 inches above the engine bay.
    Record 5–10 seconds at idle. WAV, MP3, M4A, and FLAC all work.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────
input_mode = st.radio("Choose input method:", ["Upload Audio File", "Record Live Audio"], horizontal=True)

audio_file = None
if input_mode == "Upload Audio File":
    audio_file = st.file_uploader("Audio recording", type=["wav", "mp3", "m4a", "flac", "ogg", "aac"])
else:
    audio_file = st.audio_input("Record live audio (max 15s)")

with st.expander("➕  Add OBD-II codes (optional)"):
    obd_input = st.text_input(
        "OBD-II codes",
        placeholder="e.g. P0300, P0301",
        help="Comma-separated DTC codes from your OBD scanner. Leave blank if you don't have any."
    )

# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def pct(p: float) -> str:
    return f"{p * 100:.1f}%"

def bar_html(label: str, p: float, color: str = "linear-gradient(90deg,#2f81f7,#60a5fa)") -> str:
    w = max(0.0, min(1.0, p)) * 100
    return f"""
    <div class="bar-row"><span>{label}</span><span>{pct(p)}</span></div>
    <div class="bar-track"><div class="bar-fill" style="width:{w}%;background:{color}"></div></div>
    """

# Human-readable label for the training category names
CAUSE_LABELS = {
    "engine_internal":    "Engine Internals (valvetrain, pistons, bearings)",
    "accessories":        "Engine Accessories (alternator, A/C, power steering)",
    "transmission":       "Transmission / Drivetrain",
    "exhaust":            "Exhaust System",
    "suspension":         "Suspension / Steering",
    "fuel_system":        "Fuel System (injectors, pump)",
    "cooling":            "Cooling System",
    "electrical":         "Electrical / Sensors",
    "brakes":             "Brakes",
    "normal":             "Normal Engine Sound",
}

def readable_cause(raw: str) -> str:
    key = raw.lower().replace(" ", "_").replace("-", "_")
    return CAUSE_LABELS.get(key, raw.replace("_", " ").title())

def cause_html(name: str, p: float, note: str | None) -> str:
    w = max(0.0, min(1.0, p)) * 100
    label = readable_cause(name)
    note_part = f"<div class='cause-note'>→ {note}</div>" if note else ""
    return f"""
    <div class="bar-row">
        <span class="cause-name">{label}</span>
        <span>{pct(p)}</span>
    </div>
    <div class="bar-track"><div class="bar-fill" style="width:{w}%"></div></div>
    {note_part}
    """

def verdict_badge(verdict_str: str) -> str:
    v = verdict_str.lower()
    cls = "verdict-fault" if v == "fault" else "verdict-normal" if v == "normal" else "verdict-uncertain"
    return f"<div class='verdict-badge {cls}'>{verdict_str}</div>"


def render_reasoning(reasoning: dict, obd_codes: list[str]) -> None:
    """Render the complete offline reasoning payload in the Carithm card style."""
    if obd_codes:
        tags = "".join(f"<span class='obd-tag'>{escape(code)}</span>" for code in obd_codes)
        st.markdown(f"<div style='margin:4px 0 12px'>{tags}</div>", unsafe_allow_html=True)

    interpretations = reasoning.get("obd_interpretations", [])
    if interpretations:
        body = "".join(
            "<p style='color:var(--muted);font-size:13px;margin:0 0 10px;line-height:1.55'>"
            f"<b style='color:var(--accent)'>{escape(item.get('code', ''))}</b> "
            f"{escape(item.get('interpretation', ''))}</p>"
            for item in interpretations
        )
        st.markdown(f"<div class='cd-card'><h4>OBD-II Interpretation</h4>{body}</div>",
                    unsafe_allow_html=True)

    assessment = reasoning.get("assessment", "")
    if assessment:
        st.markdown(
            "<div class='cd-card assessment-card'>"
            "<h4>Current Assessment</h4>"
            f"<p class='assessment-copy'>{escape(assessment)}</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    components = reasoning.get("components", [])
    if components:
        cards = []
        for rank, component in enumerate(components[:5], start=1):
            name = escape(component.get("name", "Unknown component"))
            probability = max(0.0, min(1.0, float(component.get("probability", 0))))
            evidence = component.get("evidence", [])
            tests = component.get("tests", [])
            evidence_html = "".join(
                f"<li style='margin:0 0 5px'>{escape(str(item))}</li>" for item in evidence[:3]
            ) or "<li>No direct corroborating evidence recorded.</li>"
            tests_html = "".join(
                f"<li style='margin:0 0 5px'>{escape(str(item))}</li>" for item in tests[:4]
            ) or "<li>Arrange a targeted inspection.</li>"
            cards.append(
                "<div class='cause-item'>"
                "<div class='cause-heading'>"
                f"<span class='cause-rank'>#{rank:02d}</span>"
                f"<div class='bar-row' style='flex:1;margin:0'><span class='cause-name'>{name}</span><span>{pct(probability)}</span></div>"
                "</div>"
                f"<div class='bar-track'><div class='bar-fill' style='width:{probability * 100:.1f}%'></div></div>"
                "<div class='cause-detail'><b>Evidence</b>"
                f"<ul style='margin:5px 0 10px 18px;padding:0'>{evidence_html}</ul>"
                "<b>Recommended checks</b>"
                f"<ol style='margin:5px 0 0 18px;padding:0'>{tests_html}</ol></div></div>"
            )
        st.markdown(
            "<div class='cd-card'><h4>Most Likely Causes and Next Checks</h4>"
            + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )

    similar_cases = reasoning.get("similar_cases", [])
    if similar_cases:
        sim_cards = []
        for case in similar_cases:
            raw_cause = case.get("cause") or case.get("l1") or case.get("kind") or "Reference Case"
            cause = escape(raw_cause.replace("_", " ").title())
            similarity = float(case.get("similarity", 0.0))
            sim_cards.append(
                f"<div style='margin-bottom:10px;padding:12px;background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:8px;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:0px;'>"
                f"<span style='font-size:14px;color:var(--text);font-weight:600;'>{cause}</span>"
                f"<span style='font-size:12px;color:var(--accent);font-weight:700;'>{similarity * 100:.0f}% Match</span>"
                f"</div>"
                f"</div>"
            )
        st.markdown(
            "<div class='cd-card'><h4>Acoustically Similar Reference Cases</h4>"
            + "".join(sim_cards) + "</div>",
            unsafe_allow_html=True,
        )


    question = reasoning.get("followup_question")
    question_id = reasoning.get("followup_id")
    if question and question_id and components:
        st.markdown("<div class='cd-card followup-card'><h4>One Follow-up Question</h4>"
                    f"<p class='followup-question'>{escape(question)}</p></div>",
                    unsafe_allow_html=True)
        yes, no, skip = st.columns(3)
        answers = ((yes, "Yes", "yes"), (no, "No", "no"), (skip, "Skip", "skip"))
        for column, label, value in answers:
            if column.button(label, key=f"followup-{question_id}-{value}"):
                from cardiag.inference.reasoning import apply_followup_answer

                asked = st.session_state["carithm_diagnosis"].get("asked_ids", [])
                update = apply_followup_answer(
                    components, question_id, value,
                    reasoning.get("primary_subsystem", ""), asked,
                )
                next_reasoning = dict(reasoning)
                next_reasoning.update(update)
                st.session_state["carithm_diagnosis"]["reasoning"] = next_reasoning
                st.session_state["carithm_diagnosis"]["asked_ids"] = update.get(
                    "asked_ids", asked + [question_id]
                )
                st.rerun()


def synthesize(result, reasoning=None) -> tuple[str, list[str]]:
    """
    Reads the Diagnosis object and returns (plain-English summary, list of next steps).
    Uses ReasoningEngine output if available for highly specific component predictions.
    """
    p_fault  = result.fault_probability
    p_knock  = result.engine_knock_probability
    verdict  = result.verdict.value.lower()
    causes   = result.causes          # list[Cause], sorted by p descending
    top_zone = result.regions[0].zone.lower() if result.regions else ""

    components = reasoning.get("components", []) if reasoning else []
    
    headline_part = ""
    headline_note = ""
    headline_name = ""
    if components:
        top_comp = components[0]
        headline_part = top_comp.get("subsystem", "").lower().replace(" ", "_").replace("-", "_")
        headline_name = top_comp.get("name", "")
    elif causes:
        headline = causes[0]
        headline_part = headline.part.lower().replace(" ", "_").replace("-", "_")
        headline_note = headline.note

    # ── Summary sentence
    if verdict == "normal":
        summary = (
            "The recording sounds like a healthy engine. "
            "No acoustic fault pattern was detected."
        )
    elif verdict == "fault":
        severity = (
            "severe"   if p_fault > 0.85 else
            "strong"   if p_fault > 0.70 else
            "moderate"
        )
        zone_text = f" in the <strong>{top_zone}</strong> area" if top_zone else ""

        knock_text = ""
        if p_knock > 0.55:
            knock_text = (
                "<br><br>⚠️ <strong>Knock Warning:</strong> The knock detector flagged a significant metallic knock — "
                "this may indicate bearing or piston wear. <strong>Stop driving and inspect immediately.</strong>"
            )
        elif p_knock > 0.30:
            knock_text = (
                f"<br><br>⚠️ <strong>Knock Warning:</strong> The model also detected a moderate knock signature ({pct(p_knock)}), "
                "which can indicate valvetrain wear or early bearing wear."
            )

        if headline_name:
            cause_text = (
                f"The diagnostic evidence strongly points to a <strong>{readable_cause(headline_part)}</strong> issue.<br>"
                f"The leading specific suspect is the <strong>{headline_name}</strong>."
            )
        elif headline_note:
            clean_hint = headline_note.strip().replace(" / ", ", ").replace("A-C", "A/C")
            clean_part = "engine accessory" if "accessor" in headline_part else headline_part.replace("_", " ")
            cause_text = (
                f"The acoustic pattern strongly points to an <strong>{clean_part}</strong> issue.<br>"
                f"Based on the specific sound signature, check the <strong>{clean_hint}</strong>."
            )
        elif "engine_internal" in headline_part or "internal" in headline_part:
            cause_text = (
                "The sound pattern most closely matches <strong>engine-internal components</strong> "
                "(valvetrain, lifters, pistons, or rod/main bearings)."
            )
        elif "accessor" in headline_part:
            cause_text = (
                "The strongest match is an <strong>accessory component</strong> — "
                "check the alternator, A/C compressor, power steering pump, or idler pulley."
            )
        elif "fuel" in headline_part:
            cause_text = (
                "The pattern resembles <strong>fuel system noise</strong> — "
                "likely injector click, fuel pump whine, or a pressure regulator issue."
            )
        elif "exhaust" in headline_part:
            cause_text = (
                "The pattern is consistent with an <strong>exhaust leak</strong> — "
                "check the manifold gasket, downpipe flange, and flex pipe."
            )
        elif "transmission" in headline_part:
            cause_text = (
                "The sound matches a <strong>drivetrain or transmission fault</strong> — "
                "check transmission fluid, CV joints, and differential."
            )
        else:
            cause_text = f"The top acoustic match was <strong>{readable_cause(headline_part)}</strong>." if headline_part else ""

        summary = (
            f"<strong>{severity.title()} fault detected</strong>{zone_text} "
            f"({pct(p_fault)} probability).<br><br>"
            f"{cause_text}{knock_text}"
        )
    else:
        summary = (
            f"The model returned an uncertain result ({pct(p_fault)} fault probability).<br><br>"
            "The recording may be too short, too noisy, or the audio level was too low "
            "for a confident call. Try a cleaner 10-second recording at idle."
        )

    steps = []
    if verdict == "fault":
        if p_knock > 0.50:
            steps.append("🔴 <strong>Do not drive</strong> — a confirmed knock can destroy the engine in minutes.")
            steps.append("Check engine oil level and pressure immediately.")
        elif p_knock > 0.25:
            steps.append("Check engine oil level — low oil is the #1 cause of valvetrain and bearing noise.")
            steps.append("Get a mechanic to do a compression and oil pressure test.")
        else:
            steps.append("Check engine oil level and condition first — it's free and takes 2 minutes.")

        if "accessor" in headline_part:
            steps.append("With engine off, spin each accessory pulley by hand and feel for roughness or grinding.")
            steps.append("Remove the serpentine belt and run briefly — if the noise disappears, an accessory is the source.")
            if headline_note and "alternator" in headline_note.lower():
                steps.append("Check alternator voltage with a multimeter (healthy: 13.5–14.5V at idle).")
        elif "engine_internal" in headline_part or "internal" in headline_part:
            steps.append("A mechanic should perform an oil pressure test and listen with an automotive stethoscope.")
        elif "exhaust" in headline_part:
            steps.append("Inspect exhaust manifold for cracks — listen for hissing at cold start.")
        elif "fuel" in headline_part:
            steps.append("Check fuel pressure; a failing pump often produces a whine audible from the tank.")

        steps.append("Add any OBD-II codes for a more targeted component breakdown.")

    elif verdict == "uncertain":
        steps.append("Re-record 8–10 seconds at idle with the phone closer to the engine.")
        steps.append("Reduce background noise (turn off A/C, radio) before recording.")
        steps.append("Add any OBD-II codes for extra diagnostic context.")

    return summary, steps


# ─────────────────────────────────────────────────────────────────────────

# Diagnose
# ─────────────────────────────────────────────────────────────────────────
if audio_file:
    tmp_path = sanitize_and_check_audio(audio_file)
    
    if tmp_path and os.path.exists(tmp_path):
        st.audio(tmp_path)

        file_name = getattr(audio_file, "name", "recorded_audio.wav") or "recorded_audio.wav"
        audio_key = f"{file_name}:{audio_file.size}:{obd_input.strip().upper()}"
        run_diagnosis = st.button("🔍  Diagnose Audio")
        saved = st.session_state.get("carithm_diagnosis")
        if run_diagnosis or (saved and saved.get("audio_key") == audio_key):
            if run_diagnosis:
                with st.spinner("Analyzing audio — this takes a few seconds…"):
                    diag_error = None
                    result = None
                    reasoning = None
                    obd_codes = [c.strip().upper() for c in obd_input.split(",") if c.strip()]
                    try:
                        result = clf.diagnose(tmp_path)
                    except BaseException as exc:  # catch EVERYTHING, not just Exception
                        import traceback
                        diag_error = exc
                        print("[diagnose] FAILED:", file=sys.stderr, flush=True)
                        traceback.print_exc(file=sys.stderr)
                        st.exception(exc)  # full traceback, visible in the UI itself
                    if result is not None:
                        try:
                            from cardiag.inference.reasoning import ReasoningEngine
                            rd = result.to_dict()
                            print(f"[debug] obd_codes={obd_codes!r} "
                                  f"verdict={rd.get('verdict')!r} "
                                  f"causes={rd.get('causes')!r}",
                                  file=sys.stderr, flush=True)
                            reasoning = ReasoningEngine().reason(result.to_dict(), obd_codes)
                        except BaseException as exc:
                            import traceback
                            reasoning = None
                            print("[reasoning] FAILED:", file=sys.stderr, flush=True)
                            traceback.print_exc(file=sys.stderr)
                            st.exception(exc)
                st.session_state["carithm_diagnosis"] = {
                    "audio_key": audio_key,
                    "result": result,
                    "reasoning": reasoning,
                    "obd_codes": obd_codes,
                    "diag_error": str(diag_error) if diag_error else None,
                    "asked_ids": [],
                }
                if diag_error:
                    st.error(f"Audio analysis failed: {diag_error}")
            else:
                result = saved["result"]
                reasoning = saved.get("reasoning")
                obd_codes = saved.get("obd_codes", [])
                diag_error = saved.get("diag_error")
                if diag_error:
                    st.error(f"Audio analysis failed: {diag_error}")

            # Always render whatever we have — never silently show only the caption
            reasoning_has_content = bool(reasoning) and any(
                reasoning.get(k) for k in
                ("obd_interpretations", "assessment", "components", "similar_cases")
            )
            if reasoning_has_content:
                try:
                    render_reasoning(reasoning, obd_codes)
                except Exception as exc:
                    st.warning(f"Result rendering error: {exc}")
            elif result is not None and not diag_error:
                # Diagnosis ran but reasoning returned nothing usable — show the
                # raw verdict/notes instead of nothing at all.
                st.info("No strong fault candidates identified from structured reasoning. "
                        "Showing the raw model output below.")
                rd = result.to_dict() if hasattr(result, "to_dict") else {}
                st.write(f"**Verdict:** {rd.get('verdict', 'unknown')}")
                st.write(f"**Fault probability:** {pct(float(rd.get('fault_probability', 0.0)))}")
                if rd.get("note"):
                    st.caption(rd["note"])
                if not reasoning_has_content and reasoning is not None:
                    print(f"[reasoning] empty content, raw dict: {reasoning}",
                          file=sys.stderr, flush=True)

            st.caption("Decision support only — not a replacement for professional inspection.")

            # Clean up the temporary file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

elif not audio_file:
    st.markdown("""
    <div style="text-align:center;padding:40px 0;color:var(--muted2);font-size:13px;">
        Upload an audio file above to get started.
    </div>
    """, unsafe_allow_html=True)
