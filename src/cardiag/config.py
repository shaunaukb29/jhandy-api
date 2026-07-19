"""Single source of truth for the pipeline: sample rates, prompts, thresholds,
search queries, and the cause taxonomy.

Every value here encodes a lesson measured while building the original corpus
(see ``docs/``); change with care. These constants were previously
duplicated in per-platform ``config.py`` files; they are identical across
platforms and now live here once.
"""
from __future__ import annotations

# --------------------------------------------------------------- sample rates
SR_CHEAP = 16_000          # cascade tiers (Silero VAD + energy)
SR_CLAP = 48_000           # CLAP input
WIN_S, HOP_S = 1.5, 0.75

# ------------------------------------------------------------------ L1 labels
# Sound types are the ONLY thing decidable from audio alone (measured: part-name
# taxonomy -> 0/15 auto-accept; sound-type -> confident & separable).
L1_PROMPTS = [
    "a grinding noise", "a squealing or squeaking noise",
    "a knocking or clunking noise", "a high-pitched whining noise",
    "a ticking or clicking noise", "a hissing noise", "a rattling noise",
    "a humming or droning roar", "a normal smooth engine idle",
]
L1_NORMAL = "normal smooth engine idle"   # see NORMAL_MARGIN below

# ------------------------------------------------ mechanical-vs-other confirm
CONFIRM_KEEP = ["a car engine or mechanical noise from a machine"]
CONFIRM_DROP = ["music or a jingle", "a person talking",
                "wind or static noise", "silence or room tone"]

# ------------------------------------------------------ fault-vs-tool gate
# Lesson: 42% of v1 auto clips were shop tools (impact wrench labeled "idle",
# grinder labeled "grinding"). KEEP spans ALL systems (wheels/brakes/bushings,
# not just engine) and leans on driving-vs-shop context.
FAULT_PROMPTS = [
    "a car engine idling or revving",
    "a car driving with a wheel bearing humming or roaring",
    "brakes squealing or grinding on a moving car",
    "a suspension or bushing clunking as a car drives over bumps",
    "a cv joint clicking as a car turns",
    "an engine belt squealing or chirping",
    "a metallic knock or tick from a running engine",
    "an exhaust leak or rumble from a running car",
]
TOOL_PROMPTS = [
    "an impact wrench or air ratchet in a workshop",
    "an electric drill or power tool",
    "hammering on metal in a workshop",
    "an angle grinder cutting metal",
    "hand tools clinking and being placed down",
    "a pneumatic air compressor in a shop",
]

# ----------------------------------------------------------------- thresholds
MECH_CONFIRM_MIN = 0.50    # below -> not mechanical
MECH_REJECT_BELOW = 0.25   # below -> reject outright (music/jingle/speech)
L1_CONF_MIN = 0.40
L1_MARGIN_MIN = 0.15
TOOL_MARGIN = 0.25         # tool_mass - fault_mass above this -> l1=shop_tool
# Lesson: CLAP defaults to "normal idle" when unsure (48% of v1 autos). "Normal"
# must be EARNED with a wide margin.
NORMAL_MARGIN = 0.30
# Tool prior is contextual: tools require a mechanic working (speech-heavy
# video). In no-narration compilations, "tool-like" = electric car parts -> review.
SHOP_SPEECH_FRAC = 0.20
MIN_REGION_S = 1.0
MAX_SPEECH_COV = 0.5
SPECTRAL_FLATNESS_MAX = 0.5   # above = wind/static

# ------------------------------------------------------------------ discovery
# Lesson: 60-180s compilation/montage videos are the sweet spot (100% auto purity
# vs 48% for 10-25min talky videos); shop-heavy repair vlogs are the main
# tool-contamination source.
FAULT_QUERIES = [
    "car sounds mechanical problems", "car noises compilation",
    "guess the car noise", "bad wheel bearing sound while driving",
    "engine knocking noise", "brake grinding noise driving",
    "bad cv joint clicking sound", "suspension clunk noise over bumps",
    "serpentine belt squeal", "bad alternator whine", "exhaust leak sound",
    "transmission whine", "power steering pump whine", "bad water pump noise",
    "lifter tick", "turbo whine", "differential whine", "rod knock sound",
    "bad strut noise", "tire noise vs wheel bearing noise",
    # Accessory-drive expansion (these components are acoustically similar
    # and critically underrepresented in the training corpus)
    # Water Pump (Goal: 150-250 clips)
    "water pump bearing failure sound", "water pump noise car",
    "bad water pump whining noise", "water pump squeal car engine",
    "water pump grinding noise diagnosis", "failing water pump sound",
    
    # Idler Pulley (Goal: 150 clips)
    "bad idler pulley noise", "idler pulley bearing noise",
    "squeaking idler pulley sound", "idler pulley squeal",
    "idler pulley grinding noise car", "failing idler pulley sound",
    
    # Belt Tensioner (Goal: 150 clips)
    "belt tensioner noise car", "belt tensioner pulley noise",
    "squeaking belt tensioner sound", "bad belt tensioner rattle",
    "belt tensioner chirp noise", "failing serpentine belt tensioner",
    
    # Alternator Bearings (Goal: 150 clips)
    "alternator bearing noise car", "alternator whining noise",
    "bad alternator grinding sound", "alternator squeal noise",
    "alternator bearing whine engine", "failing alternator sound",
    
    # Power Steering Pump (Goal: 100 clips)
    "power steering pump whining grinding", "power steering groan noise",
    "bad power steering pump sound", "power steering whine cold start",
    "power steering pump squeal turning", "low power steering fluid noise",
    
    # AC Compressor Clutch (Goal: 100 clips)
    "AC compressor clutch noise", "AC compressor bearing noise car",
    "ac compressor squeal engaged", "bad ac clutch grinding noise",
    "ac compressor rattling sound", "ac compressor whine noise",
    
    "front engine whining noise RPM", "accessory belt noise diagnosis",
    
    # Valvetrain & Timing
    "timing chain rattle on startup", "timing belt tensioner noise",
    "cam phaser rattle", "valve clearance tick",
    
    # Bottom End & Engine Internal
    "piston slap sound cold start", "crankshaft bearing knock", 
    "engine misfire sound", "spark plug misfire noise popping",
    
    # Transmission & Clutch (Weak Class)
    "torque converter shudder noise", "transmission slipping sound",
    "throwout bearing squeal clutch", "clutch chatter noise",
    "CVT transmission whine failure", "bad automatic transmission whine",
    "transmission grinding noise shifting", "gearbox whine sound",
    
    # Driveline & Differential (Weak Class)
    "bad u-joint clunk", "center support bearing noise",
    "rear differential howling sound", "driveshaft clunk",
    "differential pinion bearing whine", "differential grinding sound",
    "rear end howl noise", "transfer case whine",
    
    # Suspension & Steering
    "sway bar link clunk", "bad ball joint squeak",
    "control arm bushing clunk", "rack and pinion clunk",
    
    # Brakes (Weak Class)
    "brake grinding noise driving", "metal on metal brake sound",
    "brake pad squeal car", "warped brake rotor noise",
    "brake caliper sticking sound", "squeaking brakes stopping",
    
    # Air, Fuel, & Exhaust
    "vacuum leak hissing noise engine", "intake manifold leak sound",
    "catalytic converter rattle", "cracked exhaust manifold tick",
    "fuel pump whining noise", 
    
    # Fuel & Ignition (Weak Class)
    "engine misfire sound", "spark plug misfire noise popping",
    "bad ignition coil engine sound", "fuel injector ticking noise",
    "engine sputtering sound bad fuel", "rough idle misfire",
    
    # Boundary-Case Resolution (Targeting model confusion points from local eval)
    "bad alternator bearing vs idler pulley noise",
    "ac compressor clutch bearing squeal",
    "serpentine belt chirp vs pulley bearing",
    "lifter tick vs rod knock sound",
    "piston slap vs lifter tick",
    "camshaft phaser noise tick",
    "spun bearing sound vs lifter tap",
    "cupped tire noise vs wheel bearing hum",
    "tire roar sound vs bearing drone",
    "wheel bearing hum vs tire road noise",
]
# Normals: enthusiast videos titled with exact year/make/model (free YMM labels).
NORMAL_QUERIES = [
    "cold start idle sound", "engine idle sound stock",
    "POV night drive no music", "highway driving sound interior",
    "stock exhaust idle rev", "engine sound after oil change smooth",
    
    # Healthy accessory drive sounds
    "healthy engine idle normal sound", "good condition serpentine belt idle",
    "new alternator sound normal", "new water pump idle sound",
    "perfect engine idle sound car", "quiet engine idle normal",
]
DUR_MIN_S, DUR_MAX_S = 30, 1500

# -------------------------------------------------------- L2 (text-only) map
# L2 part candidates come from TEXT (title/transcript/OCR), never audio.
L2_KEYWORDS = {
    "wheel bearing": ["wheel bearing", "hub bearing"],
    "brakes": ["brake"], "cv joint": ["cv joint", "cv axle"],
    "belt": ["belt", "serpentine"], "alternator": ["alternator"],
    "water pump": ["water pump"], "power steering": ["power steering"],
    "suspension": ["suspension", "strut", "shock", "control arm", "sway bar",
                   "bushing", "ball joint"],
    "exhaust": ["exhaust", "muffler"], "transmission": ["transmission", "gearbox"],
    "differential": ["differential"], "turbo": ["turbo"],
    "engine internal": ["lifter", "rod knock", "knock", "misfire", "tick"],
    "fuel pump": ["fuel pump"],
    "ac compressor": ["a/c", "ac compressor", "air conditioning"],
    "tires": ["tire", "tyre"],
}

# --------------------------------------------------------------- model names
CLAP_MODEL = "laion/clap-htsat-unfused"
HAIKU_MODEL = "claude-haiku-4-5"      # routine LLM work
SONNET_MODEL = "claude-sonnet-4-6"    # multi-signal cause fusion / diagnostic reasoning
OLLAMA_MODEL = "qwen2.5:7b-instruct"  # local $0 backend
