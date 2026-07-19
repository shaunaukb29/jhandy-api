"""Component-level automotive knowledge graph.

Maps diagnostic subsystems to specific mechanical components with Bayesian priors,
evidence boosters, ordered test sequences, and follow-up question pools.

Usage:
    from cardiag.inference.components import SUBSYSTEM_COMPONENTS, FOLLOWUP_QUESTIONS
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComponentDef:
    """A specific mechanical component within a diagnostic subsystem."""
    name: str
    prior: float
    boosting_codes: list[str] = field(default_factory=list)
    boosting_symptoms: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    subsystem: str = ""
    parent_group: str = ""
    severity: str = "moderate"
    driveability: str = "monitor"
    acoustic: bool = True
    typical_sounds: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    related_components: list[str] = field(default_factory=list)


@dataclass
class FollowupQuestion:
    """A targeted yes/no question that updates the component distribution."""
    id: str
    text: str
    options: list[str]
    yes_multipliers: dict[str, float]
    no_multipliers: dict[str, float]
    # Which top component names make this question relevant (empty = always eligible)
    target_components: list[str] = field(default_factory=list)


SUBSYSTEM_COMPONENTS: dict[str, list[ComponentDef]] = {

    "engine_internal": [
        ComponentDef(
            name="VVT / variable valve timing actuator",
            prior=0.18,  # Lowered: VVT is cold-start-only; shouldn't dominate warm/persistent knocks
            boosting_codes=["P0010", "P0011", "P0012", "P0013", "P0014",
                            "P0020", "P0021", "P0022", "P0023", "P0024"],
            boosting_symptoms=["vvt", "vanos", "variable valve", "camshaft actuator",
                               "phaser rattle", "vvti", "variable timing",
                               "cold start rattle", "rattle fades as it warms up",
                               "rattles then quiets down", "noise for a few seconds then gone",
                               "morning rattle", "cold morning noise",
                               "engine rattle when first started"],
            tests=[
                "Check engine oil level and condition — VVT systems are oil-pressure driven",
                "Monitor live camshaft timing offset values with a scan tool at idle",
                "Inspect VVT solenoid oil control valve screens for debris (common on high-mileage engines)",
                "Listen near the camshaft cover with a stethoscope at cold idle",
                "Clear active timing diagnostic codes (if any) and verify if they return within 3–5 cold start cycles",
            ],
        ),
        ComponentDef(
            name="Timing chain tensioner",
            prior=0.22,
            boosting_codes=["P0016", "P0017", "P0018", "P0019"],
            boosting_symptoms=["chain rattle", "rattle", "cold", "cold start", "startup",
                               "first start", "morning", "timing chain", "chain", "rattling on start", "clatter on startup", "rattle fades as it warms up", "noise for a few seconds then gone", "diesel-like clatter", "marbles rattling", "rattle when first starting", "brief rattle at startup", "morning noise", "goes away after a few seconds"],
            tests=[
                "Listen for metallic rattle from the front of the engine on cold start",
                "Check if rattle disappears above 1,500 RPM (oil pressure restores tensioner tension)",
                "Verify oil pressure at idle with a mechanical gauge — low pressure starves tensioner",
                "Check cam-to-crank correlation codes (if active) for chain stretch",
            ],
        ),
        ComponentDef(
            name="Timing chain / chain guides",
            prior=0.14,
            boosting_codes=["P0016", "P0017"],
            boosting_symptoms=["chain", "rattle", "timing", "slack", "worn chain", "chain noise", "loose chain", "engine rattle", "clatter", "guide wear", "plastic guide failure", "metallic rattle from front of engine"],
            tests=[
                "Check cam-crank correlation fault patterns (if active) using a scan tool",
                "Inspect timing cover area for oil seepage (worn guides shed plastic debris)",
                "Check magnetic drain plug or oil filter for metallic particles",
                "Measure chain stretch using cam-crank offset live data",
            ],
        ),
        ComponentDef(
            name="Low oil pressure / oil pump",
            prior=0.16,
            boosting_codes=["P0520", "P0521", "P0522", "P0523"],
            boosting_symptoms=["oil pressure", "low oil", "oil light", "tick", "ticking",
                               "lifter noise", "valvetrain", "oil pressure warning", "cold start", "startup", "warm up", "warms up", "goes away", "cold", "grinding", "grinding noise", "noise on startup that disappears", "loud on cold start", "clears up once warm", "thick oil noise", "startup noise then quiet", "oil light flicker", "pressure warning at idle"],
            tests=[
                "Check oil level immediately on the dipstick",
                "Test oil pressure with a mechanical gauge at idle and 2,000 RPM",
                "Inspect oil pressure sensor accuracy (compare mechanical vs. sensor reading)",
                "Check for sludge inside the engine (remove oil filler cap and inspect)",
            ],
        ),
        ComponentDef(
            name="Hydraulic valve lifter",
            prior=0.12,
            boosting_codes=["P0300"],
            boosting_symptoms=["tick", "ticking", "lifter", "valve", "tap", "tapping",
                               "valvetrain", "clicking", "ticking noise", "engine tick", "top end noise", "noisy lifters", "clicking at idle", "quiets down at higher rpm", "sewing machine noise"],
            tests=[
                "Listen at the top of the engine with a mechanic's stethoscope — lifter tick is rapid and rhythmic",
                "Check if tick is present at idle but quiets at 2,000+ RPM (pressure restores lifter)",
                "Verify tick is RPM-synchronised (lifters run at cam speed = half engine RPM)",
                "Check oil viscosity — overly thin oil can cause lifter bleed-down",
                "Inspect rocker arms and camshaft lobes through the rocker cover",
            ],
        ),
        ComponentDef(
            name="Connecting rod bearing / bottom-end knock",
            prior=0.22,  # Raised: rod knock is a common and serious fault for persistent under-load engine knock
            boosting_codes=["P0011"],
            boosting_symptoms=["rod knock", "engine knock", "knocking noise", "deep knock",
                               "heavy knock", "metallic knock", "knock", "knocking",
                               "knock gets worse with rpm", "worse under acceleration",
                               "knock under load", "knock follows rpm", "knock at higher rpm",
                               "loud knock", "knock from bottom", "bottom end knock",
                               "knock persists", "knock when warm", "bearing", "big end",
                               "under load", "accelerat", "deep", "heavy"],
            tests=[
                "Avoid high RPM/load to prevent catastrophic engine failure",
                "Check oil level and pressure immediately",
                "Inspect oil filter and drain pan for bearing material (metal flakes)",
                "Listen from underneath the oil pan — rod knock is loudest from the bottom block",
            ],
        ),
        ComponentDef(
            name="Piston slap",
            prior=0.08,
            boosting_codes=[],
            boosting_symptoms=["piston slap", "cold only", "disappears when warm",
                               "bottom of engine", "deep knock", "slap", "cold engine noise", "slapping sound", "engine noise when cold", "goes away when warm", "hollow knock", "cylinder wall noise"],
            tests=[
                "Listen from underneath — piston slap is a deep, hollow sound lower in the block than valvetrain noise",
                "Check if noise stops completely when the engine reaches operating temperature",
                "Deactivate one cylinder at a time (pull injector fuse) to identify which piston",
                "Measure compression — worn cylinder bore causes slap plus low compression",
            ],
        ),
        ComponentDef(
            name="Main bearing",
            prior=0.16,  # Raised: main bearing failure is serious and common in high-mileage/low-oil engines
            boosting_codes=["P0520", "P0521"],
            boosting_symptoms=["main bearing", "deep knock", "heavy knock", "low oil pressure",
                               "knock under load", "thud", "bottom end", "loud knock",
                               "rumble under load", "engine rumble", "deep rumble",
                               "knock gets worse under load", "low frequency knock",
                               "knock follows rpm", "knock persists warm", "knock when warm",
                               "bearing", "knock", "knocking"],
            tests=[
                "Check oil pressure mechanically — main bearing failure is almost always accompanied by low oil pressure",
                "Listen from beneath the oil pan — main bearing knock is slower and lower-pitched than rod knock",
                "Inspect oil and oil filter for bearing material (silvery metallic particles)",
                "Do a cylinder drop test — main bearing knock does not change pitch when cylinders are killed one at a time",
            ],
        ),
        ComponentDef(
            name="Wrist pin (piston pin)",
            prior=0.06,
            boosting_codes=[],
            boosting_symptoms=["double knock", "wrist pin", "piston pin", "knock on deceleration",
                               "double tap", "knocking on deceleration", "clacking noise", "piston noise", "knock when letting off gas"],
            tests=[
                "Listen for a characteristic double-knock on engine deceleration (the wrist pin pattern)",
                "Deactivate cylinders one at a time — wrist pin knock disappears when the affected cylinder is killed",
                "Check piston pin clearance using micrometer and bore gauge after disassembly",
            ],
        ),
        ComponentDef(
            name="Cam phaser / variable cam timing solenoid",
            prior=0.09,
            boosting_codes=["P0010", "P0011", "P0012", "P0013", "P0014",
                             "P0020", "P0021", "P0022", "P0023", "P0024"],
            boosting_symptoms=["cam phaser", "phaser", "vanos", "timing rattle", "cold rattle",
                               "tick on startup", "cold start tick", "rattle on cold start", "startup rattle", "morning rattle", "clatter that goes away", "cold clatter", "variable cam noise"],
            tests=[
                "Read cam timing offset values live — phaser stuck or slow to respond indicates solenoid or oil-flow fault",
                "Inspect VVT solenoid screens for debris — blocked screens starve the phaser of oil",
                "Test solenoid resistance and duty-cycle response with a scan tool",
                "Check oil viscosity — overly thick oil (wrong spec or overdue change) delays phaser response on cold starts",
            ],
        ),
        ComponentDef(
            name="Harmonic balancer / crankshaft damper",
            prior=0.07,
            boosting_codes=[],
            boosting_symptoms=["harmonic balancer", "crankshaft damper", "front of engine wobble",
                               "vibration at idle", "belt wobble", "damper", "vibration", "shaking at idle", "wobble", "front pulley noise", "belt squeal from wobble"],
            tests=[
                "Inspect the front of the crankshaft snout for visible wobble while idling",
                "Check the rubber bonding ring — a separated damper causes the outer ring to orbit eccentrically",
                "Look for shiny marks on the accessory belt from a wobbling balancer",
                "Replacement is the only repair if the damper is delaminated",
            ],
        ),
        ComponentDef(
            name="Dual-mass flywheel",
            prior=0.07,
            boosting_codes=[],
            boosting_symptoms=["dual mass flywheel", "dmf", "shudder at idle", "rattle at idle",
                               "rattle in neutral", "clunk on load", "diesel rattle", "rattle in neutral at idle", "clutch rattle", "gearbox rattle", "vibration through pedal"],
            tests=[
                "Check for rattle in neutral with clutch engaged that disappears when clutch is depressed",
                "Measure flywheel rotational play — excessive movement (>15 mm on the ring gear) indicates wear",
                "Inspect for oil contamination at the flywheel/clutch interface",
            ],
        ),
    ],

    "fuel_ignition": [
        ComponentDef(
            name="Spark plugs",
            prior=0.30,
            boosting_codes=["P0300", "P0301", "P0302", "P0303", "P0304",
                            "P0305", "P0306", "P0307", "P0308"],
            boosting_symptoms=["misfire", "rough idle", "rough", "stutter", "hesitation",
                               "spark", "ignition", "plug", "cylinder", "engine shaking", "check engine light", "bucking", "jerking", "rough running", "poor acceleration", "loss of power"],
            tests=[
                "Remove and inspect spark plugs — check gap, electrode condition, and deposits",
                "Note if P030x specifies a cylinder — replace that plug first",
                "Test spark plug boot/wire insulation for cracks and tracking marks",
                "Compression test on the misfiring cylinder to rule out mechanical cause",
            ],
        ),
        ComponentDef(
            name="Ignition coil",
            prior=0.25,
            boosting_codes=["P0350", "P0351", "P0352", "P0353", "P0354",
                            "P0355", "P0356", "P0357", "P0358"],
            boosting_symptoms=["misfire", "stutter", "hesitation", "coil", "ignition", "cylinder", "check engine light", "engine shaking", "bucking", "loss of power", "jerking on acceleration", "intermittent misfire"],
            tests=[
                "Swap the suspected coil with an adjacent cylinder — if misfire follows the coil, it's faulty",
                "Measure primary resistance (typically 0.5–2 Ω) and secondary resistance",
                "Inspect coil connector for pushed-back pins, corrosion, or moisture",
            ],
        ),
        ComponentDef(
            name="Fuel injector",
            prior=0.22,
            boosting_codes=["P0200", "P0201", "P0202", "P0203", "P0204",
                            "P0205", "P0206", "P0207", "P0208"],
            boosting_symptoms=["rough", "stutter", "fuel", "injector", "lean", "rich", "misfire", "ticking noise from engine", "check engine light", "poor mileage", "smell of fuel", "clicking injector"],
            tests=[
                "Run injector balance test with scan tool — compare per-cylinder fuel trim",
                "Listen to each injector with a stethoscope — all should click at the same rate and volume",
                "Test injector resistance (high-impedance: 11–17 Ω; low-impedance: 1–5 Ω)",
            ],
        ),
        ComponentDef(
            name="Fuel pump / fuel pressure",
            prior=0.15,
            boosting_codes=["P0087", "P0088", "P0190", "P0191"],
            boosting_symptoms=["hesitation", "stutter", "fuel pressure", "pump",
                               "hard start", "won't start", "cuts out", "surging", "sputtering", "loses power on hills", "dies at idle", "whining from tank", "hard to start when hot", "engine dies while driving"],
            tests=[
                "Measure fuel rail pressure at idle and under snap throttle",
                "Check pressure drop after engine off — slow drop indicates failing check valve",
                "Listen for fuel pump hum in tank when ignition is switched to ON (before cranking)",
                "Check fuel filter restriction if serviceable",
            ],
        ),
        ComponentDef(
            name="MAF / MAP sensor",
            prior=0.08, acoustic=False,
            boosting_codes=["P0100", "P0101", "P0102", "P0103", "P0106", "P0107"],
            boosting_symptoms=["rough idle", "hesitation", "lean", "stall", "air", "maf", "check engine light", "poor fuel economy", "hesitation on acceleration", "surging idle", "black smoke"],
            tests=[
                "Check MAF reading at idle and WOT against reference spec for your engine",
                "Clean MAF sensor with MAF cleaner — common fix on high-mileage engines",
                "Check for air leaks between MAF and throttle body (cracked intake hose)",
            ],
        ),
        ComponentDef(
            name="EGR valve (stuck open)",
            prior=0.07, acoustic=False,
            boosting_codes=["P0400", "P0401", "P0402", "P0403", "P0404", "P0405"],
            boosting_symptoms=["rough idle", "stall at idle", "egr", "lean at idle", "idle surge", "check engine light", "rough idle when cold", "engine stalls at stop"],
            tests=[
                "Command EGR fully closed with scan tool — rough idle that clears confirms stuck-open EGR",
                "Inspect EGR valve and passages for heavy carbon buildup preventing full closure",
                "Check EGR position sensor for correct range and response",
            ],
        ),
        ComponentDef(
            name="Throttle body",
            prior=0.05,
            boosting_codes=["P0120", "P0121", "P0122", "P0123", "P2111", "P2112"],
            boosting_symptoms=["surging idle", "hunting idle", "throttle", "idle hunts", "stall",
                               "hesitation from stop", "check engine light", "rpm fluctuates", "idle surges up and down", "delayed response to gas pedal"],
            tests=[
                "Check for carbon buildup on the throttle plate and bore (common at 60–80k miles)",
                "Verify TPS output with scan tool — should move smoothly from 0 to WOT",
                "Perform throttle body relearn procedure after cleaning",
            ],
        ),
        ComponentDef(
            name="PCV valve / breather",
            prior=0.04,
            boosting_codes=["P0171", "P0174"],
            boosting_symptoms=["rough idle", "oil consumption", "blue smoke", "pcv",
                               "oil in intake", "hissing at idle", "whistling noise", "sucking sound under hood", "oil smell", "check engine light"],
            tests=[
                "Remove PCV valve and shake — a working valve rattles; a stuck valve does not",
                "Check for heavy oil in the intake manifold or intercooler",
                "Look for crankcase pressure blowing oil out of the dipstick tube or filler cap",
            ],
        ),
    ],

    "exhaust": [
        ComponentDef(
            name="Exhaust manifold gasket",
            prior=0.32,
            boosting_codes=[],
            boosting_symptoms=["tick", "ticking", "hiss", "exhaust", "cold", "worse when cold",
                               "manifold", "blowing", "puffing", "ticking noise on cold start", "tapping noise", "louder when cold", "quiets down when warm", "popping noise"],
            tests=[
                "Listen at the manifold when engine is cold — gasket leaks tick louder when cold (metal contracts)",
                "Look and feel for black soot deposits around manifold joints",
                "Run finger carefully near manifold face — feel for exhaust pulse",
                "Use cold water spray near manifold while engine runs — sizzle/steam marks the leak",
            ],
        ),
        ComponentDef(
            name="Catalytic converter",
            prior=0.22, acoustic=False,
            boosting_codes=["P0420", "P0421", "P0430", "P0431"],
            boosting_symptoms=["rattle", "cat", "catalytic", "exhaust rattle", "heat shield", "sulfur", "rattle when cold", "shaking rattle", "sulfur smell", "rotten egg smell", "loss of power", "check engine light"],
            tests=[
                "Tap the converter with a rubber mallet — loose internal substrate produces a hollow rattle",
                "Check if rattle is louder when cold and diminishes at operating temperature",
                "Compare upstream vs downstream O2 sensor activity — minimal downstream switching = dead converter",
                "Inspect heat shield attachment points for loose bolts or cracks",
            ],
        ),
        ComponentDef(
            name="Flex pipe / exhaust joint",
            prior=0.20,
            boosting_codes=[],
            boosting_symptoms=["hiss", "exhaust smell", "pipe", "flex", "joint", "connection",
                               "underneath", "exhaust smell inside", "loud exhaust", "raspy exhaust note", "exhaust leak sound", "louder under acceleration"],
            tests=[
                "Run engine and listen for hissing from the flex section (usually behind manifold or before cat)",
                "Look for rust-through, cracks, or separation at pipe joints",
                "Feel for exhaust pulse at suspect leak points with the back of your hand",
            ],
        ),
        ComponentDef(
            name="Oxygen sensor",
            prior=0.15, acoustic=False,
            boosting_codes=["P0130", "P0131", "P0132", "P0133", "P0134",
                            "P0135", "P0136", "P0137", "P0138", "P0140"],
            boosting_symptoms=["fuel economy", "lean", "rich", "lambda", "o2 sensor", "check engine light", "poor fuel economy", "rough idle", "failed emissions test"],
            tests=[
                "Check O2 sensor live data — upstream sensor should oscillate 0.1–0.9V",
                "Test heater resistance (typically 5–20 Ω depending on sensor)",
                "Check for exhaust leaks near the sensor (contaminate readings and cause rich/lean codes)",
            ],
        ),
        ComponentDef(
            name="EGR valve",
            prior=0.11, acoustic=False,
            boosting_codes=["P0400", "P0401", "P0402", "P0403", "P0404", "P0405"],
            boosting_symptoms=["hiss", "rough idle", "egr", "vacuum", "stall at idle", "check engine light", "stalling at idle"],
            tests=[
                "Command EGR valve open with scan tool — idle should roughen if EGR flow is present",
                "Inspect EGR passages and valve seat for heavy carbon buildup",
                "Check EGR position sensor signal for correct range",
            ],
        ),
    ],

    "belt": [
        ComponentDef(
            name="Belt tensioner / idler pulley bearing",
            prior=0.35,
            boosting_codes=[],
            boosting_symptoms=["squeal", "chirp", "squealing", "belt", "tensioner", "pulley",
                               "intermittent", "chirping", "squealing on startup", "chirping at idle", "belt noise when cold", "squeak that comes and goes"],
            tests=[
                "Check tensioner arm sweep — arm should move freely and snap back firmly",
                "Spin each idler pulley by hand — bearing should feel perfectly smooth with no drag or roughness",
                "Spray a short burst of water on the belt while engine runs — squeal reducing = belt slip; no change = bearing",
                "Check all pulley alignments with a straightedge across belt-driven accessories",
            ],
        ),
        ComponentDef(
            name="Serpentine / drive belt",
            prior=0.28,
            boosting_codes=[],
            boosting_symptoms=["squeal", "chirp", "worn belt", "cracked belt", "belt squeal",
                               "glazed belt", "squeal on startup", "squeal in the rain", "belt slipping noise", "chirping under load"],
            tests=[
                "Inspect belt surface for cracking, glazing, or missing ribs",
                "Check belt tension (should deflect ~10 mm under firm thumb pressure mid-span)",
                "Look for oil contamination on the belt (causes slipping and squealing)",
            ],
        ),
        ComponentDef(
            name="Power steering pump",
            prior=0.18,
            boosting_codes=["P0562"],
            boosting_symptoms=["whine", "groan", "power steering", "squeal turning",
                               "turning", "lock to lock", "squeal when turning", "moaning while parking", "whine at idle", "noise turning wheel"],
            tests=[
                "Check PS fluid level and condition",
                "Listen for whine that increases when turning to full lock",
                "Check for aeration (froth) in PS fluid reservoir — indicates air in system",
            ],
        ),
        ComponentDef(
            name="AC compressor clutch",
            prior=0.19,
            boosting_codes=["P0645", "P0646", "P0647"],
            boosting_symptoms=["squeal", "clicking", "clunk", "ac", "air conditioning",
                               "ac on", "compressor", "clunk when ac turns on", "squeal with ac on", "chirping when ac engages"],
            tests=[
                "Turn AC off — if noise disappears immediately, compressor or clutch is the source",
                "Check AC clutch air gap (typically 0.4–0.8 mm)",
                "Inspect clutch plate and rotor for oil contamination or wear",
            ],
        ),
    ],

    "brakes": [
        ComponentDef(
            name="Brake pad wear indicators",
            prior=0.35,
            boosting_codes=["P0571"],
            boosting_symptoms=["squeal", "squealing", "squeak", "metal on metal",
                               "when braking", "brake squeal", "brake noise", "squeaking when braking", "squeal that stops when braking harder", "high pitched squeal", "annoying squeak"],
            tests=[
                "Visually inspect pad thickness through the wheel — measure remaining material",
                "Check if squeal stops when brake pedal is lightly applied (wear-indicator squeal does)",
                "Inspect rotor surface for scoring or heavy corrosion",
            ],
        ),
        ComponentDef(
            name="Wheel bearing",
            prior=0.28,
            boosting_codes=["C0035", "C0040", "C0045", "C0050"],
            boosting_symptoms=["rumble", "grinding", "hum", "speed dependent", "bearing",
                               "changes when turning", "highway noise", "drone", "roaring noise", "humming that gets louder with speed", "noise changes when turning", "growling from wheel"],
            tests=[
                "Swerve gently at low speed — bearing noise changes pitch when weight shifts left/right",
                "Jack up each corner and spin the wheel by hand — rough or loose is abnormal",
                "Check for wheel play by gripping at 12 and 6 o'clock and rocking",
                "Check ABS reluctor ring / tone ring for damage on the affected corner",
            ],
        ),
        ComponentDef(
            name="Brake rotor",
            prior=0.20,
            boosting_codes=[],
            boosting_symptoms=["grinding", "pulsation", "vibration", "pedal pulsation",
                               "warped", "rotor", "shudder when braking", "shaking when braking", "steering wheel shakes", "pedal pulsates", "vibration when stopping"],
            tests=[
                "Measure rotor thickness with a micrometer — compare against minimum specification",
                "Check rotor runout with a dial indicator (typically max 0.05 mm)",
                "Inspect rotor face for deep scoring, cracks, or heat bluing",
            ],
        ),
        ComponentDef(
            name="Brake caliper (seized)",
            prior=0.12,
            boosting_codes=[],
            boosting_symptoms=["grinding", "pulling", "dragging", "one side",
                               "hot wheel", "one wheel braking", "car pulls to one side", "burning smell", "one wheel hot", "grinding while driving"],
            tests=[
                "Drive and check if one wheel rim is significantly hotter than the opposite side",
                "Inspect caliper slide pins — should push in and spring back freely",
                "Check if piston can be retracted with a C-clamp — seized pistons require caliper replacement",
            ],
        ),
        ComponentDef(
            name="ABS wheel speed sensor",
            prior=0.05, acoustic=False,
            boosting_codes=["C0035", "C0040", "C0045", "C0050", "C0060"],
            boosting_symptoms=["abs light", "abs", "abs warning", "traction control", "abs light on", "traction control light on"],
            tests=[
                "Read ABS-specific codes with a compatible scan tool (generic OBD readers often miss C-codes)",
                "Check wheel speed sensor wiring and connector at the affected corner",
                "Inspect reluctor ring for cracks or missing teeth",
            ],
        ),
    ],

    "transmission": [
        ComponentDef(
            name="Transmission fluid",
            prior=0.28, acoustic=False,
            boosting_codes=["P0700", "P0730"],
            boosting_symptoms=["slip", "slipping", "shudder", "shift", "delayed shift",
                               "harsh shift", "fluid", "transmission", "hard shifting", "clunk between gears", "delayed engagement", "jerky shifting"],
            tests=[
                "Check fluid level and condition — should be red, not brown, burnt, or black",
                "Note shift quality after fluid and filter change — erratic shifts often improve",
                "Check for stored TCM codes with an enhanced or dealer-level scan tool",
            ],
        ),
        ComponentDef(
            name="Torque converter",
            prior=0.22,
            boosting_codes=["P0740", "P0741", "P0742", "P0743"],
            boosting_symptoms=["shudder", "vibration", "converter", "lockup", "40 mph", "50 mph",
                               "light throttle", "shudder at highway speed", "vibration around 45 mph", "shaking cruising", "feels like driving on rumble strip"],
            tests=[
                "Check for shudder at 40–55 mph under light steady throttle (TCC lockup shudder)",
                "Verify TCC engagement/disengagement with scan tool live data",
                "Check if shudder clears by adding a TCC friction modifier to the transmission fluid",
            ],
        ),
        ComponentDef(
            name="Shift solenoid",
            prior=0.20, acoustic=False,
            boosting_codes=["P0750", "P0751", "P0752", "P0755", "P0756",
                            "P0760", "P0765", "P0770"],
            boosting_symptoms=["harsh shift", "stuck gear", "no upshift", "no downshift",
                               "wrong gear", "transmission", "stuck in gear", "won't shift", "delayed shift", "gets stuck in one gear", "check engine light with shifting problem"],
            tests=[
                "Read solenoid-specific fault codes with dealer-level scan tool",
                "Measure solenoid resistance (typically 10–30 Ω depending on type)",
                "Verify line pressure at the affected solenoid bore with a pressure gauge",
            ],
        ),
        ComponentDef(
            name="Input / output shaft bearing",
            prior=0.17,
            boosting_codes=[],
            boosting_symptoms=["whine", "rumble", "noise increases with speed", "bearing",
                               "transmission whine", "whining that changes with speed", "growl from transmission", "howling noise"],
            tests=[
                "Confirm noise changes with road speed, not engine RPM (transmission bearing pattern)",
                "Drain transmission fluid and inspect magnetic plug for metallic particles",
                "Check fluid level — low fluid accelerates bearing wear",
            ],
        ),
        ComponentDef(
            name="Clutch / friction plates",
            prior=0.13,
            boosting_codes=["P0730"],
            boosting_symptoms=["slip", "clutch", "grinding", "manual", "gear change",
                               "clutch slip", "cvt", "flare", "burning smell", "engine revs without moving", "slipping clutch", "gears feel loose"],
            tests=[
                "Clutch slip test: engage 3rd gear at low speed, increase throttle — RPM rising without vehicle acceleration = slip",
                "Check clutch pedal free-play (manual: typically 15–25 mm at pedal)",
                "Inspect clutch fluid reservoir level and condition",
            ],
        ),
    ],

    "power_steering": [
        ComponentDef(
            name="Power steering pump",
            prior=0.40,
            boosting_codes=["P0562"],
            boosting_symptoms=["whine", "groan", "moan", "power steering", "turning",
                               "full lock", "noisy steering", "steering whine", "moaning noise turning", "whine when turning wheel", "groan on full lock", "screech when turning"],
            tests=[
                "Check PS fluid level and condition in the reservoir",
                "Listen for whine that intensifies when turning to full lock",
                "Check for aeration (frothy fluid) — indicates air in the system",
                "Measure pump pressure with a gauge (most pumps: 1,000–1,500 psi at idle)",
            ],
        ),
        ComponentDef(
            name="Steering rack / gearbox",
            prior=0.30,
            boosting_codes=[],
            boosting_symptoms=["clunk", "knock", "vibration", "wander", "play",
                               "rack", "knocking over bumps", "clunk over bumps while turning", "knock in steering", "loose steering feel"],
            tests=[
                "Check inner tie rod play — reach under car, grab inner tie rod, feel for knock",
                "Check for PS fluid leaks at rack bellows or end seals",
                "Check both tie rod ends for play",
            ],
        ),
        ComponentDef(
            name="Low PS fluid / air in system",
            prior=0.18,
            boosting_codes=[],
            boosting_symptoms=["whine", "groan", "intermittent", "fluid low", "bubbles", "whine that comes and goes", "noise only sometimes", "growling steering"],
            tests=[
                "Check and top up PS fluid to max line",
                "Bleed the system: turn steering lock-to-lock 10 times with front wheels off the ground",
                "Replace fluid if dark or contaminated",
            ],
        ),
        ComponentDef(
            name="EPS motor / control module",
            prior=0.12, acoustic=False,
            boosting_codes=["C0460", "C0475", "P0562"],
            boosting_symptoms=["eps", "electric steering", "steering warning light",
                               "heavy steering", "no power assist", "steering feels heavy", "assist cuts out"],
            tests=[
                "Read EPS-specific fault codes with a compatible scan tool",
                "Perform steering angle sensor calibration procedure",
                "Check EPS supply voltage under load — low voltage causes loss of assist",
            ],
        ),
    ],

    "low_oil": [
        ComponentDef(
            name="Low engine oil level",
            prior=0.40, acoustic=False,
            boosting_codes=["P0521"],
            boosting_symptoms=["oil", "low oil", "oil light", "level", "tick", "tapping",
                               "oil pressure warning", "oil warning light", "burning oil smell", "low oil light flickers", "ticking that low oil would cause"],
            tests=[
                "Check dipstick immediately — do not run the engine if oil is at minimum or below",
                "Look for oil puddles under the car or visible external leaks",
                "Check for blue exhaust smoke (oil consumption / burning)",
                "Track oil consumption rate over 1,000 miles if level is marginal",
            ],
        ),
        ComponentDef(
            name="Oil pressure sensor (faulty reading)",
            prior=0.25, acoustic=False,
            boosting_codes=["P0520", "P0521", "P0522", "P0523"],
            boosting_symptoms=["warning light only", "oil pressure warning", "sensor",
                               "intermittent warning", "light comes on then off", "warning light flickers", "gauge reads zero"],
            tests=[
                "Rule out actual low oil first — check dipstick before anything else",
                "Confirm actual oil pressure with a mechanical gauge at the sender port",
                "Check sensor wiring and connector for shorts or open circuits",
            ],
        ),
        ComponentDef(
            name="Oil pump",
            prior=0.20,
            boosting_codes=["P0520"],
            boosting_symptoms=["oil pressure low", "pump", "tick", "knock", "confirmed low pressure", "cold start", "startup", "warm up", "grinding on startup", "noise that goes away after warming up", "startup noise then quiet", "loud on cold start", "clears up once warm"],
            tests=[
                "Measure oil pressure mechanically at idle and 2,000 RPM — compare to spec",
                "Check oil pump pressure relief valve for sticking open",
                "Inspect oil pump pickup tube for clogging or cracked O-ring",
            ],
        ),
        ComponentDef(
            name="Oil pickup tube / sump strainer",
            prior=0.15,
            boosting_codes=[],
            boosting_symptoms=["sludge", "oil pressure drops under load", "oil pickup",
                               "cold pressure ok warm pressure drops", "pressure drops on hard turns", "light flickers cornering", "warning light under braking"],
            tests=[
                "Check engine oil for sludge (remove filler cap, inspect with flashlight)",
                "Test oil pressure under load — pickup issues worsen with acceleration",
                "Consider oil pan removal if sludge contamination is suspected",
            ],
        ),
    ],

    "drivetrain": [
        ComponentDef(
            name="CV joint (outer)",
            prior=0.32,
            boosting_codes=[],
            boosting_symptoms=["clicking when turning", "clicking on turns", "clicking under load",
                               "clicking accelerating", "cv", "cv joint", "clicking corner", "clicking noise turning", "popping when turning sharp", "clicking in parking lot"],
            tests=[
                "Drive in tight circles at low speed — outer CV clicking is loudest when turning hard under throttle",
                "Inspect CV boot for splits or missing grease — a torn boot destroys the joint within weeks",
                "Lift the car and rotate the driveshaft by hand while feeling for clicks or binding",
            ],
        ),
        ComponentDef(
            name="CV joint (inner / tripod)",
            prior=0.20,
            boosting_codes=[],
            boosting_symptoms=["clunk on acceleration", "clunk on deceleration", "thud under load",
                               "inner cv", "plunge", "shudder accelerating", "clunk shifting from drive to reverse", "thud accelerating", "clunking under load"],
            tests=[
                "Check for clunk when transitioning from acceleration to deceleration (coast)",
                "Inspect inner CV/tripod boot for grease contamination or tearing",
                "Check inboard joint for excessive plunge travel",
            ],
        ),
        ComponentDef(
            name="Driveshaft / propshaft U-joint",
            prior=0.22,
            boosting_codes=[],
            boosting_symptoms=["vibration under acceleration", "clunk at takeoff", "u-joint",
                               "propshaft", "driveshaft", "vibration at speed", "clunk taking off from stop", "vibration that gets worse with speed"],
            tests=[
                "Grip the driveshaft and twist — any rotational play greater than a few degrees indicates worn U-joints",
                "Inspect U-joint bearing caps for rust, missing grease, or discolouration",
                "Check driveshaft balance by looking for missing balance weights",
            ],
        ),
        ComponentDef(
            name="Differential bearing",
            prior=0.16,
            boosting_codes=[],
            boosting_symptoms=["rear whine", "differential whine", "speed dependent rear noise",
                               "differential", "rear axle noise", "whine at highway", "whine at highway speed", "howling from rear", "gets louder with speed"],
            tests=[
                "Confirm noise tracks vehicle speed not engine RPM",
                "Check differential fluid level and condition (dark or metallic = bearing wear)",
                "Drain fluid and inspect magnetic plug for metallic particles",
            ],
        ),
        ComponentDef(
            name="Transfer case",
            prior=0.10,
            boosting_codes=[],
            boosting_symptoms=["4wd noise", "awd noise", "transfer case", "binding",
                               "4x4", "all wheel drive noise", "clunk changing 4wd", "grinding shifting 4wd", "binding in turns", "noise in 4wd only"],
            tests=[
                "Check transfer case fluid level and condition",
                "Test 2WD vs 4WD engagement — noise changing mode indicates internal fault",
                "Verify front/rear driveshaft runout and U-joint condition before condemning the case",
            ],
        ),
    ],

    "suspension": [
        ComponentDef(
            name="Strut / shock absorber",
            prior=0.28,
            boosting_codes=[],
            boosting_symptoms=["clunk over bumps", "clunk on bumps", "thud over bumps",
                               "bouncy", "knocking over bumps", "shock", "strut", "clunk hitting a pothole", "bouncy ride", "knock over speed bumps"],
            tests=[
                "Bounce test: push down hard on each corner — should rebound once and stop",
                "Inspect strut mount / top bearing for wear and play",
                "Check damper body for external oil leaks (oil streak = damper failure)",
                "Listen for the exact clunk with a helper — confirms suspension vs other source",
            ],
        ),
        ComponentDef(
            name="Control arm bushing",
            prior=0.25,
            boosting_codes=[],
            boosting_symptoms=["clunk turning", "clunk over bumps", "clunk when braking",
                               "bushing", "control arm", "vague steering", "clunk over bumps while turning", "vague steering feel", "knocking sound cornering"],
            tests=[
                "Lift each corner and use a pry bar to lever the control arm up/down — any movement is abnormal",
                "Visually inspect rubber bushings for cracking, tearing, or displaced rubber",
                "Have a helper rock the steering while you feel the control arm joints",
            ],
        ),
        ComponentDef(
            name="Ball joint",
            prior=0.22,
            boosting_codes=[],
            boosting_symptoms=["clunk turning", "play in steering", "steering wander",
                               "ball joint", "clunk on steering input", "clunk turning at low speed", "knocking in front end", "loose feeling steering"],
            tests=[
                "Jack under the lower control arm (not the chassis) to unload the ball joint, then check for play",
                "Check for axial and radial play using a dial indicator",
                "A worn ball joint is a serious safety item — do not delay replacement",
            ],
        ),
        ComponentDef(
            name="Sway bar link / end link",
            prior=0.18,
            boosting_codes=[],
            boosting_symptoms=["clunk over bumps", "rattle over bumps", "sway bar",
                               "anti-roll", "end link", "rattle corner", "rattle over rough roads", "clunk on uneven pavement", "rattling front end"],
            tests=[
                "Lift the car and check sway bar end links for play by hand — they should be firm",
                "Look for cracked or collapsed end link bushings",
                "Disconnect the end link and re-check — clunk disappearing confirms sway bar link",
            ],
        ),
        ComponentDef(
            name="Strut mount / top bearing",
            prior=0.07,
            boosting_codes=[],
            boosting_symptoms=["clunk when turning", "clunk at full lock", "steering clunk",
                               "strut top", "mount bearing", "clunk turning the wheel while parked", "creak turning the wheel"],
            tests=[
                "Turn steering lock-to-lock with car stationary — click/clunk from top of strut = worn mount bearing",
                "Inspect mount plate for cracking or crushing",
                "Check bearing plate for corrosion (especially in salty climates)",
            ],
        ),
    ],

    "cooling": [
        ComponentDef(
            name="Thermostat",
            prior=0.35, acoustic=False,
            boosting_codes=["P0125", "P0128"],
            boosting_symptoms=["overheating", "always cold", "poor heater", "temperature gauge",
                               "thermostat", "takes long to warm up", "runs cold", "heater blows cold", "temp gauge stays low", "takes forever to warm up", "no heat from vents"],
            tests=[
                "Monitor coolant temperature with a scan tool — should reach operating temp (typically 85–95°C) within 5 minutes",
                "Feel the upper radiator hose — it should stay cold until thermostat opens, then go hot quickly",
                "A thermostat stuck open keeps engine cold; stuck closed causes rapid overheating",
            ],
        ),
        ComponentDef(
            name="Head gasket",
            prior=0.20,
            boosting_codes=["P0300", "P0125"],
            boosting_symptoms=["white smoke", "coolant loss", "overheating", "milky oil",
                               "head gasket", "bubbling coolant", "sweet smell exhaust", "losing coolant with no leak", "milky oil cap", "overheating on highway", "bubbles in coolant reservoir"],
            tests=[
                "Check oil filler cap for white or cream-coloured emulsion — indicates coolant in oil",
                "Perform a combustion gas test on the coolant (block tester) — CO2 in coolant confirms head gasket",
                "Check for white smoke with a sweet smell at the exhaust tailpipe",
                "Monitor coolant level — a head gasket leak causes steady coolant consumption with no external leak",
            ],
        ),
        ComponentDef(
            name="Radiator",
            prior=0.20, acoustic=False,
            boosting_codes=[],
            boosting_symptoms=["overheating", "coolant loss", "coolant leak", "radiator",
                               "gurgling", "steam from front", "coolant puddle under car", "steam from front of car", "green puddle on driveway"],
            tests=[
                "Pressure-test the cooling system to 1.0–1.2 bar — drops indicate an external leak",
                "Inspect radiator end tanks and core for cracks, swelling, or weeping",
                "Check cooling fans engage at operating temperature (electric) or are correctly pitched (mechanical)",
            ],
        ),
        ComponentDef(
            name="Coolant hose / connection",
            prior=0.15, acoustic=False,
            boosting_codes=[],
            boosting_symptoms=["coolant leak", "hose", "smell", "steam", "puddle",
                               "low coolant", "coolant smell", "coolant smell inside car", "steam under hood", "puddle after parking"],
            tests=[
                "Visually inspect all hoses for cracks, swelling, or soft spots when squeezed",
                "Check hose clamps for looseness or corrosion at each connection",
                "Pressure-test the system to locate weeping joints",
            ],
        ),
        ComponentDef(
            name="Water pump (cooling)",
            prior=0.10,
            boosting_codes=["P0125", "P0128"],
            boosting_symptoms=["overheating", "coolant loss from weep hole", "water pump",
                               "bearing noise front", "coolant leak front", "coolant drip from front of engine", "squealing and overheating together"],
            tests=[
                "Check weep hole below the pump body for active coolant dripping",
                "Grip the pump pulley and check for radial play — any movement = bearing failure",
                "Confirm coolant is circulating: squeeze upper rad hose at operating temp — should feel flow pulses",
            ],
        ),
    ],

    "accessories": [
        ComponentDef(
            name="Alternator",
            prior=0.35,
            boosting_codes=["P0620", "P0621", "P0622", "P0623",
                            "P0624", "P0625", "P0626"],
            boosting_symptoms=["whine", "charging", "battery", "alternator", "electrical",
                               "dimming lights", "charging light", "battery light on", "dimming headlights", "whining that changes with electrical load", "dashboard lights flicker"],
            tests=[
                "Measure charging voltage at idle (should be 13.5–14.8 V)",
                "Listen for bearing whine that changes pitch when electrical load changes (headlights, rear defroster)",
                "Check alternator belt tension and condition",
                "Load-test battery — a failing battery forces the alternator to work harder and run hotter",
            ],
        ),
        ComponentDef(
            name="Water pump",
            prior=0.25,
            boosting_codes=["P0125", "P0128"],
            boosting_symptoms=["whine", "coolant", "overheating", "water pump",
                               "coolant leak", "temperature rising", "coolant leak under car", "squeal that gets worse with rpm", "overheating with squeal"],
            tests=[
                "Check coolant level and look for external leaks around the pump weep hole",
                "Grip water pump pulley and rock it — any radial play indicates bearing failure",
                "Listen for bearing whine that increases with RPM",
                "Inspect the weep hole for coolant staining (bearing seal failure)",
            ],
        ),
        ComponentDef(
            name="AC compressor",
            prior=0.22,
            boosting_codes=["P0645", "P0646", "P0647"],
            boosting_symptoms=["clicking", "squeal", "clunk", "ac", "air conditioning",
                               "ac noise", "compressor", "loud clunk when ac turns on", "grinding with ac on", "no cold air with noise"],
            tests=[
                "Switch AC off — if noise stops immediately, compressor or clutch is confirmed",
                "Check clutch engagement — should engage smoothly without a loud clunk",
                "Measure AC clutch air gap (typically 0.4–0.8 mm)",
            ],
        ),
        ComponentDef(
            name="Idler / tensioner bearing",
            prior=0.18,
            boosting_codes=[],
            boosting_symptoms=["squeal", "chirp", "bearing noise", "pulley", "chirping", "squeaking that comes and goes", "chirp at idle", "rattling pulley noise"],
            tests=[
                "Spin each free-spinning pulley by hand — rough or noisy bearings fail",
                "Use a mechanic's stethoscope on each pulley while engine runs to isolate the source",
                "Check for pulley wobble — a wobbling pulley indicates a worn bearing",
            ],
        ),
    ],
}


FOLLOWUP_QUESTIONS: dict[str, list[FollowupQuestion]] = {

    "engine_internal": [
        FollowupQuestion(
            id="ei_rpm_dependent",
            text="Does the noise significantly reduce or disappear above approximately 1,500 RPM?",
            options=["Yes — quieter at higher RPM", "No — present at all RPM", "Skip"],
            yes_multipliers={
                "Timing chain tensioner": 1.7,
                "Timing chain / chain guides": 1.5,
                "VVT / variable valve timing actuator": 1.3,
                "Piston slap": 0.4,
                "Hydraulic valve lifter": 0.6,
            },
            no_multipliers={
                "Piston slap": 1.9,
                "Hydraulic valve lifter": 1.5,
                "Low oil pressure / oil pump": 1.4,
                "Timing chain tensioner": 0.5,
            },
            target_components=["Timing chain tensioner", "VVT / variable valve timing actuator",
                               "Piston slap"],
        ),
        FollowupQuestion(
            id="ei_cold_only",
            text="Is the noise only present on cold starts and does it disappear once the engine fully warms up?",
            options=["Yes — only when cold", "No — also present when warm", "Skip"],
            yes_multipliers={
                "VVT / variable valve timing actuator": 1.8,
                "Timing chain tensioner": 1.5,
                "Piston slap": 1.7,
                "Hydraulic valve lifter": 0.6,
                "Low oil pressure / oil pump": 0.8,
            },
            no_multipliers={
                "Hydraulic valve lifter": 1.6,
                "Low oil pressure / oil pump": 1.5,
                "Timing chain / chain guides": 1.4,
                "Piston slap": 0.3,
                "VVT / variable valve timing actuator": 0.5,
            },
            target_components=["VVT / variable valve timing actuator", "Piston slap",
                               "Hydraulic valve lifter"],
        ),
    ],

    "fuel_ignition": [
        FollowupQuestion(
            id="fi_cylinder_specific",
            text="Does the rough running feel like a single cylinder cutting out at idle, or is it a general hesitation across the RPM range?",
            options=["Single cylinder — noticeable misfire at idle", "General hesitation across all RPM", "Skip"],
            yes_multipliers={
                "Spark plugs": 1.5,
                "Ignition coil": 1.7,
                "Fuel injector": 1.4,
                "Fuel pump / fuel pressure": 0.5,
                "MAF / MAP sensor": 0.6,
            },
            no_multipliers={
                "Fuel pump / fuel pressure": 1.7,
                "MAF / MAP sensor": 1.6,
                "Spark plugs": 0.7,
                "Ignition coil": 0.7,
            },
            target_components=["Spark plugs", "Ignition coil", "Fuel pump / fuel pressure"],
        ),
    ],

    "exhaust": [
        FollowupQuestion(
            id="ex_cold_louder",
            text="Is the noise louder when the engine is cold and does it quiet down as the exhaust reaches operating temperature?",
            options=["Yes — louder when cold", "No — same volume hot or cold", "Skip"],
            yes_multipliers={
                "Exhaust manifold gasket": 1.9,
                "Flex pipe / exhaust joint": 1.3,
                "Catalytic converter": 0.4,
            },
            no_multipliers={
                "Catalytic converter": 1.7,
                "Oxygen sensor": 1.4,
                "Exhaust manifold gasket": 0.5,
            },
            target_components=["Exhaust manifold gasket", "Catalytic converter"],
        ),
    ],

    "belt": [
        FollowupQuestion(
            id="belt_rpm_vs_speed",
            text="Does the noise pitch track engine RPM (faster idle = higher pitch), or does it track vehicle speed regardless of gear?",
            options=["Tracks engine RPM", "Tracks vehicle speed", "Skip"],
            yes_multipliers={
                "Belt tensioner / idler pulley bearing": 1.6,
                "Serpentine / drive belt": 1.6,
                "Power steering pump": 1.4,
                "AC compressor clutch": 1.4,
            },
            no_multipliers={
                "Belt tensioner / idler pulley bearing": 0.4,
                "Serpentine / drive belt": 0.4,
            },
            target_components=["Belt tensioner / idler pulley bearing",
                               "Serpentine / drive belt"],
        ),
    ],

    "brakes": [
        FollowupQuestion(
            id="brk_only_braking",
            text="Does the noise occur only when applying the brakes, or also while driving without braking?",
            options=["Only when braking", "Also present without braking", "Skip"],
            yes_multipliers={
                "Brake pad wear indicators": 1.9,
                "Brake rotor": 1.6,
                "Brake caliper (seized)": 1.5,
                "Wheel bearing": 0.3,
            },
            no_multipliers={
                "Wheel bearing": 2.2,
                "ABS wheel speed sensor": 1.4,
                "Brake pad wear indicators": 0.4,
            },
            target_components=["Brake pad wear indicators", "Wheel bearing"],
        ),
    ],

    "transmission": [
        FollowupQuestion(
            id="trans_speed_vs_rpm",
            text="Does the symptom occur at a specific vehicle speed (e.g. 40–60 mph) regardless of gear, or at a specific engine RPM?",
            options=["Specific vehicle speed", "Specific engine RPM", "Skip"],
            yes_multipliers={
                "Torque converter": 1.8,
                "Input / output shaft bearing": 1.6,
                "Shift solenoid": 0.6,
                "Transmission fluid": 0.9,
            },
            no_multipliers={
                "Shift solenoid": 1.8,
                "Transmission fluid": 1.5,
                "Torque converter": 0.5,
            },
            target_components=["Torque converter", "Input / output shaft bearing"],
        ),
    ],

    "power_steering": [
        FollowupQuestion(
            id="ps_turning_only",
            text="Is the noise or heaviness only present when turning the steering wheel, or also at straight-ahead driving?",
            options=["Only when turning", "Also at straight-ahead", "Skip"],
            yes_multipliers={
                "Power steering pump": 1.8,
                "Low PS fluid / air in system": 1.6,
                "Steering rack / gearbox": 1.4,
                "EPS motor / control module": 0.5,
            },
            no_multipliers={
                "EPS motor / control module": 2.0,
                "Steering rack / gearbox": 1.5,
                "Power steering pump": 0.6,
            },
            target_components=["Power steering pump", "Steering rack / gearbox"],
        ),
    ],

    "low_oil": [
        FollowupQuestion(
            id="lo_warning_light",
            text="Is the oil pressure warning light illuminated on the dashboard?",
            options=["Yes — warning light is on", "No warning light", "Skip"],
            yes_multipliers={
                "Low engine oil level": 1.7,
                "Oil pump": 1.6,
                "Oil pressure sensor (faulty reading)": 0.7,
            },
            no_multipliers={
                "Oil pressure sensor (faulty reading)": 1.9,
                "Low engine oil level": 0.5,
                "Oil pickup tube / sump strainer": 1.4,
            },
            target_components=["Low engine oil level", "Oil pressure sensor (faulty reading)"],
        ),
    ],

    "accessories": [
        FollowupQuestion(
            id="acc_tracks_rpm",
            text="Does the noise pitch change directly with engine RPM — higher revs produce a higher-pitched sound?",
            options=["Yes — pitch tracks RPM", "No — constant pitch", "Skip"],
            yes_multipliers={
                "Alternator": 1.6,
                "Water pump": 1.6,
                "AC compressor": 1.3,
                "Idler / tensioner bearing": 1.4,
            },
            no_multipliers={
                "AC compressor": 1.8,
                "Alternator": 0.7,
                "Water pump": 0.7,
            },
            target_components=["Alternator", "Water pump", "AC compressor"],
        ),
    ],

    "drivetrain": [
        FollowupQuestion(
            id="dt_turning_click",
            text="Does the clicking noise only occur when turning, or is it present when driving straight?",
            options=["Only when turning", "Also when driving straight", "Skip"],
            yes_multipliers={
                "CV joint (outer)": 2.2,
                "CV joint (inner / tripod)": 0.4,
                "Driveshaft / propshaft U-joint": 0.3,
                "Differential bearing": 0.2,
            },
            no_multipliers={
                "CV joint (outer)": 0.2,
                "CV joint (inner / tripod)": 1.5,
                "Driveshaft / propshaft U-joint": 1.8,
                "Differential bearing": 1.6,
            },
            target_components=["CV joint (outer)", "CV joint (inner / tripod)", "Driveshaft / propshaft U-joint"],
        ),
    ],

    "suspension": [
        FollowupQuestion(
            id="sp_bumps_clunk",
            text="Does the clunking sound occur specifically when driving over bumps, or does it happen when turning the steering wheel?",
            options=["Mainly over bumps", "Mainly when turning", "Skip"],
            yes_multipliers={
                "Strut / shock absorber": 1.8,
                "Sway bar link / end link": 1.9,
                "Strut mount / top bearing": 0.4,
                "Ball joint": 0.6,
            },
            no_multipliers={
                "Strut / shock absorber": 0.4,
                "Sway bar link / end link": 0.3,
                "Strut mount / top bearing": 2.0,
                "Ball joint": 1.7,
                "Control arm bushing": 1.5,
            },
            target_components=["Strut / shock absorber", "Sway bar link / end link", "Strut mount / top bearing", "Ball joint"],
        ),
    ],

    "cooling": [
        FollowupQuestion(
            id="cl_head_gasket",
            text="Is there white smoke from the exhaust or does the oil look milky under the filler cap?",
            options=["Yes — white smoke or milky oil", "No — normal oil and exhaust", "Skip"],
            yes_multipliers={
                "Head gasket": 2.5,
                "Thermostat": 0.2,
                "Radiator": 0.2,
                "Coolant hose / connection": 0.3,
                "Water pump (cooling)": 0.3,
            },
            no_multipliers={
                "Head gasket": 0.1,
                "Thermostat": 1.6,
                "Radiator": 1.5,
                "Coolant hose / connection": 1.4,
                "Water pump (cooling)": 1.4,
            },
            target_components=["Head gasket"],
        ),
    ],
}


def get_followup_question(question_id: str) -> FollowupQuestion | None:
    """Retrieve a follow-up question definition by its ID across all subsystems."""
    for questions in FOLLOWUP_QUESTIONS.values():
        for q in questions:
            if q.id == question_id:
                return q
    return None