"""
config.py - Taxonomies, enums, and constraints for car cabin prompt generation.

Two-level pattern: det_class (stable OD label) + gen_variant (richer phrasing).
"""

# ---------------------------------------------------------------------------
# VLM / API settings
# ---------------------------------------------------------------------------
VLM_MODEL = "gpt-5.1"

# ---------------------------------------------------------------------------
# Clothing taxonomy (7 det_classes, silhouette-based)
# ---------------------------------------------------------------------------
CLOTHING_TAXONOMY = {
    "sleeveless_top": {
        "thermal": "light",
        "gen_variants": [
            "tank top",
            "sleeveless top",
            "athletic tank top",
            "racerback tank top",
            "scoop-neck tank top",
        ],
    },
    "short_sleeve_top": {
        "thermal": "light",
        "gen_variants": [
            "t-shirt",
            "casual tee",
            "v-neck t-shirt",
            "short sleeve polo",
            "striped t-shirt",
            "henley shirt",
        ],
    },
    "long_sleeve_plain": {
        "thermal": "mid",
        "gen_variants": [
            "long sleeve shirt",
            "long sleeve t-shirt",
            "thin pullover",
            "crew neck sweater",
            "v-neck sweater",
            "ribbed long sleeve top",
        ],
    },
    "long_sleeve_collared": {
        "thermal": "mid",
        "gen_variants": [
            "long sleeve polo",
            "collared dress shirt",
            "white button-up shirt",
            "oxford shirt",
            "flannel shirt",
        ],
    },
    "hoodie_sweatshirt": {
        "thermal": "mid",
        "gen_variants": [
            "hoodie",
            "pullover hoodie",
            "sweatshirt",
            "zip-up hoodie",
            "hooded sweatshirt",
        ],
    },
    "light_jacket": {
        "thermal": "mid",
        "gen_variants": [
            "light jacket",
            "denim jacket",
            "bomber jacket",
            "windbreaker",
            "fleece jacket",
            "unzipped light jacket",
        ],
    },
    "heavy_jacket_coat": {
        "thermal": "heavy",
        "gen_variants": [
            "heavy jacket",
            "winter coat",
            "puffer jacket",
            "parka",
            "wool coat",
            "insulated coat",
        ],
    },
}

# Thermal tier lookup (det_class -> thermal weight)
THERMAL_TIERS = {
    "light": ["sleeveless_top", "short_sleeve_top"],
    "mid": [
        "long_sleeve_plain",
        "long_sleeve_collared",
        "hoodie_sweatshirt",
        "light_jacket",
    ],
    "heavy": ["heavy_jacket_coat"],
}

# ---------------------------------------------------------------------------
# Glasses taxonomy (3 manifest classes, 2 OD bbox classes)
# ---------------------------------------------------------------------------
GLASSES_TAXONOMY = {
    "no_glasses": {
        "gen_variants": ["no glasses", "face without glasses"],
        "is_od_class": False,
    },
    "eyeglasses": {
        "gen_variants": [
            "clear prescription glasses",
            "thin wire-frame glasses",
            "black rectangular frame glasses",
            "round metal frame glasses",
        ],
        "is_od_class": True,
    },
    "sunglasses": {
        "gen_variants": [
            "dark sunglasses",
            "aviator sunglasses",
            "large black sunglasses",
            "tinted sunglasses",
        ],
        "is_od_class": True,
    },
}

# ---------------------------------------------------------------------------
# Headwear taxonomy (5 manifest classes, 3 OD bbox classes)
# ---------------------------------------------------------------------------
HEADWEAR_TAXONOMY = {
    "no_headwear": {
        "gen_variants": ["no headwear", "uncovered hair"],
        "is_od_class": False,
        "has_bare_scalp": False,
    },
    "bare_head": {
        "gen_variants": ["bald head", "shaved head", "bald scalp visible"],
        "is_od_class": False,
        "has_bare_scalp": True,
    },
    "cap": {
        "gen_variants": ["baseball cap", "cap worn forward", "cap worn backward"],
        "is_od_class": True,
        "has_bare_scalp": False,
    },
    "beanie": {
        "gen_variants": ["beanie", "wool beanie", "fitted beanie"],
        "is_od_class": True,
        "has_bare_scalp": False,
    },
    "hijab_headscarf": {
        "gen_variants": ["hijab", "headscarf", "wrapped headscarf"],
        "is_od_class": True,
        "has_bare_scalp": False,
    },
}

# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------
LAYERING_MODES = ["single", "open_outer", "closed_outer"]

LAYERING_DISTRIBUTION = {
    "single": 0.65,
    "open_outer": 0.20,
    "closed_outer": 0.15,
}

# Valid outer classes for layering (must be >= mid thermal)
VALID_OUTER_CLASSES = ["hoodie_sweatshirt", "light_jacket", "heavy_jacket_coat"]

# Valid inner classes for layering
VALID_INNER_CLASSES = [
    "sleeveless_top",
    "short_sleeve_top",
    "long_sleeve_plain",
    "long_sleeve_collared",
]

# ---------------------------------------------------------------------------
# Person demographics
# ---------------------------------------------------------------------------
GENDERS = ["male", "female"]
ETHNICITIES = [
    "Caucasian",
    "African",
    "Asian",
    "Hispanic",
    "Middle Eastern",
    "South Asian",
]
AGES = ["teen", "young adult", "middle-aged", "elderly"]

# ---------------------------------------------------------------------------
# Scene variables (global defaults; per-ref whitelists override)
# ---------------------------------------------------------------------------
SEATBELT_OPTIONS = ["wearing seatbelt", "seatbelt not visible"]
LIGHTING_CONDITIONS = [
    "bright sunlight",
    "overcast",
    "night with interior lights",
    "dim cabin",
]
WEATHER_CONDITIONS = ["rainy", "snowy", "sunny", "foggy"]
TIMES_OF_DAY = ["morning", "noon", "evening", "night"]
SEAT_TYPES = ["leather seats", "fabric seats", "sports seats"]
WINDOW_TINTS = ["clear windows", "lightly tinted windows", "heavily tinted windows"]
CABIN_COLORS = ["black", "beige", "gray", "brown", "white"]

# Garment colors (applied to clothing gen_variant)
GARMENT_COLORS = [
    "red", "blue", "black", "white", "green", "yellow",
    "brown", "gray", "purple", "orange", "navy", "olive",
]

# ---------------------------------------------------------------------------
# Sampling distributions
# ---------------------------------------------------------------------------
# Glasses presence: ~50% no_glasses, ~25% eyeglasses, ~25% sunglasses
GLASSES_DISTRIBUTION = {
    "no_glasses": 0.50,
    "eyeglasses": 0.25,
    "sunglasses": 0.25,
}

# Headwear presence
HEADWEAR_DISTRIBUTION = {
    "no_headwear": 0.55,
    "bare_head": 0.07,
    "cap": 0.15,
    "beanie": 0.11,
    "hijab_headscarf": 0.12,
}

# Sleeveless share (only on arms_visible refs)
SLEEVELESS_SHARE = 0.12

# ---------------------------------------------------------------------------
# Weather-clothing coupling (soft constraint, §7.4)
# ---------------------------------------------------------------------------
WEATHER_CLOTHING_BIAS = {
    "snowy": {"heavy": 0.70, "mid": 0.20, "light": 0.10},
    "rainy": {"heavy": 0.30, "mid": 0.50, "light": 0.20},
    "foggy": {"heavy": 0.30, "mid": 0.40, "light": 0.30},
    "sunny": {"heavy": 0.20, "mid": 0.40, "light": 0.40},
}

# Cold weather biases toward layering
WEATHER_LAYERING_BIAS = {
    "snowy": {"single": 0.40, "open_outer": 0.30, "closed_outer": 0.30},
    "rainy": {"single": 0.50, "open_outer": 0.25, "closed_outer": 0.25},
    "foggy": {"single": 0.60, "open_outer": 0.20, "closed_outer": 0.20},
    "sunny": {"single": 0.75, "open_outer": 0.15, "closed_outer": 0.10},
}

# ---------------------------------------------------------------------------
# Color contrast — minimum fraction of samples with visible contrast (§7.2)
# ---------------------------------------------------------------------------
MIN_COLOR_CONTRAST_RATIO = 0.35
COLOR_REROLL_PROBABILITY = 0.50

# ---------------------------------------------------------------------------
# Output / scale
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 1000  # Development placeholder; production targets per §1.6
