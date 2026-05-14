from __future__ import annotations


COARSE_LABELS = ("cooking", "eating", "cleanup", "other")
COARSE_LABEL_TO_ID = {label: idx for idx, label in enumerate(COARSE_LABELS)}

EATING_KEYWORDS = (
    "eat",
    "chew",
    "drink",
    "mouth",
    "fork",
    "spoon",
    "glass",
    "napkin",
    "converse",
    "plate_away",
    "pasta",
    "salad",
    "soup",
)
CLEANUP_KEYWORDS = (
    "clean",
    "dishwasher",
    "dish",
    "scrub",
    "rinse",
    "drying_plate",
    "drying_hands",
    "wipe",
    "crumb",
    "waste",
    "bin",
    "leftover",
    "sponge",
    "towel",
    "faucet",
    "sink",
    "plates",
    "silverware",
    "washing_hands",
)
COOKING_KEYWORDS = (
    "batter",
    "basting",
    "bread",
    "cake",
    "cook",
    "dough",
    "egg",
    "flour",
    "food",
    "garlic",
    "herb",
    "knife",
    "meat",
    "oven",
    "pan",
    "pancake",
    "pot",
    "soup",
    "spice",
    "stove",
    "vegetable",
    "zest",
    "zesting",
    "bowl",
    "kettle",
    "ladling",
    "pouring",
    "slicing",
    "stirring",
    "whisking",
)


def classify_coarse_label(name: str, desc: str = "", env: str = "") -> str:
    text = f"{name} {desc} {env}".lower()
    name_text = name.lower()

    if any(keyword in name_text for keyword in EATING_KEYWORDS) or "dining room" in text:
        if not any(keyword in name_text for keyword in CLEANUP_KEYWORDS):
            return "eating"
    if any(keyword in name_text for keyword in CLEANUP_KEYWORDS):
        return "cleanup"
    if any(keyword in name_text for keyword in COOKING_KEYWORDS):
        return "cooking"
    return "other"
