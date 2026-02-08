NORMAL_STATE_WORDS = [
    # neutral (no state adjective) is a strong positive prior in CLIP ZSAD
    # and helps prevent "abnormal always wins" similarity collapse.
    "good",
    "normal",
    "perfect",
    "clean",
    "intact",
]

# A commonly used "state_level_prompts" style negative prompt set in ZSAD literature (e.g., WinCLIP/AnomalyCLIP-like).
# Keep this list centralized so downstream changes are 1-line edits.
ABNORMAL_STATE_WORDS = [
    "damaged",
    "broken",
    "defective",
    "flawed",
    "corrupted",
    "scratched",
    "stained",
    "contaminated",
    "blemished",
    "cracked",
]

NORMAL_STATE_TEMPLATES = ["{}"] + [f"{w} {{}}" for w in NORMAL_STATE_WORDS]
ABNORMAL_STATE_TEMPLATES = [f"{w} {{}}" for w in ABNORMAL_STATE_WORDS]
