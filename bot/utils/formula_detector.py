import re

FORMULA_PATTERNS = [
    r"\\\\frac",
    r"\\\\sqrt",
    r"\\^",
    r"_",
    r"?",
    r"?",
    r"?",
]


def contains_formula(text: str) -> bool:
    for pattern in FORMULA_PATTERNS:
        if re.search(pattern, text):
            return True
    return False
