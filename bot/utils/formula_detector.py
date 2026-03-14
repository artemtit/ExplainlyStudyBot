MATH_TOKENS = [
    "^",
    "=",
    "sqrt",
    "√",
    "/",
    "+",
    "-",
    "*",
]


def contains_formula(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    for token in MATH_TOKENS:
        if token in text_lower:
            return True
    return False
