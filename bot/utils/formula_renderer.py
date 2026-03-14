import io
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bot.utils.formula_parser import wrap_legacy_formulas

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["mathtext.rm"] = "serif"

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]"
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def convert_to_latex(text: str) -> str:
    clean = _strip_emoji(text).strip()
    if not clean:
        return clean

    clean = re.sub(r"(?<!\\)cos\s*([A-Za-z\u0391-\u03A9\u03B1-\u03C9]+)", r"\\cos{\1}", clean)
    clean = re.sub(r"(?<!\\)sin\s*([A-Za-z\u0391-\u03A9\u03B1-\u03C9]+)", r"\\sin{\1}", clean)
    clean = re.sub(r"(?<!\\)tan\s*([A-Za-z\u0391-\u03A9\u03B1-\u03C9]+)", r"\\tan{\1}", clean)
    clean = re.sub(r"sqrt\(([^()]+)\)", r"\\sqrt{\1}", clean)
    clean = re.sub(
        r"\(\s*([^()]+?)\s*\)\s*/\s*([A-Za-z0-9]+)",
        r"\\frac{\1}{\2}",
        clean,
    )
    clean = clean.replace("±", r"\\pm")
    return clean


def render_formula_image(text: str) -> bytes | list[bytes]:
    normalized = wrap_legacy_formulas(text)
    latex = convert_to_latex(normalized)
    latex = re.sub(r"\$\$(.*?)\$\$", r"$\1$", latex)
    latex = re.sub(r"\${2,}", "$", latex)
    if "$" not in latex and any(token in latex for token in ("^", "sqrt", "=", "/")):
        latex = f"${latex}$"

    raw_lines = latex.split("\n")
    lines: list[str] = []
    for line in raw_lines:
        line = re.sub(r"([a-zA-Z]\(x\)\s*=\s*[^ ,.;]+)", r"$\1$", line)
        lowered = line.lower()
        if "$" not in line and any(token in lowered for token in ("^", "/", "sqrt")):
            line = f"${line}$"
        if "$" in line:
            lines.append(line)
        else:
            lines.extend(textwrap.fill(line, width=70).split("\n"))

    chunks = [lines[i : i + 60] for i in range(0, len(lines), 60)] or [[]]
    images: list[bytes] = []

    for chunk in chunks:
        line_count = max(1, len(chunk))
        height = max(5.0, line_count * 0.35)
        height = min(height, 12)

        fig = plt.figure(figsize=(12, height))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis("off")

        y = 0.98
        for line in chunk:
            stripped = line.strip()
            if not stripped:
                y -= 0.04
                continue

            if "$$" in stripped:
                fontsize = 26
                x = 0.5
                align = "center"
            elif stripped.startswith("$") and stripped.endswith("$"):
                fontsize = 22
                x = 0.5
                align = "center"
            elif "$" in stripped:
                fontsize = 22
                x = 0.02
                align = "left"
            else:
                fontsize = 18
                x = 0.02
                align = "left"

            plt.text(
                x,
                y,
                line,
                fontsize=fontsize,
                ha=align,
                va="top",
            )
            line_height = fontsize / 150
            y -= line_height
            if y < 0.05:
                break

        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            dpi=220,
            pad_inches=0.5,
        )
        plt.close(fig)
        buf.seek(0)
        images.append(buf.getvalue())

    if len(images) == 1:
        return images[0]
    return images
