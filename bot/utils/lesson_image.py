from __future__ import annotations

from io import BytesIO
import textwrap

import matplotlib.pyplot as plt


def render_lesson_image(text: str) -> BytesIO:

    wrapped = "\n".join(textwrap.wrap(text, 80))

    lines = wrapped.count("\n") + 1
    height = max(6, lines * 0.45)

    fig = plt.figure(figsize=(8, height), facecolor="white")

    plt.text(
        0.05,
        0.95,
        wrapped,
        fontsize=14,
        va="top",
        wrap=True,
    )

    plt.axis("off")

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    buffer.seek(0)
    return buffer


__all__ = ["render_lesson_image"]
