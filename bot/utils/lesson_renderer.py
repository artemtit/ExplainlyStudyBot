from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from bot.utils.lesson_validation import LessonRenderData, wrap_text


def _draw_text(
    ax,
    *,
    x: float,
    y: float,
    text: str,
    fontsize: int,
    line_height: float,
    fontweight: str | None = None,
    fontfamily: str | None = None,
) -> float:
    dpi = ax.figure.dpi
    px_per_pt = dpi / 72.0
    step = fontsize * px_per_pt * line_height

    for line in text.splitlines():
        ax.text(
            x,
            y,
            line,
            fontsize=fontsize,
            fontweight=fontweight,
            fontfamily=fontfamily,
            ha="left",
            va="top",
            color="#111111",
        )
        y -= step
    return y


def render_lesson_card(data: LessonRenderData) -> bytes:
    width_px = 1200
    height_px = 1600
    dpi = 100

    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("#f7f7f7")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(0, height_px)
    ax.axis("off")

    container_x = 80
    container_y = 80
    container_w = width_px - 160
    container_h = height_px - 160

    container = FancyBboxPatch(
        (container_x, container_y),
        container_w,
        container_h,
        boxstyle="round,pad=0,rounding_size=30",
        linewidth=0,
        facecolor="white",
    )
    ax.add_patch(container)

    padding = 80
    content_left = container_x + padding
    content_right = container_x + container_w - padding
    content_top = container_y + container_h - padding

    y = content_top

    ax.hlines(y, content_left, content_right, colors="#e0e0e0", linewidth=2)
    y -= 32

    y = _draw_text(
        ax,
        x=content_left,
        y=y,
        text="📘 Тема",
        fontsize=36,
        line_height=1.2,
        fontweight="bold",
    )
    y -= 8

    topic_text = wrap_text(data.topic, 32)
    y = _draw_text(
        ax,
        x=content_left,
        y=y,
        text=topic_text,
        fontsize=60,
        line_height=1.1,
        fontweight="bold",
    )

    y -= 24
    ax.hlines(y, content_left, content_right, colors="#e0e0e0", linewidth=2)
    y -= 48

    def draw_section(title: str, body: str, *, monospace: bool = False) -> None:
        nonlocal y
        y = _draw_text(
            ax,
            x=content_left,
            y=y,
            text=title,
            fontsize=36,
            line_height=1.2,
            fontweight="bold",
        )
        y -= 12
        wrapped_body = wrap_text(body, 42)
        y = _draw_text(
            ax,
            x=content_left,
            y=y,
            text=wrapped_body,
            fontsize=32,
            line_height=1.4,
            fontfamily="DejaVu Sans Mono" if monospace else None,
        )
        y -= 40

    draw_section("Краткое объяснение", data.explanation)
    draw_section("Формулы", data.formulas, monospace=True)
    draw_section("Пример", data.example)

    ax.hlines(max(y, container_y + 40), content_left, content_right, colors="#e0e0e0", linewidth=2)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def render_solution_card(
    *,
    topic: str,
    problem: str,
    steps: list[dict],
) -> bytes:
    width_px = 1200
    height_px = 1600
    dpi = 100

    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("#f7f7f7")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_px)
    ax.set_ylim(0, height_px)
    ax.axis("off")

    container_x = 80
    container_y = 80
    container_w = width_px - 160
    container_h = height_px - 160

    container = FancyBboxPatch(
        (container_x, container_y),
        container_w,
        container_h,
        boxstyle="round,pad=0,rounding_size=30",
        linewidth=0,
        facecolor="white",
    )
    ax.add_patch(container)

    padding = 80
    content_left = container_x + padding
    content_right = container_x + container_w - padding
    content_top = container_y + container_h - padding

    y = content_top

    ax.hlines(y, content_left, content_right, colors="#e0e0e0", linewidth=2)
    y -= 32

    y = _draw_text(
        ax,
        x=content_left,
        y=y,
        text="Тема",
        fontsize=36,
        line_height=1.2,
        fontweight="bold",
    )
    y -= 8

    topic_text = wrap_text(topic, 32)
    y = _draw_text(
        ax,
        x=content_left,
        y=y,
        text=topic_text,
        fontsize=60,
        line_height=1.1,
        fontweight="bold",
    )

    y -= 24
    ax.hlines(y, content_left, content_right, colors="#e0e0e0", linewidth=2)
    y -= 48

    def draw_section(title: str, body: str, *, monospace: bool = False) -> None:
        nonlocal y
        y = _draw_text(
            ax,
            x=content_left,
            y=y,
            text=title,
            fontsize=36,
            line_height=1.2,
            fontweight="bold",
        )
        y -= 12
        wrapped_body = wrap_text(body, 42)
        y = _draw_text(
            ax,
            x=content_left,
            y=y,
            text=wrapped_body,
            fontsize=32,
            line_height=1.4,
            fontfamily="DejaVu Sans Mono" if monospace else None,
        )
        y -= 40

    draw_section("Задача", problem)

    step_lines: list[str] = []
    formulas: list[str] = []
    for idx, step in enumerate(steps, start=1):
        explanation = str(step.get("explanation") or "").strip()
        result = str(step.get("result") or "").strip()
        formula = str(step.get("formula") or "").strip()
        if explanation:
            step_lines.append(f"Шаг {idx}. {explanation}")
        if result:
            step_lines.append(f"Результат: {result}")
        if formula:
            formulas.append(formula)

    steps_body = "\n".join(step_lines) if step_lines else "-"
    draw_section("Шаги решения", steps_body)

    if formulas:
        draw_section("Формулы", "\n".join(formulas), monospace=True)

    ax.hlines(max(y, container_y + 40), content_left, content_right, colors="#e0e0e0", linewidth=2)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


__all__ = ["render_lesson_card", "render_solution_card"]
