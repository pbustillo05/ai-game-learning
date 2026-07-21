#!/usr/bin/env python3
"""Create decision plots for the HU 32 bb fixed-beta branching sweep."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parents[1]
EXPERIMENT = ROOT / "runs/experiments/branching_hu32_ckpt814"
OUTPUT = EXPERIMENT / "plots"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BG = "#f7f8fa"
PANEL = "#ffffff"
INK = "#17212b"
MUTED = "#667482"
GRID = "#dce2e8"
TARGET = "#c43c39"
COLORS = [
    "#4e79a7",
    "#59a14f",
    "#f28e2b",
    "#b07aa1",
    "#e15759",
    "#76b7b2",
    "#edc948",
]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def load_completed_samples() -> dict[float, list[dict[str, object]]]:
    samples: dict[float, list[dict[str, object]]] = {}
    for name in ("coarse_geometry.json", "extended_geometry.json", "high_geometry.json"):
        payload = json.loads((EXPERIMENT / name).read_text())
        for record in payload["records"]:
            beta = float(record["beta"])
            samples[beta] = [
                sample
                for sample in record["samples"]
                if not sample.get("upper_cutoff_activated", 0)
            ]
    for name in (
        "fixed_beta_stop10k_coarse64.json.samples.jsonl",
        "fixed_beta_stop50k_coarse64.json.samples.jsonl",
    ):
        for line in (EXPERIMENT / name).read_text().splitlines():
            sample = json.loads(line)
            if sample.get("stop_cutoff_reached", 0):
                continue
            samples.setdefault(float(sample["beta"]), []).append(sample)
    return dict(sorted(samples.items()))


def load_pilot(name: str) -> dict[str, object]:
    payload = json.loads((EXPERIMENT / name).read_text())
    sample = dict(payload["records"][0]["samples"][0])
    sample["stop_cutoff"] = payload["stop_cutoff"]
    return sample


def load_all_sweep_samples() -> dict[float, list[dict[str, object]]]:
    """Load every non-pilot sample, deduplicated by ``(beta, repeat)``."""

    keyed: dict[tuple[float, int], dict[str, object]] = {}
    for name in ("coarse_geometry.json", "extended_geometry.json", "high_geometry.json"):
        payload = json.loads((EXPERIMENT / name).read_text())
        for record in payload["records"]:
            beta = float(record["beta"])
            for repeat, raw in enumerate(record["samples"]):
                sample = dict(raw)
                sample.setdefault("beta", beta)
                sample.setdefault("repeat", repeat)
                sample.setdefault("stop_cutoff_reached", 0.0)
                sample.setdefault("pending_nodes", 0.0)
                keyed[(beta, repeat)] = sample
    for name in (
        "fixed_beta_stop10k_coarse64.json.samples.jsonl",
        "fixed_beta_stop50k_coarse64.json.samples.jsonl",
        "fixed_beta_stop50k_b225_resume39_streets.json.samples.jsonl",
        "fixed_beta_stop50k_b250_400_coarse64_streets.json.samples.jsonl",
    ):
        path = EXPERIMENT / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            keyed[(float(sample["beta"]), int(sample["repeat"]))] = sample
    grouped: dict[float, list[dict[str, object]]] = {}
    for (beta, _), sample in sorted(keyed.items()):
        grouped.setdefault(beta, []).append(sample)
    return grouped


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1800, 1100), BG)
    draw = ImageDraw.Draw(image)
    draw.text((90, 55), title, fill=INK, font=font(42, bold=True))
    draw.text((90, 112), subtitle, fill=MUTED, font=font(23))
    return image, draw


def footer(draw: ImageDraw.ImageDraw, text: str) -> None:
    draw.text((90, 1060), text, fill=MUTED, font=font(17), anchor="ls")


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#e3e7eb", width=2)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str,
    width: int = 3,
    dash: int = 12,
) -> None:
    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    position = 0.0
    while position < length:
        finish = min(position + dash, length)
        draw.line(
            (
                x1 + ux * position,
                y1 + uy * position,
                x1 + ux * finish,
                y1 + uy * finish,
            ),
            fill=fill,
            width=width,
        )
        position += dash * 1.75


def save(image: Image.Image, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT / name, optimize=True)


def plot_survival(samples: dict[float, list[dict[str, object]]]) -> None:
    selected = [1.0, 1.24, 1.5, 1.75, 2.0, 2.25]
    image, draw = canvas(
        "How rapidly does fixed branching grow the tree?",
        "Empirical survival curve: fraction of completed trees with at least N allocated nodes",
    )
    box = (75, 165, 1725, 1015)
    panel(draw, box)
    left, top, right, bottom = 175, 225, 1655, 910
    xmin, xmax = 1.0, 50_000.0

    def sx(value: float) -> float:
        return left + (math.log10(value) - math.log10(xmin)) / (
            math.log10(xmax) - math.log10(xmin)
        ) * (right - left)

    def sy(value: float) -> float:
        return bottom - value * (bottom - top)

    for probability in (0, 0.25, 0.5, 0.75, 1.0):
        y = sy(probability)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text(
            (left - 20, y), f"{probability:.0%}", fill=MUTED, font=font(19), anchor="rm"
        )
    for value, label in ((1, "1"), (10, "10"), (100, "100"), (1000, "1k"), (10_000, "10k"), (50_000, "50k")):
        x = sx(value)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((x, bottom + 22), label, fill=MUTED, font=font(19), anchor="ma")
    target_x = sx(10_000)
    dashed_line(draw, (target_x, top), (target_x, bottom), fill=TARGET, width=4)
    draw.text((target_x - 12, top + 4), "10k target", fill=TARGET, font=font(19, bold=True), anchor="ra")

    log_grid = [10 ** (math.log10(xmin) + i / 240 * (math.log10(xmax) - math.log10(xmin))) for i in range(241)]
    for index, beta in enumerate(selected):
        values = [float(sample["nodes"]) for sample in samples[beta]]
        points = []
        for value in log_grid:
            survival = sum(node >= value for node in values) / len(values)
            points.append((sx(value), sy(survival)))
        color = COLORS[index]
        draw.line(points, fill=color, width=5, joint="curve")
        legend_x = 175 + index * 240
        legend_y = 980
        draw.line((legend_x, legend_y, legend_x + 55, legend_y), fill=color, width=6)
        label = f"β={beta:g}  (n={len(values)})"
        draw.text((legend_x + 68, legend_y), label, fill=INK, font=font(19), anchor="lm")

    draw.text((915, 945), "Allocated nodes N (log scale)", fill=INK, font=font(21, bold=True), anchor="ma")
    draw.text((left, 190), "Fraction of trees", fill=INK, font=font(20, bold=True), anchor="la")
    footer(draw, "HU 32 bb · checkpoint 814 · fixed β per tree · completed, uncensored samples only")
    save(image, "01_tree_size_survival.png")


def plot_quantiles(samples: dict[float, list[dict[str, object]]]) -> None:
    image, draw = canvas(
        "Tree-size distribution by branching intensity",
        "Median, central 50%, central 90%, and observed maximum; node axis is logarithmic",
    )
    box = (75, 165, 1725, 1015)
    panel(draw, box)
    left, top, right, bottom = 165, 225, 1660, 900
    betas = list(samples)
    ymin, ymax = 1.0, 50_000.0

    def sx(index: int) -> float:
        return left + index / (len(betas) - 1) * (right - left)

    def sy(value: float) -> float:
        return bottom - (math.log10(value) - math.log10(ymin)) / (
            math.log10(ymax) - math.log10(ymin)
        ) * (bottom - top)

    for value, label in ((1, "1"), (10, "10"), (100, "100"), (1000, "1k"), (10_000, "10k"), (50_000, "50k")):
        y = sy(value)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 18, y), label, fill=MUTED, font=font(18), anchor="rm")
    dashed_line(draw, (left, sy(10_000)), (right, sy(10_000)), fill=TARGET, width=4)
    draw.text((right - 5, sy(10_000) - 10), "10k target", fill=TARGET, font=font(18, bold=True), anchor="rb")

    medians: list[tuple[float, float]] = []
    for index, beta in enumerate(betas):
        values = [float(sample["nodes"]) for sample in samples[beta]]
        q05, q25, q50, q75, q95 = [quantile(values, p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
        maximum = max(values)
        x = sx(index)
        draw.line((x, sy(q05), x, sy(q95)), fill="#8bb4d8", width=6)
        draw.line((x, sy(q25), x, sy(q75)), fill="#3478ad", width=18)
        draw.ellipse((x - 9, sy(q50) - 9, x + 9, sy(q50) + 9), fill=PANEL, outline=INK, width=4)
        draw.polygon(
            ((x, sy(maximum) - 10), (x - 9, sy(maximum) + 8), (x + 9, sy(maximum) + 8)),
            fill="#e15759",
        )
        medians.append((x, sy(q50)))
        label = f"{beta:g}\nn={len(values)}"
        draw.multiline_text((x, bottom + 25), label, fill=INK, font=font(17), anchor="ma", align="center", spacing=3)
    draw.line(medians, fill=INK, width=3)

    legend_y = 975
    draw.line((360, legend_y - 18, 360, legend_y + 18), fill="#8bb4d8", width=6)
    draw.text((382, legend_y), "5th–95th", fill=INK, font=font(18), anchor="lm")
    draw.line((600, legend_y - 18, 600, legend_y + 18), fill="#3478ad", width=18)
    draw.text((622, legend_y), "25th–75th", fill=INK, font=font(18), anchor="lm")
    draw.ellipse((850 - 9, legend_y - 9, 850 + 9, legend_y + 9), fill=PANEL, outline=INK, width=4)
    draw.text((872, legend_y), "median", fill=INK, font=font(18), anchor="lm")
    draw.polygon(((1080, legend_y - 10), (1071, legend_y + 8), (1089, legend_y + 8)), fill="#e15759")
    draw.text((1102, legend_y), "observed max", fill=INK, font=font(18), anchor="lm")
    draw.text((left, 190), "Allocated nodes (log scale)", fill=INK, font=font(20, bold=True), anchor="la")
    footer(draw, "HU 32 bb · checkpoint 814 · fixed β per tree · β=2.25 currently has 39 samples; others have 64")
    save(image, "02_tree_size_quantiles.png")


def plot_cutoff_pilots() -> None:
    pilots = [
        load_pilot("fixed_beta_stop10k_pilot_b400.json"),
        load_pilot("fixed_beta_stop50k_pilot_b400.json"),
    ]
    image, draw = canvas(
        "What remains live when β=4 reaches the safety cutoff?",
        "Two single-tree safety pilots; these diagnose cutoff behavior and are not distribution estimates",
    )
    left_box = (75, 165, 880, 1015)
    right_box = (920, 165, 1725, 1015)
    panel(draw, left_box)
    panel(draw, right_box)

    draw.text((115, 215), "Allocated vs. pending nodes", fill=INK, font=font(25, bold=True))
    left, top, right, bottom = 155, 285, 820, 875
    maximum = 90_000
    for value in (0, 20_000, 40_000, 60_000, 80_000):
        y = bottom - value / maximum * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 15, y), f"{value // 1000}k", fill=MUTED, font=font(17), anchor="rm")
    for index, pilot in enumerate(pilots):
        center = 315 + index * 325
        width = 92
        allocated = float(pilot["nodes"])
        queued = float(pilot["queued_nodes"])
        frontier = float(pilot["frontier_nodes"])
        for offset, value, color, name in (
            (-58, allocated, "#4e79a7", "allocated"),
            (58, queued, "#f28e2b", "queued"),
        ):
            x1, x2 = center + offset - width / 2, center + offset + width / 2
            y = bottom - value / maximum * (bottom - top)
            draw.rectangle((x1, y, x2, bottom), fill=color)
            if name == "queued" and frontier:
                frontier_height = frontier / maximum * (bottom - top)
                draw.rectangle((x1, y - frontier_height, x2, y), fill="#e15759")
            draw.text((center + offset, y - 12 - (frontier / maximum * (bottom - top) if name == "queued" else 0)), f"{int(value + (frontier if name == 'queued' else 0)):,}", fill=INK, font=font(17, bold=True), anchor="mb")
        draw.text((center, bottom + 28), f"stop={int(pilot['stop_cutoff']) // 1000}k", fill=INK, font=font(19, bold=True), anchor="ma")
    draw.rectangle((225, 944, 249, 968), fill="#4e79a7")
    draw.text((260, 956), "allocated", fill=INK, font=font(17), anchor="lm")
    draw.rectangle((390, 944, 414, 968), fill="#f28e2b")
    draw.text((425, 956), "family queues", fill=INK, font=font(17), anchor="lm")
    draw.rectangle((590, 944, 614, 968), fill="#e15759")
    draw.text((625, 956), "unrouted frontier", fill=INK, font=font(17), anchor="lm")

    draw.text((960, 215), "Measured CUDA peak", fill=INK, font=font(25, bold=True))
    left, top, right, bottom = 1000, 285, 1665, 875
    maximum_gb = 10.0
    for value in (0, 2, 4, 6, 8, 10):
        y = bottom - value / maximum_gb * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 15, y), f"{value} GB", fill=MUTED, font=font(17), anchor="rm")
    limit_y = bottom - 9.5 / maximum_gb * (bottom - top)
    dashed_line(draw, (left, limit_y), (right, limit_y), fill=TARGET, width=4)
    draw.text((right, limit_y + 8), "9.5 GB hard limit", fill=TARGET, font=font(17, bold=True), anchor="ra")
    for index, pilot in enumerate(pilots):
        center = 1160 + index * 330
        for offset, key, color in (
            (-58, "peak_allocated_bytes", "#59a14f"),
            (58, "peak_reserved_bytes", "#b07aa1"),
        ):
            value = float(pilot[key]) / 1e9
            x1, x2 = center + offset - 46, center + offset + 46
            y = bottom - value / maximum_gb * (bottom - top)
            draw.rectangle((x1, y, x2, bottom), fill=color)
            draw.text((center + offset, y - 10), f"{value:.2f}", fill=INK, font=font(17, bold=True), anchor="mb")
        draw.text((center, bottom + 28), f"stop={int(pilot['stop_cutoff']) // 1000}k", fill=INK, font=font(19, bold=True), anchor="ma")
    draw.rectangle((1140, 944, 1164, 968), fill="#59a14f")
    draw.text((1175, 956), "allocated", fill=INK, font=font(17), anchor="lm")
    draw.rectangle((1370, 944, 1394, 968), fill="#b07aa1")
    draw.text((1405, 956), "reserved", fill=INK, font=font(17), anchor="lm")
    footer(draw, "HU 32 bb · checkpoint 814 · β=4 fixed until stop · pending = family queues + unrouted frontier")
    save(image, "03_cutoff_queue_memory_pilots.png")


def plot_per_beta_histograms(samples: dict[float, list[dict[str, object]]]) -> None:
    complete = [(beta, rows) for beta, rows in samples.items() if len(rows) == 64]
    width, height = 1800, 1420
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((90, 52), "Distribution of sampled nodes for each fixed β", fill=INK, font=font(42, bold=True))
    draw.text(
        (90, 110),
        "One histogram per completed 64-tree block · common logarithmic node axis · bar height is fraction of trees",
        fill=MUTED,
        font=font(22),
    )

    columns, rows_count = 4, 3
    outer_left, outer_top = 75, 165
    gap_x, gap_y = 22, 24
    panel_width = (1650 - gap_x * (columns - 1)) / columns
    panel_height = (1135 - gap_y * (rows_count - 1)) / rows_count
    xmin, xmax = 1.0, 50_000.0
    bin_count = 22
    edges = [
        10 ** (math.log10(xmin) + index / bin_count * (math.log10(xmax) - math.log10(xmin)))
        for index in range(bin_count + 1)
    ]
    all_fractions: list[float] = []
    histogram_rows: list[tuple[float, list[dict[str, object]], list[int]]] = []
    for beta, beta_samples in complete:
        values = [float(sample["nodes"]) for sample in beta_samples]
        counts = [0] * bin_count
        for value in values:
            index = min(
                bin_count - 1,
                max(
                    0,
                    int(
                        (math.log10(value) - math.log10(xmin))
                        / (math.log10(xmax) - math.log10(xmin))
                        * bin_count
                    ),
                ),
            )
            counts[index] += 1
        all_fractions.extend(count / 64 for count in counts)
        histogram_rows.append((beta, beta_samples, counts))
    ymax = max(0.25, math.ceil(max(all_fractions) * 20) / 20)

    for panel_index in range(columns * rows_count):
        row, column = divmod(panel_index, columns)
        x1 = outer_left + column * (panel_width + gap_x)
        y1 = outer_top + row * (panel_height + gap_y)
        x2, y2 = x1 + panel_width, y1 + panel_height
        panel(draw, (int(x1), int(y1), int(x2), int(y2)))
        if panel_index >= len(histogram_rows):
            if panel_index == len(histogram_rows):
                draw.text(
                    ((x1 + x2) / 2, (y1 + y2) / 2 - 16),
                    "β=2.25: 39/64 complete",
                    fill=MUTED,
                    font=font(21, bold=True),
                    anchor="mm",
                )
                draw.text(
                    ((x1 + x2) / 2, (y1 + y2) / 2 + 22),
                    "omitted until the block is complete",
                    fill=MUTED,
                    font=font(17),
                    anchor="mm",
                )
            continue
        beta, beta_samples, counts = histogram_rows[panel_index]
        values = [float(sample["nodes"]) for sample in beta_samples]
        plot_left, plot_right = x1 + 52, x2 - 22
        plot_top, plot_bottom = y1 + 92, y2 - 47

        def sx(value: float) -> float:
            return plot_left + (math.log10(value) - math.log10(xmin)) / (
                math.log10(xmax) - math.log10(xmin)
            ) * (plot_right - plot_left)

        def sy(fraction: float) -> float:
            return plot_bottom - fraction / ymax * (plot_bottom - plot_top)

        for fraction in (0, ymax / 2, ymax):
            y = sy(fraction)
            draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        for index, count in enumerate(counts):
            fraction = count / 64
            bx1 = sx(edges[index]) + 1
            bx2 = sx(edges[index + 1]) - 1
            draw.rectangle((bx1, sy(fraction), bx2, plot_bottom), fill="#4e79a7")
        median_value = quantile(values, 0.5)
        median_x = sx(median_value)
        dashed_line(draw, (median_x, plot_top), (median_x, plot_bottom), fill="#e15759", width=3, dash=8)
        draw.text((x1 + 24, y1 + 18), f"β={beta:g} · n=64", fill=INK, font=font(21, bold=True))
        draw.text(
            (x1 + 24, y1 + 53),
            f"p50 {median_value:,.0f}  ·  p90 {quantile(values, .9):,.0f}  ·  max {max(values):,.0f}",
            fill=MUTED,
            font=font(14),
            anchor="la",
        )
        draw.text((plot_left - 6, plot_top), f"{ymax:.0%}", fill=MUTED, font=font(14), anchor="ra")
        draw.text((plot_left - 6, plot_bottom), "0%", fill=MUTED, font=font(14), anchor="ra")
        for value, label in ((1, "1"), (10, "10"), (100, "100"), (1000, "1k"), (10_000, "10k")):
            x = sx(value)
            draw.line((x, plot_bottom, x, plot_bottom + 5), fill=MUTED, width=1)
            draw.text((x, plot_bottom + 10), label, fill=MUTED, font=font(14), anchor="ma")

    draw.rectangle((80, 1346, 104, 1370), fill="#4e79a7")
    draw.text((115, 1358), "tree-count fraction", fill=INK, font=font(17), anchor="lm")
    dashed_line(draw, (330, 1346), (330, 1370), fill="#e15759", width=3, dash=6)
    draw.text((345, 1358), "median", fill=INK, font=font(17), anchor="lm")
    draw.text(
        (1710, 1358),
        "HU 32 bb · checkpoint 814 · fixed β per tree · sampled nodes per tree",
        fill=MUTED,
        font=font(16),
        anchor="rm",
    )
    save(image, "04_nodes_per_tree_facets_64.png")


def plot_constant_beta_growth(samples: dict[float, list[dict[str, object]]]) -> None:
    image, draw = canvas(
        "Constant β has a sharp tree-growth transition",
        "Each point is one tree · triangles reached the 50k safety stop and are right-censored",
    )
    box = (75, 165, 1725, 1015)
    panel(draw, box)
    left, top, right, bottom = 165, 225, 1660, 895
    betas = list(samples)
    ymin, ymax = 1.0, 60_000.0

    def sx(index: int) -> float:
        return left + index / max(1, len(betas) - 1) * (right - left)

    def sy(value: float) -> float:
        value = max(ymin, min(ymax, value))
        return bottom - (math.log10(value) - math.log10(ymin)) / (
            math.log10(ymax) - math.log10(ymin)
        ) * (bottom - top)

    for value, label in ((1, "1"), (10, "10"), (100, "100"), (1000, "1k"), (10_000, "10k"), (50_000, "50k")):
        y = sy(value)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 18, y), label, fill=MUTED, font=font(18), anchor="rm")
    dashed_line(draw, (left, sy(10_000)), (right, sy(10_000)), fill=TARGET, width=4)
    draw.text((right, sy(10_000) - 9), "10k target", fill=TARGET, font=font(18, bold=True), anchor="rb")
    dashed_line(draw, (left, sy(50_000)), (right, sy(50_000)), fill="#8f1d1d", width=3)
    draw.text((right, sy(50_000) + 9), "50k censoring boundary", fill="#8f1d1d", font=font(17, bold=True), anchor="ra")

    rng = random.Random(814)
    median_points = []
    for index, beta in enumerate(betas):
        rows = samples[beta]
        center = sx(index)
        spread = min(42, (right - left) / max(1, len(betas)) * 0.35)
        values = [float(row["nodes"]) for row in rows]
        for row in rows:
            x = center + rng.uniform(-spread, spread)
            y = sy(float(row["nodes"]))
            if row.get("stop_cutoff_reached", 0):
                draw.polygon(((x, y - 8), (x - 7, y + 7), (x + 7, y + 7)), fill="#c43c39")
            else:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#4e79a7", outline="#315e85")
        observed_median = quantile(values, 0.5)
        median_points.append((center, sy(observed_median)))
        draw.line((center - 30, sy(observed_median), center + 30, sy(observed_median)), fill=INK, width=5)
        draw.multiline_text(
            (center, bottom + 25),
            f"{beta:g}\nn={len(rows)}",
            fill=INK,
            font=font(16),
            anchor="ma",
            align="center",
            spacing=2,
        )
    draw.line(median_points, fill=INK, width=2)
    draw.text((left, 190), "Observed allocated nodes (log scale)", fill=INK, font=font(20, bold=True))
    draw.ellipse((500, 960, 510, 970), fill="#4e79a7", outline="#315e85")
    draw.text((525, 965), "completed tree", fill=INK, font=font(18), anchor="lm")
    draw.polygon(((780, 957), (773, 972), (787, 972)), fill="#c43c39")
    draw.text((800, 965), "right-censored at 50k", fill=INK, font=font(18), anchor="lm")
    draw.line((1120, 965, 1175, 965), fill=INK, width=5)
    draw.text((1190, 965), "observed median", fill=INK, font=font(18), anchor="lm")
    counts = ", ".join(f"β={beta:g}: {len(rows)}" for beta, rows in samples.items() if beta >= 2.25)
    footer(draw, f"HU 32 bb · checkpoint 814 · snapshot counts: {counts}")
    save(image, "05_constant_beta_tree_growth.png")


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def plot_growth_probabilities(samples: dict[float, list[dict[str, object]]]) -> None:
    image, draw = canvas(
        "Probability of tiny, target-sized, and censored trees",
        "Observed rates with 95% Wilson intervals · cutoff hits imply true size ≥50k",
    )
    box = (75, 165, 1725, 1015)
    panel(draw, box)
    left, top, right, bottom = 165, 230, 1650, 885
    betas = list(samples)

    def sx(index: int) -> float:
        return left + index / max(1, len(betas) - 1) * (right - left)

    def sy(value: float) -> float:
        return bottom - value * (bottom - top)

    for probability in (0, 0.25, 0.5, 0.75, 1.0):
        y = sy(probability)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 20, y), f"{probability:.0%}", fill=MUTED, font=font(18), anchor="rm")
    definitions = (
        ("tiny: ≤10 nodes", "#6b7785", lambda row: float(row["nodes"]) <= 10),
        ("target: ≥10k nodes", "#2f8f46", lambda row: float(row["nodes"]) >= 10_000),
        ("censored: ≥50k nodes", "#c43c39", lambda row: bool(row.get("stop_cutoff_reached", 0))),
    )
    for label, color, predicate in definitions:
        points = []
        for index, beta in enumerate(betas):
            rows = samples[beta]
            successes = sum(predicate(row) for row in rows)
            rate = successes / len(rows)
            low, high = wilson_interval(successes, len(rows))
            x = sx(index)
            draw.line((x, sy(low), x, sy(high)), fill=color, width=4)
            draw.line((x - 7, sy(low), x + 7, sy(low)), fill=color, width=3)
            draw.line((x - 7, sy(high), x + 7, sy(high)), fill=color, width=3)
            draw.ellipse((x - 7, sy(rate) - 7, x + 7, sy(rate) + 7), fill=PANEL, outline=color, width=4)
            points.append((x, sy(rate)))
        draw.line(points, fill=color, width=4)
    for index, beta in enumerate(betas):
        draw.multiline_text(
            (sx(index), bottom + 25), f"{beta:g}\nn={len(samples[beta])}", fill=INK, font=font(16), anchor="ma", align="center"
        )
    legend_x = 350
    for index, (label, color, _) in enumerate(definitions):
        x = legend_x + index * 420
        draw.line((x, 965, x + 55, 965), fill=color, width=5)
        draw.ellipse((x + 21, 958, x + 35, 972), fill=PANEL, outline=color, width=3)
        draw.text((x + 70, 965), label, fill=INK, font=font(18), anchor="lm")
    draw.text((left, 195), "Fraction of sampled trees", fill=INK, font=font(20, bold=True))
    footer(draw, "HU 32 bb · checkpoint 814 · constant β · intervals reflect finite tree count, not censoring uncertainty")
    save(image, "06_constant_beta_outcome_probabilities.png")


def plot_street_distributions(samples: dict[float, list[dict[str, object]]]) -> None:
    street_labels = ("preflop", "flop", "turn", "river")
    eligible = {
        beta: [row for row in rows if all(f"nodes_{street}" in row for street in street_labels)]
        for beta, rows in samples.items()
    }
    eligible = {beta: rows for beta, rows in eligible.items() if rows}
    image, draw = canvas(
        "Where constant-β growth accumulates in the poker tree",
        "Per-tree node counts by street · bars show 10th–90th and 25th–75th percentiles · circles are medians",
    )
    box = (75, 165, 1725, 1015)
    panel(draw, box)
    left, top, right, bottom = 170, 230, 1650, 885
    betas = list(eligible)
    ymin, ymax = 0.8, 60_000.0

    def group_x(index: int) -> float:
        return left + (index + 0.5) / len(betas) * (right - left)

    def sy(value: float) -> float:
        value = max(ymin, min(ymax, value))
        return bottom - (math.log10(value) - math.log10(ymin)) / (
            math.log10(ymax) - math.log10(ymin)
        ) * (bottom - top)

    for value, label in ((1, "1"), (10, "10"), (100, "100"), (1000, "1k"), (10_000, "10k"), (50_000, "50k")):
        y = sy(value)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 18, y), label, fill=MUTED, font=font(18), anchor="rm")
    offsets = (-120, -40, 40, 120)
    colors = ("#4e79a7", "#59a14f", "#f28e2b", "#b07aa1")
    for beta_index, beta in enumerate(betas):
        rows = eligible[beta]
        center = group_x(beta_index)
        for street_index, street in enumerate(street_labels):
            values = [float(row[f"nodes_{street}"]) for row in rows]
            q10, q25, q50, q75, q90 = [quantile(values, p) for p in (.1, .25, .5, .75, .9)]
            x = center + offsets[street_index]
            color = colors[street_index]
            draw.line((x, sy(q10), x, sy(q90)), fill=color, width=6)
            draw.line((x, sy(q25), x, sy(q75)), fill=color, width=18)
            draw.ellipse((x - 8, sy(q50) - 8, x + 8, sy(q50) + 8), fill=PANEL, outline=color, width=4)
        censored = sum(bool(row.get("stop_cutoff_reached", 0)) for row in rows)
        draw.multiline_text(
            (center, bottom + 28),
            f"β={beta:g} · n={len(rows)}\ncensored={censored}",
            fill=INK,
            font=font(18, bold=True),
            anchor="ma",
            align="center",
            spacing=4,
        )
    for index, (street, color) in enumerate(zip(street_labels, colors, strict=True)):
        x = 380 + index * 300
        draw.line((x, 970 - 15, x, 970 + 15), fill=color, width=14)
        draw.ellipse((x - 7, 963, x + 7, 977), fill=PANEL, outline=color, width=3)
        draw.text((x + 22, 970), street, fill=INK, font=font(18), anchor="lm")
    draw.text((left, 195), "Allocated nodes per street (log scale)", fill=INK, font=font(20, bold=True))
    footer(draw, "Counts are measured at natural completion or at the 50k stop; zeros are shown at the plot floor")
    save(image, "07_nodes_by_street_constant_beta.png")


def main() -> None:
    random.seed(0)
    samples = load_completed_samples()
    plot_survival(samples)
    plot_quantiles(samples)
    plot_cutoff_pilots()
    plot_per_beta_histograms(samples)
    all_samples = load_all_sweep_samples()
    plot_constant_beta_growth(all_samples)
    plot_growth_probabilities(all_samples)
    plot_street_distributions(all_samples)
    for path in sorted(OUTPUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
