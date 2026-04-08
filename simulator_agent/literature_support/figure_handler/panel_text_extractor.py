"""Extract per-panel conditions (T, P) from figure images using OCR.

Uses pytesseract with spatial clustering to group text annotations
into panel-level condition sets.

Only runs as a targeted fallback when:
- Figure is identified as multi-panel
- Caption suggests experimental results
- Per-panel conditions are missing from text extraction
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

try:
    from PIL import Image
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


@dataclass
class PanelCondition:
    """Per-panel condition extracted from figure annotation."""
    panel_label: str | None     # "a", "b", "c", etc.
    temperature_text: str | None
    pressure_text: str | None
    bbox_x: float               # approximate x position (for panel grouping)
    bbox_y: float               # approximate y position


# ── Patterns for filtering OCR output ────────────────────────────────────

_TEMP_PATTERN = re.compile(r'(\d{3,4})\s*K\b')
_PRESS_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*atm\b')
_PANEL_LABEL = re.compile(r'^\s*\(?([a-z])\)?\s*$')


def ocr_available() -> bool:
    """Check if OCR dependencies are installed."""
    if not _OCR_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_panel_conditions(
    image_bytes: bytes,
) -> list[PanelCondition]:
    """Run OCR on a figure image and extract per-panel T/P annotations.

    Returns a list of PanelCondition objects, one per detected panel.
    Returns empty list if OCR is unavailable or finds nothing useful.
    """
    if not ocr_available():
        return []

    image = Image.open(io.BytesIO(image_bytes))

    # Get detailed OCR output with bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Extract text with positions
    text_boxes: list[dict] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        conf = int(data["conf"][i])
        if conf < 30:  # skip low-confidence OCR results
            continue
        text_boxes.append({
            "text": text,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
            "conf": conf,
        })

    if not text_boxes:
        return []

    # Reconstruct nearby text fragments into lines
    # Then filter for temperature and pressure patterns
    conditions = _extract_conditions_from_boxes(text_boxes, image.width, image.height)
    return conditions


def _extract_conditions_from_boxes(
    boxes: list[dict],
    img_width: int,
    img_height: int,
) -> list[PanelCondition]:
    """Group OCR text boxes into panel conditions using spatial clustering."""
    # First pass: find all temperature and pressure mentions
    temp_hits: list[dict] = []
    press_hits: list[dict] = []
    panel_labels: list[dict] = []

    # Combine nearby text boxes into larger strings for pattern matching
    # Sort by y then x to reconstruct reading order
    sorted_boxes = sorted(boxes, key=lambda b: (b["y"], b["x"]))

    # Group boxes into lines (similar y coordinate, within 15px)
    lines: list[list[dict]] = []
    current_line: list[dict] = []
    current_y = -999

    for box in sorted_boxes:
        if abs(box["y"] - current_y) > 15:
            if current_line:
                lines.append(current_line)
            current_line = [box]
            current_y = box["y"]
        else:
            current_line.append(box)
    if current_line:
        lines.append(current_line)

    # For each line, concatenate text and check for patterns
    for line in lines:
        line_text = " ".join(b["text"] for b in line)
        avg_x = sum(b["x"] for b in line) / len(line)
        avg_y = sum(b["y"] for b in line) / len(line)

        # Check for temperature
        t_match = _TEMP_PATTERN.search(line_text)
        if t_match:
            temp_hits.append({
                "text": t_match.group(0),
                "value": float(t_match.group(1)),
                "x": avg_x,
                "y": avg_y,
            })

        # Check for pressure
        p_match = _PRESS_PATTERN.search(line_text)
        if p_match:
            press_hits.append({
                "text": p_match.group(0),
                "value": float(p_match.group(1)),
                "x": avg_x,
                "y": avg_y,
            })

        # Check for panel labels
        for box in line:
            lbl = _PANEL_LABEL.match(box["text"])
            if lbl:
                panel_labels.append({
                    "label": lbl.group(1),
                    "x": box["x"],
                    "y": box["y"],
                })

    if not temp_hits:
        return []

    # Pair temperatures with pressures by spatial proximity
    panels = _pair_conditions(temp_hits, press_hits, panel_labels)
    return panels


def _pair_conditions(
    temps: list[dict],
    pressures: list[dict],
    labels: list[dict],
) -> list[PanelCondition]:
    """Pair temperature and pressure hits by spatial proximity."""
    panels: list[PanelCondition] = []
    used_pressures: set[int] = set()

    for temp in temps:
        # Find nearest pressure (within 100px vertically)
        best_press = None
        best_dist = 100.0  # max pairing distance in pixels

        for j, press in enumerate(pressures):
            if j in used_pressures:
                continue
            dist = abs(temp["y"] - press["y"]) + abs(temp["x"] - press["x"]) * 0.3
            if dist < best_dist:
                best_dist = dist
                best_press = (j, press)

        # Find nearest panel label
        panel_label = None
        for lbl in labels:
            if abs(lbl["x"] - temp["x"]) < 200 and abs(lbl["y"] - temp["y"]) < 200:
                panel_label = lbl["label"]
                break

        press_text = None
        if best_press:
            used_pressures.add(best_press[0])
            press_text = best_press[1]["text"]

        panels.append(PanelCondition(
            panel_label=panel_label,
            temperature_text=temp["text"],
            pressure_text=press_text,
            bbox_x=temp["x"],
            bbox_y=temp["y"],
        ))

    # Sort by position (top-left to bottom-right, row by row)
    panels.sort(key=lambda p: (p.bbox_y, p.bbox_x))

    # If no panel labels, assign alphabetically by position
    if not any(p.panel_label for p in panels):
        for i, p in enumerate(panels):
            p.panel_label = chr(ord("a") + i)

    return panels
