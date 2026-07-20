"""Shared drawing / annotation helpers for the FIDO visualization scripts."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Bundled assets (Onest font + SynthesEyes logo).
ASSETS = Path(__file__).resolve().parent / "assets"
ONEST_PATH = ASSETS / "onest.ttf"
LOGO_PATH = ASSETS / "syntheseyes_logo.png"

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}
_logo_cache = None
_logo_loaded = False

# Shared layout: both task windows render at this height; the logo is a fixed
# width so it does not scale up on the wider task 2 canvas.
CANVAS_H = 768
LOGO_W = 300


def fit_height(img: np.ndarray, height: int = CANVAS_H) -> np.ndarray:
    """Resize `img` to `height` px tall, preserving aspect ratio."""
    if img.shape[0] == height:
        return img
    scale = height / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), height),
                      interpolation=cv2.INTER_AREA)


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Return a cached Onest font of the given pixel size."""
    key = int(size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(str(ONEST_PATH), key)
    return _font_cache[key]


def draw_text(img: np.ndarray, text: str, org, size: int = 22,
              color=(255, 255, 255), anchor: str = "la") -> np.ndarray:
    """Draw `text` in the Onest font on a BGR image; returns a new array.

    `org` is (x, y); `anchor` follows PIL conventions ("la" = top-left).
    """
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((int(org[0]), int(org[1])), text, font=_font(size),
              fill=(int(color[2]), int(color[1]), int(color[0])), anchor=anchor)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_point(stereo_bgr: np.ndarray, x: float, y: float) -> np.ndarray:
    """Draw a crosshair marker at (x, y) on a copy of the stereo image."""
    out = stereo_bgr.copy()
    px, py = int(round(x)), int(round(y))
    color = (200, 200, 200)
    cv2.drawMarker(out, (px, py), color, markerType=cv2.MARKER_CROSS,
                   markerSize=40, thickness=2, line_type=cv2.LINE_AA)
    cv2.circle(out, (px, py), 10, color, 2, lineType=cv2.LINE_AA)
    return draw_text(out, f"({x:.1f}, {y:.1f})", (px + 14, py - 28),
                     size=22, color=color)


def draw_instrument_point(bscan_bgr: np.ndarray, pt) -> np.ndarray:
    """Draw a marker at the lowest-x instrument point on a copy of the B-scan."""
    out = bscan_bgr.copy()
    if pt is None:
        return out
    x, y = pt
    color = (200, 200, 200)  # green (BGR)
    cv2.drawMarker(out, (x, y), color, markerType=cv2.MARKER_TILTED_CROSS,
                   markerSize=26, thickness=2, line_type=cv2.LINE_AA)
    cv2.circle(out, (x, y), 8, color, 2, lineType=cv2.LINE_AA)
    return draw_text(out, f"({x}, {y})", (x + 12, y - 10),
                     size=20, color=color)


# Colors (BGR) for the two OCT crosshair lines and their matching B-scans.
# Index 0 -> Start 0/End 0 line & Bscan 00; index 1 -> Start 1/End 1 & Bscan 01.
CROSSHAIR_COLORS = ((252, 3, 194), (248, 252, 3))  # cyan, magenta


def draw_crosshair_lines(img: np.ndarray, lines, colors=CROSSHAIR_COLORS,
                         thickness: int = 2) -> np.ndarray:
    """Draw each (start, end) line in `lines`; a dot marks the Start point."""
    out = img.copy()
    for (start, end), color in zip(lines, colors):
        p0 = (int(round(start[0])), int(round(start[1])))
        p1 = (int(round(end[0])), int(round(end[1])))
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.circle(out, p0, 5, color, -1, cv2.LINE_AA)
    return out


def banner(img: np.ndarray, text: str) -> np.ndarray:
    """Return a copy of `img` with a label drawn across the top."""
    return draw_text(img, text, (8, 6), size=22, color=(255, 255, 255))


def _logo() -> np.ndarray:
    """Load (and cache) the SynthesEyes logo as BGRA, or None if missing."""
    global _logo_cache, _logo_loaded
    if not _logo_loaded:
        _logo_cache = cv2.imread(str(LOGO_PATH), cv2.IMREAD_UNCHANGED)
        _logo_loaded = True
    return _logo_cache


def paste_logo(img: np.ndarray, target_w: int = None, margin: int = 14,
               corner: str = "tr") -> np.ndarray:
    """Alpha-composite the SynthesEyes logo into a corner of `img`.

    corner: 'tr' (top-right), 'tl', 'br', 'bl'. `target_w` defaults to a
    quarter of the image width.
    """
    logo = _logo()
    if logo is None:
        return img
    out = img.copy()
    H, W = out.shape[:2]
    lh, lw = logo.shape[:2]

    if target_w is None:
        target_w = W // 4
    scale = target_w / lw
    nw, nh = max(1, int(lw * scale)), max(1, int(lh * scale))
    logo_r = cv2.resize(logo, (nw, nh), interpolation=cv2.INTER_AREA)

    x0 = margin if "l" in corner else W - nw - margin
    y0 = margin if "t" in corner else H - nh - margin
    x0, y0 = max(0, x0), max(0, y0)

    if logo_r.shape[2] == 4:
        rgb = logo_r[..., :3].astype(np.float32)
        a = logo_r[..., 3:4].astype(np.float32) / 255.0
    else:
        rgb = logo_r.astype(np.float32)
        a = np.ones((nh, nw, 1), np.float32)

    roi = out[y0:y0 + nh, x0:x0 + nw].astype(np.float32)
    out[y0:y0 + nh, x0:x0 + nw] = (roi * (1 - a) + rgb * a).astype(np.uint8)
    return out


def overlay(stereo_bgr: np.ndarray, warped_gray: np.ndarray,
            alpha: float = 0.6, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Alpha-blend a colorized warped projection onto the stereo image where it is non-zero."""
    colored = cv2.applyColorMap(warped_gray, colormap)
    mask = (warped_gray > 0).astype(np.float32)[..., None] * alpha
    return (stereo_bgr * (1 - mask) + colored * mask).astype(np.uint8)




def compose_phase1_image(stereo_marked: np.ndarray, scans, z: float,
                         scan_colors=CROSSHAIR_COLORS) -> np.ndarray:
    """Lay the marked stereo image beside the two B-scans, with the z label.

    Each B-scan gets a colored bar along its bottom edge matching the color of
    its corresponding OCT crosshair line in the stereo image.
    """
    s_h, s_w = stereo_marked.shape[:2]

    # Stack the two B-scans vertically into a column the same height as stereo.
    scan_w = max(sc.shape[1] for sc in scans)
    col_h = s_h // 2
    bar_h = 14
    panels = []
    for i, sc in enumerate(scans):
        panel = cv2.resize(sc, (scan_w, col_h), interpolation=cv2.INTER_AREA)
        panel = banner(panel, f"Bscan {i:02d}")
        color = scan_colors[i % len(scan_colors)]
        # Colored bar along the bottom, matching this scan's crosshair line.
        cv2.rectangle(panel, (0, col_h - bar_h), (scan_w, col_h), color, -1)
        panels.append(panel)
    scan_col = np.vstack(panels)

    # Match heights exactly before hstack (integer-division rounding guard).
    if scan_col.shape[0] != s_h:
        scan_col = cv2.resize(scan_col, (scan_w, s_h), interpolation=cv2.INTER_AREA)

    stereo_panel = banner(
        stereo_marked, f"Stereo Left\nRetina distance: {z:.2f} microns")
    composite = fit_height(np.hstack([stereo_panel, scan_col]), CANVAS_H)
    return paste_logo(composite, target_w=LOGO_W, corner="bl")


def compose_phase2_image(overlay_bgr: np.ndarray, warped_gray: np.ndarray,
                         bscan_grid: np.ndarray = None) -> np.ndarray:
    """Lay the overlay, warped projection, and B-scan montage in one image.

    All panels are aligned to the overlay's height and hstacked; the whole
    canvas is then resized to the shared CANVAS_W width.
    """
    over = banner(overlay_bgr, "Stereo Left + OCT overlay")
    h = over.shape[0]

    def _panel(gray, label):
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gh, gw = bgr.shape[:2]
        if gh != h:
            bgr = cv2.resize(bgr, (max(1, int(gw * h / gh)), h),
                             interpolation=cv2.INTER_AREA)
        return banner(bgr, label)

    panels = [over, _panel(warped_gray, "Warped OCT projection (Average Projection)")]
    if bscan_grid is not None:
        panels.append(_panel(bscan_grid, "128 B-scans"))

    # Resize the whole canvas to the shared height, then brand it.
    composite = fit_height(np.hstack(panels), CANVAS_H)
    return paste_logo(composite, target_w=LOGO_W, corner="bl")
