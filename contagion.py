"""
Contagion: shape detection by recursive pixel spreading.

The image is pushed to infinite contrast (every pixel becomes 0 or 1). A scan
runs over the rows; the first black pixel it meets "infects" every black pixel
within a radius, which infects its own neighbours in turn, until the spread
finds nothing left to infect. Each infected pixel is set back to 0, so when the
scan resumes the shape is gone. When the image is all zeros, every shape has
been collected as a cluster of pixels.

Clusters smaller than a sieve size are dropped as noise. Each remaining cluster
is classified from its barycentre: the distance from the barycentre to every
contour pixel, its mean and standard deviation, and the number of corners that
show up as peaks when that distance is plotted against the angle.
"""

from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageOps


# ------------------------------------------------------------------ step 1
def binarize(image: Image.Image, threshold: int | None = None) -> np.ndarray:
    """Infinite contrast: 1 where the pixel is dark, 0 elsewhere.

    With no threshold given, Otsu's method picks the one that best separates
    the dark and light populations of the histogram.
    """
    grey = np.asarray(image.convert("L"), dtype=np.uint8)
    if threshold is None:
        threshold = _otsu(grey)
    return (grey < threshold).astype(np.uint8)


def _otsu(grey: np.ndarray) -> int:
    hist = np.bincount(grey.ravel(), minlength=256).astype(np.float64)
    total = grey.size
    levels = np.arange(256)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(hist * levels)
    mean_bg = np.divide(sum_bg, weight_bg, out=np.zeros(256), where=weight_bg > 0)
    mean_fg = np.divide(sum_bg[-1] - sum_bg, weight_fg, out=np.zeros(256), where=weight_fg > 0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(between)) + 1


# ------------------------------------------------------------- steps 2 to 4
def find_clusters(binary: np.ndarray, radius: int = 1) -> list[list[tuple[int, int]]]:
    """Scan the rows; every black pixel found starts a contagion.

    `radius` is how far a pixel infects: 1 means the 8 touching neighbours,
    a larger radius jumps small gaps so a dashed outline stays one shape.
    The input array is not modified.
    """
    grid = binary.copy()
    height, width = grid.shape
    offsets = [(dy, dx)
               for dy in range(-radius, radius + 1)
               for dx in range(-radius, radius + 1)
               if (dy, dx) != (0, 0)]
    clusters: list[list[tuple[int, int]]] = []

    def spread(y: int, x: int, cluster: list[tuple[int, int]]) -> None:
        # Exit condition: no infected neighbour left, the loop just ends.
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and grid[ny, nx]:
                grid[ny, nx] = 0
                cluster.append((ny, nx))
                spread(ny, nx, cluster)

    def scan() -> None:
        for y in range(height):
            row = grid[y]
            for x in range(width):
                if row[x]:
                    row[x] = 0
                    cluster = [(y, x)]
                    spread(y, x, cluster)
                    clusters.append(cluster)

    _run_deep(scan)
    return clusters


def _run_deep(fn) -> None:
    """Run `fn` with room for deep recursion.

    The contagion recurses once per pixel, so a large shape goes tens of
    thousands of calls deep: far past Python's default limit of 1000, and past
    what the main thread's stack can hold. A worker thread gets a big stack.
    """
    old_limit = sys.getrecursionlimit()
    old_stack = threading.stack_size()
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(512 * 1024 * 1024)
    errors: list[BaseException] = []

    def target() -> None:
        try:
            fn()
        except BaseException as exc:  # re-raised in the caller's thread
            errors.append(exc)

    try:
        worker = threading.Thread(target=target)
        worker.start()
        worker.join()
    finally:
        threading.stack_size(old_stack)
        sys.setrecursionlimit(old_limit)
    if errors:
        raise errors[0]


# ------------------------------------------------------------------ step 5
def sieve(clusters: list[list[tuple[int, int]]], min_pixels: int) -> list[list[tuple[int, int]]]:
    """Drop every cluster smaller than `min_pixels`: specks, dust, noise."""
    return [c for c in clusters if len(c) >= min_pixels]


# ------------------------------------------------------------------ step 6
@dataclass
class Shape:
    label: str
    barycentre: tuple[float, float]          # (x, y)
    contour: list[tuple[int, int]]           # (x, y) contour pixels
    pixels: int
    mean_distance: float
    std_distance: float
    corners: int
    profile: np.ndarray = field(repr=False)


ANGLE_BINS = 90                # 4 degrees per bin
CIRCLE_MAX_CV = 0.05           # std / mean of the contour distance
PEAK_HALF_WINDOW = 5           # a corner is the highest point within +/- 20 degrees
PEAK_MIN_RISE = 0.04           # ... and stands 4 % above the mean distance
SQUARE_MIN_SIDE_RATIO = 0.85   # shorter half-side / longer half-side
MIN_SOLIDITY = 0.9             # filled area / convex hull area


def classify(cluster: list[tuple[int, int]]) -> Shape:
    # Holes do not change what a shape is: a ring is a circle, an icon on a
    # round button is still a round button. Work on the filled silhouette, and
    # only its outer contour.
    filled = _fill_holes(cluster)
    pts = np.array(filled)                       # rows of (y, x)
    cy, cx = pts.mean(axis=0)
    contour = _contour(filled)
    cont = np.array(contour, dtype=np.float64)   # rows of (y, x)
    dy, dx = cont[:, 0] - cy, cont[:, 1] - cx
    dist = np.hypot(dx, dy)
    mean, std = float(dist.mean()), float(dist.std())
    cv = std / mean if mean else 0.0

    profile = _radial_profile(np.arctan2(dy, dx), dist)
    peaks = _peaks(profile)

    if _solidity(filled, contour) < MIN_SOLIDITY:
        label = "unknown"     # concave: a letter, a blob, two shapes merged
    elif cv < CIRCLE_MAX_CV:
        label = "circle"
    elif len(peaks) == 2:
        label = "ellipse"
    elif len(peaks) == 3:
        label = "triangle"
    elif len(peaks) == 4:
        label = "square" if _even_sides(profile, peaks) else "rectangle"
    else:
        label = "unknown"

    return Shape(
        label=label,
        barycentre=(float(cx), float(cy)),
        contour=[(x, y) for y, x in contour],
        pixels=len(cluster),
        mean_distance=mean,
        std_distance=std,
        corners=len(peaks),
        profile=profile,
    )


def _fill_holes(cluster: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Add every background pixel the outside cannot reach: the holes."""
    pts = np.array(cluster)
    y0, x0 = pts.min(axis=0) - 1
    y1, x1 = pts.max(axis=0) + 1
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
    mask[pts[:, 0] - y0, pts[:, 1] - x0] = True
    outside = np.zeros_like(mask)
    outside[0, 0] = True
    stack = [(0, 0)]                     # the frame is one pixel wider than the shape
    h, w = mask.shape
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    ys, xs = np.nonzero(~outside)
    return list(zip((ys + y0).tolist(), (xs + x0).tolist()))


def _solidity(filled: list[tuple[int, int]], contour: list[tuple[int, int]]) -> float:
    """Filled area over the area of its convex hull: 1 for any convex shape."""
    hull = _convex_hull([(x, y) for y, x in contour])
    if len(hull) < 3:
        return 1.0
    xs, ys = np.array(hull, dtype=np.float64).T
    area = 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    # Pixel centres sit half a pixel inside the drawn edge: pad the hull by
    # half its perimeter so small convex shapes are not read as concave.
    perimeter = float(np.hypot(np.diff(xs, append=xs[0]), np.diff(ys, append=ys[0])).sum())
    return min(1.0, len(filled) / (area + perimeter / 2 + 1))


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Andrew's monotone chain."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _contour(cluster: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Pixels of the cluster with at least one side touching the outside."""
    inside = set(cluster)
    return [(y, x) for y, x in cluster
            if (y - 1, x) not in inside or (y + 1, x) not in inside
            or (y, x - 1) not in inside or (y, x + 1) not in inside]


def _radial_profile(angles: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Farthest contour point per angular slice, lightly smoothed.

    Plotted against the angle, this distance is flat for a circle and rises to
    one bump per corner for a polygon, wherever the shape is rotated.
    """
    bins = ((angles + math.pi) / (2 * math.pi) * ANGLE_BINS).astype(int) % ANGLE_BINS
    profile = np.full(ANGLE_BINS, np.nan)
    for b, d in zip(bins, dist):
        if np.isnan(profile[b]) or d > profile[b]:
            profile[b] = d
    # Small shapes leave some slices empty: fill them from their neighbours.
    known = ~np.isnan(profile)
    if not known.all():
        idx = np.arange(ANGLE_BINS)
        profile = np.interp(idx, idx[known], profile[known], period=ANGLE_BINS)
    kernel = np.ones(3) / 3
    return np.convolve(np.concatenate([profile[-1:], profile, profile[:1]]), kernel, "valid")


def _peaks(profile: np.ndarray) -> list[int]:
    n = len(profile)
    floor = profile.mean() * (1 + PEAK_MIN_RISE)
    peaks = []
    for i in range(n):
        window = [profile[(i + k) % n] for k in range(-PEAK_HALF_WINDOW, PEAK_HALF_WINDOW + 1)]
        if profile[i] >= floor and profile[i] == max(window):
            # A flat top would count twice: keep its first bin only.
            if not peaks or (i - peaks[-1]) > PEAK_HALF_WINDOW:
                peaks.append(i)
    if len(peaks) > 1 and (peaks[0] + n - peaks[-1]) <= PEAK_HALF_WINDOW:
        peaks.pop()
    return peaks


def _even_sides(profile: np.ndarray, peaks: list[int]) -> bool:
    """For four corners, the dips between them are the half-sides."""
    n = len(profile)
    dips = []
    for a, b in zip(peaks, peaks[1:] + [peaks[0] + n]):
        dips.append(min(profile[i % n] for i in range(a, b + 1)))
    return min(dips) / max(dips) >= SQUARE_MIN_SIDE_RATIO


# ------------------------------------------------------------------ step 7
COLOURS = {
    "circle": (230, 57, 70), "ellipse": (244, 162, 97), "triangle": (42, 157, 143),
    "square": (38, 70, 83), "rectangle": (69, 123, 157), "unknown": (150, 150, 150),
}


def overlay(image: Image.Image, shapes: list[Shape]) -> Image.Image:
    """Draw each contour and its label at the barycentre over the original."""
    # Fade the original so the overlay reads on top of solid black shapes.
    out = Image.blend(image.convert("RGB"), Image.new("RGB", image.size, (255, 255, 255)), 0.75)
    draw = ImageDraw.Draw(out)
    for s in shapes:
        colour = COLOURS.get(s.label, (255, 0, 255))
        for x, y in s.contour:
            draw.rectangle((x - 1, y - 1, x + 1, y + 1), fill=colour)
        x, y = s.barycentre
        draw.line((x - 6, y, x + 6, y), fill=colour, width=2)
        draw.line((x, y - 6, x, y + 6), fill=colour, width=2)
        text = s.label
        left, top, right, bottom = draw.textbbox((x + 10, y - 7), text)
        draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=colour)
        draw.text((x + 10, y - 7), text, fill=(255, 255, 255))
    return out


# ------------------------------------------------------------------ all
def detect(image: Image.Image, radius: int = 1, min_pixels: int = 50,
           threshold: int | None = None) -> list[Shape]:
    binary = binarize(image, threshold)
    return [classify(c) for c in sieve(find_clusters(binary, radius), min_pixels)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect shapes by contagion.")
    parser.add_argument("image")
    parser.add_argument("-o", "--output", default="contagion_out.png")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=50)
    parser.add_argument("--threshold", type=int, default=None)
    args = parser.parse_args()

    # Phone photos are stored sideways with an EXIF flag saying how to turn them.
    img = ImageOps.exif_transpose(Image.open(args.image))
    found = detect(img, args.radius, args.min_pixels, args.threshold)
    for s in found:
        print(f"{s.label:9s} at ({s.barycentre[0]:.0f}, {s.barycentre[1]:.0f})  "
              f"{s.pixels} px  corners={s.corners}  cv={s.std_distance / s.mean_distance:.3f}")
    overlay(img, found).save(args.output)
    print(f"-> {args.output}")
