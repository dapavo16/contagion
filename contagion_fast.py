"""
Contagion, compiled: the same algorithm with its hot loops run through Numba.

Pure Python spends its time in two loops that touch every pixel one by one:
the contagion itself, and filling each shape's holes before classification.
Here both are compiled to machine code by Numba, and shapes travel as label
images and bounding boxes instead of Python lists of coordinates. The steps
and the classification rules are unchanged: `describe` is shared with
`contagion.py`, so both versions label shapes the same way.

The compiled contagion runs on an explicit stack: Numba compiles recursion,
but the depth limit of a real photo (a shape of a million pixels) would still
apply.

    pip install numba
    python contagion_fast.py photo.jpg -o out.png
"""

from __future__ import annotations

import numpy as np
from numba import njit
from PIL import Image

from contagion import Shape, binarize, describe, overlay


# ------------------------------------------------------------- steps 2 to 4
@njit(cache=True)
def label_clusters(binary: np.ndarray, radius: int):
    """Scan the rows; every black pixel starts a contagion.

    Returns a label image (0 = background, 1..n = cluster number), each
    cluster's pixel count and bounding box.
    """
    height, width = binary.shape
    grid = binary.copy()
    labels = np.zeros((height, width), dtype=np.int32)
    stack = np.empty(height * width, dtype=np.int64)      # flat indices
    sizes = [0]                                           # index 0: background
    boxes = [(0, 0, 0, 0)]                                # y0, x0, y1, x1
    n = 0
    for y in range(height):
        for x in range(width):
            if grid[y, x] == 0:
                continue
            n += 1
            grid[y, x] = 0
            labels[y, x] = n
            top = 0
            stack[top] = y * width + x
            top += 1
            count, y0, x0, y1, x1 = 0, y, x, y, x
            while top > 0:
                top -= 1
                cy, cx = divmod(stack[top], width)
                count += 1
                y0, y1 = min(y0, cy), max(y1, cy)
                x0, x1 = min(x0, cx), max(x1, cx)
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and grid[ny, nx]:
                            grid[ny, nx] = 0
                            labels[ny, nx] = n
                            stack[top] = ny * width + nx
                            top += 1
            sizes.append(count)
            boxes.append((y0, x0, y1, x1))
    return labels, np.array(sizes), np.array(boxes)


@njit(cache=True)
def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood the outside from the frame; whatever it cannot reach is the shape."""
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=np.bool_)
    stack = np.empty(h * w, dtype=np.int64)
    top = 0
    for y in range(h):
        for x in range(w):
            if (y == 0 or x == 0 or y == h - 1 or x == w - 1) and not mask[y, x]:
                outside[y, x] = True
                stack[top] = y * w + x
                top += 1
    while top > 0:
        top -= 1
        y, x = divmod(stack[top], w)
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack[top] = ny * w + nx
                top += 1
    return ~outside


# --------------------------------------------------------- steps 5 and 6
def detect(image: Image.Image, radius: int = 1, min_pixels: int = 50,
           threshold: int | None = None) -> list[Shape]:
    binary = binarize(image, threshold)
    labels, sizes, boxes = label_clusters(binary, radius)
    shapes = []
    for n in np.flatnonzero(sizes >= min_pixels):        # the sieve
        if n == 0:
            continue
        y0, x0, y1, x1 = boxes[n]
        crop = np.pad(labels[y0:y1 + 1, x0:x1 + 1] == n, 1)
        filled = fill_holes(crop)
        ys, xs = np.nonzero(filled)
        # Outer contour: filled pixels with a side touching the outside.
        inner = filled[1:-1, 1:-1] & filled[:-2, 1:-1] & filled[2:, 1:-1] \
            & filled[1:-1, :-2] & filled[1:-1, 2:]
        edge = filled.copy()
        edge[1:-1, 1:-1] &= ~inner
        cy, cx = np.nonzero(edge)
        contour = np.column_stack([cy + y0 - 1, cx + x0 - 1]).astype(np.float64)
        shapes.append(describe(int(sizes[n]), len(ys),
                               float(ys.mean() + y0 - 1), float(xs.mean() + x0 - 1),
                               contour))
    return shapes


if __name__ == "__main__":
    import argparse
    from PIL import ImageOps

    parser = argparse.ArgumentParser(description="Detect shapes by contagion (compiled).")
    parser.add_argument("image")
    parser.add_argument("-o", "--output", default="contagion_out.png")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--min-pixels", type=int, default=50)
    parser.add_argument("--threshold", type=int, default=None)
    args = parser.parse_args()

    img = ImageOps.exif_transpose(Image.open(args.image))
    found = detect(img, args.radius, args.min_pixels, args.threshold)
    for s in found:
        print(f"{s.label:9s} at ({s.barycentre[0]:.0f}, {s.barycentre[1]:.0f})  "
              f"{s.pixels} px  corners={s.corners}  cv={s.std_distance / s.mean_distance:.3f}")
    overlay(img, found).save(args.output)
    print(f"-> {args.output}")
