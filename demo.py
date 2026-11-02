"""Draw a test sheet of shapes, run the contagion on it, save the overlay."""

import math
import random
import time

from PIL import Image, ImageDraw

from contagion import detect, overlay


def rotated(points, angle_deg, cx, cy):
    a = math.radians(angle_deg)
    return [(cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a),
             cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a)) for x, y in points]


def test_sheet(seed: int = 7) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("L", (900, 520), 255)
    d = ImageDraw.Draw(img)

    d.ellipse((60, 60, 220, 220), fill=0)                                   # circle
    d.ellipse((290, 80, 520, 200), fill=0)                                  # ellipse
    d.polygon(rotated([(600, 60), (760, 60), (760, 220), (600, 220)], 20, 680, 140), fill=0)
    d.polygon([(80, 460), (230, 460), (155, 300)], fill=0)                  # triangle
    d.polygon(rotated([(300, 330), (540, 330), (540, 440), (300, 440)], -12, 420, 385), fill=0)
    d.ellipse((620, 310, 740, 430), fill=0)                                 # small circle
    d.polygon(rotated([(780, 300), (860, 300), (820, 380)], 35, 820, 340), fill=0)

    for _ in range(120):                                                    # dust for the sieve
        x, y = rng.randrange(900), rng.randrange(520)
        d.rectangle((x, y, x + rng.randrange(1, 4), y + rng.randrange(1, 4)), fill=0)
    return img


if __name__ == "__main__":
    sheet = test_sheet()
    sheet.save("examples/input.png")
    t0 = time.perf_counter()
    shapes = detect(sheet, radius=1, min_pixels=80)
    elapsed = time.perf_counter() - t0
    for s in shapes:
        print(f"{s.label:9s} at ({s.barycentre[0]:4.0f}, {s.barycentre[1]:4.0f})  "
              f"{s.pixels:6d} px  corners={s.corners}  cv={s.std_distance / s.mean_distance:.3f}")
    print(f"{len(shapes)} shapes in {elapsed * 1000:.0f} ms ({sheet.width}x{sheet.height})")
    overlay(sheet, shapes).save("examples/output.png")
