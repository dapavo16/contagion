import numpy as np
import pytest
from PIL import Image, ImageDraw

numba = pytest.importorskip("numba")

import contagion
import contagion_fast
from demo import test_sheet as demo_sheet


def summary(shapes):
    return sorted((s.label, round(s.barycentre[0], 3), round(s.barycentre[1], 3), s.pixels)
                  for s in shapes)


def test_same_clusters_as_pure_python():
    rng = np.random.default_rng(5)
    b = (rng.random((150, 200)) > 0.55).astype(np.uint8)
    for r in (1, 2):
        labels, sizes, _ = contagion_fast.label_clusters(b, r)
        pure = contagion.find_clusters(b, r, recursive=False)
        assert sorted(sizes[1:].tolist()) == sorted(len(c) for c in pure)
        assert labels.max() == len(pure)


def test_same_shapes_as_pure_python_on_the_demo_sheet():
    sheet = demo_sheet()
    assert summary(contagion_fast.detect(sheet, min_pixels=80)) == \
        summary(contagion.detect(sheet, min_pixels=80))


def test_ring_is_a_circle():
    img = Image.new("L", (300, 300), 255)
    d = ImageDraw.Draw(img)
    d.ellipse((50, 50, 250, 250), fill=0)
    d.ellipse((90, 90, 210, 210), fill=255)
    [shape] = contagion_fast.detect(img, min_pixels=30)
    assert shape.label == "circle"
