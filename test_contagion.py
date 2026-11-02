import numpy as np
import pytest
from PIL import Image, ImageDraw

from contagion import binarize, classify, detect, find_clusters, sieve
from demo import rotated


def canvas():
    img = Image.new("L", (300, 300), 255)
    return img, ImageDraw.Draw(img)


def only_shape(img):
    shapes = detect(img, min_pixels=30)
    assert len(shapes) == 1
    return shapes[0]


def test_binarize_gives_zeros_and_ones():
    img, d = canvas()
    d.rectangle((10, 10, 50, 50), fill=40)
    b = binarize(img)
    assert set(np.unique(b)) == {0, 1}
    assert b[30, 30] == 1 and b[200, 200] == 0


def test_contagion_separates_shapes_and_empties_nothing_in_place():
    b = np.zeros((20, 20), dtype=np.uint8)
    b[2:5, 2:5] = 1
    b[10:15, 10:12] = 1
    before = b.copy()
    clusters = find_clusters(b)
    assert sorted(len(c) for c in clusters) == [9, 10]
    assert (b == before).all()


def test_radius_bridges_gaps():
    b = np.zeros((5, 12), dtype=np.uint8)
    b[2, [1, 3, 5, 7, 9]] = 1           # a dotted line
    assert len(find_clusters(b, radius=1)) == 5
    assert len(find_clusters(b, radius=2)) == 1


def test_deep_recursion_on_a_big_blob():
    b = np.ones((250, 250), dtype=np.uint8)
    assert len(find_clusters(b)[0]) == 250 * 250


def test_sieve_drops_small_clusters():
    assert len(sieve([[(0, 0)] * 3, [(0, 0)] * 80], min_pixels=50)) == 1


def test_circle():
    img, d = canvas()
    d.ellipse((50, 50, 250, 250), fill=0)
    s = only_shape(img)
    assert s.label == "circle"
    assert s.barycentre == pytest.approx((150, 150), abs=1)


def test_ellipse():
    img, d = canvas()
    d.ellipse((20, 100, 280, 200), fill=0)
    assert only_shape(img).label == "ellipse"


@pytest.mark.parametrize("angle", [0, 15, 30, 45])
def test_square_any_rotation(angle):
    img, d = canvas()
    d.polygon(rotated([(80, 80), (220, 80), (220, 220), (80, 220)], angle, 150, 150), fill=0)
    assert only_shape(img).label == "square"


@pytest.mark.parametrize("angle", [0, 25])
def test_rectangle(angle):
    img, d = canvas()
    d.polygon(rotated([(40, 110), (260, 110), (260, 190), (40, 190)], angle, 150, 150), fill=0)
    assert only_shape(img).label == "rectangle"


@pytest.mark.parametrize("angle", [0, 40, 90])
def test_triangle(angle):
    img, d = canvas()
    d.polygon(rotated([(60, 240), (240, 240), (150, 70)], angle, 150, 180), fill=0)
    assert only_shape(img).label == "triangle"
