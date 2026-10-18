import importlib.util
from pathlib import Path

import pytest
np = pytest.importorskip('numpy')


SPEC = importlib.util.spec_from_file_location(
    "prepare_reference_crop", Path(__file__).resolve().parents[1] / "scripts" / "prepare_reference_crop.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
square_crop = PREPARE.square_crop


def image(height=100, width=120):
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def assert_original_pixels(original, cropped, box):
    x1, y1, x2, y2 = box
    assert all(isinstance(value, int) for value in box)
    assert 0 <= x1 < x2 <= original.shape[1]
    assert 0 <= y1 < y2 <= original.shape[0]
    assert x2 - x1 == y2 - y1
    np.testing.assert_array_equal(cropped, original[y1:y2, x1:x2])
    assert not np.shares_memory(cropped, original)


def test_margin_expands_each_side_before_square():
    original = image()
    cropped, box = square_crop(original, [40, 40, 60, 60], margin=0.3)
    assert box == (34, 34, 66, 66)
    assert_original_pixels(original, cropped, box)


def test_rectangular_bbox_uses_longer_expanded_edge():
    original = image()
    cropped, box = square_crop(original, [40, 20, 80, 40], margin=0.3)
    assert box == (28, 0, 92, 64)
    assert_original_pixels(original, cropped, box)


@pytest.mark.parametrize("bbox,expected", [
    ([0, 0, 10, 20], (0, 0, 32, 32)),
    ([110, 0, 120, 20], (88, 0, 120, 32)),
    ([0, 80, 10, 100], (0, 68, 32, 100)),
    ([110, 80, 120, 100], (88, 68, 120, 100)),
])
def test_edge_square_shifts_instead_of_padding(bbox, expected):
    original = image()
    cropped, box = square_crop(original, bbox)
    assert box == expected
    assert_original_pixels(original, cropped, box)


@pytest.mark.parametrize("bbox,expected", [
    ([-5, -8, 15, 12], (0, 0, 32, 32)),
    ([110, 90, 135, 115], (79, 59, 120, 100)),
])
def test_partly_out_of_bounds_bbox_is_clamped(bbox, expected):
    original = image()
    cropped, box = square_crop(original, bbox)
    assert box == expected
    assert_original_pixels(original, cropped, box)


def test_rectangular_image_clamps_square_side_to_shorter_edge():
    original = image(height=30, width=100)
    cropped, box = square_crop(original, [70, 4, 90, 24], margin=1)
    assert box == (65, 0, 95, 30)
    assert_original_pixels(original, cropped, box)


def test_one_pixel_image_remains_one_pixel_without_padding():
    original = image(height=1, width=1)
    cropped, box = square_crop(original, [0, 0, 1, 1])
    assert box == (0, 0, 1, 1)
    assert_original_pixels(original, cropped, box)


@pytest.mark.parametrize("margin,expected", [(0, (3, 4, 4, 5)), (0.3, (2, 3, 5, 6))])
def test_one_pixel_bbox_has_positive_integral_crop(margin, expected):
    original = image()
    cropped, box = square_crop(original, [3, 4, 4, 5], margin=margin)
    assert box == expected
    assert_original_pixels(original, cropped, box)


def test_subpixel_bbox_rounds_side_up_not_to_zero():
    original = image()
    cropped, box = square_crop(original, [3.2, 4.2, 3.3, 4.3], margin=0)
    assert box == (3, 4, 4, 5)
    assert_original_pixels(original, cropped, box)


def test_cropped_copy_cannot_mutate_original():
    original = image()
    unchanged = original.copy()
    cropped, box = square_crop(original, [40, 40, 60, 60])
    assert_original_pixels(original, cropped, box)
    cropped[:] = 0
    np.testing.assert_array_equal(original, unchanged)


@pytest.mark.parametrize("bbox", [
    [-20, 10, -1, 30], [120, 10, 140, 30], [10, -20, 30, 0], [10, 100, 30, 120],
    [0, 0, 0, 1], [0, 0, 1, 0], [20, 10, 10, 30], [10, 20, 30, 10],
    [0, 0, float("nan"), 1], [0, 0, float("inf"), 1],
    [0, 0, 1], [[0, 0, 1, 1]], None, [0, 0, "bad", 1],
])
def test_invalid_or_fully_outside_bbox_fails(bbox):
    with pytest.raises(ValueError, match="bbox"):
        square_crop(image(), bbox)


@pytest.mark.parametrize("margin", [-0.01, 1.01, float("nan"), float("inf"), True, "0.3", None])
def test_invalid_margin_fails(margin):
    with pytest.raises(ValueError, match="margin"):
        square_crop(image(), [0, 0, 10, 10], margin=margin)


@pytest.mark.parametrize("margin", [0, 1, np.float32(0.3)])
def test_margin_range_endpoints_and_numpy_scalar_are_valid(margin):
    original = image()
    cropped, box = square_crop(original, [40, 40, 60, 60], margin=margin)
    assert_original_pixels(original, cropped, box)


@pytest.mark.parametrize("invalid", [
    None, np.zeros((10, 10), dtype=np.uint8), np.zeros((0, 10, 3), dtype=np.uint8),
    np.zeros((10, 0, 3), dtype=np.uint8), np.zeros((10, 10, 4), dtype=np.uint8),
])
def test_invalid_image_shape_fails(invalid):
    with pytest.raises(ValueError, match="image"):
        square_crop(invalid, [0, 0, 1, 1])
