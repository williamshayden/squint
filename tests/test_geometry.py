import numpy as np
import pytest

from edge_perception.contracts import Box, Detection, Region
from edge_perception.geometry import (
    crop_region,
    full_frame_region,
    map_detection_to_source,
    validate_region,
)


def test_full_frame_region_is_valid_identity_region() -> None:
    region = full_frame_region(3840, 2160)

    assert region == Region("full-frame", 0, 0, 3840, 2160)
    assert validate_region(region, 3840, 2160) is None


def test_full_frame_detection_remains_in_source_coordinates() -> None:
    detection = Detection(Box(10.5, 20.25, 110.5, 220.25), 1, 0.9, "person")

    mapped = map_detection_to_source(detection, full_frame_region(3840, 2160), 3840, 2160)

    assert mapped == detection


def test_crop_detection_maps_once_to_source_pixels() -> None:
    local = Detection(Box(10.5, 20.25, 110.5, 220.25), 1, 0.9, "person")
    region = Region("left", 960, 540, 1280, 1080)

    mapped = map_detection_to_source(local, region, 3840, 2160)

    assert mapped.box == Box(970.5, 560.25, 1070.5, 760.25)
    assert mapped.class_id == 1
    assert mapped.score == 0.9
    assert mapped.label == "person"


def test_mapping_clips_continuous_box_edges_at_frame_boundary() -> None:
    local = Detection(Box(50.5, 10.25, 150.75, 120.5), 2, 0.8, "car")
    region = Region("bottom-right", 900, 900, 100, 100)

    mapped = map_detection_to_source(local, region, 1000, 1000)

    assert mapped.box == Box(950.5, 910.25, 1000.0, 1000.0)


@pytest.mark.parametrize(
    ("region", "frame_width", "frame_height"),
    [
        (Region("negative", -1, 0, 1, 1), 10, 10),
        (Region("over-wide", 9, 0, 2, 1), 10, 10),
        (Region("over-tall", 0, 9, 1, 2), 10, 10),
    ],
)
def test_validate_region_rejects_regions_outside_frame(
    region: Region, frame_width: int, frame_height: int
) -> None:
    with pytest.raises(ValueError):
        validate_region(region, frame_width, frame_height)


def test_crop_region_returns_contiguous_rgb_pixels_from_non_contiguous_input() -> None:
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)[:, ::2, :]
    region = Region("center", 1, 2, 2, 3)

    cropped = crop_region(image, region)

    assert cropped.flags.c_contiguous
    assert np.array_equal(cropped, image[2:5, 1:3])
