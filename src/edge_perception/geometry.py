"""Geometry helpers for validating crops and restoring source coordinates."""

import numpy as np

from edge_perception.contracts import Box, Detection, Region


def _validate_frame_size(frame_width: int, frame_height: int) -> None:
    if (
        not isinstance(frame_width, int)
        or isinstance(frame_width, bool)
        or not isinstance(frame_height, int)
        or isinstance(frame_height, bool)
        or frame_width <= 0
        or frame_height <= 0
    ):
        raise ValueError("frame dimensions must be positive integers")


def full_frame_region(frame_width: int, frame_height: int) -> Region:
    """Return the region spanning every pixel in a source frame."""
    _validate_frame_size(frame_width, frame_height)
    return Region("full-frame", 0, 0, frame_width, frame_height)


def validate_region(region: Region, frame_width: int, frame_height: int) -> None:
    """Raise when a region is not wholly contained by a source frame."""
    if not isinstance(region, Region):
        raise TypeError("region must be a Region")
    _validate_frame_size(frame_width, frame_height)
    if region.x < 0 or region.y < 0:
        raise ValueError("region origin must not be negative")
    if region.x + region.width > frame_width or region.y + region.height > frame_height:
        raise ValueError("region must fit within the frame")


def crop_region(image: np.ndarray, region: Region) -> np.ndarray:
    """Return a contiguous RGB crop after ensuring the region fits the image."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB array")
    frame_height, frame_width, _ = image.shape
    validate_region(region, frame_width, frame_height)
    return np.ascontiguousarray(image[region.y : region.y + region.height, region.x : region.x + region.width])


def map_detection_to_source(
    detection: Detection, region: Region, frame_width: int, frame_height: int
) -> Detection:
    """Translate a crop-relative detection into clipped source-frame coordinates."""
    if not isinstance(detection, Detection):
        raise TypeError("detection must be a Detection")
    validate_region(region, frame_width, frame_height)

    mapped = Box(
        max(0.0, min(float(frame_width), detection.box.x1 + region.x)),
        max(0.0, min(float(frame_height), detection.box.y1 + region.y)),
        max(0.0, min(float(frame_width), detection.box.x2 + region.x)),
        max(0.0, min(float(frame_height), detection.box.y2 + region.y)),
    )
    return Detection(mapped, detection.class_id, detection.score, detection.label)
