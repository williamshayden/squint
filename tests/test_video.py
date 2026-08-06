from __future__ import annotations

import wave
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from edge_perception.video import iter_video


def _write_mpeg4_video(path: Path) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 30)

        for frame_index, red_value in enumerate((20, 120, 220)):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[..., 0] = red_value
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = frame_index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def _write_audio_only_container(path: Path) -> None:
    with wave.open(str(path), mode="wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8_000)
        audio.writeframes(b"\0\0" * 80)


def test_iter_video_decodes_encoded_frames_in_chronological_order(tmp_path: Path) -> None:
    video_path = tmp_path / "chronological.mp4"
    _write_mpeg4_video(video_path)

    decoded = list(iter_video(video_path))

    assert [frame.frame_index for frame in decoded] == [0, 1, 2]
    timestamps = [frame.source_time_ms for frame in decoded]
    assert all(timestamp is not None for timestamp in timestamps)
    assert timestamps == sorted(timestamps)  # type: ignore[type-var]
    assert all(frame.image.shape == (48, 64, 3) for frame in decoded)
    assert all(frame.image.dtype == np.uint8 for frame in decoded)
    red_means = [float(frame.image[..., 0].mean()) for frame in decoded]
    assert red_means[0] + 50 < red_means[1]
    assert red_means[1] + 50 < red_means[2]


def test_iter_video_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_video(tmp_path / "missing.mp4"))


def test_iter_video_rejects_container_without_video_stream(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio-only.wav"
    _write_audio_only_container(audio_path)

    with pytest.raises(ValueError, match="video stream"):
        list(iter_video(audio_path))
