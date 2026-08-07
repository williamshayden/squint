from __future__ import annotations

from dataclasses import dataclass
from math import inf, nan
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal

from edge_perception.config import CaptureRequest, CaptureResult


class FakeTimer(QObject):
    timeout = Signal()

    def __init__(self, *, emit_timeout: bool = True, start_error: Exception | None = None) -> None:
        super().__init__()
        self.single_shot = False
        self.emit_timeout = emit_timeout
        self.start_error = start_error
        self.intervals: list[int] = []
        self.stop_calls = 0

    def setSingleShot(self, value: bool) -> None:
        self.single_shot = value

    def start(self, interval: int) -> None:
        self.intervals.append(interval)
        if self.start_error is not None:
            raise self.start_error
        if self.emit_timeout:
            self.timeout.emit()

    def stop(self) -> None:
        self.stop_calls += 1


class FakeLoop:
    def __init__(self, *, exec_error: Exception | None = None, run_callbacks_before_error: bool = False) -> None:
        self.callbacks: list[object] = []
        self.quit_calls = 0
        self.exec_error = exec_error
        self.run_callbacks_before_error = run_callbacks_before_error

    def call_soon(self, callback: object) -> None:
        self.callbacks.append(callback)

    def exec(self) -> int:
        while self.callbacks and self.quit_calls == 0:
            callback = self.callbacks.pop(0)
            assert callable(callback)
            callback()
            if self.exec_error is not None and self.run_callbacks_before_error:
                raise self.exec_error
        if self.exec_error is not None:
            raise self.exec_error
        return 0

    def quit(self) -> None:
        self.quit_calls += 1


class FakeController(QObject):
    recordingStarted = Signal()
    recordingFinished = Signal(object)
    errorOccurred = Signal(str)

    def __init__(
        self,
        result: CaptureResult,
        *,
        error: str | None = None,
        late_error: str | None = None,
        duplicate_result: bool = False,
        stop_preview_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.result = result
        self.error = error
        self.late_error = late_error
        self.duplicate_result = duplicate_result
        self.stop_preview_error = stop_preview_error
        self.preview_requests: list[tuple[CaptureRequest, object]] = []
        self.record_paths: list[Path | None] = []
        self.stop_recording_calls = 0
        self.stop_preview_calls = 0
        self.recording_active = False

    def start_preview(self, request: CaptureRequest, video_sink: object) -> None:
        self.preview_requests.append((request, video_sink))

    def start_recording(self, path: Path | None) -> None:
        self.record_paths.append(path)
        if self.error is not None:
            self.errorOccurred.emit(self.error)
            return
        self.recording_active = True
        self.recordingStarted.emit()

    def stop_recording(self) -> None:
        self.stop_recording_calls += 1
        self.recording_active = False
        self.recordingFinished.emit(self.result)
        if self.duplicate_result:
            self.recordingFinished.emit(self.result)
        if self.late_error is not None:
            self.errorOccurred.emit(self.late_error)

    def stop_preview(self) -> None:
        self.stop_preview_calls += 1
        self.recording_active = False
        if self.stop_preview_error is not None:
            raise self.stop_preview_error


@dataclass
class FakeRuntime:
    controller: FakeController
    timer: FakeTimer
    loop: FakeLoop
    video_sink: object
    widgets_application_created: bool = False
    schedule_error: Exception | None = None

    def schedule(self, callback: object) -> None:
        if self.schedule_error is not None:
            raise self.schedule_error
        self.loop.call_soon(callback)


def _request() -> CaptureRequest:
    return CaptureRequest("camera-1", "Research camera", 1920, 1080, 30.0, True)


def _result(request: CaptureRequest, path: Path) -> CaptureResult:
    return CaptureResult(
        request=request,
        selected_width=1920,
        selected_height=1080,
        selected_min_fps=30.0,
        selected_max_fps=30.0,
        selected_pixel_format="NV12",
        actual_width=1920,
        actual_height=1080,
        actual_fps=30.0,
        container="mp4",
        codec="h264",
        duration_seconds=0.05,
        has_audio=False,
        file_size_bytes=7,
        path=path,
        sha256="a" * 64,
    )


def _runtime(request: CaptureRequest, path: Path, *, error: str | None = None) -> FakeRuntime:
    return FakeRuntime(FakeController(_result(request, path), error=error), FakeTimer(), FakeLoop(), object())


def test_capture_camera_starts_preview_records_for_duration_and_returns_result(tmp_path: Path) -> None:
    from edge_perception.camera_cli import capture_camera

    request = _request()
    output = tmp_path / "capture.mp4"
    fake_runtime = _runtime(request, output)
    expected_capture_result = fake_runtime.controller.result

    result = capture_camera(
        request,
        duration_seconds=0.05,
        output=output,
        _runtime=fake_runtime,  # type: ignore[arg-type]
    )

    assert fake_runtime.controller.preview_requests == [(request, fake_runtime.video_sink)]
    assert fake_runtime.controller.record_paths == [output.resolve()]
    assert fake_runtime.timer.intervals == [50]
    assert fake_runtime.controller.stop_recording_calls == 1
    assert result == expected_capture_result
    assert fake_runtime.widgets_application_created is False
    assert fake_runtime.loop.quit_calls == 1
    assert fake_runtime.timer.stop_calls == 1


def test_capture_camera_passes_none_to_shared_controller_when_output_is_omitted(tmp_path: Path) -> None:
    from edge_perception.camera_cli import capture_camera

    request = _request()
    fake_runtime = _runtime(request, tmp_path / "chosen-by-controller.mp4")

    assert capture_camera(
        request,
        duration_seconds=0.05,
        output=None,
        _runtime=fake_runtime,  # type: ignore[arg-type]
    ) == fake_runtime.controller.result
    assert fake_runtime.controller.record_paths == [None]


def test_capture_camera_raises_controller_error_and_releases_preview(tmp_path: Path) -> None:
    from edge_perception.camera_cli import capture_camera

    request = _request()
    fake_runtime = _runtime(request, tmp_path / "capture.mp4", error="camera disconnected")

    with pytest.raises(RuntimeError, match="camera disconnected"):
        capture_camera(
            request,
            duration_seconds=0.05,
            output=tmp_path / "capture.mp4",
            _runtime=fake_runtime,  # type: ignore[arg-type]
        )

    assert fake_runtime.controller.stop_preview_calls == 1
    assert fake_runtime.loop.quit_calls == 1
    assert fake_runtime.timer.stop_calls == 1


def test_capture_camera_discards_active_capture_when_loop_execution_raises(tmp_path: Path) -> None:
    from edge_perception.camera_cli import capture_camera

    request = _request()
    fake_runtime = _runtime(request, tmp_path / "capture.mp4")
    fake_runtime.timer.emit_timeout = False
    fake_runtime.loop.exec_error = RuntimeError("event loop failed")
    fake_runtime.loop.run_callbacks_before_error = True

    with pytest.raises(RuntimeError, match="event loop failed"):
        capture_camera(
            request,
            duration_seconds=0.05,
            output=tmp_path / "capture.mp4",
            _runtime=fake_runtime,  # type: ignore[arg-type]
        )

    assert fake_runtime.timer.stop_calls == 1
    assert fake_runtime.controller.stop_preview_calls == 1
    assert fake_runtime.controller.recording_active is False


def test_capture_camera_preserves_schedule_failure_when_preview_cleanup_also_fails(tmp_path: Path) -> None:
    from edge_perception.camera_cli import capture_camera

    request = _request()
    fake_runtime = _runtime(request, tmp_path / "capture.mp4")
    fake_runtime.schedule_error = RuntimeError("schedule failed")
    fake_runtime.controller.stop_preview_error = OSError("cleanup failed")

    with pytest.raises(RuntimeError, match="schedule failed; cleanup failed: cleanup failed"):
        capture_camera(
            request,
            duration_seconds=0.05,
            output=tmp_path / "capture.mp4",
            _runtime=fake_runtime,  # type: ignore[arg-type]
        )

    assert fake_runtime.timer.stop_calls == 1
    assert fake_runtime.controller.stop_preview_calls == 1


def test_capture_camera_routes_timer_start_failure_to_terminal_cleanup(tmp_path: Path) -> None:
    from edge_perception.camera_cli import capture_camera

    request = _request()
    fake_runtime = _runtime(request, tmp_path / "capture.mp4")
    fake_runtime.timer.start_error = RuntimeError("timer cannot start")

    with pytest.raises(RuntimeError, match="timer cannot start"):
        capture_camera(
            request,
            duration_seconds=0.05,
            output=tmp_path / "capture.mp4",
            _runtime=fake_runtime,  # type: ignore[arg-type]
        )

    assert fake_runtime.loop.quit_calls == 1
    assert fake_runtime.timer.stop_calls == 1
    assert fake_runtime.controller.stop_preview_calls == 1


def test_capture_camera_ignores_late_error_after_completed_result(tmp_path: Path) -> None:
    from edge_perception.camera_cli import capture_camera

    request = _request()
    fake_runtime = _runtime(request, tmp_path / "capture.mp4")
    fake_runtime.controller.late_error = "late error"

    assert capture_camera(
        request,
        duration_seconds=0.05,
        output=tmp_path / "capture.mp4",
        _runtime=fake_runtime,  # type: ignore[arg-type]
    ) == fake_runtime.controller.result
    assert fake_runtime.loop.quit_calls == 1
    assert fake_runtime.controller.stop_preview_calls == 0


def test_capture_camera_ignores_duplicate_finished_signal_after_completed_result(tmp_path: Path) -> None:
    from edge_perception.camera_cli import capture_camera

    request = _request()
    fake_runtime = _runtime(request, tmp_path / "capture.mp4")
    fake_runtime.controller.duplicate_result = True

    assert capture_camera(
        request,
        duration_seconds=0.05,
        output=tmp_path / "capture.mp4",
        _runtime=fake_runtime,  # type: ignore[arg-type]
    ) == fake_runtime.controller.result
    assert fake_runtime.loop.quit_calls == 1


@pytest.mark.parametrize("duration", [0.0, -0.1, nan, inf, 0.0001, 2_147_483.648])
def test_capture_camera_rejects_non_positive_or_non_finite_duration_before_preview(
    tmp_path: Path,
    duration: float,
) -> None:
    from edge_perception.camera_cli import capture_camera

    request = _request()
    fake_runtime = _runtime(request, tmp_path / "capture.mp4")

    with pytest.raises(ValueError, match="duration must be finite and positive"):
        capture_camera(
            request,
            duration_seconds=duration,
            output=tmp_path / "capture.mp4",
            _runtime=fake_runtime,  # type: ignore[arg-type]
        )

    assert fake_runtime.controller.preview_requests == []
