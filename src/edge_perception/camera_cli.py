"""Headless camera discovery and timed recording orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Protocol, cast

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
from PySide6.QtMultimedia import QVideoSink

from edge_perception.capture import CameraDeviceInfo, QtCaptureController
from edge_perception.config import CaptureRequest, CaptureResult


class _SignalLike(Protocol):
    def connect(self, slot: object) -> object: ...


class _CaptureControllerLike(Protocol):
    recordingStarted: _SignalLike
    recordingFinished: _SignalLike
    errorOccurred: _SignalLike

    def devices(self) -> tuple[CameraDeviceInfo, ...]: ...

    def start_preview(self, request: CaptureRequest, video_output: object) -> object: ...

    def start_recording(self, final_path: Path | None = None) -> None: ...

    def stop_recording(self) -> None: ...

    def stop_preview(self) -> None: ...


class _TimerLike(Protocol):
    timeout: _SignalLike

    def setSingleShot(self, value: bool) -> None: ...

    def start(self, interval: int) -> None: ...

    def stop(self) -> None: ...


class _EventLoopLike(Protocol):
    def exec(self) -> int: ...

    def quit(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _CameraRuntime:
    controller: _CaptureControllerLike
    video_sink: object
    timer: _TimerLike
    loop: _EventLoopLike
    schedule: Callable[[Callable[[], None]], None]


def list_cameras() -> tuple[CameraDeviceInfo, ...]:
    """Discover currently available cameras without constructing a GUI."""

    application = _core_application()
    controller = QtCaptureController(parent=application)
    return controller.devices()


def capture_camera(
    request: CaptureRequest,
    *,
    duration_seconds: float,
    output: Path | None,
    _runtime: _CameraRuntime | None = None,
) -> CaptureResult:
    """Record one bounded, finalized camera capture through the shared controller."""

    if not isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration must be finite and positive")
    runtime = _runtime or _production_runtime()
    output_path = None if output is None else Path(output).resolve()
    interval_ms = round(duration_seconds * 1000)
    result: CaptureResult | None = None
    error_message: str | None = None

    def complete(capture_result: object) -> None:
        nonlocal result
        if result is None and error_message is None:
            if not isinstance(capture_result, CaptureResult):
                fail("camera capture returned an invalid result")
                return
            result = capture_result
            runtime.loop.quit()

    def fail(message: object) -> None:
        nonlocal error_message
        if result is None and error_message is None:
            error_message = str(message) or "camera capture failed"
            runtime.loop.quit()

    def stop_recording() -> None:
        try:
            runtime.controller.stop_recording()
        except Exception as error:  # noqa: BLE001 - terminal adapter boundary
            fail(error)

    def start_capture() -> None:
        try:
            runtime.controller.start_preview(request, runtime.video_sink)
            runtime.controller.start_recording(output_path)
        except Exception as error:  # noqa: BLE001 - terminal adapter boundary
            fail(error)

    runtime.timer.setSingleShot(True)
    runtime.controller.recordingStarted.connect(lambda: runtime.timer.start(interval_ms))
    runtime.timer.timeout.connect(stop_recording)
    runtime.controller.recordingFinished.connect(complete)
    runtime.controller.errorOccurred.connect(fail)
    try:
        runtime.schedule(start_capture)
        runtime.loop.exec()
    finally:
        runtime.timer.stop()

    if result is not None:
        return result
    try:
        runtime.controller.stop_preview()
    except Exception as cleanup_error:  # noqa: BLE001 - preserve terminal failure
        if error_message is None:
            error_message = str(cleanup_error) or "camera preview cleanup failed"
        else:
            error_message += f"; cleanup failed: {cleanup_error}"
    if error_message is not None:
        raise RuntimeError(error_message)
    raise RuntimeError("camera capture ended without a result")


def _core_application() -> QCoreApplication:
    application = QCoreApplication.instance()
    if application is None:
        application = QCoreApplication(["edge-perception-camera"])
    return application


def _production_runtime() -> _CameraRuntime:
    application = _core_application()
    controller = QtCaptureController(parent=application)
    loop = QEventLoop(application)
    video_sink = QVideoSink(application)
    timer = QTimer(application)
    return _CameraRuntime(
        controller=cast(_CaptureControllerLike, controller),
        video_sink=video_sink,
        timer=cast(_TimerLike, timer),
        loop=loop,
        schedule=lambda callback: QTimer.singleShot(0, callback),
    )
