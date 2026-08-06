"""Injected Qt camera ownership with pure selection and validation helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from PySide6.QtCore import QByteArray, QObject, QSize, QStandardPaths, QUrl, Signal
from PySide6.QtMultimedia import (
    QCamera,
    QCameraDevice,
    QMediaCaptureSession,
    QMediaDevices,
    QMediaFormat,
    QMediaRecorder,
    QVideoFrameFormat,
)

from edge_perception.config import CaptureRequest, CaptureResult
from edge_perception.video import VideoMetadata, first_video_frame, probe_video


@dataclass(frozen=True, slots=True)
class CameraFormatInfo:
    width: int
    height: int
    min_fps: float
    max_fps: float
    pixel_format: str
    qt_format: object = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class CameraDeviceInfo:
    device_id: str
    description: str
    formats: tuple[CameraFormatInfo, ...]
    qt_device: object = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class RecordingProfile:
    file_format: object
    video_codec: object


def select_camera_format(
    formats: Sequence[CameraFormatInfo],
    request: CaptureRequest,
) -> CameraFormatInfo:
    """Select one camera format using the documented independent constraints."""

    candidates = tuple(formats)
    if request.strict:
        candidates = tuple(
            camera_format
            for camera_format in candidates
            if (
                request.requested_width is None
                or camera_format.width == request.requested_width
            )
            and (
                request.requested_height is None
                or camera_format.height == request.requested_height
            )
            and (
                request.requested_fps is None
                or camera_format.min_fps <= request.requested_fps <= camera_format.max_fps
            )
        )
        if not candidates:
            raise ValueError("requested camera mode is unavailable")
    elif not candidates:
        raise ValueError("camera has no supported video formats")

    def rank(indexed: tuple[int, CameraFormatInfo]) -> tuple[float, ...]:
        index, camera_format = indexed
        supplied_matches = 0
        if request.requested_width is not None:
            supplied_matches += camera_format.width == request.requested_width
        if request.requested_height is not None:
            supplied_matches += camera_format.height == request.requested_height
        if request.requested_fps is not None:
            supplied_matches += (
                camera_format.min_fps
                <= request.requested_fps
                <= camera_format.max_fps
            )
        width_distance = (
            0.0
            if request.requested_width is None
            else float(abs(camera_format.width - request.requested_width))
        )
        height_distance = (
            0.0
            if request.requested_height is None
            else float(abs(camera_format.height - request.requested_height))
        )
        fps_distance = _fps_range_distance(camera_format, request.requested_fps)
        return (
            -float(supplied_matches),
            width_distance,
            height_distance,
            fps_distance,
            -float(camera_format.width * camera_format.height),
            -camera_format.max_fps,
            float(index),
        )

    return min(enumerate(candidates), key=rank)[1]


def _fps_range_distance(camera_format: CameraFormatInfo, requested_fps: float | None) -> float:
    if requested_fps is None or camera_format.min_fps <= requested_fps <= camera_format.max_fps:
        return 0.0
    return min(
        abs(requested_fps - camera_format.min_fps),
        abs(requested_fps - camera_format.max_fps),
    )


def select_recording_profile(
    file_formats: Sequence[object],
    video_codecs: Sequence[object],
) -> RecordingProfile:
    """Choose deterministic video encode settings from Qt-reported capabilities."""

    if not file_formats or not video_codecs:
        raise ValueError("no supported video recording profile")
    file_format = _preferred_enum(file_formats, ("MPEG4", "Matroska"))
    video_codec = _preferred_enum(video_codecs, ("H264", "H265", "VP9", "AV1"))
    return RecordingProfile(file_format, video_codec)


def _preferred_enum(values: Sequence[object], preferred_names: Sequence[str]) -> object:
    by_name = {_enum_name(value): value for value in values}
    for name in preferred_names:
        if name in by_name:
            return by_name[name]
    return min(values, key=_enum_sort_key)


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value).rsplit(".", 1)[-1]


def _enum_sort_key(value: object) -> tuple[int, str]:
    raw_value = getattr(value, "value", value)
    number = raw_value if isinstance(raw_value, int) else 0
    return number, _enum_name(value)


def validate_capture_result(request: CaptureRequest, metadata: VideoMetadata) -> None:
    """Validate video-only output and each supplied strict capture constraint."""

    if metadata.has_audio:
        raise ValueError("captured video contains audio")
    if not request.strict:
        return
    if request.requested_width is not None and metadata.width != request.requested_width:
        raise ValueError("captured width differs from strict request")
    if request.requested_height is not None and metadata.height != request.requested_height:
        raise ValueError("captured height differs from strict request")
    if request.requested_fps is not None:
        tolerance = max(0.1, request.requested_fps * 0.005)
        if abs(metadata.average_fps - request.requested_fps) > tolerance:
            raise ValueError("captured FPS differs from strict request")


class _SignalLike(Protocol):
    def connect(self, slot: object) -> object: ...


class _CameraFormatLike(Protocol):
    def resolution(self) -> QSize: ...

    def minFrameRate(self) -> float: ...

    def maxFrameRate(self) -> float: ...

    def pixelFormat(self) -> object: ...


class _CameraDeviceLike(Protocol):
    def id(self) -> QByteArray | bytes: ...

    def description(self) -> str: ...

    def videoFormats(self) -> Sequence[_CameraFormatLike]: ...


class _MediaDevicesLike(Protocol):
    videoInputsChanged: _SignalLike

    def videoInputs(self) -> Sequence[_CameraDeviceLike]: ...


class _CameraLike(Protocol):
    errorOccurred: _SignalLike

    def setCameraFormat(self, camera_format: object) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def deleteLater(self) -> None: ...


class _CaptureSessionLike(Protocol):
    def setCamera(self, camera: object) -> None: ...

    def setRecorder(self, recorder: object) -> None: ...

    def setVideoOutput(self, video_output: object) -> None: ...

    def deleteLater(self) -> None: ...


class _RecorderLike(Protocol):
    recorderStateChanged: _SignalLike
    errorOccurred: _SignalLike

    def setMediaFormat(self, media_format: object) -> None: ...

    def setOutputLocation(self, location: QUrl) -> None: ...

    def actualLocation(self) -> QUrl: ...

    def record(self) -> None: ...

    def stop(self) -> None: ...

    def deleteLater(self) -> None: ...


CameraFactory = Callable[[_CameraDeviceLike, QObject], _CameraLike]
ObjectFactory = Callable[[QObject], _CaptureSessionLike]
RecorderFactory = Callable[[QObject], _RecorderLike]
MediaFormatFactory = Callable[[RecordingProfile], object]
FormatCapabilities = Callable[[], tuple[Sequence[object], Sequence[object]]]
Probe = Callable[[Path], VideoMetadata]
DecodeValidator = Callable[[Path], object]


class QtCaptureController(QObject):
    """Own one injected Qt camera graph and atomically publish validated video."""

    devicesChanged = Signal()
    previewStarted = Signal(object)
    previewStopped = Signal()
    recordingStarted = Signal()
    recordingFinished = Signal(object)
    errorOccurred = Signal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        media_devices: _MediaDevicesLike | None = None,
        camera_factory: CameraFactory | None = None,
        capture_session_factory: ObjectFactory | None = None,
        recorder_factory: RecorderFactory | None = None,
        media_format_factory: MediaFormatFactory | None = None,
        format_capabilities: FormatCapabilities | None = None,
        probe: Probe = probe_video,
        decode_validator: DecodeValidator = first_video_frame,
        capture_directory: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._media_devices = media_devices or cast(_MediaDevicesLike, QMediaDevices(self))
        self._camera_factory = camera_factory or _create_qt_camera
        self._capture_session_factory = capture_session_factory or _create_qt_capture_session
        self._recorder_factory = recorder_factory or _create_qt_recorder
        self._media_format_factory = media_format_factory or _create_qt_media_format
        self._format_capabilities = format_capabilities or _qt_format_capabilities
        self._probe = probe
        self._decode_validator = decode_validator
        self._capture_directory = (
            Path(capture_directory)
            if capture_directory is not None
            else Path(
                QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
            )
            / "captures"
        )
        self._camera: _CameraLike | None = None
        self._session: _CaptureSessionLike | None = None
        self._recorder: _RecorderLike | None = None
        self._request: CaptureRequest | None = None
        self._selected_format: CameraFormatInfo | None = None
        self._temporary_path: Path | None = None
        self._final_path: Path | None = None
        self._recording_pending = False
        self._recording_active = False
        self._discarding = False
        self._media_devices.videoInputsChanged.connect(self._on_devices_changed)

    @property
    def is_previewing(self) -> bool:
        return self._camera is not None

    @property
    def is_recording(self) -> bool:
        return self._recording_active

    def devices(self) -> tuple[CameraDeviceInfo, ...]:
        """Return stable descriptors without transferring Qt object ownership."""

        return tuple(self._describe_device(device) for device in self._media_devices.videoInputs())

    def start_preview(
        self,
        request: CaptureRequest,
        video_output: object,
    ) -> CameraFormatInfo:
        """Build one camera graph, apply the selected format, and start raw preview."""

        if self._recording_pending or self._recording_active:
            raise RuntimeError("cannot replace preview while recording")
        if self._camera is not None:
            self.stop_preview()
        device = next(
            (candidate for candidate in self.devices() if candidate.device_id == request.device_id),
            None,
        )
        if device is None:
            raise ValueError("camera device is unavailable")
        selected = select_camera_format(device.formats, request)
        camera = self._camera_factory(cast(_CameraDeviceLike, device.qt_device), self)
        session = self._capture_session_factory(self)
        recorder = self._recorder_factory(self)
        self._camera = camera
        self._session = session
        self._recorder = recorder
        self._request = request
        self._selected_format = selected
        camera.errorOccurred.connect(self._on_camera_error)
        recorder.recorderStateChanged.connect(self._on_recorder_state_changed)
        recorder.errorOccurred.connect(self._on_recorder_error)
        camera.setCameraFormat(selected.qt_format)
        session.setCamera(camera)
        session.setRecorder(recorder)
        session.setVideoOutput(video_output)
        camera.start()
        self.previewStarted.emit(selected)
        return selected

    def stop_preview(self) -> None:
        """Stop preview, discarding an incomplete recording when necessary."""

        if self._recording_pending or self._recording_active:
            self.discard()
            return
        if self._camera is None:
            return
        self._release_media()
        self.previewStopped.emit()

    def start_recording(self, final_path: Path | None = None) -> None:
        """Start video-only recording to an owned sibling temporary path."""

        if self._recorder is None or self._request is None or self._selected_format is None:
            raise RuntimeError("camera preview is not active")
        if self._recording_pending or self._recording_active:
            raise RuntimeError("camera recording is already active")
        profile = select_recording_profile(*self._format_capabilities())
        destination = self._resolve_final_path(final_path, profile)
        if destination.exists():
            raise FileExistsError(f"capture destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary_path = destination.parent / (
            f".{destination.stem}.{uuid4().hex}.part{destination.suffix or _profile_extension(profile)}"
        )
        self._temporary_path = temporary_path.resolve()
        self._final_path = destination.resolve()
        self._discarding = False
        self._recording_pending = True
        self._recording_active = False
        self._recorder.setMediaFormat(self._media_format_factory(profile))
        self._recorder.setOutputLocation(QUrl.fromLocalFile(str(self._temporary_path)))
        self._recorder.record()

    def stop_recording(self) -> None:
        """Request recorder finalization; publication follows StoppedState."""

        if not (self._recording_pending or self._recording_active) or self._recorder is None:
            raise RuntimeError("camera recording is not active")
        self._recorder.stop()

    def discard(self) -> None:
        """Stop owned media and remove only this controller's incomplete path."""

        self._discarding = True
        self._recording_pending = False
        self._recording_active = False
        recorder = self._recorder
        if recorder is not None:
            recorder.stop()
        had_preview = self._camera is not None
        self._cleanup_owned_temporary()
        self._release_media()
        self._discarding = False
        if had_preview:
            self.previewStopped.emit()

    def _describe_device(self, device: _CameraDeviceLike) -> CameraDeviceInfo:
        formats: list[CameraFormatInfo] = []
        for qt_format in device.videoFormats():
            resolution = qt_format.resolution()
            width = int(resolution.width())
            height = int(resolution.height())
            pixel_format = QVideoFrameFormat.pixelFormatToString(
                cast(QVideoFrameFormat.PixelFormat, qt_format.pixelFormat())
            )
            formats.append(
                CameraFormatInfo(
                    width,
                    height,
                    float(qt_format.minFrameRate()),
                    float(qt_format.maxFrameRate()),
                    pixel_format,
                    qt_format,
                )
            )
        raw_device_id = device.id()
        device_id = bytes(raw_device_id.data()) if isinstance(raw_device_id, QByteArray) else raw_device_id
        return CameraDeviceInfo(
            device_id.hex(),
            str(device.description()),
            tuple(formats),
            device,
        )

    def _resolve_final_path(
        self,
        final_path: Path | None,
        profile: RecordingProfile,
    ) -> Path:
        if final_path is not None:
            return Path(final_path).resolve()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        filename = f"capture-{timestamp}-{uuid4().hex}{_profile_extension(profile)}"
        return (self._capture_directory / filename).resolve()

    def _on_devices_changed(self) -> None:
        self.devicesChanged.emit()

    def _on_camera_error(self, _error: object, message: str) -> None:
        self._fail(message or "camera error")

    def _on_recorder_error(self, _error: object, message: str) -> None:
        self._fail(message or "camera recorder error")

    def _on_recorder_state_changed(self, state: object) -> None:
        if state == QMediaRecorder.RecorderState.RecordingState:
            if self._recording_pending and not self._recording_active:
                self._recording_active = True
                self.recordingStarted.emit()
            return
        if state != QMediaRecorder.RecorderState.StoppedState:
            return
        if self._discarding or not (self._recording_pending or self._recording_active):
            return
        self._recording_pending = False
        self._recording_active = False
        self._finalize_recording()

    def _finalize_recording(self) -> None:
        recorder = self._recorder
        request = self._request
        selected = self._selected_format
        temporary_path = self._temporary_path
        final_path = self._final_path
        if None in (recorder, request, selected, temporary_path, final_path):
            self._fail("camera recording state is incomplete")
            return
        assert recorder is not None
        assert request is not None
        assert selected is not None
        assert temporary_path is not None
        assert final_path is not None
        actual_location = recorder.actualLocation().toLocalFile()
        actual_path = Path(actual_location).resolve() if actual_location else None
        if actual_path != temporary_path:
            self._fail("recorder reported an unexpected output location")
            return
        try:
            self._decode_validator(temporary_path)
            metadata = self._probe(temporary_path)
            validate_capture_result(request, metadata)
            digest = _sha256_file(temporary_path)
            result = CaptureResult(
                request=request,
                selected_width=selected.width,
                selected_height=selected.height,
                selected_min_fps=selected.min_fps,
                selected_max_fps=selected.max_fps,
                selected_pixel_format=selected.pixel_format,
                actual_width=metadata.width,
                actual_height=metadata.height,
                actual_fps=metadata.average_fps,
                container=metadata.container,
                codec=metadata.codec,
                duration_seconds=metadata.duration_seconds,
                has_audio=metadata.has_audio,
                file_size_bytes=metadata.file_size_bytes,
                path=final_path,
                sha256=digest,
            )
            if final_path.exists():
                raise FileExistsError(f"capture destination already exists: {final_path}")
            os.replace(temporary_path, final_path)
        except Exception as error:  # noqa: BLE001 - terminal boundary must clean injected failures
            self._fail(str(error))
            return
        self._temporary_path = None
        self._final_path = None
        had_preview = self._camera is not None
        self._release_media()
        if had_preview:
            self.previewStopped.emit()
        self.recordingFinished.emit(result)

    def _fail(self, message: str) -> None:
        self._discarding = True
        self._recording_pending = False
        self._recording_active = False
        recorder = self._recorder
        if recorder is not None:
            recorder.stop()
        had_preview = self._camera is not None
        self._cleanup_owned_temporary()
        self._release_media()
        self._discarding = False
        if had_preview:
            self.previewStopped.emit()
        self.errorOccurred.emit(message)

    def _cleanup_owned_temporary(self) -> None:
        temporary_path = self._temporary_path
        self._temporary_path = None
        self._final_path = None
        if temporary_path is None:
            return
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass

    def _release_media(self) -> None:
        camera = self._camera
        session = self._session
        recorder = self._recorder
        self._camera = None
        self._session = None
        self._recorder = None
        self._request = None
        self._selected_format = None
        self._recording_pending = False
        self._recording_active = False
        if camera is not None:
            camera.stop()
        for media_object in (recorder, session, camera):
            if media_object is not None:
                media_object.deleteLater()


def _create_qt_camera(device: _CameraDeviceLike, parent: QObject) -> _CameraLike:
    return cast(_CameraLike, QCamera(cast(QCameraDevice, device), parent))


def _create_qt_capture_session(parent: QObject) -> _CaptureSessionLike:
    return cast(_CaptureSessionLike, QMediaCaptureSession(parent))


def _create_qt_recorder(parent: QObject) -> _RecorderLike:
    return cast(_RecorderLike, QMediaRecorder(parent))


def _create_qt_media_format(profile: RecordingProfile) -> object:
    media_format = QMediaFormat()
    media_format.setFileFormat(cast(QMediaFormat.FileFormat, profile.file_format))
    media_format.setVideoCodec(cast(QMediaFormat.VideoCodec, profile.video_codec))
    return media_format


def _qt_format_capabilities() -> tuple[Sequence[object], Sequence[object]]:
    media_format = QMediaFormat()
    conversion_mode = QMediaFormat.ConversionMode.Encode
    return (
        tuple(media_format.supportedFileFormats(conversion_mode)),
        tuple(media_format.supportedVideoCodecs(conversion_mode)),
    )


def _profile_extension(profile: RecordingProfile) -> str:
    return {
        "WMV": ".wmv",
        "AVI": ".avi",
        "Matroska": ".mkv",
        "MPEG4": ".mp4",
        "Ogg": ".ogv",
        "QuickTime": ".mov",
        "WebM": ".webm",
    }.get(_enum_name(profile.file_format), ".video")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as capture_file:
        for block in iter(lambda: capture_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
