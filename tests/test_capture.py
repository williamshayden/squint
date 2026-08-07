from __future__ import annotations

import errno
import os
import subprocess
import sys
from dataclasses import replace
from enum import IntEnum
from hashlib import sha256
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, QSize, QUrl, Signal
from PySide6.QtMultimedia import QMediaRecorder, QVideoFrameFormat
from pytestqt.qtbot import QtBot

from edge_perception.capture import (
    CameraDeviceInfo,
    CameraFormatInfo,
    QtCaptureController,
    RecordingProfile,
    select_camera_format,
    select_recording_profile,
    validate_capture_result,
)
from edge_perception.config import CaptureRequest, CaptureResult
from edge_perception.video import VideoMetadata


def test_shared_capture_import_does_not_load_qtwidgets() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import edge_perception.capture; "
                "print('PySide6.QtWidgets' in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


class FakeFileFormat(IntEnum):
    AVI = 1
    Matroska = 2
    MPEG4 = 3


class FakeVideoCodec(IntEnum):
    MPEG4 = 2
    H264 = 3
    H265 = 4
    VP9 = 6
    AV1 = 7


class FakeCameraFormat:
    def __init__(
        self,
        width: int,
        height: int,
        min_fps: float,
        max_fps: float,
        pixel_format: QVideoFrameFormat.PixelFormat,
    ) -> None:
        self._resolution = QSize(width, height)
        self._min_fps = min_fps
        self._max_fps = max_fps
        self._pixel_format = pixel_format

    def resolution(self) -> QSize:
        return self._resolution

    def minFrameRate(self) -> float:
        return self._min_fps

    def maxFrameRate(self) -> float:
        return self._max_fps

    def pixelFormat(self) -> QVideoFrameFormat.PixelFormat:
        return self._pixel_format


class FakeCameraDevice:
    def __init__(
        self,
        device_id: bytes,
        description: str,
        formats: tuple[FakeCameraFormat, ...],
    ) -> None:
        self._device_id = device_id
        self._description = description
        self._formats = formats

    def id(self) -> bytes:
        return self._device_id

    def description(self) -> str:
        return self._description

    def videoFormats(self) -> list[FakeCameraFormat]:
        return list(self._formats)


class FakeMediaDevices(QObject):
    videoInputsChanged = Signal()

    def __init__(self, devices: tuple[FakeCameraDevice, ...]) -> None:
        super().__init__()
        self.devices = devices

    def videoInputs(self) -> list[FakeCameraDevice]:
        return list(self.devices)


class FakeCamera(QObject):
    errorOccurred = Signal(object, str)

    def __init__(
        self,
        device: object,
        parent: QObject,
        *,
        start_error: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.device = device
        self.start_error = start_error
        self.camera_format: object | None = None
        self.start_count = 0
        self.stop_count = 0

    def setCameraFormat(self, camera_format: object) -> None:
        self.camera_format = camera_format

    def start(self) -> None:
        self.start_count += 1
        if self.start_error is not None:
            self.errorOccurred.emit(QMediaRecorder.Error.ResourceError, self.start_error)

    def stop(self) -> None:
        self.stop_count += 1


class FakeCaptureSession(QObject):
    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self.camera: object | None = None
        self.recorder: object | None = None
        self.video_output: object | None = None
        self.audio_inputs: list[object] = []

    def setCamera(self, camera: object) -> None:
        self.camera = camera

    def setRecorder(self, recorder: object) -> None:
        self.recorder = recorder

    def setVideoOutput(self, video_output: object) -> None:
        self.video_output = video_output

    def setAudioInput(self, audio_input: object) -> None:
        self.audio_inputs.append(audio_input)


class FakeRecorder(QObject):
    recorderStateChanged = Signal(object)
    errorOccurred = Signal(object, str)

    def __init__(
        self,
        parent: QObject,
        *,
        output_error: str | None = None,
        record_error: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.output_error = output_error
        self.record_error = record_error
        self.media_format: object | None = None
        self.output_location = QUrl()
        self.actual_location: QUrl | None = None
        self.record_count = 0
        self.stop_count = 0

    def setMediaFormat(self, media_format: object) -> None:
        self.media_format = media_format

    def setOutputLocation(self, location: QUrl) -> None:
        self.output_location = location
        if self.output_error is not None:
            self.errorOccurred.emit(QMediaRecorder.Error.ResourceError, self.output_error)

    def actualLocation(self) -> QUrl:
        return self.actual_location or self.output_location

    def record(self) -> None:
        self.record_count += 1
        if self.record_error is not None:
            self.errorOccurred.emit(QMediaRecorder.Error.ResourceError, self.record_error)

    def stop(self) -> None:
        self.stop_count += 1


class ControllerHarness:
    def __init__(
        self,
        tmp_path: Path,
        metadata: VideoMetadata | None = None,
        *,
        decode_error: Exception | None = None,
        camera_start_error: str | None = None,
        recorder_output_error: str | None = None,
        recorder_record_error: str | None = None,
    ) -> None:
        self.qt_formats = (
            FakeCameraFormat(
                1280,
                720,
                15.0,
                60.0,
                QVideoFrameFormat.PixelFormat.Format_NV12,
            ),
            FakeCameraFormat(
                1920,
                1080,
                30.0,
                30.0,
                QVideoFrameFormat.PixelFormat.Format_YUYV,
            ),
        )
        self.device = FakeCameraDevice(b"camera-1", "EMEET", self.qt_formats)
        self.media_devices = FakeMediaDevices((self.device,))
        self.cameras: list[FakeCamera] = []
        self.sessions: list[FakeCaptureSession] = []
        self.recorders: list[FakeRecorder] = []
        self.probed_paths: list[Path] = []
        self.decoded_paths: list[Path] = []
        self.decode_error = decode_error
        self.camera_start_error = camera_start_error
        self.recorder_output_error = recorder_output_error
        self.recorder_record_error = recorder_record_error
        self.metadata = metadata or VideoMetadata(
            1280,
            720,
            60.0,
            "mp4",
            "h264",
            2.5,
            False,
            7,
        )
        self.controller = QtCaptureController(
            media_devices=self.media_devices,
            camera_factory=self._camera_factory,
            capture_session_factory=self._session_factory,
            recorder_factory=self._recorder_factory,
            media_format_factory=lambda profile: profile,
            format_capabilities=lambda: (
                (FakeFileFormat.MPEG4,),
                (FakeVideoCodec.H264,),
            ),
            profile_is_supported=lambda _profile: True,
            probe=self._probe,
            decode_validator=self._decode,
            capture_directory=tmp_path / "captures",
        )

    @property
    def device_id(self) -> str:
        return b"camera-1".hex()

    @property
    def camera(self) -> FakeCamera:
        return self.cameras[-1]

    @property
    def session(self) -> FakeCaptureSession:
        return self.sessions[-1]

    @property
    def recorder(self) -> FakeRecorder:
        return self.recorders[-1]

    def request(
        self,
        *,
        width: int | None = 1280,
        height: int | None = 720,
        fps: float | None = 60.0,
        strict: bool = True,
    ) -> CaptureRequest:
        return CaptureRequest(self.device_id, "EMEET", width, height, fps, strict)

    def start_preview(self, request: CaptureRequest | None = None) -> CameraFormatInfo:
        return self.controller.start_preview(request or self.request(), object())

    def start_recording(self, final_path: Path) -> Path:
        self.controller.start_recording(final_path)
        return Path(self.recorder.output_location.toLocalFile())

    def _camera_factory(self, device: object, parent: QObject) -> FakeCamera:
        camera = FakeCamera(device, parent, start_error=self.camera_start_error)
        self.cameras.append(camera)
        return camera

    def _session_factory(self, parent: QObject) -> FakeCaptureSession:
        session = FakeCaptureSession(parent)
        self.sessions.append(session)
        return session

    def _recorder_factory(self, parent: QObject) -> FakeRecorder:
        recorder = FakeRecorder(
            parent,
            output_error=self.recorder_output_error,
            record_error=self.recorder_record_error,
        )
        self.recorders.append(recorder)
        return recorder

    def _probe(self, path: Path) -> VideoMetadata:
        self.probed_paths.append(path)
        return self.metadata

    def _decode(self, path: Path) -> None:
        self.decoded_paths.append(path)
        if self.decode_error is not None:
            raise self.decode_error


def test_select_camera_format_treats_fps_independently() -> None:
    formats = (
        CameraFormatInfo(1280, 720, 15.0, 60.0, "NV12", object()),
        CameraFormatInfo(1920, 1080, 30.0, 30.0, "YUYV", object()),
    )
    request = CaptureRequest("camera-1", "EMEET", None, None, 60.0, False)

    selected = select_camera_format(formats, request)

    assert (selected.width, selected.height, selected.max_fps) == (1280, 720, 60.0)


def test_select_camera_format_ranks_each_supplied_constraint() -> None:
    formats = (
        CameraFormatInfo(1280, 1080, 30.0, 30.0, "NV12", object()),
        CameraFormatInfo(1920, 720, 15.0, 60.0, "YUYV", object()),
        CameraFormatInfo(1600, 900, 24.0, 30.0, "RGB", object()),
    )
    request = CaptureRequest("camera-1", "EMEET", 1920, 1080, 60.0, False)

    selected = select_camera_format(formats, request)

    assert selected == CameraFormatInfo(1920, 720, 15.0, 60.0, "YUYV", object())


def test_select_camera_format_without_constraints_prefers_pixels_then_fps() -> None:
    formats = (
        CameraFormatInfo(1920, 1080, 15.0, 30.0, "NV12", object()),
        CameraFormatInfo(1280, 720, 15.0, 120.0, "YUYV", object()),
        CameraFormatInfo(1920, 1080, 15.0, 60.0, "RGB", object()),
    )
    request = CaptureRequest("camera-1", "EMEET", None, None, None, False)

    selected = select_camera_format(formats, request)

    assert selected == CameraFormatInfo(1920, 1080, 15.0, 60.0, "RGB", object())


@pytest.mark.parametrize(
    ("capture_request", "expected"),
    [
        (CaptureRequest("camera-1", "EMEET", 1920, None, None, True), (1920, 720)),
        (CaptureRequest("camera-1", "EMEET", None, 1080, None, True), (1280, 1080)),
        (CaptureRequest("camera-1", "EMEET", None, None, 60.0, True), (1920, 720)),
    ],
)
def test_strict_camera_format_filters_every_supplied_field_independently(
    capture_request: CaptureRequest,
    expected: tuple[int, int],
) -> None:
    formats = (
        CameraFormatInfo(1280, 1080, 15.0, 30.0, "NV12", object()),
        CameraFormatInfo(1920, 720, 15.0, 60.0, "YUYV", object()),
    )

    selected = select_camera_format(formats, capture_request)

    assert (selected.width, selected.height) == expected


def test_strict_camera_format_rejects_unavailable_mode() -> None:
    request = CaptureRequest("camera-1", "EMEET", 1920, 1080, 60.0, True)
    formats = (CameraFormatInfo(1920, 1080, 30.0, 30.0, "YUYV", object()),)

    with pytest.raises(ValueError, match="^requested camera mode is unavailable$"):
        select_camera_format(formats, request)


def test_strict_capture_uses_documented_fps_tolerance() -> None:
    request = CaptureRequest("camera-1", "EMEET", 1920, 1080, 30.0, True)
    accepted = VideoMetadata(1920, 1080, 29.95, "mp4", "h264", 5.0, False, 100)
    rejected = replace(accepted, average_fps=29.7)

    validate_capture_result(request, accepted)
    with pytest.raises(ValueError, match="FPS"):
        validate_capture_result(request, rejected)


def test_capture_validation_requires_video_only() -> None:
    request = CaptureRequest("camera-1", "EMEET", None, None, None, False)
    metadata = VideoMetadata(640, 480, 30.0, "mp4", "h264", 1.0, True, 100)

    with pytest.raises(ValueError, match="audio"):
        validate_capture_result(request, metadata)


def test_recording_profile_uses_documented_preference_order() -> None:
    profile = select_recording_profile(
        (FakeFileFormat.AVI, FakeFileFormat.Matroska, FakeFileFormat.MPEG4),
        (
            FakeVideoCodec.MPEG4,
            FakeVideoCodec.AV1,
            FakeVideoCodec.VP9,
            FakeVideoCodec.H265,
            FakeVideoCodec.H264,
        ),
    )

    assert profile == RecordingProfile(FakeFileFormat.MPEG4, FakeVideoCodec.H264)


def test_recording_profile_selects_first_compatible_preferred_pair() -> None:
    checked: list[RecordingProfile] = []

    def is_supported(profile: RecordingProfile) -> bool:
        checked.append(profile)
        return profile == RecordingProfile(FakeFileFormat.Matroska, FakeVideoCodec.H264)

    profile = select_recording_profile(
        (FakeFileFormat.Matroska, FakeFileFormat.MPEG4),
        (FakeVideoCodec.VP9, FakeVideoCodec.H265, FakeVideoCodec.H264),
        is_supported=is_supported,
    )

    assert profile == RecordingProfile(FakeFileFormat.Matroska, FakeVideoCodec.H264)
    assert checked == [
        RecordingProfile(FakeFileFormat.MPEG4, FakeVideoCodec.H264),
        RecordingProfile(FakeFileFormat.MPEG4, FakeVideoCodec.H265),
        RecordingProfile(FakeFileFormat.MPEG4, FakeVideoCodec.VP9),
        RecordingProfile(FakeFileFormat.Matroska, FakeVideoCodec.H264),
    ]


def test_recording_profile_rejects_nonempty_but_incompatible_capabilities() -> None:
    with pytest.raises(ValueError, match="^no supported video recording profile$"):
        select_recording_profile(
            (FakeFileFormat.MPEG4,),
            (FakeVideoCodec.H264,),
            is_supported=lambda _profile: False,
        )


def test_recording_profile_falls_back_to_enum_order() -> None:
    profile = select_recording_profile(
        (FakeFileFormat.AVI,),
        (FakeVideoCodec.MPEG4,),
    )

    assert profile == RecordingProfile(FakeFileFormat.AVI, FakeVideoCodec.MPEG4)


@pytest.mark.parametrize(
    ("file_formats", "video_codecs"),
    [
        ((), (FakeVideoCodec.H264,)),
        ((FakeFileFormat.MPEG4,), ()),
    ],
)
def test_recording_profile_requires_encodable_video(
    file_formats: tuple[FakeFileFormat, ...],
    video_codecs: tuple[FakeVideoCodec, ...],
) -> None:
    with pytest.raises(ValueError, match="^no supported video recording profile$"):
        select_recording_profile(file_formats, video_codecs)


def test_controller_enumerates_stable_device_descriptors(tmp_path: Path) -> None:
    harness = ControllerHarness(tmp_path)
    changed: list[tuple[CameraDeviceInfo, ...]] = []
    harness.controller.devicesChanged.connect(lambda: changed.append(harness.controller.devices()))

    devices = harness.controller.devices()
    harness.media_devices.videoInputsChanged.emit()

    assert devices == (
        CameraDeviceInfo(
            b"camera-1".hex(),
            "EMEET",
            (
                CameraFormatInfo(1280, 720, 15.0, 60.0, "NV12", object()),
                CameraFormatInfo(1920, 1080, 30.0, 30.0, "YUYV", object()),
            ),
            object(),
        ),
    )
    assert changed == [devices]


def test_preview_applies_selected_format(tmp_path: Path) -> None:
    harness = ControllerHarness(tmp_path)
    video_output = object()
    selected_events: list[CameraFormatInfo] = []
    harness.controller.previewStarted.connect(selected_events.append)

    selected = harness.controller.start_preview(
        harness.request(width=None, height=None, fps=60.0, strict=False),
        video_output,
    )

    assert selected == CameraFormatInfo(1280, 720, 15.0, 60.0, "NV12", object())
    assert harness.camera.camera_format is harness.qt_formats[0]
    assert harness.camera.start_count == 1
    assert harness.session.camera is harness.camera
    assert harness.session.recorder is harness.recorder
    assert harness.session.video_output is video_output
    assert harness.session.audio_inputs == []
    assert selected_events == [selected]


def test_synchronous_camera_error_does_not_emit_preview_success(tmp_path: Path) -> None:
    harness = ControllerHarness(tmp_path, camera_start_error="sync camera failed")
    previews: list[CameraFormatInfo] = []
    errors: list[str] = []
    harness.controller.previewStarted.connect(previews.append)
    harness.controller.errorOccurred.connect(errors.append)

    with pytest.raises(RuntimeError, match="^sync camera failed$"):
        harness.start_preview()

    assert previews == []
    assert errors == ["sync camera failed"]
    assert harness.controller.is_previewing is False


@pytest.mark.parametrize(
    ("error_stage", "message"),
    [
        ("output", "sync output failed"),
        ("record", "sync record failed"),
    ],
)
def test_synchronous_recorder_error_aborts_start_without_stale_dereference(
    tmp_path: Path,
    error_stage: str,
    message: str,
) -> None:
    harness = ControllerHarness(
        tmp_path,
        recorder_output_error=message if error_stage == "output" else None,
        recorder_record_error=message if error_stage == "record" else None,
    )
    harness.start_preview()
    started: list[None] = []
    errors: list[str] = []
    harness.controller.recordingStarted.connect(lambda: started.append(None))
    harness.controller.errorOccurred.connect(errors.append)

    with pytest.raises(RuntimeError, match=f"^{message}$"):
        harness.controller.start_recording(tmp_path / "finished.mp4")

    assert started == []
    assert errors == [message]
    assert harness.controller.is_previewing is False
    assert not list(tmp_path.glob(".capture-*"))


def test_recording_state_is_explicit(tmp_path: Path) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    started: list[None] = []
    harness.controller.recordingStarted.connect(lambda: started.append(None))

    harness.controller.start_recording(tmp_path / "finished.mp4")

    assert harness.recorder.record_count == 1
    assert harness.controller.is_recording is False
    assert started == []
    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.RecordingState)
    assert harness.controller.is_recording is True
    assert started == [None]
    harness.controller.stop_recording()
    assert harness.recorder.stop_count == 1
    harness.controller.discard()


def test_discard_removes_only_owned_temporary_file(tmp_path: Path) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    temporary_path = harness.start_recording(final_path)
    temporary_path.write_bytes(b"partial")
    unrelated = final_path.parent / "unrelated.part.mp4"
    unrelated.write_bytes(b"keep")

    harness.controller.discard()

    assert not temporary_path.exists()
    assert unrelated.read_bytes() == b"keep"
    assert not (tmp_path / "finished.mp4").exists()


def test_recorder_error_cleans_temporary_file(tmp_path: Path) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    temporary_path = harness.start_recording(final_path)
    temporary_path.write_bytes(b"partial")
    unrelated = final_path.parent / "keep.mp4"
    unrelated.write_bytes(b"keep")
    errors: list[str] = []
    finished: list[CaptureResult] = []
    harness.controller.errorOccurred.connect(errors.append)
    harness.controller.recordingFinished.connect(finished.append)

    harness.recorder.errorOccurred.emit(QMediaRecorder.Error.ResourceError, "camera failed")

    assert not temporary_path.exists()
    assert unrelated.read_bytes() == b"keep"
    assert errors == ["camera failed"]
    assert finished == []


def test_cleanup_permission_error_preserves_primary_failure_and_owned_file(
    monkeypatch: pytest.MonkeyPatch,
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    staged_path = harness.start_recording(tmp_path / "finished.mp4")
    staged_path.write_bytes(b"partial")
    errors: list[str] = []
    harness.controller.errorOccurred.connect(errors.append)

    def deny_unlink(_path: Path) -> None:
        raise PermissionError("access denied")

    monkeypatch.setattr("edge_perception.capture._unlink_file", deny_unlink)
    with qtbot.captureExceptions() as exceptions:
        harness.recorder.errorOccurred.emit(
            QMediaRecorder.Error.ResourceError,
            "camera failed",
        )

    assert exceptions == []
    assert errors == [
        f"camera failed; cleanup failed: could not remove staged file {staged_path}: access denied"
    ]
    assert staged_path.read_bytes() == b"partial"
    assert harness.controller.is_previewing is False

    monkeypatch.setattr("edge_perception.capture._unlink_file", lambda path: path.unlink())
    harness.controller.discard()

    assert not staged_path.parent.exists()


def test_cleanup_retains_owned_staging_directory_until_rmdir_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    staged_path = harness.start_recording(tmp_path / "finished.mp4")
    staged_path.write_bytes(b"partial")
    errors: list[str] = []
    harness.controller.errorOccurred.connect(errors.append)

    def deny_rmdir(_path: Path) -> None:
        raise PermissionError("directory busy")

    monkeypatch.setattr("edge_perception.capture._remove_directory", deny_rmdir)
    harness.recorder.errorOccurred.emit(QMediaRecorder.Error.ResourceError, "camera failed")

    assert not staged_path.exists()
    assert staged_path.parent.is_dir()
    assert errors == [
        (
            "camera failed; cleanup failed: "
            f"could not remove staging directory {staged_path.parent}: directory busy"
        )
    ]

    monkeypatch.setattr(
        "edge_perception.capture._remove_directory", lambda path: path.rmdir()
    )
    harness.controller.discard()

    assert not staged_path.parent.exists()


def test_retained_cleanup_failure_blocks_reuse_without_losing_ownership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    old_staged_path = harness.start_recording(tmp_path / "first.mp4")
    old_staged_path.write_bytes(b"partial")

    def deny_unlink(_path: Path) -> None:
        raise PermissionError("still locked")

    monkeypatch.setattr("edge_perception.capture._unlink_file", deny_unlink)
    harness.recorder.errorOccurred.emit(QMediaRecorder.Error.ResourceError, "camera failed")
    harness.start_preview()
    replacement_recorder = harness.recorder

    with pytest.raises(RuntimeError) as raised:
        harness.controller.start_recording(tmp_path / "second.mp4")

    assert str(raised.value) == (
        "cannot start recording while prior capture cleanup is incomplete: "
        f"could not remove staged file {old_staged_path}: still locked"
    )
    assert old_staged_path.read_bytes() == b"partial"
    assert set(tmp_path.glob(".capture-*")) == {old_staged_path.parent}
    assert replacement_recorder.record_count == 0
    assert replacement_recorder.output_location.isEmpty()

    monkeypatch.setattr("edge_perception.capture._unlink_file", lambda path: path.unlink())
    harness.controller.discard()

    assert not old_staged_path.parent.exists()


def test_reuse_proceeds_after_retained_cleanup_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    old_staged_path = harness.start_recording(tmp_path / "first.mp4")
    old_staged_path.write_bytes(b"partial")

    def deny_unlink(_path: Path) -> None:
        raise PermissionError("temporarily locked")

    monkeypatch.setattr("edge_perception.capture._unlink_file", deny_unlink)
    harness.recorder.errorOccurred.emit(QMediaRecorder.Error.ResourceError, "camera failed")
    harness.start_preview()
    monkeypatch.setattr("edge_perception.capture._unlink_file", lambda path: path.unlink())

    new_staged_path = harness.start_recording(tmp_path / "second.mp4")

    assert not old_staged_path.parent.exists()
    assert new_staged_path.parent != old_staged_path.parent
    assert set(tmp_path.glob(".capture-*")) == {new_staged_path.parent}
    assert harness.recorder.record_count == 1

    harness.controller.discard()

    assert not list(tmp_path.glob(".capture-*"))


def test_success_atomically_publishes_capture(tmp_path: Path) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    temporary_path = harness.start_recording(final_path)
    temporary_path.write_bytes(b"payload")
    finished: list[CaptureResult] = []
    harness.controller.recordingFinished.connect(finished.append)
    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.RecordingState)

    assert temporary_path.parent.parent == final_path.parent
    assert temporary_path.parent != final_path.parent
    assert not final_path.exists()
    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert final_path.read_bytes() == b"payload"
    assert not temporary_path.exists()
    assert harness.decoded_paths == [temporary_path]
    assert harness.probed_paths == [temporary_path]
    assert len(finished) == 1
    assert finished[0].path == final_path.resolve()


def test_owned_rewritten_actual_location_is_probed_hashed_and_published(tmp_path: Path) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    requested_path = harness.start_recording(final_path)
    actual_path = requested_path.parent / "backend-rewritten.mp4"
    actual_path.write_bytes(b"payload")
    harness.recorder.actual_location = QUrl.fromLocalFile(str(actual_path))
    finished: list[CaptureResult] = []
    harness.controller.recordingFinished.connect(finished.append)

    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert final_path.read_bytes() == b"payload"
    assert harness.decoded_paths == [actual_path]
    assert harness.probed_paths == [actual_path]
    assert finished[0].sha256 == sha256(b"payload").hexdigest()
    assert not requested_path.parent.exists()


def test_collision_race_preserves_existing_completed_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    staged_path = harness.start_recording(final_path)
    staged_path.write_bytes(b"new capture")
    real_link = os.link

    def collide_then_link(source: Path, destination: Path) -> None:
        destination.write_bytes(b"completed capture")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", collide_then_link)
    errors: list[str] = []
    harness.controller.errorOccurred.connect(errors.append)

    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert final_path.read_bytes() == b"completed capture"
    assert errors == [f"capture destination already exists: {final_path.resolve()}"]
    assert not staged_path.parent.exists()


def test_unsupported_hard_links_fail_without_copy_or_replace_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    staged_path = harness.start_recording(final_path)
    staged_path.write_bytes(b"payload")

    def unsupported_link(_source: Path, _destination: Path) -> None:
        raise OSError(errno.EPERM, "hard links unsupported")

    monkeypatch.setattr(os, "link", unsupported_link)
    errors: list[str] = []
    harness.controller.errorOccurred.connect(errors.append)

    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert not final_path.exists()
    assert errors == ["atomic capture publication is unavailable: hard links unsupported"]
    assert not staged_path.parent.exists()


def test_capture_result_uses_probed_metadata(tmp_path: Path) -> None:
    metadata = VideoMetadata(640, 480, 29.97, "matroska", "vp9", 3.25, False, 7)
    harness = ControllerHarness(tmp_path, metadata)
    request = harness.request(width=1920, height=1080, fps=60.0, strict=False)
    selected = harness.start_preview(request)
    final_path = tmp_path / "finished.mkv"
    temporary_path = harness.start_recording(final_path)
    temporary_path.write_bytes(b"payload")
    finished: list[CaptureResult] = []
    harness.controller.recordingFinished.connect(finished.append)
    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.RecordingState)

    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert finished == [
        CaptureResult(
            request=request,
            selected_width=selected.width,
            selected_height=selected.height,
            selected_min_fps=selected.min_fps,
            selected_max_fps=selected.max_fps,
            selected_pixel_format=selected.pixel_format,
            actual_width=640,
            actual_height=480,
            actual_fps=29.97,
            container="matroska",
            codec="vp9",
            duration_seconds=3.25,
            has_audio=False,
            file_size_bytes=7,
            path=final_path,
            sha256=sha256(b"payload").hexdigest(),
        )
    ]


def test_failed_validation_never_publishes_capture(tmp_path: Path) -> None:
    metadata = VideoMetadata(1280, 720, 60.0, "mp4", "h264", 1.0, True, 7)
    harness = ControllerHarness(tmp_path, metadata)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    temporary_path = harness.start_recording(final_path)
    temporary_path.write_bytes(b"payload")
    errors: list[str] = []
    harness.controller.errorOccurred.connect(errors.append)

    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert not final_path.exists()
    assert not temporary_path.exists()
    assert errors == ["captured video contains audio"]


def test_decode_failure_cleans_temporary_and_never_publishes(
    qtbot: QtBot,
    tmp_path: Path,
) -> None:
    harness = ControllerHarness(tmp_path, decode_error=RuntimeError("decode failed"))
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    temporary_path = harness.start_recording(final_path)
    temporary_path.write_bytes(b"malformed")
    errors: list[str] = []
    harness.controller.errorOccurred.connect(errors.append)

    with qtbot.captureExceptions() as exceptions:
        harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert exceptions == []
    assert not temporary_path.exists()
    assert not final_path.exists()
    assert errors == ["decode failed"]


def test_unexpected_actual_location_is_not_owned_or_deleted(tmp_path: Path) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    temporary_path = harness.start_recording(final_path)
    temporary_path.write_bytes(b"partial")
    foreign_path = tmp_path / "foreign.mp4"
    foreign_path.write_bytes(b"foreign")
    harness.recorder.actual_location = QUrl.fromLocalFile(str(foreign_path))
    errors: list[str] = []
    harness.controller.errorOccurred.connect(errors.append)

    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert not temporary_path.exists()
    assert foreign_path.read_bytes() == b"foreign"
    assert not final_path.exists()
    assert errors == ["recorder reported an unexpected output location"]


def test_traversing_actual_location_is_rejected_and_owned_stage_is_removed(
    tmp_path: Path,
) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    requested_path = harness.start_recording(final_path)
    requested_path.write_bytes(b"partial")
    nested_directory = requested_path.parent / "nested"
    nested_directory.mkdir()
    traversing_path = nested_directory / ".." / requested_path.name
    harness.recorder.actual_location = QUrl.fromLocalFile(str(traversing_path))
    errors: list[str] = []
    harness.controller.errorOccurred.connect(errors.append)

    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert not final_path.exists()
    assert not requested_path.parent.exists()
    assert errors == ["recorder reported an unexpected output location"]


def test_symlinked_actual_location_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    harness = ControllerHarness(tmp_path)
    harness.start_preview()
    final_path = tmp_path / "finished.mp4"
    requested_path = harness.start_recording(final_path)
    requested_path.write_bytes(b"partial")
    real_is_symlink = Path.is_symlink

    def report_actual_as_symlink(path: Path) -> bool:
        return path == requested_path or real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_actual_as_symlink)
    errors: list[str] = []
    harness.controller.errorOccurred.connect(errors.append)

    harness.recorder.recorderStateChanged.emit(QMediaRecorder.RecorderState.StoppedState)

    assert not final_path.exists()
    assert not requested_path.parent.exists()
    assert errors == ["recorder reported an unexpected output location"]
