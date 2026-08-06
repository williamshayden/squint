from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QByteArray, QObject, QProcess, Signal

from edge_perception.config import RunConfig, load_run_config
from edge_perception.contracts import Region
from edge_perception.gui.run_controller import (
    MALFORMED_PROGRESS_ERROR,
    MISSING_TERMINAL_PROGRESS_ERROR,
    RunController,
)
from edge_perception.progress import ProgressEvent


class FakeProcess(QObject):
    readyReadStandardOutput = Signal()
    readyReadStandardError = Signal()
    finished = Signal(int, object)
    errorOccurred = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.program: str | None = None
        self.arguments: list[str] | None = None
        self.start_count = 0
        self.kill_count = 0
        self._stdout = bytearray()
        self._stderr = bytearray()

    @property
    def started_once(self) -> bool:
        return self.start_count == 1

    def setProgram(self, program: str) -> None:
        self.program = program

    def setArguments(self, arguments: list[str]) -> None:
        self.arguments = list(arguments)

    def start(self) -> None:
        self.start_count += 1

    def kill(self) -> None:
        self.kill_count += 1

    def readAllStandardOutput(self) -> QByteArray:
        data = QByteArray(bytes(self._stdout))
        self._stdout.clear()
        return data

    def readAllStandardError(self) -> QByteArray:
        data = QByteArray(bytes(self._stderr))
        self._stderr.clear()
        return data

    def emit_stdout(self, data: str | bytes) -> None:
        self._stdout.extend(data.encode("utf-8") if isinstance(data, str) else data)
        self.readyReadStandardOutput.emit()

    def emit_stderr(self, data: str | bytes) -> None:
        self._stderr.extend(data.encode("utf-8") if isinstance(data, str) else data)
        self.readyReadStandardError.emit()

    def finish(
        self,
        exit_code: int = 0,
        exit_status: QProcess.ExitStatus = QProcess.ExitStatus.NormalExit,
    ) -> None:
        self.finished.emit(exit_code, exit_status)


@pytest.fixture
def fake_process() -> FakeProcess:
    return FakeProcess()


def make_config(tmp_path: Path, *, output: str = "run") -> RunConfig:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    return RunConfig(
        input_path=input_path,
        output_dir=tmp_path / output,
        regions=(Region("roi", 1, 2, 30, 40),),
        threshold=0.3,
        max_frames=3,
        warmup_runs=1,
        annotate_every=1,
        detector_id="dfine-nano-coco",
        device="cpu",
    )


def event_json(event: ProgressEvent) -> str:
    return json.dumps(event.to_dict(), allow_nan=False, sort_keys=True)


def test_run_controller_starts_worker_without_shell(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    config = make_config(tmp_path)

    controller.start(config)

    assert fake_process.program == sys.executable
    assert fake_process.arguments == [
        "-m",
        "edge_perception.worker",
        "--config",
        str((config.output_dir.parent / f"{config.output_dir.name}.experiment.json").resolve()),
        "--cancel-file",
        str((config.output_dir.parent / f".{config.output_dir.name}.cancel").resolve()),
    ]
    assert fake_process.started_once


def test_run_controller_rejects_second_active_run(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    controller.start(make_config(tmp_path, output="run-a"))

    with pytest.raises(RuntimeError, match="already active"):
        controller.start(make_config(tmp_path, output="run-b"))


def test_partial_stdout_is_buffered_until_newline(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    progress: list[ProgressEvent] = []
    controller.progressChanged.connect(progress.append)
    controller.start(make_config(tmp_path))
    event = ProgressEvent("running", 2, 3, 4.5, None)
    record = event_json(event)

    fake_process.emit_stdout(record[: len(record) // 2])

    assert progress == []

    fake_process.emit_stdout(record[len(record) // 2 :] + "\n")

    assert progress == [event]


def test_one_stdout_chunk_can_contain_multiple_events(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    progress: list[ProgressEvent] = []
    controller.progressChanged.connect(progress.append)
    controller.start(make_config(tmp_path))
    first = ProgressEvent("validating", 0, 0, 0.5, None)
    second = ProgressEvent("running", 2, 3, 4.5, None)

    fake_process.emit_stdout(f"{event_json(first)}\n{event_json(second)}\n")

    assert progress == [first, second]


def test_non_json_stdout_fails_the_run(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    controller.runFailed.connect(failures.append)
    controller.start(make_config(tmp_path))

    fake_process.emit_stdout("not-json\n")

    assert MALFORMED_PROGRESS_ERROR == "worker emitted malformed progress"
    assert failures == [MALFORMED_PROGRESS_ERROR]
    assert fake_process.kill_count == 1


def test_stderr_is_bounded_and_included_in_failure(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    controller.runFailed.connect(failures.append)
    controller.start(make_config(tmp_path))
    discarded = b"discarded-prefix"
    retained = b"z" * (16 * 1024)

    fake_process.emit_stderr(discarded + retained)
    fake_process.finish(7, QProcess.ExitStatus.CrashExit)

    assert len(failures) == 1
    assert retained.decode("ascii") in failures[0]
    assert discarded.decode("ascii") not in failures[0]


def test_cancel_atomically_publishes_owned_file(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    config = make_config(tmp_path)
    controller.start(config)
    cancel_path = (config.output_dir.parent / f".{config.output_dir.name}.cancel").resolve()

    controller.cancel()

    assert cancel_path.exists()
    assert cancel_path.read_bytes() == b""
    assert list(cancel_path.parent.glob(f"{cancel_path.name}.*.tmp")) == []


def test_zero_exit_without_terminal_event_is_failure(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    finished: list[tuple[Path, dict[str, object]]] = []
    controller.runFailed.connect(failures.append)
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.start(make_config(tmp_path))
    fake_process.emit_stdout(event_json(ProgressEvent("running", 1, 1, 1.0, None)) + "\n")

    fake_process.finish(0, QProcess.ExitStatus.NormalExit)

    assert finished == []
    assert failures == [MISSING_TERMINAL_PROGRESS_ERROR]


def test_crash_preserves_run_and_experiment_config(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    config = make_config(tmp_path)
    controller.start(config)
    assert controller.config_path is not None
    config.output_dir.mkdir()
    artifact = config.output_dir / "partial.jsonl"
    artifact.write_text("partial\n", encoding="utf-8")

    fake_process.finish(9, QProcess.ExitStatus.CrashExit)

    assert config.output_dir.is_dir()
    assert artifact.read_text(encoding="utf-8") == "partial\n"
    assert controller.config_path.exists()


def test_terminal_cleanup_removes_only_cancel_file(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    config = make_config(tmp_path)
    controller.start(config)
    assert controller.config_path is not None
    cancel_path = config.output_dir.parent / f".{config.output_dir.name}.cancel"
    controller.cancel()
    config.output_dir.mkdir()
    artifact = config.output_dir / "summary.json"
    artifact.write_text("{}\n", encoding="utf-8")
    keep = config.output_dir.parent / "keep.tmp"
    keep.write_text("keep", encoding="utf-8")
    terminal = ProgressEvent("cancelled", 2, 2, 5.0, None)

    fake_process.emit_stdout(event_json(terminal) + "\n")
    fake_process.finish(0, QProcess.ExitStatus.NormalExit)

    assert not cancel_path.exists()
    assert controller.config_path.exists()
    assert artifact.exists()
    assert keep.read_text(encoding="utf-8") == "keep"


def test_persisted_gui_config_loads_as_last_config(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    finished: list[tuple[Path, dict[str, object]]] = []
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    config = make_config(tmp_path)
    controller.start(config)
    terminal = ProgressEvent("complete", 3, 4, 6.5, None)

    fake_process.emit_stdout(event_json(terminal) + "\n")
    fake_process.finish(0, QProcess.ExitStatus.NormalExit)

    assert controller.config_path is not None
    assert load_run_config(controller.config_path) == controller.last_config
    assert finished == [(config.output_dir, terminal.to_dict())]


def test_existing_experiment_config_is_never_overwritten(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    config = make_config(tmp_path)
    config_path = config.output_dir.parent / f"{config.output_dir.name}.experiment.json"
    config_path.write_text("immutable\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="experiment config already exists"):
        controller.start(config)

    assert config_path.read_text(encoding="utf-8") == "immutable\n"
    assert fake_process.start_count == 0


def test_start_removes_only_the_exact_stale_cancel_file(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    config = make_config(tmp_path)
    cancel_path = config.output_dir.parent / f".{config.output_dir.name}.cancel"
    sibling = config.output_dir.parent / f"{cancel_path.name}.keep"
    cancel_path.write_text("stale", encoding="utf-8")
    sibling.write_text("keep", encoding="utf-8")

    controller.start(config)

    assert not cancel_path.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"
