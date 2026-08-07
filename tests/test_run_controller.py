from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import QByteArray, QObject, QProcess, Signal

import edge_perception.config as config_module
import edge_perception.gui.run_controller as run_controller_module
from edge_perception.config import RunConfig, load_run_config
from edge_perception.contracts import Region
from edge_perception.gui.run_controller import (
    MALFORMED_PROGRESS_ERROR,
    MISSING_TERMINAL_PROGRESS_ERROR,
    RunController,
)
from edge_perception.progress import ProgressEvent, ProgressPhase


class FakeProcess(QObject):
    readyReadStandardOutput = Signal()
    readyReadStandardError = Signal()
    finished = Signal(int, QProcess.ExitStatus)
    errorOccurred = Signal(QProcess.ProcessError)

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

    def emit_error(self, error: QProcess.ProcessError) -> None:
        self.errorOccurred.emit(error)


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


def install_owned_quarantine_unlink_failure(
    monkeypatch: pytest.MonkeyPatch,
    cancel_path: Path,
) -> list[Path]:
    actual_replace = os.replace
    actual_unlink = Path.unlink
    quarantines: list[Path] = []

    def capture_quarantine(source: Path | str, destination: Path | str) -> None:
        actual_replace(source, destination)
        if Path(source) == cancel_path:
            quarantines.append(Path(destination))

    def fail_quarantine(path: Path, *, missing_ok: bool = False) -> None:
        if quarantines and path == quarantines[-1]:
            raise OSError("locked")
        actual_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(run_controller_module.os, "replace", capture_quarantine)
    monkeypatch.setattr(Path, "unlink", fail_quarantine)
    return quarantines


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


def test_config_publication_race_preserves_winner_and_does_not_start_loser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    winner_process = FakeProcess()
    loser_process = FakeProcess()
    winner = RunController(process=winner_process)
    loser = RunController(process=loser_process)
    loser_config = make_config(tmp_path, output="race")
    winner_config = replace(loser_config, threshold=0.7)
    config_path = tmp_path / "race.experiment.json"
    actual_link = os.link
    interleaved = False
    winner_bytes: list[bytes] = []

    def publish_with_winner(source: str | bytes, destination: str | bytes) -> None:
        nonlocal interleaved
        if Path(destination) == config_path and not interleaved:
            interleaved = True
            winner.start(winner_config)
            winner_bytes.append(config_path.read_bytes())
        actual_link(source, destination)

    monkeypatch.setattr(config_module.os, "link", publish_with_winner)

    with pytest.raises(FileExistsError):
        loser.start(loser_config)

    assert interleaved is True
    assert winner_process.start_count == 1
    assert loser_process.start_count == 0
    assert load_run_config(config_path) == winner_config
    assert config_path.read_bytes() == winner_bytes[0]
    assert list(tmp_path.glob(".race.experiment.json.*.tmp")) == []


def test_hard_link_unavailable_fails_without_config_or_owned_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    config = make_config(tmp_path, output="unsupported")
    config_path = tmp_path / "unsupported.experiment.json"
    cancel_path = tmp_path / ".unsupported.cancel"
    cancel_path.write_text("stale", encoding="utf-8")

    def reject_link(_source: str | bytes, _destination: str | bytes) -> None:
        raise OSError("hard links disabled")

    monkeypatch.setattr(config_module.os, "link", reject_link)

    with pytest.raises(OSError, match="exclusive config publication requires hard-link support"):
        controller.start(config)

    assert not config_path.exists()
    assert cancel_path.read_text(encoding="utf-8") == "stale"
    assert fake_process.start_count == 0
    assert list(tmp_path.glob(".unsupported.experiment.json.*.tmp")) == []


def test_existing_config_error_survives_owned_temp_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    first = make_config(tmp_path, output="race-cleanup")
    second = make_config(tmp_path, output="after-race-cleanup")
    config_path = tmp_path / "race-cleanup.experiment.json"
    sentinel = tmp_path / "keep.tmp"
    sentinel.write_bytes(b"keep")
    actual_link = os.link
    actual_unlink = Path.unlink
    link_sources: list[Path] = []

    def publish_winner_then_link(source: str | bytes, destination: str | bytes) -> None:
        source_path = Path(source)
        link_sources.append(source_path)
        Path(destination).write_bytes(b"winner")
        actual_link(source, destination)

    def fail_owned_temp(path: Path, *, missing_ok: bool = False) -> None:
        if link_sources and path == link_sources[0]:
            raise OSError("temp locked")
        actual_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(config_module.os, "link", publish_winner_then_link)
        context.setattr(Path, "unlink", fail_owned_temp)

        with pytest.raises(FileExistsError) as error:
            controller.start(first)

        retained = link_sources[0]
        cleanup_error = (
            f"config publication cleanup failed while removing {retained}: temp locked"
        )
        assert str(error.value) == (
            f"experiment config already exists: {config_path}\n{cleanup_error}"
        )
        assert config_path.read_bytes() == b"winner"
        assert load_run_config(retained) == first
        assert retained.read_bytes() != b"winner"
        assert sentinel.read_bytes() == b"keep"
        assert fake_process.start_count == 0

        with pytest.raises(
            RuntimeError,
            match="previous config publication cleanup is unresolved",
        ):
            controller.start(second)

        assert link_sources == [retained]
        assert not (tmp_path / "after-race-cleanup.experiment.json").exists()

    controller.start(second)

    assert not retained.exists()
    assert sentinel.read_bytes() == b"keep"
    assert fake_process.start_count == 1


def test_hard_link_error_survives_owned_temp_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    config = make_config(tmp_path, output="unsupported-cleanup")
    config_path = tmp_path / "unsupported-cleanup.experiment.json"
    actual_unlink = Path.unlink
    link_sources: list[Path] = []

    def reject_link(source: str | bytes, _destination: str | bytes) -> None:
        link_sources.append(Path(source))
        raise OSError("hard links disabled")

    def fail_owned_temp(path: Path, *, missing_ok: bool = False) -> None:
        if link_sources and path == link_sources[0]:
            raise OSError("temp locked")
        actual_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(config_module.os, "link", reject_link)
    monkeypatch.setattr(Path, "unlink", fail_owned_temp)

    with pytest.raises(OSError) as error:
        controller.start(config)

    retained = link_sources[0]
    assert str(error.value) == (
        "exclusive config publication requires hard-link support: "
        f"{config_path}\n"
        f"config publication cleanup failed while removing {retained}: temp locked"
    )
    assert not config_path.exists()
    assert load_run_config(retained) == config
    assert fake_process.start_count == 0


def test_successful_config_reservation_survives_owned_temp_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    finished: list[tuple[Path, dict[str, object]]] = []
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    first = make_config(tmp_path, output="reserved-cleanup")
    second = make_config(tmp_path, output="after-reserved-cleanup")
    config_path = tmp_path / "reserved-cleanup.experiment.json"
    sentinel = tmp_path / "keep.tmp"
    sentinel.write_bytes(b"keep")
    actual_link = os.link
    actual_unlink = Path.unlink
    link_sources: list[Path] = []

    def capture_link(source: str | bytes, destination: str | bytes) -> None:
        link_sources.append(Path(source))
        actual_link(source, destination)

    def fail_owned_temp(path: Path, *, missing_ok: bool = False) -> None:
        if link_sources and path == link_sources[0]:
            raise OSError("temp locked")
        actual_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(config_module.os, "link", capture_link)
        context.setattr(Path, "unlink", fail_owned_temp)

        controller.start(first)

        retained = link_sources[0]
        cleanup_error = (
            f"config publication cleanup failed while removing {retained}: temp locked"
        )
        assert controller.config_path == config_path
        assert controller.last_config == first
        assert load_run_config(config_path) == first
        assert config_path.read_bytes() == retained.read_bytes()
        assert sentinel.read_bytes() == b"keep"
        assert fake_process.start_count == 1

        terminal = ProgressEvent("complete", 3, 4, 5.0, None)
        fake_process.emit_stdout(event_json(terminal) + "\n")
        fake_process.finish()

        assert finished == []
        assert failures == [cleanup_error]
        assert terminated == [None]
        assert controller.is_active is False
        assert retained.exists()

        with pytest.raises(
            RuntimeError,
            match="previous config publication cleanup is unresolved",
        ):
            controller.start(second)

        assert link_sources == [retained]
        assert not (tmp_path / "after-reserved-cleanup.experiment.json").exists()

    controller.start(second)

    assert not retained.exists()
    assert config_path.exists()
    assert sentinel.read_bytes() == b"keep"
    assert fake_process.start_count == 2


def test_terminal_cleanup_preserves_replaced_retained_config_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    finished: list[tuple[Path, dict[str, object]]] = []
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    first = make_config(tmp_path, output="terminal-replacement")
    second = make_config(tmp_path, output="after-terminal-replacement")
    config_path = tmp_path / "terminal-replacement.experiment.json"
    actual_link = os.link
    actual_unlink = Path.unlink
    link_sources: list[Path] = []

    def capture_link(source: str | bytes, destination: str | bytes) -> None:
        link_sources.append(Path(source))
        actual_link(source, destination)

    def fail_initial_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if link_sources and path == link_sources[0]:
            raise OSError("temp locked")
        actual_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(config_module.os, "link", capture_link)
        context.setattr(Path, "unlink", fail_initial_cleanup)
        controller.start(first)

    retained = link_sources[0]
    replacement = tmp_path / "terminal-replacement.foreign"
    replacement.write_bytes(b"replacement")
    os.replace(replacement, retained)
    terminal = ProgressEvent("complete", 3, 4, 5.0, None)
    cleanup_error = (
        "config publication cleanup refused to remove changed retained temporary: "
        f"{retained}"
    )

    fake_process.emit_stdout(event_json(terminal) + "\n")
    fake_process.finish()

    assert finished == []
    assert failures == [cleanup_error]
    assert terminated == [None]
    assert controller.is_active is False
    assert controller.config_path == config_path
    assert controller.last_config == first
    assert load_run_config(config_path) == first
    assert retained.read_bytes() == b"replacement"
    assert fake_process.start_count == 1

    with pytest.raises(
        RuntimeError,
        match="previous config publication cleanup is unresolved",
    ):
        controller.start(second)

    assert retained.read_bytes() == b"replacement"
    assert not (tmp_path / "after-terminal-replacement.experiment.json").exists()
    assert fake_process.start_count == 1

    retained.unlink()
    os.link(config_path, retained)
    controller.start(second)

    assert not retained.exists()
    assert load_run_config(config_path) == first
    assert controller.last_config == second
    assert fake_process.start_count == 2


def test_pre_reuse_cleanup_preserves_replaced_retained_config_temporary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    first = make_config(tmp_path, output="failed-publication-replacement")
    second = make_config(tmp_path, output="after-failed-publication-replacement")
    first_config_path = tmp_path / "failed-publication-replacement.experiment.json"
    second_config_path = tmp_path / "after-failed-publication-replacement.experiment.json"
    actual_unlink = Path.unlink
    link_sources: list[Path] = []

    def reject_link(source: str | bytes, _destination: str | bytes) -> None:
        link_sources.append(Path(source))
        raise OSError("hard links disabled")

    def fail_initial_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if link_sources and path == link_sources[0]:
            raise OSError("temp locked")
        actual_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(config_module.os, "link", reject_link)
        context.setattr(Path, "unlink", fail_initial_cleanup)

        with pytest.raises(OSError) as error:
            controller.start(first)

        retained = link_sources[0]
        assert str(error.value) == (
            "exclusive config publication requires hard-link support: "
            f"{first_config_path}\n"
            f"config publication cleanup failed while removing {retained}: temp locked"
        )
        owned_backup = tmp_path / "owned-config-temporary"
        os.replace(retained, owned_backup)
        retained.write_bytes(b"replacement")

    cleanup_error = (
        "config publication cleanup refused to remove changed retained temporary: "
        f"{retained}"
    )
    for _ in range(2):
        with pytest.raises(RuntimeError) as error:
            controller.start(second)
        assert str(error.value) == (
            f"previous config publication cleanup is unresolved: {cleanup_error}"
        )
        assert retained.read_bytes() == b"replacement"

    assert load_run_config(owned_backup) == first
    assert not first_config_path.exists()
    assert not second_config_path.exists()
    assert controller.config_path is None
    assert controller.last_config is None
    assert fake_process.start_count == 0

    retained.unlink()
    os.replace(owned_backup, retained)
    controller.start(second)

    assert not retained.exists()
    assert not owned_backup.exists()
    assert load_run_config(second_config_path) == second
    assert controller.config_path == second_config_path
    assert controller.last_config == second
    assert fake_process.start_count == 1


def test_protocol_failure_appends_retained_config_cleanup_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    config = make_config(tmp_path, output="protocol-cleanup")
    actual_link = os.link
    actual_unlink = Path.unlink
    link_sources: list[Path] = []

    def capture_link(source: str | bytes, destination: str | bytes) -> None:
        link_sources.append(Path(source))
        actual_link(source, destination)

    def fail_owned_temp(path: Path, *, missing_ok: bool = False) -> None:
        if link_sources and path == link_sources[0]:
            raise OSError("temp locked")
        actual_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(config_module.os, "link", capture_link)
    monkeypatch.setattr(Path, "unlink", fail_owned_temp)
    controller.start(config)
    retained = link_sources[0]
    cleanup_error = f"config publication cleanup failed while removing {retained}: temp locked"

    fake_process.emit_stdout("not-json\n")

    assert failures == []
    assert controller.is_active is True
    assert fake_process.kill_count == 1

    fake_process.finish(1, QProcess.ExitStatus.CrashExit)

    assert failures == [f"{MALFORMED_PROGRESS_ERROR}\n{cleanup_error}"]
    assert terminated == [None]
    assert controller.is_active is False


def test_cancel_replacement_is_preserved_and_controller_can_be_reused(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    finished: list[tuple[Path, dict[str, object]]] = []
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    first = make_config(tmp_path, output="first")
    controller.start(first)
    controller.cancel()
    cancel_path = tmp_path / ".first.cancel"
    replacement = tmp_path / "replacement.cancel"
    replacement.write_bytes(b"replacement")
    os.replace(replacement, cancel_path)
    terminal = ProgressEvent("complete", 3, 4, 5.0, None)

    fake_process.emit_stdout(event_json(terminal) + "\n")
    fake_process.finish()

    assert finished == [(first.output_dir, terminal.to_dict())]
    assert failures == []
    assert terminated == [None]
    assert cancel_path.read_bytes() == b"replacement"
    assert controller.is_active is False

    controller.start(make_config(tmp_path, output="second"))

    assert fake_process.start_count == 2


def test_cancel_cleanup_quarantines_before_inspection_and_preserves_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    finished: list[tuple[Path, dict[str, object]]] = []
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    config = make_config(tmp_path, output="interleaved")
    controller.start(config)
    controller.cancel()
    cancel_path = tmp_path / ".interleaved.cancel"
    actual_replace = os.replace
    quarantines: list[Path] = []

    def replace_then_interleave(source: Path | str, destination: Path | str) -> None:
        actual_replace(source, destination)
        if Path(source) == cancel_path:
            quarantines.append(Path(destination))
            cancel_path.write_bytes(b"replacement")

    monkeypatch.setattr(run_controller_module.os, "replace", replace_then_interleave)
    terminal = ProgressEvent("complete", 3, 4, 5.0, None)

    fake_process.emit_stdout(event_json(terminal) + "\n")
    fake_process.finish()

    assert finished == [(config.output_dir, terminal.to_dict())]
    assert failures == []
    assert terminated == [None]
    assert controller.is_active is False
    assert cancel_path.read_bytes() == b"replacement"
    assert len(quarantines) == 1
    assert not quarantines[0].exists()


def test_owned_cancel_quarantine_unlink_failure_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    finished: list[tuple[Path, dict[str, object]]] = []
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    first = make_config(tmp_path, output="quarantine-retry")
    second = make_config(tmp_path, output="after-quarantine-retry")
    sentinel = tmp_path / "keep.cancel"
    sentinel.write_bytes(b"keep")
    controller.start(first)
    controller.cancel()
    cancel_path = tmp_path / ".quarantine-retry.cancel"

    with monkeypatch.context() as context:
        quarantines = install_owned_quarantine_unlink_failure(context, cancel_path)
        terminal = ProgressEvent("complete", 3, 4, 5.0, None)
        fake_process.emit_stdout(event_json(terminal) + "\n")
        fake_process.finish()

        assert len(quarantines) == 1
        quarantine = quarantines[0]
        cleanup_error = (
            f"cancellation cleanup failed while removing {quarantine}: locked"
        )
        assert finished == []
        assert failures == [cleanup_error]
        assert terminated == [None]
        assert controller.is_active is False
        assert quarantine.read_bytes() == b""
        assert not cancel_path.exists()
        assert sentinel.read_bytes() == b"keep"

        with pytest.raises(
            RuntimeError,
            match="previous cancellation cleanup is unresolved",
        ):
            controller.start(second)

        assert len(quarantines) == 1
        assert fake_process.start_count == 1

    controller.start(second)

    assert not quarantine.exists()
    assert sentinel.read_bytes() == b"keep"
    assert fake_process.start_count == 2


def test_failed_cancel_quarantine_retains_owned_private_placeholder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    first = make_config(tmp_path, output="quarantine-placeholder")
    second = make_config(tmp_path, output="after-quarantine-placeholder")
    controller.start(first)
    controller.cancel()
    cancel_path = tmp_path / ".quarantine-placeholder.cancel"
    actual_replace = os.replace
    actual_unlink = Path.unlink
    quarantines: list[Path] = []

    def fail_quarantine_rename(source: Path | str, destination: Path | str) -> None:
        if Path(source) == cancel_path:
            quarantines.append(Path(destination))
            raise OSError("rename locked")
        actual_replace(source, destination)

    def fail_private_placeholder(path: Path, *, missing_ok: bool = False) -> None:
        if path in quarantines:
            raise OSError("temp locked")
        actual_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(run_controller_module.os, "replace", fail_quarantine_rename)
        context.setattr(Path, "unlink", fail_private_placeholder)
        terminal = ProgressEvent("complete", 3, 4, 5.0, None)
        fake_process.emit_stdout(event_json(terminal) + "\n")
        fake_process.finish()

        assert len(quarantines) == 1
        quarantine = quarantines[0]
        assert failures == [
            (
                "cancellation cleanup failed while quarantining "
                f"{cancel_path}: rename locked\n"
                "cancellation cleanup failed while removing private quarantine "
                f"{quarantine}: temp locked"
            )
        ]
        assert terminated == [None]
        assert cancel_path.read_bytes() == b""
        assert quarantine.read_bytes() == b""

        with pytest.raises(
            RuntimeError,
            match="previous cancellation cleanup is unresolved",
        ):
            controller.start(second)

        assert quarantines == [quarantine]
        assert fake_process.start_count == 1

    controller.start(second)

    assert not quarantine.exists()
    assert not cancel_path.exists()
    assert fake_process.start_count == 2


def test_foreign_cancel_quarantine_restore_collision_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    finished: list[tuple[Path, dict[str, object]]] = []
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    first = make_config(tmp_path, output="foreign-collision")
    second = make_config(tmp_path, output="after-foreign-collision")
    controller.start(first)
    controller.cancel()
    cancel_path = tmp_path / ".foreign-collision.cancel"
    replacement = tmp_path / "foreign.cancel"
    replacement.write_bytes(b"foreign")
    os.replace(replacement, cancel_path)
    actual_replace = os.replace
    quarantines: list[Path] = []

    def quarantine_then_collide(source: Path | str, destination: Path | str) -> None:
        actual_replace(source, destination)
        if Path(source) == cancel_path:
            quarantines.append(Path(destination))
            cancel_path.write_bytes(b"collision")

    with monkeypatch.context() as context:
        context.setattr(run_controller_module.os, "replace", quarantine_then_collide)
        terminal = ProgressEvent("complete", 3, 4, 5.0, None)
        fake_process.emit_stdout(event_json(terminal) + "\n")
        fake_process.finish()

        assert len(quarantines) == 1
        quarantine = quarantines[0]
        assert finished == []
        assert len(failures) == 1
        assert failures[0].startswith(
            f"cancellation cleanup failed while restoring {quarantine} to {cancel_path}:"
        )
        assert terminated == [None]
        assert controller.is_active is False
        assert quarantine.read_bytes() == b"foreign"
        assert cancel_path.read_bytes() == b"collision"

        with pytest.raises(
            RuntimeError,
            match="previous cancellation cleanup is unresolved",
        ):
            controller.start(second)

        assert quarantine.read_bytes() == b"foreign"
        assert cancel_path.read_bytes() == b"collision"
        assert fake_process.start_count == 1

    cancel_path.unlink()
    controller.start(second)

    assert cancel_path.read_bytes() == b"foreign"
    assert not quarantine.exists()
    assert fake_process.start_count == 2


def test_cleanup_failure_replaces_success_and_blocks_reuse_until_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    finished: list[tuple[Path, dict[str, object]]] = []
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    first = make_config(tmp_path, output="first")
    controller.start(first)
    controller.cancel()
    cancel_path = tmp_path / ".first.cancel"

    with monkeypatch.context() as context:
        quarantines = install_owned_quarantine_unlink_failure(context, cancel_path)
        terminal = ProgressEvent("complete", 3, 4, 5.0, None)
        fake_process.emit_stdout(event_json(terminal) + "\n")
        fake_process.finish()

        quarantine = quarantines[0]
        cleanup_error = f"cancellation cleanup failed while removing {quarantine}: locked"
        assert finished == []
        assert failures == [cleanup_error]
        assert terminated == [None]
        assert controller.is_active is False
        second = make_config(tmp_path, output="second")
        with pytest.raises(
            RuntimeError,
            match="previous cancellation cleanup is unresolved",
        ):
            controller.start(second)
        assert fake_process.start_count == 1
        assert not (tmp_path / "second.experiment.json").exists()

    controller.start(second)

    assert fake_process.start_count == 2
    assert not quarantine.exists()
    assert not cancel_path.exists()


def test_failed_to_start_cleanup_failure_emits_once_and_ignores_late_finished(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    finished: list[tuple[Path, dict[str, object]]] = []
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    config = make_config(tmp_path)
    controller.start(config)
    controller.cancel()
    cancel_path = tmp_path / ".run.cancel"
    quarantines = install_owned_quarantine_unlink_failure(monkeypatch, cancel_path)

    fake_process.emit_error(QProcess.ProcessError.FailedToStart)

    quarantine = quarantines[0]
    cleanup_error = f"cancellation cleanup failed while removing {quarantine}: locked"
    assert finished == []
    assert failures == [f"worker failed to start\n{cleanup_error}"]
    assert terminated == [None]
    assert controller.is_active is False

    fake_process.finish(1, QProcess.ExitStatus.CrashExit)

    assert failures == [f"worker failed to start\n{cleanup_error}"]
    assert terminated == [None]


def test_worker_failure_keeps_error_and_appends_cleanup_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    controller.start(make_config(tmp_path))
    controller.cancel()
    cancel_path = tmp_path / ".run.cancel"
    quarantines = install_owned_quarantine_unlink_failure(monkeypatch, cancel_path)
    fake_process.emit_stdout(
        event_json(ProgressEvent("failed", 1, 1, 2.0, "model failed")) + "\n"
    )

    fake_process.finish()

    quarantine = quarantines[0]
    cleanup_error = f"cancellation cleanup failed while removing {quarantine}: locked"
    assert failures == [f"model failed\n{cleanup_error}"]
    assert terminated == [None]


def test_protocol_failure_keeps_error_and_appends_cleanup_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    controller.start(make_config(tmp_path))
    controller.cancel()
    cancel_path = tmp_path / ".run.cancel"
    quarantines = install_owned_quarantine_unlink_failure(monkeypatch, cancel_path)

    fake_process.emit_stdout("not-json\n")

    assert controller.is_active is True
    assert failures == []

    fake_process.finish(1, QProcess.ExitStatus.CrashExit)

    quarantine = quarantines[0]
    cleanup_error = f"cancellation cleanup failed while removing {quarantine}: locked"
    assert failures == [f"{MALFORMED_PROGRESS_ERROR}\n{cleanup_error}"]
    assert terminated == [None]


def test_failed_to_start_then_finished_emits_one_failure_and_one_termination(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    finished: list[tuple[Path, dict[str, object]]] = []
    terminated: list[None] = []
    controller.runFailed.connect(failures.append)
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.processTerminated.connect(lambda: terminated.append(None))
    controller.start(make_config(tmp_path))

    fake_process.emit_error(QProcess.ProcessError.FailedToStart)
    fake_process.finish(1, QProcess.ExitStatus.CrashExit)

    assert failures == ["worker failed to start"]
    assert finished == []
    assert terminated == [None]
    assert controller.is_active is False


def test_crash_error_waits_for_finished_before_one_terminal_outcome(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    controller.start(make_config(tmp_path))

    fake_process.emit_error(QProcess.ProcessError.Crashed)

    assert controller.is_active is True
    assert failures == []
    assert terminated == []

    fake_process.finish(7, QProcess.ExitStatus.CrashExit)

    assert failures == ["worker exited abnormally with code 7"]
    assert terminated == [None]
    assert controller.is_active is False


def test_invalid_utf8_fails_once_but_remains_active_until_finished(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    failures: list[str] = []
    terminated: list[None] = []
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    controller.start(make_config(tmp_path))

    fake_process.emit_stdout(b"\xff\n")

    assert failures == [MALFORMED_PROGRESS_ERROR]
    assert terminated == []
    assert controller.is_active is True
    assert fake_process.kill_count == 1

    fake_process.finish(1, QProcess.ExitStatus.CrashExit)

    assert failures == [MALFORMED_PROGRESS_ERROR]
    assert terminated == [None]
    assert controller.is_active is False


def test_split_multibyte_utf8_sequence_is_decoded_incrementally(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    progress: list[ProgressEvent] = []
    failures: list[str] = []
    finished: list[tuple[Path, dict[str, object]]] = []
    terminated: list[None] = []
    controller.progressChanged.connect(progress.append)
    controller.runFailed.connect(failures.append)
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.processTerminated.connect(lambda: terminated.append(None))
    config = make_config(tmp_path)
    controller.start(config)
    event = ProgressEvent("running", 1, 1, 2.0, "café")
    encoded = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
    split_at = encoded.index(b"\xc3") + 1

    fake_process.emit_stdout(encoded[:split_at])

    assert progress == []
    assert failures == []

    fake_process.emit_stdout(encoded[split_at:] + b"\n")
    terminal = ProgressEvent("complete", 2, 2, 3.0, None)
    fake_process.emit_stdout(event_json(terminal) + "\n")
    fake_process.finish()

    assert progress == [event, terminal]
    assert failures == []
    assert finished == [(config.output_dir, terminal.to_dict())]
    assert terminated == [None]


def test_final_unterminated_jsonl_record_is_malformed_progress(
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    progress: list[ProgressEvent] = []
    failures: list[str] = []
    terminated: list[None] = []
    controller.progressChanged.connect(progress.append)
    controller.runFailed.connect(failures.append)
    controller.processTerminated.connect(lambda: terminated.append(None))
    controller.start(make_config(tmp_path))
    fake_process.emit_stdout(event_json(ProgressEvent("running", 1, 1, 2.0, None)))

    fake_process.finish()

    assert progress == []
    assert failures == [MALFORMED_PROGRESS_ERROR]
    assert terminated == [None]
    assert controller.is_active is False


@pytest.mark.parametrize("post_phase", ["complete", "running"])
def test_duplicate_or_post_terminal_record_is_one_protocol_failure(
    post_phase: ProgressPhase,
    tmp_path: Path,
    fake_process: FakeProcess,
) -> None:
    controller = RunController(process=fake_process)
    progress: list[ProgressEvent] = []
    failures: list[str] = []
    finished: list[tuple[Path, dict[str, object]]] = []
    terminated: list[None] = []
    controller.progressChanged.connect(progress.append)
    controller.runFailed.connect(failures.append)
    controller.runFinished.connect(lambda path, payload: finished.append((path, payload)))
    controller.processTerminated.connect(lambda: terminated.append(None))
    controller.start(make_config(tmp_path))
    terminal = ProgressEvent("complete", 2, 2, 3.0, None)
    post = ProgressEvent(post_phase, 2, 2, 3.1, None)

    fake_process.emit_stdout(f"{event_json(terminal)}\n{event_json(post)}\n")

    assert progress == [terminal]
    assert failures == [MALFORMED_PROGRESS_ERROR]
    assert finished == []
    assert terminated == []
    assert controller.is_active is True

    fake_process.finish()

    assert failures == [MALFORMED_PROGRESS_ERROR]
    assert finished == []
    assert terminated == [None]
