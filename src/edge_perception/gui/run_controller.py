"""One-process native GUI adapter for the canonical JSONL worker."""

from __future__ import annotations

import codecs
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PySide6.QtCore import QObject, QProcess, Signal

from edge_perception.config import RunConfig, publish_run_config
from edge_perception.progress import ProgressEvent, ProgressPhase
from edge_perception.runner import validate_output_directory

MALFORMED_PROGRESS_ERROR = "worker emitted malformed progress"
MISSING_TERMINAL_PROGRESS_ERROR = "worker exited without terminal progress"
_STDERR_LIMIT_BYTES = 16 * 1024
_TERMINAL_PHASES = {"complete", "cancelled", "failed"}


@dataclass(frozen=True, slots=True)
class _OwnedCancellation:
    path: Path
    identity: tuple[int, int]


class RunController(QObject):
    """Own and observe exactly one reusable ``QProcess`` worker."""

    progressChanged = Signal(object)
    runFinished = Signal(Path, dict)
    runFailed = Signal(str)
    processTerminated = Signal()

    def __init__(self, process: QProcess | None = None) -> None:
        super().__init__()
        self._process = QProcess(self) if process is None else process
        self._active = False
        self._config_path: Path | None = None
        self._cancel_path: Path | None = None
        self._owned_cancel: _OwnedCancellation | None = None
        self._last_config: RunConfig | None = None
        self._stdout_decoder: codecs.IncrementalDecoder
        self._stdout_buffer = ""
        self._stderr_tail = bytearray()
        self._terminal_event: ProgressEvent | None = None
        self._protocol_failed = False
        self._outcome_emitted = False
        self._process_terminated_emitted = False
        self._reset_stream_state()
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.finished.connect(self._process_finished)
        self._process.errorOccurred.connect(self._process_error)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def config_path(self) -> Path | None:
        return self._config_path

    @property
    def last_config(self) -> RunConfig | None:
        return self._last_config

    def start(self, config: RunConfig) -> None:
        """Persist ``config`` and start the canonical worker without a shell."""

        if self._active:
            raise RuntimeError("a run is already active")
        if not isinstance(config, RunConfig):
            raise TypeError("config must be a RunConfig")
        self._retry_retained_cancel_cleanup()
        validate_output_directory(config.output_dir)
        config_path = _experiment_config_path(config.output_dir)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        publish_run_config(config_path, config)
        self._config_path = config_path
        self._last_config = config
        cancel_path = _cancel_file_path(config.output_dir)
        self._cancel_path = cancel_path
        cancel_path.unlink(missing_ok=True)

        self._reset_stream_state()
        self._active = True
        try:
            self._process.setProgram(sys.executable)
            self._process.setArguments(
                [
                    "-m",
                    "edge_perception.worker",
                    "--config",
                    str(config_path),
                    "--cancel-file",
                    str(cancel_path),
                ]
            )
            self._process.start()
        except BaseException:
            self._active = False
            raise

    def cancel(self) -> None:
        """Atomically publish this active run's private cancellation file."""

        if not self._active or self._cancel_path is None or self._outcome_emitted:
            return
        owned = self._owned_cancel
        if owned is not None:
            try:
                current = os.stat(owned.path, follow_symlinks=False)
            except FileNotFoundError:
                self._owned_cancel = None
            except OSError:
                raise
            else:
                if _file_identity(current) == owned.identity:
                    return
                self._owned_cancel = None
                return
        cancel_path = self._cancel_path
        descriptor, temporary_name = tempfile.mkstemp(
            dir=cancel_path.parent,
            prefix=f"{cancel_path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            os.fsync(descriptor)
            identity = _file_identity(os.fstat(descriptor))
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, cancel_path)
            self._owned_cancel = _OwnedCancellation(cancel_path, identity)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def kill(self) -> None:
        """Force the active worker to stop; completion still waits for ``finished``."""

        if self._active:
            self._process.kill()

    def _reset_stream_state(self) -> None:
        self._stdout_decoder = codecs.getincrementaldecoder("utf-8")()
        self._stdout_buffer = ""
        self._stderr_tail.clear()
        self._terminal_event = None
        self._protocol_failed = False
        self._outcome_emitted = False
        self._process_terminated_emitted = False

    def _read_stdout(self) -> None:
        if not self._active:
            return
        raw = self._process.readAllStandardOutput().data()
        if not raw:
            return
        try:
            decoded = self._stdout_decoder.decode(raw)
        except UnicodeDecodeError:
            self._malformed_progress()
            return
        self._consume_stdout(decoded)

    def _consume_stdout(self, decoded: str) -> None:
        self._stdout_buffer += decoded
        while "\n" in self._stdout_buffer:
            record, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            record = record.removesuffix("\r")
            self._parse_progress(record)
            if self._protocol_failed:
                return

    def _parse_progress(self, record: str) -> None:
        try:
            payload = json.loads(record, parse_constant=_reject_json_constant)
            if not isinstance(payload, dict):
                raise TypeError("progress record must be an object")
            values = cast(dict[str, object], payload)
            if set(values) != {
                "phase",
                "frames_processed",
                "inference_count",
                "elapsed_ms",
                "error",
            }:
                raise ValueError("progress record fields do not match the contract")
            event = ProgressEvent(
                phase=cast(ProgressPhase, values["phase"]),
                frames_processed=cast(int, values["frames_processed"]),
                inference_count=cast(int, values["inference_count"]),
                elapsed_ms=cast(float, values["elapsed_ms"]),
                error=cast(str | None, values["error"]),
            )
            if self._terminal_event is not None:
                raise ValueError("progress emitted after a terminal event")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._malformed_progress()
            return
        self.progressChanged.emit(event)
        if event.phase in _TERMINAL_PHASES:
            self._terminal_event = event

    def _malformed_progress(self, *, kill_process: bool = True) -> None:
        if self._protocol_failed:
            return
        self._protocol_failed = True
        if self._owned_cancel is None:
            self._emit_failure_once(MALFORMED_PROGRESS_ERROR)
        if kill_process and self._active:
            self._process.kill()

    def _read_stderr(self) -> None:
        if not self._active:
            return
        self._stderr_tail.extend(self._process.readAllStandardError().data())
        if len(self._stderr_tail) > _STDERR_LIMIT_BYTES:
            del self._stderr_tail[:-_STDERR_LIMIT_BYTES]

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if not self._active:
            return
        self._read_stdout()
        self._read_stderr()
        try:
            final_text = self._stdout_decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            self._malformed_progress(kill_process=False)
        else:
            self._consume_stdout(final_text)
            if self._stdout_buffer and not self._protocol_failed:
                self._malformed_progress(kill_process=False)

        self._active = False
        terminal = self._terminal_event
        failure: str | None = None
        success: tuple[Path, dict[str, object]] | None = None
        if self._protocol_failed:
            failure = MALFORMED_PROGRESS_ERROR
        elif terminal is not None and terminal.phase == "failed":
            failure = self._failure_with_stderr(terminal.error or "worker reported failure")
        elif exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            failure = self._failure_with_stderr(
                f"worker exited abnormally with code {exit_code}"
            )
        elif terminal is None:
            failure = self._failure_with_stderr(MISSING_TERMINAL_PROGRESS_ERROR)
        else:
            config = self._last_config
            if config is None:
                failure = "worker configuration is unavailable"
            else:
                success = (config.output_dir, terminal.to_dict())
        self._finalize_stopped(failure=failure, success=success)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if not self._active or error != QProcess.ProcessError.FailedToStart:
            return
        self._read_stderr()
        self._active = False
        self._finalize_stopped(
            failure=self._failure_with_stderr("worker failed to start"),
            success=None,
        )

    def _failure_with_stderr(self, message: str) -> str:
        if not self._stderr_tail:
            return message
        return f"{message}\n{self._stderr_tail.decode('utf-8', errors='replace')}"

    def _emit_failure_once(self, message: str) -> None:
        if self._outcome_emitted:
            return
        self._outcome_emitted = True
        self.runFailed.emit(message)

    def _emit_success_once(self, success: tuple[Path, dict[str, object]]) -> None:
        if self._outcome_emitted:
            return
        self._outcome_emitted = True
        self.runFinished.emit(*success)

    def _finalize_stopped(
        self,
        *,
        failure: str | None,
        success: tuple[Path, dict[str, object]] | None,
    ) -> None:
        try:
            try:
                cleanup_failure = self._cleanup_owned_cancel()
            except Exception as error:  # noqa: BLE001 - terminal signals must survive cleanup
                cleanup_failure = f"cancellation cleanup failed unexpectedly: {error}"
            if cleanup_failure is not None:
                failure = (
                    cleanup_failure if failure is None else f"{failure}\n{cleanup_failure}"
                )
                success = None
            if failure is not None:
                self._emit_failure_once(failure)
            elif success is not None:
                self._emit_success_once(success)
        finally:
            self._emit_process_terminated_once()

    def _emit_process_terminated_once(self) -> None:
        if self._process_terminated_emitted:
            return
        self._process_terminated_emitted = True
        self.processTerminated.emit()

    def _cleanup_owned_cancel(self) -> str | None:
        owned = self._owned_cancel
        if owned is None:
            return None
        try:
            current = os.stat(owned.path, follow_symlinks=False)
        except FileNotFoundError:
            self._owned_cancel = None
            return None
        except OSError as error:
            return f"cancellation cleanup failed while inspecting {owned.path}: {error}"
        if _file_identity(current) != owned.identity:
            self._owned_cancel = None
            return None
        try:
            owned.path.unlink()
        except FileNotFoundError:
            self._owned_cancel = None
            return None
        except OSError as error:
            return f"cancellation cleanup failed while removing {owned.path}: {error}"
        self._owned_cancel = None
        return None

    def _retry_retained_cancel_cleanup(self) -> None:
        cleanup_failure = self._cleanup_owned_cancel()
        if cleanup_failure is not None:
            raise RuntimeError(
                f"previous cancellation cleanup is unresolved: {cleanup_failure}"
            )


def _experiment_config_path(output_dir: Path) -> Path:
    output = Path(output_dir).resolve()
    return output.parent / f"{output.name}.experiment.json"


def _cancel_file_path(output_dir: Path) -> Path:
    output = Path(output_dir).resolve()
    return output.parent / f".{output.name}.cancel"


def _file_identity(stat_result: os.stat_result) -> tuple[int, int]:
    return stat_result.st_dev, stat_result.st_ino


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")
