"""Partial, non-fatal host and NVIDIA hardware telemetry."""

from __future__ import annotations

import platform
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from types import TracebackType
from typing import Any, Final, Self

import psutil  # type: ignore[import-untyped]

_NVML_AUTO: Final = object()
_SAMPLE_FIELDS: Final = (
    "process_rss_bytes",
    "system_memory_used_bytes",
    "gpu_utilization_percent",
    "gpu_memory_used_bytes",
    "gpu_power_watts",
    "gpu_temperature_c",
)


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """One immutable timestamped snapshot of available hardware sensors."""

    timestamp_ns: int
    process_rss_bytes: int | None
    system_memory_used_bytes: int | None
    gpu_utilization_percent: float | None
    gpu_memory_used_bytes: int | None
    gpu_power_watts: float | None
    gpu_temperature_c: float | None

    @classmethod
    def unavailable(cls, timestamp_ns: int) -> TelemetrySample:
        """Create a timestamped record when a probe read fails."""

        return cls(timestamp_ns, None, None, None, None, None, None)

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "process_rss_bytes": self.process_rss_bytes,
            "system_memory_used_bytes": self.system_memory_used_bytes,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "gpu_memory_used_bytes": self.gpu_memory_used_bytes,
            "gpu_power_watts": self.gpu_power_watts,
            "gpu_temperature_c": self.gpu_temperature_c,
        }


class HardwareProbe:
    """Read psutil and, when available, the first NVIDIA device through NVML."""

    def __init__(
        self,
        *,
        psutil_module: Any = psutil,
        nvml_module: Any = _NVML_AUTO,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self._psutil = psutil_module
        self._clock = clock
        self._process = self._read(lambda: self._psutil.Process())
        self._nvml: Any | None = None
        self._nvml_handle: Any | None = None
        self._nvml_owner: Any | None = None
        self._gpu_identity: dict[str, str] | None = None
        self._capability_messages: list[str] = []
        self._initialize_nvml(nvml_module)

    @staticmethod
    def _read(operation: Callable[[], Any]) -> Any | None:
        try:
            return operation()
        except Exception:  # noqa: BLE001 - sensors are optional capability boundaries
            return None

    @staticmethod
    def _text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _initialize_nvml(self, nvml_module: Any) -> None:
        if nvml_module is _NVML_AUTO:
            try:
                nvml_module = import_module("pynvml")
            except Exception as error:  # noqa: BLE001 - NVML must remain optional
                self._mark_nvml_unavailable(error)
                return
        elif nvml_module is None:
            self._mark_nvml_unavailable(None)
            return

        try:
            nvml_module.nvmlInit()
            self._nvml_owner = nvml_module
            if int(nvml_module.nvmlDeviceGetCount()) < 1:
                raise RuntimeError("no NVIDIA devices detected")
            handle = nvml_module.nvmlDeviceGetHandleByIndex(0)
        except Exception as error:  # noqa: BLE001 - NVML failures must not abort a run
            self._mark_nvml_unavailable(error)
            self.close()
            return

        self._nvml = nvml_module
        self._nvml_handle = handle
        name = self._read(lambda: nvml_module.nvmlDeviceGetName(handle))
        driver = self._read(nvml_module.nvmlSystemGetDriverVersion)
        if name is not None:
            self._gpu_identity = {"name": self._text(name)}
            if driver is not None:
                self._gpu_identity["driver_version"] = self._text(driver)

    def _mark_nvml_unavailable(self, error: Exception | None) -> None:
        detail = "not installed or disabled" if error is None else f"{type(error).__name__}: {error}"
        self._capability_messages.append(f"NVIDIA telemetry unavailable: {detail}")

    def close(self) -> None:
        """Release an NVML initialization owned by this probe exactly once."""

        nvml_owner = self._nvml_owner
        if nvml_owner is None:
            return
        self._nvml_owner = None
        self._nvml = None
        self._nvml_handle = None
        try:
            nvml_owner.nvmlShutdown()
        except Exception:  # noqa: BLE001 - cleanup must never abort a run
            return

    @property
    def capability_messages(self) -> tuple[str, ...]:
        return tuple(self._capability_messages)

    @property
    def gpu_identity(self) -> dict[str, str] | None:
        return None if self._gpu_identity is None else dict(self._gpu_identity)

    def logical_cpu_count(self) -> int | None:
        value = self._read(lambda: self._psutil.cpu_count(logical=True))
        return None if value is None else int(value)

    def total_ram_bytes(self) -> int | None:
        value = self._read(lambda: self._psutil.virtual_memory().total)
        return None if value is None else int(value)

    def sample(self, *, timestamp_ns: int | None = None) -> TelemetrySample:
        """Return one common-shape record, leaving failed sensors unavailable."""

        timestamp = self._clock() if timestamp_ns is None else timestamp_ns
        rss = None
        process = self._process
        if process is not None:
            rss = self._read(lambda: process.memory_info().rss)
        system_used = self._read(lambda: self._psutil.virtual_memory().used)

        nvml = self._nvml
        handle = self._nvml_handle
        if nvml is None or handle is None:
            utilization = memory_used = power_milliwatts = temperature = None
        else:
            utilization = self._read(lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            memory_used = self._read(lambda: nvml.nvmlDeviceGetMemoryInfo(handle).used)
            power_milliwatts = self._read(lambda: nvml.nvmlDeviceGetPowerUsage(handle))
            temperature = self._read(
                lambda: nvml.nvmlDeviceGetTemperature(
                    handle,
                    nvml.NVML_TEMPERATURE_GPU,
                )
            )

        return TelemetrySample(
            timestamp_ns=int(timestamp),
            process_rss_bytes=None if rss is None else int(rss),
            system_memory_used_bytes=None if system_used is None else int(system_used),
            gpu_utilization_percent=None if utilization is None else float(utilization),
            gpu_memory_used_bytes=None if memory_used is None else int(memory_used),
            gpu_power_watts=None if power_milliwatts is None else float(power_milliwatts) / 1_000.0,
            gpu_temperature_c=None if temperature is None else float(temperature),
        )


def collect_host_report(probe: HardwareProbe | None = None) -> dict[str, object]:
    """Collect JSON-native host identity and telemetry capability information."""

    if probe is None:
        hardware = HardwareProbe()
        owns_probe = True
    else:
        hardware = probe
        owns_probe = False
    try:
        return {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "logical_cpu_count": hardware.logical_cpu_count(),
            "total_ram_bytes": hardware.total_ram_bytes(),
            "gpu": hardware.gpu_identity,
            "capability_messages": list(hardware.capability_messages),
        }
    finally:
        if owns_probe:
            hardware.close()


class TelemetryMonitor:
    """Sample a hardware probe periodically for the lifetime of a context."""

    def __init__(
        self,
        *,
        probe: HardwareProbe | Any | None = None,
        interval_seconds: float = 0.05,
        clock: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive")
        self._owns_probe = probe is None
        self._probe: Any = HardwareProbe() if probe is None else probe
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._samples: list[TelemetrySample] = []
        self._samples_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def samples(self) -> tuple[TelemetrySample, ...]:
        with self._samples_lock:
            return tuple(self._samples)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while True:
            timestamp_ns = self._clock()
            try:
                sample = self._probe.sample(timestamp_ns=timestamp_ns)
            except Exception:  # noqa: BLE001 - record unavailable sensors and keep sampling
                sample = TelemetrySample.unavailable(timestamp_ns)
            with self._samples_lock:
                self._samples.append(sample)
            if self._stop_event.wait(self._interval_seconds):
                return

    def __enter__(self) -> Self:
        if self.is_running:
            raise RuntimeError("telemetry monitor is already running")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="hardware-telemetry",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop sampling and release only a probe owned by this monitor."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._owns_probe:
            self._probe.close()

    def peaks(self) -> dict[str, int | float | None]:
        """Return maxima for every sampled sensor, ignoring unavailable values."""

        samples = self.samples
        peaks: dict[str, int | float | None] = {}
        for field_name in _SAMPLE_FIELDS:
            values = [getattr(sample, field_name) for sample in samples]
            available = [value for value in values if value is not None]
            peaks[field_name] = max(available) if available else None
        return peaks
