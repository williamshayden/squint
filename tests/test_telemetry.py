from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from edge_perception.telemetry import (
    HardwareProbe,
    TelemetryMonitor,
    TelemetrySample,
    collect_host_report,
)


class FakePsutil:
    def __init__(self, rss_values: tuple[int, ...] = (100,)) -> None:
        self._rss_values: Iterator[int] = iter(rss_values)
        self._last_rss = rss_values[-1]

    def Process(self) -> FakePsutil:
        return self

    def memory_info(self) -> SimpleNamespace:
        self._last_rss = next(self._rss_values, self._last_rss)
        return SimpleNamespace(rss=self._last_rss)

    def virtual_memory(self) -> SimpleNamespace:
        return SimpleNamespace(used=600, total=1_000)

    def cpu_count(self, *, logical: bool) -> int:
        assert logical is True
        return 8


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0

    def __init__(self) -> None:
        self.initialized = False

    def nvmlInit(self) -> None:
        self.initialized = True

    def nvmlDeviceGetCount(self) -> int:
        return 1

    def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
        assert index == 0
        return "gpu-0"

    def nvmlDeviceGetName(self, handle: str) -> bytes:
        assert handle == "gpu-0"
        return b"Fake GPU"

    def nvmlSystemGetDriverVersion(self) -> bytes:
        return b"999.1"

    def nvmlDeviceGetUtilizationRates(self, handle: str) -> SimpleNamespace:
        assert handle == "gpu-0"
        return SimpleNamespace(gpu=75)

    def nvmlDeviceGetMemoryInfo(self, handle: str) -> SimpleNamespace:
        assert handle == "gpu-0"
        return SimpleNamespace(used=400)

    def nvmlDeviceGetPowerUsage(self, handle: str) -> int:
        assert handle == "gpu-0"
        return 12_500

    def nvmlDeviceGetTemperature(self, handle: str, sensor: int) -> int:
        assert handle == "gpu-0"
        assert sensor == self.NVML_TEMPERATURE_GPU
        return 55


def test_hardware_probe_returns_one_common_timestamped_record_shape() -> None:
    clock_values = iter((10_000_000_001, 10_000_000_002))
    with_gpu = HardwareProbe(
        psutil_module=FakePsutil(),
        nvml_module=FakeNvml(),
        clock=lambda: next(clock_values),
    ).sample()
    without_gpu = HardwareProbe(
        psutil_module=FakePsutil(),
        nvml_module=None,
        clock=lambda: next(clock_values),
    ).sample()

    assert with_gpu.timestamp_ns == 10_000_000_001
    assert without_gpu.timestamp_ns == 10_000_000_002
    assert with_gpu.to_dict().keys() == without_gpu.to_dict().keys()
    assert with_gpu.process_rss_bytes == 100
    assert with_gpu.system_memory_used_bytes == 600
    assert with_gpu.gpu_utilization_percent == 75.0
    assert with_gpu.gpu_memory_used_bytes == 400
    assert with_gpu.gpu_power_watts == 12.5
    assert with_gpu.gpu_temperature_c == 55.0
    assert without_gpu.gpu_utilization_percent is None
    assert without_gpu.gpu_memory_used_bytes is None
    assert without_gpu.gpu_power_watts is None
    assert without_gpu.gpu_temperature_c is None


def test_collect_host_report_includes_identity_and_one_capability_message() -> None:
    available = HardwareProbe(psutil_module=FakePsutil(), nvml_module=FakeNvml())
    unavailable = HardwareProbe(psutil_module=FakePsutil(), nvml_module=None)

    available_report = collect_host_report(available)
    unavailable_report = collect_host_report(unavailable)

    assert isinstance(available_report["os"], str)
    assert isinstance(available_report["python_version"], str)
    assert available_report["logical_cpu_count"] == 8
    assert available_report["total_ram_bytes"] == 1_000
    assert available_report["gpu"] == {"name": "Fake GPU", "driver_version": "999.1"}
    assert available_report["capability_messages"] == []
    assert unavailable_report["gpu"] is None
    assert len(unavailable_report["capability_messages"]) == 1
    assert "NVIDIA telemetry unavailable" in unavailable_report["capability_messages"][0]


class SequenceProbe:
    def __init__(self) -> None:
        self._samples: Iterator[TelemetrySample | Exception] = iter(
            (
                TelemetrySample(1, 100, 500, None, None, None, None),
                RuntimeError("sensor read failed"),
                TelemetrySample(3, 250, 700, 90.0, 800, 20.0, 60.0),
            )
        )
        self.third_sample_seen = threading.Event()

    def sample(self, *, timestamp_ns: int | None = None) -> TelemetrySample:
        result = next(self._samples, TelemetrySample(3, 250, 700, 90.0, 800, 20.0, 60.0))
        if isinstance(result, Exception):
            raise result
        if result.timestamp_ns == 3:
            self.third_sample_seen.set()
        if timestamp_ns is None:
            return result
        return TelemetrySample(
            timestamp_ns,
            result.process_rss_bytes,
            result.system_memory_used_bytes,
            result.gpu_utilization_percent,
            result.gpu_memory_used_bytes,
            result.gpu_power_watts,
            result.gpu_temperature_c,
        )


@contextmanager
def _running_monitor(monitor: TelemetryMonitor) -> Iterator[TelemetryMonitor]:
    with monitor:
        yield monitor


def test_monitor_continues_after_probe_error_joins_and_calculates_peaks() -> None:
    probe = SequenceProbe()
    timestamp = 0

    def clock() -> int:
        nonlocal timestamp
        timestamp += 1
        return timestamp

    monitor = TelemetryMonitor(probe=probe, interval_seconds=0.001, clock=clock)
    with _running_monitor(monitor):
        assert monitor.is_running
        assert probe.third_sample_seen.wait(timeout=1.0)

    assert not monitor.is_running
    assert isinstance(monitor.samples, tuple)
    assert [sample.timestamp_ns for sample in monitor.samples[:3]] == [1, 2, 3]
    assert monitor.samples[1].to_dict() == {
        "timestamp_ns": 2,
        "process_rss_bytes": None,
        "system_memory_used_bytes": None,
        "gpu_utilization_percent": None,
        "gpu_memory_used_bytes": None,
        "gpu_power_watts": None,
        "gpu_temperature_c": None,
    }
    assert monitor.peaks() == {
        "process_rss_bytes": 250,
        "system_memory_used_bytes": 700,
        "gpu_utilization_percent": 90.0,
        "gpu_memory_used_bytes": 800,
        "gpu_power_watts": 20.0,
        "gpu_temperature_c": 60.0,
    }


def test_telemetry_samples_are_immutable() -> None:
    sample = TelemetrySample(1, 2, 3, None, None, None, None)

    with pytest.raises(AttributeError):
        sample.process_rss_bytes = 4  # type: ignore[misc]
