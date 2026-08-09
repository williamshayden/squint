import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    reserve_ms: float
    capacity_ms: float
    refill_ms_per_s: float

    def __post_init__(self) -> None:
        values = (self.reserve_ms, self.capacity_ms, self.refill_ms_per_s)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("budget values must be finite and positive")
        if self.capacity_ms != 2.0 * self.reserve_ms:
            raise ValueError("capacity_ms must equal 2 * reserve_ms")

    @classmethod
    def for_rate(
        cls, *, reserve_ms: float, source_fps: float, nominal_rate: float
    ) -> "BudgetConfig":
        if not math.isfinite(nominal_rate) or not 0.0 < nominal_rate <= 1.0:
            raise ValueError("nominal_rate must be in (0, 1]")
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError("source_fps must be finite and positive")
        return cls(
            reserve_ms=reserve_ms,
            capacity_ms=2.0 * reserve_ms,
            refill_ms_per_s=nominal_rate * source_fps * reserve_ms,
        )


class TokenBucket:
    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self.balance_ms = config.reserve_ms
        self._timestamp_s: float | None = None

    @property
    def affordable(self) -> bool:
        return self.balance_ms >= self.config.reserve_ms

    @property
    def normalized_balance(self) -> float:
        return self.balance_ms / self.config.capacity_ms

    def reset(self, *, timestamp_s: float) -> None:
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        self.balance_ms = self.config.reserve_ms
        self._timestamp_s = timestamp_s

    def refill(self, *, timestamp_s: float) -> None:
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._timestamp_s is None:
            raise RuntimeError("token bucket must be reset before refill")
        elapsed = timestamp_s - self._timestamp_s
        if elapsed < 0.0:
            raise ValueError("timestamps must be monotonic")
        self.balance_ms = min(
            self.config.capacity_ms,
            self.balance_ms + elapsed * self.config.refill_ms_per_s,
        )
        self._timestamp_s = timestamp_s

    def charge(self, actual_ms: float) -> None:
        if not math.isfinite(actual_ms) or actual_ms < 0.0:
            raise ValueError("actual_ms must be finite and nonnegative")
        balance_ms = self.balance_ms - actual_ms
        if not math.isfinite(balance_ms) or not math.isfinite(
            balance_ms / self.config.capacity_ms
        ):
            raise ValueError("charge would make balance non-finite")
        self.balance_ms = balance_ms
