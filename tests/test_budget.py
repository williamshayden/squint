from __future__ import annotations

import importlib
import math
from types import ModuleType

import pytest
from hypothesis import given
from hypothesis import strategies as st


def _budget_module() -> ModuleType | None:
    try:
        return importlib.import_module("squint_rl.budget")
    except ModuleNotFoundError:
        return None


def _budget_types() -> tuple[type[object], type[object]]:
    module = _budget_module()
    assert module is not None, "Task 4 budget module must exist"
    return module.BudgetConfig, module.TokenBucket


def test_budget_module_is_available() -> None:
    assert _budget_module() is not None, "Task 4 budget module must exist"


def test_bucket_refills_in_source_time_and_charges_actual_cost() -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(
        budget_config(reserve_ms=10.0, capacity_ms=20.0, refill_ms_per_s=5.0)
    )
    bucket.reset(timestamp_s=0.0)
    assert bucket.balance_ms == 10.0
    bucket.refill(timestamp_s=1.0)
    assert bucket.balance_ms == 15.0
    assert bucket.affordable
    bucket.charge(12.0)
    assert bucket.balance_ms == 3.0
    assert not bucket.affordable


def test_refill_caps_balance_at_capacity() -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    bucket.reset(timestamp_s=0.0)
    bucket.refill(timestamp_s=100.0)
    assert bucket.balance_ms == 20.0
    assert bucket.normalized_balance == 1.0


def test_sequential_refills_credit_only_each_source_time_delta() -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 1.0))
    bucket.reset(timestamp_s=0.0)
    bucket.refill(timestamp_s=1.0)
    assert bucket.balance_ms == 11.0
    bucket.refill(timestamp_s=3.0)
    assert bucket.balance_ms == 13.0


def test_p95_overrun_creates_debt_and_blocks_calls() -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    bucket.reset(timestamp_s=0.0)
    bucket.charge(14.0)
    assert bucket.balance_ms == -4.0
    bucket.refill(timestamp_s=2.0)
    assert bucket.balance_ms == 6.0
    assert not bucket.affordable


def test_affordability_includes_exact_reserve_boundary() -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    bucket.reset(timestamp_s=0.0)
    bucket.charge(0.0)
    assert bucket.affordable
    bucket.charge(0.000001)
    assert not bucket.affordable


def test_charge_subtracts_exact_fractional_cost() -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    bucket.reset(timestamp_s=0.0)
    bucket.charge(2.375)
    assert bucket.balance_ms == 7.625


def test_reset_restores_reserve_and_restarts_source_clock() -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    bucket.reset(timestamp_s=5.0)
    bucket.charge(8.0)
    bucket.refill(timestamp_s=6.0)
    bucket.reset(timestamp_s=100.0)
    assert bucket.balance_ms == 10.0
    bucket.refill(timestamp_s=101.0)
    assert bucket.balance_ms == 15.0


def test_refill_requires_reset_first() -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    with pytest.raises(RuntimeError, match="reset"):
        bucket.refill(timestamp_s=1.0)


def test_refill_rejects_time_travel_without_mutating_balance() -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    bucket.reset(timestamp_s=2.0)
    with pytest.raises(ValueError, match="monotonic"):
        bucket.refill(timestamp_s=1.0)
    assert bucket.balance_ms == 10.0
    bucket.refill(timestamp_s=2.0)
    assert bucket.balance_ms == 10.0


@pytest.mark.parametrize(
    ("reserve_ms", "capacity_ms", "refill_ms_per_s"),
    [
        (0.0, 0.0, 1.0),
        (-1.0, 2.0, 1.0),
        (math.nan, 20.0, 1.0),
        (10.0, math.inf, 1.0),
        (10.0, 21.0, 1.0),
        (10.0, 20.0, 0.0),
        (10.0, 20.0, math.nan),
    ],
)
def test_config_rejects_invalid_values(
    reserve_ms: float, capacity_ms: float, refill_ms_per_s: float
) -> None:
    budget_config, _token_bucket = _budget_types()
    with pytest.raises(ValueError):
        budget_config(reserve_ms, capacity_ms, refill_ms_per_s)


def test_for_rate_computes_detector_millisecond_refill_rate() -> None:
    budget_config, _token_bucket = _budget_types()
    config = budget_config.for_rate(
        reserve_ms=12.5, source_fps=30.0, nominal_rate=0.25
    )
    assert config.reserve_ms == 12.5
    assert config.capacity_ms == 25.0
    assert config.refill_ms_per_s == 93.75


@pytest.mark.parametrize("nominal_rate", [0.0, -0.1, 1.1, math.nan, math.inf])
def test_for_rate_rejects_invalid_nominal_rate(nominal_rate: float) -> None:
    budget_config, _token_bucket = _budget_types()
    with pytest.raises(ValueError, match="nominal_rate"):
        budget_config.for_rate(
            reserve_ms=10.0, source_fps=30.0, nominal_rate=nominal_rate
        )


@pytest.mark.parametrize("source_fps", [0.0, -30.0, math.nan, math.inf])
def test_for_rate_rejects_invalid_source_fps(source_fps: float) -> None:
    budget_config, _token_bucket = _budget_types()
    with pytest.raises(ValueError):
        budget_config.for_rate(
            reserve_ms=10.0, source_fps=source_fps, nominal_rate=0.25
        )


@pytest.mark.parametrize("actual_ms", [-0.001, math.nan, math.inf])
def test_charge_rejects_invalid_cost(actual_ms: float) -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    bucket.reset(timestamp_s=0.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        bucket.charge(actual_ms)
    assert bucket.balance_ms == 10.0


@pytest.mark.parametrize("timestamp_s", [math.nan, math.inf, -math.inf])
def test_reset_rejects_nonfinite_timestamp(timestamp_s: float) -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    with pytest.raises(ValueError, match="finite"):
        bucket.reset(timestamp_s=timestamp_s)
    assert bucket.balance_ms == 10.0


@pytest.mark.parametrize("timestamp_s", [math.nan, math.inf, -math.inf])
def test_refill_rejects_nonfinite_timestamp(timestamp_s: float) -> None:
    budget_config, token_bucket = _budget_types()
    bucket = token_bucket(budget_config(10.0, 20.0, 5.0))
    bucket.reset(timestamp_s=0.0)
    with pytest.raises(ValueError, match="finite"):
        bucket.refill(timestamp_s=timestamp_s)
    assert bucket.balance_ms == 10.0
    bucket.refill(timestamp_s=0.0)
    assert math.isfinite(bucket.balance_ms)


@st.composite
def _accounting_values(draw: st.DrawFn) -> tuple[float, float, float, float]:
    reserve_ms = draw(st.floats(min_value=0.01, max_value=1_000_000.0))
    refill_ms_per_s = draw(st.floats(min_value=0.01, max_value=1_000_000.0))
    elapsed_s = draw(st.floats(min_value=0.0, max_value=1_000.0))
    actual_ms = draw(st.floats(min_value=0.0, max_value=1_000_000.0))
    return reserve_ms, refill_ms_per_s, elapsed_s, actual_ms


@given(_accounting_values())
def test_hypothesis_conserves_refill_and_charge_accounting(
    values: tuple[float, float, float, float],
) -> None:
    reserve_ms, refill_ms_per_s, elapsed_s, actual_ms = values
    budget_config, token_bucket = _budget_types()
    config = budget_config(reserve_ms, 2.0 * reserve_ms, refill_ms_per_s)
    bucket = token_bucket(config)
    bucket.reset(timestamp_s=0.0)

    old_balance = bucket.balance_ms
    bucket.refill(timestamp_s=elapsed_s)
    expected_balance = min(
        config.capacity_ms, old_balance + refill_ms_per_s * elapsed_s
    )
    assert bucket.balance_ms == expected_balance

    pre_charge_balance = bucket.balance_ms
    bucket.charge(actual_ms)
    assert bucket.balance_ms == pre_charge_balance - actual_ms
