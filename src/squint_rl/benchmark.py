from __future__ import annotations

import json
import math
import platform
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from pathlib import Path
from typing import Any, TypeAlias, cast

from .artifacts import (
    AtomicRun,
    CurveRow,
    write_curve_csv,
    write_json,
    write_mot_ground_truth,
    write_mot_tracks,
)
from .budget import BudgetConfig
from .config import BenchmarkConfig, ConfigurationError, PolicySpec, load_factory
from .env import RUN_DETECTOR, SKIP, SquintEnv
from .episode import Episode, EpisodeValidationError
from .metrics import CurvePoint, MetricReport, common_support_areas, run_trackeval
from .policies import Policy, reset_policy
from .tracker import Observation, PolicyContext, TrackBatch, Tracker

_ACTION_FORMAT = b"squint.action.v1"
_METRIC_FORMAT = b"squint.metric-input.v1"
_RESERVED_POLICY_IDENTIFIERS = frozenset({"all-frame", "first-frame-only", "anchors"})
_WINDOWS_RESERVED_COMPONENTS = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    output_dir: Path
    results: Mapping[str, object]
    action_sha256: str
    metric_input_sha256: str


@dataclass(frozen=True, slots=True)
class _Rollout:
    tracks: tuple[TrackBatch, ...]
    detector_calls: int
    charged_ms: float
    reward_total: float
    action_count: int
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class _Candidate:
    identifier: str
    rate: float | None
    rollouts: Mapping[str, _Rollout]
    report: MetricReport


class _FirstFrameOnly:
    def __init__(self) -> None:
        self._first = True

    def reset(self, *, seed: int) -> None:
        del seed
        self._first = True

    def __call__(self, _observation: Observation) -> int:
        action = RUN_DETECTOR if self._first else SKIP
        self._first = False
        return action


def evaluate(
    config: BenchmarkConfig | str | Path,
    *,
    policy_factory: Callable[..., Policy] | None = None,
) -> BenchmarkResult:
    """Replay configured tracking schedules and publish deterministic artifacts."""
    if not isinstance(config, BenchmarkConfig):
        config = BenchmarkConfig.load(config)
    episodes = tuple(Episode.open(path) for path in config.episodes)
    _validate_episodes(episodes)
    policy_specs = _effective_policy_specs(config, policy_factory)
    _validate_benchmark_configuration(config, policy_specs)

    with AtomicRun(config.output_dir) as work:
        started_at = datetime.now(timezone.utc)
        result = _run_all(
            config,
            episodes,
            work,
            policy_specs=policy_specs,
            policy_factory=policy_factory,
        )
        _write_complete_artifacts(
            config,
            episodes,
            result,
            work,
            policy_specs=policy_specs,
            started_at=started_at,
        )

    return BenchmarkResult(
        output_dir=config.output_dir,
        results=cast(Mapping[str, object], result["results"]),
        action_sha256=cast(str, result["action_sha256"]),
        metric_input_sha256=cast(str, result["metric_input_sha256"]),
    )


def _effective_policy_specs(
    config: BenchmarkConfig, policy_factory: Callable[..., Policy] | None
) -> tuple[PolicySpec, ...]:
    if policy_factory is None:
        return config.policies
    if not callable(policy_factory):
        raise ConfigurationError("policy_factory must be callable")
    return (PolicySpec("external", "<callable>", {}),)


def _validate_benchmark_configuration(
    config: BenchmarkConfig, policy_specs: Sequence[PolicySpec]
) -> None:
    _validate_factory_path(config.tracker.factory, "tracker.factory")
    if not policy_specs:
        raise ConfigurationError("policies must be nonempty")
    identifiers: set[str] = set()
    for index, policy in enumerate(policy_specs):
        field = f"policies[{index}]"
        _validate_policy_identifier(policy.identifier, field)
        identifier_key = policy.identifier.casefold()
        if identifier_key in identifiers:
            raise ConfigurationError(f"{field}.id must be unique")
        identifiers.add(identifier_key)
        if policy.factory != "<callable>":
            _validate_factory_path(policy.factory, f"{field}.factory")


def _validate_factory_path(path: str, field: str) -> None:
    try:
        load_factory(path)
    except ConfigurationError as exc:
        raise ConfigurationError(f"{field} is invalid: {exc}") from exc


def _validate_policy_identifier(identifier: object, field: str) -> None:
    if (
        not _portable_component(identifier)
        or cast(str, identifier).casefold() in _RESERVED_POLICY_IDENTIFIERS
    ):
        raise ConfigurationError(
            f"{field}.id must be a nonempty artifact-safe identifier"
        )


def _portable_component(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    if value[0] in ". " or value[-1] in ". ":
        return False
    if not all(
        character.isascii() and (character.isalnum() or character in "._-")
        for character in value
    ):
        return False
    return value.split(".", 1)[0].casefold() not in _WINDOWS_RESERVED_COMPONENTS


def _validate_episodes(episodes: Sequence[Episode]) -> None:
    if not episodes:
        raise EpisodeValidationError("benchmark requires at least one episode")
    identifiers: set[str] = set()
    detector_profile: str | None = None
    hardware_profile: str | None = None
    cost_unit: str | None = None
    for episode in episodes:
        manifest = episode.manifest
        episode_object = _manifest_mapping(manifest, "episode")
        identifier = episode_object.get("id")
        if not _portable_component(identifier):
            raise EpisodeValidationError(
                "manifest episode.id must be an artifact-safe portable component"
            )
        identifier_key = cast(str, identifier).casefold()
        if identifier_key in identifiers:
            raise EpisodeValidationError(
                "manifest episode.id must be unique across the benchmark"
            )
        identifiers.add(identifier_key)
        detector = _manifest_mapping(manifest, "detector")
        profile = json.dumps(_jsonable(detector), sort_keys=True, separators=(",", ":"))
        if detector_profile is None:
            detector_profile = profile
        elif detector_profile != profile:
            raise EpisodeValidationError(
                "episodes must share the same detector profile"
            )
        hardware = json.dumps(
            _jsonable(_manifest_mapping(manifest, "hardware")),
            sort_keys=True,
            separators=(",", ":"),
        )
        if hardware_profile is None:
            hardware_profile = hardware
        elif hardware_profile != hardware:
            raise EpisodeValidationError(
                "episodes must share the same hardware profile"
            )
        unit = _manifest_mapping(manifest, "cost_profile").get("unit")
        if not isinstance(unit, str) or not unit:
            raise EpisodeValidationError(
                "manifest cost_profile.unit must be a nonempty string"
            )
        if cost_unit is None:
            cost_unit = unit
        elif cost_unit != unit:
            raise EpisodeValidationError(
                "episodes must share the same detector-cost unit"
            )
        _reserve_ms(episode)


def _manifest_mapping(
    manifest: Mapping[str, object], name: str
) -> Mapping[str, object]:
    value = manifest.get(name)
    if not isinstance(value, Mapping):
        raise EpisodeValidationError(f"manifest {name} must be an object")
    return cast(Mapping[str, object], value)


def _episode_id(episode: Episode) -> str:
    return cast(str, _manifest_mapping(episode.manifest, "episode")["id"])


def _reserve_ms(episode: Episode) -> float:
    reserve = _manifest_mapping(episode.manifest, "cost_profile").get("reserve_ms")
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)):
        raise EpisodeValidationError(
            "manifest cost_profile.reserve_ms must be a positive finite number"
        )
    value = float(reserve)
    if not math.isfinite(value) or value <= 0.0:
        raise EpisodeValidationError(
            "manifest cost_profile.reserve_ms must be a positive finite number"
        )
    return value


def _run_all(
    config: BenchmarkConfig,
    episodes: Sequence[Episode],
    work: Path,
    *,
    policy_specs: Sequence[PolicySpec],
    policy_factory: Callable[..., Policy] | None,
) -> dict[str, object]:
    ordered_episodes = tuple(sorted(episodes, key=_episode_id))
    staging = work / ".trackeval"
    gt_root = staging / "gt"
    sequence_lengths = {
        _episode_id(episode): episode.frame_count for episode in ordered_episodes
    }
    for episode in ordered_episodes:
        write_mot_ground_truth(
            gt_root / _episode_id(episode) / "gt" / "gt.txt", episode
        )

    action_digest = sha256(_ACTION_FORMAT)
    action_count = 0
    candidates: list[_Candidate] = []
    all_frame = _all_frame_candidate(
        config, ordered_episodes, work, gt_root, sequence_lengths, action_digest
    )
    action_count += sum(rollout.action_count for rollout in all_frame.rollouts.values())
    candidates.append(all_frame)
    all_frame_cost = sum(rollout.charged_ms for rollout in all_frame.rollouts.values())
    if all_frame_cost <= 0.0:
        raise RuntimeError("all-frame detector cost must be positive")

    first_frame = _first_frame_candidate(
        config, ordered_episodes, work, gt_root, sequence_lengths, action_digest
    )
    action_count += sum(
        rollout.action_count for rollout in first_frame.rollouts.values()
    )
    candidates.append(first_frame)

    constrained: dict[str, list[_Candidate]] = {}
    for policy in sorted(policy_specs, key=lambda item: item.identifier):
        policy_candidates: list[_Candidate] = []
        for rate_index, rate in enumerate(config.budget_rates):
            candidate = _constrained_candidate(
                config,
                ordered_episodes,
                work,
                gt_root,
                sequence_lengths,
                action_digest,
                policy,
                rate_index,
                rate,
                policy_factory=policy_factory,
            )
            action_count += sum(
                rollout.action_count for rollout in candidate.rollouts.values()
            )
            policy_candidates.append(candidate)
            candidates.append(candidate)
        constrained[policy.identifier] = policy_candidates

    metric_input_sha256 = _metric_input_hash(gt_root, work / "tracks")
    curves = {
        identifier: [
            CurvePoint(
                _realized_compute(candidate, all_frame_cost),
                candidate.report.combined.hota,
            )
            for candidate in values
        ]
        for identifier, values in constrained.items()
    }
    curve_areas: dict[str, JsonValue] | None
    try:
        areas = common_support_areas(curves, grid_size=101)
    except ValueError:
        curve_areas = None
    else:
        curve_areas = {
            "support": [areas.support[0], areas.support[1]],
            "values": dict(areas.values),
        }

    curve_rows = [
        CurveRow(
            policy.identifier,
            cast(float, candidate.rate),
            _realized_compute(candidate, all_frame_cost),
            candidate.report.combined.hota,
        )
        for policy in sorted(policy_specs, key=lambda item: item.identifier)
        for candidate in constrained[policy.identifier]
    ]
    write_curve_csv(work / "curve.csv", curve_rows)
    results: dict[str, JsonValue] = {
        "schema_version": 1,
        "status": "complete",
        "actions": {"count": action_count, "sha256": action_digest.hexdigest()},
        "metric_inputs": {"sha256": metric_input_sha256},
        "anchors": {
            "all_frame": _candidate_summary(all_frame, all_frame_cost),
            "first_frame_only": _candidate_summary(first_frame, all_frame_cost),
        },
        "policies": {
            identifier: {
                "points": [
                    _candidate_summary(candidate, all_frame_cost)
                    for candidate in constrained[identifier]
                ]
            }
            for identifier in sorted(constrained)
        },
        "curve_areas": curve_areas,
    }
    replay_frames = sum(
        len(rollout.tracks)
        for candidate in candidates
        for rollout in candidate.rollouts.values()
    )
    replay_seconds = sum(
        rollout.wall_seconds
        for candidate in candidates
        for rollout in candidate.rollouts.values()
    )
    shutil.rmtree(staging)
    return {
        "results": results,
        "action_sha256": action_digest.hexdigest(),
        "metric_input_sha256": metric_input_sha256,
        "replay": {
            "source_frames": replay_frames,
            "wall_seconds": replay_seconds,
            "throughput_fps": replay_frames / replay_seconds
            if replay_seconds > 0.0
            else None,
        },
    }


def _all_frame_candidate(
    config: BenchmarkConfig,
    episodes: Sequence[Episode],
    work: Path,
    gt_root: Path,
    sequence_lengths: Mapping[str, int],
    digest: Any,
) -> _Candidate:
    rollouts: dict[str, _Rollout] = {}
    for episode in episodes:
        tracker = _new_tracker(config, episode)
        tracker.reset()
        tracks: list[TrackBatch] = []
        started = time.perf_counter()
        for frame_index in range(episode.frame_count):
            frame = episode.frame(frame_index)
            tracks.append(tracker.step(frame.detections, frame.timestamp_s))
        wall_seconds = time.perf_counter() - started
        rollout = _Rollout(
            tracks=tuple(tracks),
            detector_calls=episode.frame_count,
            charged_ms=sum(
                float(episode.frame(index).detector_latency_ms)
                for index in range(episode.frame_count)
            ),
            reward_total=0.0,
            action_count=episode.frame_count,
            wall_seconds=wall_seconds,
        )
        identifier = _episode_id(episode)
        rollouts[identifier] = rollout
        _record_actions(
            digest,
            policy_id="all-frame",
            rate_index=-1,
            episode=episode,
            records=tuple((RUN_DETECTOR, RUN_DETECTOR, False) for _ in tracks),
        )
        write_mot_tracks(
            work / "tracks" / "anchors" / "all-frame" / f"{identifier}.txt", tracks
        )
    report = _evaluate_candidate(
        work, gt_root, sequence_lengths, "all-frame", None, rollouts
    )
    return _Candidate("all-frame", None, rollouts, report)


def _first_frame_candidate(
    config: BenchmarkConfig,
    episodes: Sequence[Episode],
    work: Path,
    gt_root: Path,
    sequence_lengths: Mapping[str, int],
    digest: Any,
) -> _Candidate:
    rollouts: dict[str, _Rollout] = {}
    for episode in episodes:
        rate = config.budget_rates[0]
        seed = _seed(config.seed, "first-frame-only", 0, episode.content_sha256)
        tracker = _new_tracker(config, episode)
        policy = cast(Policy, _FirstFrameOnly())
        rollout, records = _roll_env(episode, tracker, policy, rate, config, seed)
        identifier = _episode_id(episode)
        rollouts[identifier] = rollout
        _record_actions(digest, "first-frame-only", -2, episode, records)
        write_mot_tracks(
            work / "tracks" / "anchors" / "first-frame-only" / f"{identifier}.txt",
            rollout.tracks,
        )
    report = _evaluate_candidate(
        work, gt_root, sequence_lengths, "first-frame-only", None, rollouts
    )
    return _Candidate("first-frame-only", None, rollouts, report)


def _constrained_candidate(
    config: BenchmarkConfig,
    episodes: Sequence[Episode],
    work: Path,
    gt_root: Path,
    sequence_lengths: Mapping[str, int],
    digest: Any,
    policy_spec: PolicySpec,
    rate_index: int,
    rate: float,
    *,
    policy_factory: Callable[..., Policy] | None,
) -> _Candidate:
    rollouts: dict[str, _Rollout] = {}
    factory = policy_factory or cast(
        Callable[..., Policy], load_factory(policy_spec.factory)
    )
    for episode in episodes:
        seed = _seed(
            config.seed, policy_spec.identifier, rate_index, episode.content_sha256
        )
        tracker = _new_tracker(config, episode)
        policy = factory(
            context=PolicyContext(
                nominal_rate=rate,
                source_fps=episode.fps,
                reserve_ms=_reserve_ms(episode),
                seed=seed,
                time_since_detector_scale_s=config.observation_scales.time_since_detector_s,
            ),
            **dict(policy_spec.parameters),
        )
        rollout, records = _roll_env(episode, tracker, policy, rate, config, seed)
        identifier = _episode_id(episode)
        rollouts[identifier] = rollout
        _record_actions(digest, policy_spec.identifier, rate_index, episode, records)
        write_mot_tracks(
            work
            / "tracks"
            / policy_spec.identifier
            / _rate_directory(rate)
            / f"{identifier}.txt",
            rollout.tracks,
        )
    report = _evaluate_candidate(
        work, gt_root, sequence_lengths, policy_spec.identifier, rate, rollouts
    )
    return _Candidate(policy_spec.identifier, rate, rollouts, report)


def _new_tracker(config: BenchmarkConfig, episode: Episode) -> Tracker:
    factory = cast(Callable[..., Tracker], load_factory(config.tracker.factory))
    return factory(episode=episode, **dict(config.tracker.parameters))


def _roll_env(
    episode: Episode,
    tracker: Tracker,
    policy: Policy,
    rate: float,
    config: BenchmarkConfig,
    seed: int,
) -> tuple[_Rollout, tuple[tuple[int, int, bool], ...]]:
    env = SquintEnv(
        episode=episode,
        tracker=tracker,
        budget=BudgetConfig.for_rate(
            reserve_ms=_reserve_ms(episode), source_fps=episode.fps, nominal_rate=rate
        ),
        observation_scales=config.observation_scales,
    )
    reset_policy(policy, seed=seed)
    observation, _ = env.reset(seed=seed)
    records: list[tuple[int, int, bool]] = []
    reward_total = 0.0
    charged_ms = 0.0
    detector_calls = 0
    started = time.perf_counter()
    while True:
        requested = policy(observation)
        observation, reward, terminated, truncated, info = env.step(requested)
        if truncated:
            raise RuntimeError("SquintEnv must not truncate replay episodes")
        reward_total += float(reward)
        charged_ms += float(info["charged_ms"])
        detector_calls += int(info["applied_action"] == RUN_DETECTOR)
        records.append(
            (
                int(info["requested_action"]),
                int(info["applied_action"]),
                bool(info["denied"]),
            )
        )
        if terminated:
            break
    wall_seconds = time.perf_counter() - started
    return (
        _Rollout(
            tracks=tuple(env.track_history),
            detector_calls=detector_calls,
            charged_ms=charged_ms,
            reward_total=reward_total,
            action_count=len(records),
            wall_seconds=wall_seconds,
        ),
        tuple(records),
    )


def _evaluate_candidate(
    work: Path,
    gt_root: Path,
    sequence_lengths: Mapping[str, int],
    identifier: str,
    rate: float | None,
    rollouts: Mapping[str, _Rollout],
) -> MetricReport:
    trackers_root = work / ".trackeval" / "trackers"
    data = trackers_root / "squint" / "data"
    if data.exists():
        shutil.rmtree(trackers_root)
    for episode_id in sorted(rollouts):
        source = _track_path(work, identifier, rate, episode_id)
        destination = data / f"{episode_id}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        return run_trackeval(
            gt_root=gt_root,
            trackers_root=trackers_root,
            output_root=work / ".trackeval" / "output",
            sequence_lengths=sequence_lengths,
        )


def _track_path(
    work: Path, identifier: str, rate: float | None, episode_id: str
) -> Path:
    if rate is None:
        return work / "tracks" / "anchors" / identifier / f"{episode_id}.txt"
    return work / "tracks" / identifier / _rate_directory(rate) / f"{episode_id}.txt"


def _rate_directory(rate: float) -> str:
    return f"rho-{rate.hex()}"


def _record_actions(
    digest: Any,
    policy_id: str,
    rate_index: int,
    episode: Episode,
    records: Sequence[tuple[int, int, bool]],
) -> None:
    _hash_fields(digest, ("unit", policy_id, str(rate_index), episode.content_sha256))
    for requested, applied, denied in records:
        digest.update(bytes((requested, applied, int(denied))))


def _seed(config_seed: int, policy_id: str, rate_index: int, content_hash: str) -> int:
    digest = sha256(b"squint.seed.v1")
    _hash_fields(digest, (str(config_seed), policy_id, str(rate_index), content_hash))
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def _hash_fields(digest: Any, fields: Sequence[str]) -> None:
    for field in fields:
        payload = field.encode("utf-8")
        digest.update(len(payload).to_bytes(4, "big"))
        digest.update(payload)


def _metric_input_hash(gt_root: Path, tracks_root: Path) -> str:
    digest = sha256(_METRIC_FORMAT)
    files = [
        *(
            ("gt", item.relative_to(gt_root).as_posix(), item)
            for item in gt_root.rglob("*")
            if item.is_file()
        ),
        *(
            ("tracks", item.relative_to(tracks_root).as_posix(), item)
            for item in tracks_root.rglob("*")
            if item.is_file()
        ),
    ]
    for category, relative, item in sorted(
        files, key=lambda value: (value[0], value[1])
    ):
        _hash_fields(digest, (category, relative))
        payload = item.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _realized_compute(candidate: _Candidate, all_frame_cost: float) -> float:
    return (
        sum(rollout.charged_ms for rollout in candidate.rollouts.values())
        / all_frame_cost
    )


def _candidate_summary(
    candidate: _Candidate, all_frame_cost: float
) -> dict[str, JsonValue]:
    total_frames = sum(len(rollout.tracks) for rollout in candidate.rollouts.values())
    report = candidate.report
    return {
        "nominal_rate": candidate.rate,
        "realized_compute": _realized_compute(candidate, all_frame_cost),
        "detector_calls": sum(
            rollout.detector_calls for rollout in candidate.rollouts.values()
        ),
        "charged_detector_ms": sum(
            rollout.charged_ms for rollout in candidate.rollouts.values()
        ),
        "reward_total": sum(
            rollout.reward_total for rollout in candidate.rollouts.values()
        ),
        "metrics": _report_json(report),
        "source_frames": total_frames,
    }


def _report_json(report: MetricReport) -> dict[str, JsonValue]:
    return {
        "combined": _summary_json(report.combined),
        "per_sequence": {
            identifier: _summary_json(summary)
            for identifier, summary in sorted(report.per_sequence.items())
        },
    }


def _summary_json(summary: object) -> dict[str, JsonValue]:
    values = cast(Any, summary)
    return {
        "hota": float(values.hota),
        "deta": float(values.deta),
        "assa": float(values.assa),
        "idf1": float(values.idf1),
        "false_positives": int(values.false_positives),
        "false_negatives": int(values.false_negatives),
        "identity_switches": int(values.identity_switches),
    }


def _write_complete_artifacts(
    config: BenchmarkConfig,
    episodes: Sequence[Episode],
    result: Mapping[str, object],
    work: Path,
    *,
    policy_specs: Sequence[PolicySpec],
    started_at: datetime,
) -> None:
    ended_at = datetime.now(timezone.utc)
    config_path = work / "config.json"
    write_json(config_path, _config_json(config, policy_specs))
    write_json(work / "results.json", result["results"])
    write_json(
        work / "provenance.json",
        {
            "package_version": _optional_version("squint-rl"),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
            "episodes": [
                {"id": _episode_id(episode), "content_sha256": episode.content_sha256}
                for episode in sorted(episodes, key=_episode_id)
            ],
            "tracker_factory": config.tracker.factory,
            "policy_factories": [
                policy.factory
                for policy in sorted(policy_specs, key=lambda item: item.identifier)
            ],
            "dependencies": {
                name: _optional_version(name)
                for name in ("numpy", "scipy", "gymnasium", "trackeval")
            },
            "started_at_utc": started_at.isoformat(),
            "ended_at_utc": ended_at.isoformat(),
            "python_executable": sys.executable,
            "replay": result["replay"],
        },
    )


def _config_json(
    config: BenchmarkConfig, policy_specs: Sequence[PolicySpec]
) -> dict[str, JsonValue]:
    return {
        "schema_version": 1,
        "episodes": [str(path) for path in config.episodes],
        "output_dir": str(config.output_dir),
        "budget_rates": list(config.budget_rates),
        "seed": config.seed,
        "observation_scales": {
            "active_tracks": config.observation_scales.active_tracks,
            "age_s": config.observation_scales.age_s,
            "motion_px_s": config.observation_scales.motion_px_s,
            "time_since_detector_s": config.observation_scales.time_since_detector_s,
        },
        "tracker": {
            "factory": config.tracker.factory,
            "parameters": _jsonable(config.tracker.parameters),
        },
        "policies": [
            {
                "id": policy.identifier,
                "factory": policy.factory,
                "parameters": _jsonable(policy.parameters),
            }
            for policy in policy_specs
        ],
    }


def _jsonable(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _optional_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None
