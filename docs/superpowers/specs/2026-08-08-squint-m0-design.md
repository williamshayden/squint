# Squint M0 Design Specification

**Status:** Approved for implementation
**Date:** 2026-08-08  
**Owner:** William Hayden  
**Product:** Squint  
**Distribution:** squint-rl  
**Python package:** squint_rl  
**CLI:** squint

## 1. The project in one sentence

Squint is a detector-neutral Gymnasium benchmark for learning and comparing policies that decide when to refresh a tracking-by-detection pipeline under a streaming compute budget.

The honest M0 claim is:

> Squint provides deterministic replay, hidden ground truth, a causal Gymnasium environment, a refillable detector-compute budget, replaceable trackers and policies, and standard multi-object-tracking evaluation.

M0 does not claim a novel scheduling algorithm, hardware transfer, detector transfer, production readiness, or live-camera performance.

### Ten-year-old explanation

A tracker guesses where objects moved. Looking again with the detector costs battery. Squint lets a small policy learn when paying to look again is worth it, then grades the result without showing the answers during the test.

## 2. Research objective

### M0 question

Can a budget-conditioned policy learn to preserve more multi-object-tracking quality than simple causal scheduling rules when every strategy receives the same detector-compute allowance?

### North-star question

Can one replay-trained, budget-conditioned scheduler transfer to unseen scenes, detectors, trackers, hardware profiles, and eventually live streams with little or no recalibration?

The M0 system is the instrument needed to investigate the north-star question. It is not itself evidence that the north-star claim is true.

## 3. M0 scope

M0 contains:

- One immutable replay episode format.
- One detector-neutral tracker protocol.
- One standards-compliant Gymnasium environment.
- One binary action: skip or run the detector.
- One token-bucket compute budget.
- One canonical dense training reward.
- One frozen-policy evaluator using HOTA and IDF1.
- Four causal baselines and one unconstrained quality anchor.
- One external reference RL training recipe.
- Python and CLI benchmark workflows.
- Plain JSON, CSV, and MOTChallenge-format outputs.

M0 explicitly excludes:

- Camera capture and live inference.
- A GUI or browser application.
- Packaged detector weights.
- A detector inside the environment.
- An RL training framework inside Squint.
- Databases, model stores, job queues, plugin registries, and report servers.
- Multiple detector, tracker, dataset, or hardware claims.
- Segmentation, embeddings, temporal detectors, visual trackers, and raw-pixel policies.

## 4. System boundaries

~~~mermaid
flowchart LR
    E["Episode: replay facts"] --> G["SquintEnv: budget, reward, Gym API"]
    P["External policy"] -->|"SKIP or RUN"| G
    G -->|"detections or no measurement"| T["Tracker adapter"]
    T -->|"tracks and causal summary"| G
    G -->|"observation and dense reward"| P
    G --> O["Run artifacts"]
    O --> V["Evaluator: HOTA, IDF1, compute curve"]
~~~

The four core public concepts are:

~~~python
Episode
Tracker
SquintEnv
evaluate()
~~~

### Episode

Episode owns immutable replay facts. It contains no policy output and no tracker state.

### Tracker

Tracker is the only stateful perception component in M0. Squint owns the protocol, not the tracking algorithm.

~~~python
class Tracker(Protocol):
    def reset(self) -> None: ...

    def step(
        self,
        detections: DetectionBatch | None,
        timestamp_s: float,
    ) -> TrackBatch: ...

    def summary(self) -> TrackerSummary: ...
~~~

None means the detector was skipped. An empty DetectionBatch means the detector ran and returned no detections. An adapter may handle both as a prediction-only tracker update, but the environment preserves the distinction for cost accounting and diagnostics.

### SquintEnv

SquintEnv owns causal step ordering, observation construction, token accounting, hidden-ground-truth reward, and Gymnasium termination semantics. It never loads or runs a detector.

### Evaluator

The evaluator receives frozen policy outputs and hidden ground truth after an episode. It writes standard tracking records and invokes the canonical MOT evaluator. Training reward is never presented as a benchmark metric.

## 5. Episode contract

Each episode is a directory:

~~~text
episode/
├── manifest.json
└── arrays.npz
~~~

### Manifest

The manifest records:

- Schema name and version.
- Episode and source sequence identifiers.
- Source hash, dimensions, frame count, frame rate, and duration.
- Dataset, split, class mapping, and ignore-region rules.
- Detector family, model identifier, immutable revision, weights hash, threshold, input size, precision, and backend.
- Hardware identity, accelerator backend, driver/runtime versions, and timing protocol.
- Detector-cost unit and profile statistics.
- Scene-feature definition.
- Aggregate latency, GPU utilization, and VRAM telemetry.
- Hashes for every data artifact other than the manifest itself, plus a canonical
  episode content hash derived from normalized manifest fields and array hashes.

GPU utilization and VRAM are provenance and diagnostics in M0. They are not added to the token budget.

### Arrays

arrays.npz stores:

- Per-frame timestamps.
- Per-frame measured detector latency.
- A fixed cheap scene-change feature.
- Cached detector boxes, scores, and classes for every frame.
- Ground-truth boxes, track identities, classes, visibility, validity, and ignore flags.

Variable-length boxes use flat arrays plus frame-offset arrays. Coordinates are source-frame pixel xyxy coordinates.

The episode does not contain:

- Raw video.
- Tracker state.
- Policy actions.
- Policy observations.
- Benchmark results.

The loader rejects unsupported versions, inconsistent frame counts, non-monotonic timestamps, non-finite coordinates, invalid offsets, negative costs, and hash mismatches before execution.

## 6. Causal policy contract

### Action space

~~~python
spaces.Discrete(2)

0 = SKIP
1 = RUN_DETECTOR
~~~

### Observation space

The observation is a named spaces.Dict, not an opaque flat vector:

~~~text
scene_change
tracker_state
compute_budget
~~~

M0 scene_change is a normalized 3 by 3 grid of mean absolute grayscale difference between the current and previous source frames. It is computed during trace generation and costs no detector inference.

M0 tracker_state contains only portable lifecycle statistics derivable by every adapter:

- Normalized active-track count.
- Confirmed-track fraction.
- Stale-track fraction.
- Normalized mean track age.
- Normalized mean box-motion magnitude.
- Mean last-measurement confidence.

M0 does not require a tracker-native uncertainty scalar. Native uncertainty has inconsistent meaning across tracker families and would undermine transfer.

All clipping and normalization constants for tracker and scene features are
computed from the training partition, frozen in the benchmark configuration,
and reused unchanged for validation and test. No held-out episode computes its
own observation scale.

M0 compute_budget contains:

- Token balance divided by capacity.
- Refill rate divided by the all-frame reference rate.
- Whether one profiled detector call is currently affordable.
- Normalized time since the last applied detector call.
- Previous applied action.

The policy never receives raw pixels, ground truth, cached current-frame detections, current-frame realized detector latency, future information, or episode progress.

### Step ordering

reset prepares the first causal decision state. For each step:

1. Refill tokens using elapsed source time.
2. Construct the current observation from the current cheap scene feature,
   tracker state after the previous processed frame, and current budget state.
3. Receive SKIP or RUN_DETECTOR.
4. If RUN is affordable, reveal the cached detections, update the tracker, and charge the recorded detector latency.
5. If RUN is not affordable, apply prediction-only tracking and record a denied request.
6. If SKIP is selected, apply prediction-only tracking.
7. Compare resulting tracks with hidden ground truth and compute dense reward.
8. Advance to the next causal decision state.

info reports requested action, applied action, denial, tokens, charged cost, detector-call count, and diagnostic matching counts. Information needed to choose the current action never appears in info after the fact.

## 7. Streaming compute budget

M0 uses a token bucket measured in detector milliseconds:

~~~text
balance = min(capacity, balance + refill_rate × source_time_delta)
~~~

The reference hardware profile is calibrated only from training data:

- Exclude detector warm-up calls.
- Synchronize the accelerator around timed inference.
- Set reserve_ms to the training-profile p95 detector latency.
- Set capacity_ms to 2 × reserve_ms.
- Start each episode with reserve_ms so initial acquisition is possible.

A requested detector call is admitted when balance is at least reserve_ms. After inference is revealed, the recorded actual latency is charged. A rare call above the p95 reserve may create bounded debt; no further call is admitted until refill repays it.

For nominal budget fraction rho:

~~~text
refill_rate_ms_per_source_second =
    rho × source_fps × reserve_ms
~~~

M0 evaluates rho in:

~~~text
0.10, 0.25, 0.50, 0.75, 1.00
~~~

The quality-compute plot uses realized measured detector cost on its x-axis, normalized by the cost of running the detector on every frame. Nominal rho is configuration, not the reported achieved cost.

For scalar curve comparison, sort each strategy's realized points, linearly
interpolate HOTA on a shared 101-point compute grid over the strategies' common
measured support, and integrate with the trapezoidal rule. Do not extrapolate
beyond a strategy's measured range.

## 8. Dense training reward

The environment matches predicted tracks to valid ground-truth boxes with Hungarian assignment at IoU 0.5 after applying the dataset's ignore rules.

For frame t:

~~~text
error_t =
    false_negatives_t
  + false_positives_t
  + identity_switches_t
  + sum(1 - IoU) over matched pairs

reward_t =
    clip(
        1 - error_t / max(1, valid_ground_truth_count_t),
        -1,
        1,
    )
~~~

This is a shaping reward, not a published tracking metric. It gives immediate localization, detection, and identity feedback while the token bucket independently enforces compute.

Researchers may replace it through Gymnasium RewardWrapper without changing Squint's environment contract.

## 9. Reference stack

M0's reference experiment uses:

- Detector trace producer: D-FINE-N.
- Model: ustc-community/dfine-nano-coco.
- Model revision: 066438d3d8f0da137a37b38fdf3368fd4afceced.
- Trace minimum confidence: 0.10, preserving low-confidence detections for ByteTrack association.
- Tracker: ByteTrack through an optional external adapter.
- Tracker implementation: pinned release of the Apache-2.0 roboflow/trackers package.
- Data: locally downloaded MOT17 training sequences.
- Reference accelerator: William's laptop NVIDIA GPU.

The core package imports none of D-FINE, Transformers, PyTorch, ByteTrack, or the dataset tooling. Reference integrations live behind an optional extra and produce the neutral episode contract.

Squint distributes no detector weights, MOT17 frames, MOT17 annotations, or generated MOT17 traces. Users prepare the scientific data locally. The wheel includes only a programmatically generated synthetic fixture.

## 10. Installation and user workflow

The unrelated PyPI project named squint prevents use of that distribution and import namespace. Squint therefore uses:

~~~text
Product:       Squint
Distribution:  squint-rl
Import:        squint_rl
Executable:    squint
Gym ID:        SquintReplay-v0
~~~

End users install with pip:

~~~bash
pip install squint-rl
pip install "squint-rl[reference]"
~~~

uv may be used for development but is never required by an end user.

The M0 CLI surface is:

~~~bash
squint episode validate EPISODE
squint benchmark CONFIG
squint benchmark CONFIG --policy python:module.path:factory
~~~

RL training remains ordinary Gymnasium usage through an external library. Squint may ship a reference training example, but no squint train abstraction is part of M0.

Built-in baseline policies and external policies implement the same callable boundary:

~~~python
action = policy(observation)
~~~

Each successful benchmark writes atomically:

~~~text
run/
├── config.json
├── provenance.json
├── results.json
├── curve.csv
└── tracks/
~~~

Invalid input fails before execution with a nonzero CLI exit code and a field-specific message. Partial output is never presented as a completed run.

## 11. M0 experiment card

### 11.1 Hypotheses

Primary hypothesis:

> A single PPO policy conditioned on tracker, scene-change, and token-bucket state achieves a larger held-out HOTA-versus-realized-compute area than the strongest validation-selected causal heuristic on the fixed D-FINE-N plus ByteTrack stack.

Null hypothesis:

> The learned scheduler does not improve the held-out quality-compute tradeoff over the strongest causal heuristic.

M0 completion does not depend on rejecting the null hypothesis. A reproducible negative result is a valid M0 result.

### 11.2 Experimental unit

The experimental unit is:

~~~text
policy × policy seed × source sequence × token-bucket rate
~~~

Detector traces, tracker implementation and parameters, episode data, reward, and evaluator are fixed across policies.

### 11.3 Dataset split

MOT17 has seven unique annotated training scenes. M0 uses a whole-sequence split so no camera scene appears in more than one partition:

| Partition | MOT17 sequences | Use |
|---|---|---|
| Train | 02, 04, 05, 10 | PPO learning and hardware-cost calibration |
| Validation | 09 | Checkpoint selection and heuristic threshold selection |
| Test | 11, 13 | One frozen final evaluation |

Official detector variants are not treated as separate videos. Squint reads each unique image sequence once and uses D-FINE-N detections generated locally.

Training samples deterministic ten-second windows with five-second stride from training sequences. Validation and test use complete sequences from a reset tracker.

This small split is sufficient to prove the workflow, not to support a broad scene-generalization claim. Per-sequence results must be reported; M0 will not claim population-level statistical significance from two held-out scenes.

### 11.4 Trace generation

For every source frame:

1. Compute the fixed 3 by 3 scene-change grid.
2. Run pinned D-FINE-N once.
3. Record source-coordinate boxes, confidence, class, synchronized inference latency, and telemetry.
4. Import MOT17 valid pedestrian ground truth and ignore metadata.
5. Write and hash the immutable episode.

Warm-up frames are excluded from the hardware profile. Replay is deterministic
after an episode has been sealed. Trace generation can vary because of backend
numerics and measured latency, so the exact cached outputs are hashed and all
software, model, data, and hardware identities are recorded.

### 11.5 Training

The reference learner is Stable-Baselines3 PPO with MultiInputPolicy. Stable-Baselines3 is a reference-development dependency, not a Squint core dependency.

Initial preregistered configuration:

| Setting | Value |
|---|---|
| Policy network | Separate 2 by 64 Tanh actor and critic MLPs |
| Learning rate | 3e-4 |
| Rollout steps | 2048 |
| Batch size | 64 |
| Discount gamma | 0.99 |
| GAE lambda | 0.95 |
| PPO clip | 0.20 |
| Training steps | 500,000 per seed |
| Seeds | 0, 1, 2, 3, 4 |

At each training-window reset, rho is sampled uniformly from 0.10 to 1.00. This trains one budget-conditioned policy rather than one policy per budget.

Every 50,000 steps, each seed is evaluated on the validation sequence over the five fixed budget rates. The checkpoint with the largest validation HOTA-compute area is frozen. Test data is never used for checkpoint or hyperparameter selection. Frozen PPO policies use deterministic action selection during benchmark evaluation.

One bounded hyperparameter revision is allowed after the first validation run. It must be documented before test evaluation. Repeated open-ended tuning is outside M0.

### 11.6 Baselines

All causal constrained strategies receive the same observations and token bucket:

1. Greedy affordable: request detection whenever a call is affordable.
2. Periodic: request at a fixed source-time interval matched to each nominal budget.
3. Scene change: request when the cheap scene-change score crosses a validation-selected threshold.
4. Track staleness: request when track staleness crosses a validation-selected threshold.

An unconstrained all-frame detector run supplies the maximum-cost quality anchor. It is not presented as a budget-matched policy.

Threshold grids and the choice of strongest heuristic family are frozen using validation data before the final test. No baseline is tuned on test sequences.

### 11.7 Evaluation

The primary tracking metric is HOTA. Secondary metrics are:

- IDF1.
- DetA.
- AssA.
- False positives and false negatives.
- Identity switches.
- Detector calls.
- Realized detector milliseconds.
- Denied detector requests.
- End-to-end replay throughput.

GPU utilization and VRAM remain provenance diagnostics.

The primary result is the HOTA-versus-realized-compute curve. A normalized trapezoidal area over the constrained range is used as a scalar summary. The primary comparison is:

~~~text
learned policy area - strongest validation-selected heuristic area
~~~

Report mean and standard deviation across the five policy seeds, plus every held-out sequence separately. Do not use the two test scenes to make a broad significance claim.

### 11.8 Positive-result criterion

M0 records a positive learned-scheduling result only if:

- Mean learned-policy curve area exceeds the frozen strongest heuristic.
- The improvement direction is positive on both held-out sequences.
- The comparison is computed only over their common realized-compute support,
  using the interpolation rule in Section 7.

Otherwise the primary hypothesis is recorded as unsupported. The environment and benchmark may still pass M0.

### 11.9 Pre-training viability gate

Do not spend time training PPO until the validation trace satisfies:

- The all-frame detector plus tracker produces finite standard MOT metrics.
- All-frame HOTA is at least 0.25.
- All-frame HOTA exceeds first-frame-only tracking by at least 0.10.
- At least three constrained budget rates produce distinct detector-call schedules.
- Repeating a baseline run with the same seed produces byte-identical actions and metric inputs.

If the gate fails, stop and diagnose detector threshold, class mapping, tracker adapter, ignore handling, or episode construction. Do not hide a broken reference stack with RL tuning.

## 12. Ordered M0 experiments

### E0 — Synthetic contract conformance

Use the bundled generated fixture to prove schema validation, Gymnasium API compliance, causal observations, reward matching, tracker reset behavior, and exact token accounting.

Pass condition: all conformance and property tests pass on Windows and Linux without model or dataset dependencies.

### E1 — Causality and leakage audit

Create paired synthetic episodes that are identical through frame t but differ afterward. Observations and actions through frame t must remain identical. Mutating hidden ground truth must change reward/evaluation but never the policy observation.

Pass condition: no future trace, detector output, cost, ground truth, or episode-progress leakage is observable.

### E2 — Reference trace viability

Generate D-FINE-N traces for train and validation data, run the all-frame and first-frame-only anchors, and apply the pre-training viability gate.

Pass condition: the reference stack has measurable scheduling headroom.

### E3 — Baseline benchmark

Run all frozen causal baselines over validation budgets, select heuristic thresholds, then freeze all non-RL settings.

Pass condition: a complete deterministic baseline quality-compute artifact exists.

### E4 — Learnability smoke test

Train one PPO seed on a deliberately simple synthetic episode where scene changes identify the useful detector-refresh moments.

Pass condition: PPO beats greedy-affordable and periodic scheduling on synthetic reward without observing ground truth.

### E5 — Reference training

Train five PPO seeds on training sequences and select one checkpoint per seed using validation area.

Pass condition: all seeds complete under the fixed protocol and produce loadable frozen policies. Winning is not required.

### E6 — Frozen held-out test

Run each frozen policy and baseline exactly once over test sequences and all five budget rates. Produce TrackEval inputs, metric tables, curve.csv, and provenance.

Pass condition: the run is complete, reproducible, and no test-informed tuning occurred.

### E7 — Minimal interpretation

Run one preregistered observation ablation by removing scene_change and retraining seeds under the same budget. This determines whether the visual change signal contributes beyond tracker and bucket state.

Pass condition: report the delta honestly. The ablation is explanatory and does not gate package release.

## 13. M0 completion gates

### Engineering gate

- A standard wheel installs with pip.
- Core installation imports no detector, tracker, Torch, Qt, or camera runtime.
- Gymnasium's environment checker passes.
- Synthetic tests pass on Windows and Linux.
- Replay is deterministic for fixed inputs and seed.
- Invalid data fails before a run begins.
- Python and CLI produce the same benchmark result.

### Experimental gate

- Reference episodes pass validation.
- Baselines and five PPO seeds run under one frozen protocol.
- Test evaluation occurs once after freezing decisions.
- The quality-compute curve and per-sequence metrics are published.
- A negative result is labeled negative rather than reframed.

### Scope gate

M0 contains no camera, GUI, live inference, detector registry, training service, database, browser server, or second reference stack.

## 14. Failure handling

- Schema or integrity failure: reject the episode before reset.
- Tracker failure: terminate the run as failed; do not score partial output.
- Unaffordable detector request: apply skip, record denial, continue.
- Missing optional reference dependency: explain the exact install extra.
- Interrupted benchmark: leave a clearly incomplete temporary directory.
- Non-finite reward or tracker output: fail immediately with frame and field context.
- Metric-tool failure: preserve raw MOT output and mark evaluation incomplete.

## 15. Testing strategy

M0 tests are layered:

- Unit tests for records, matching, reward, token bucket, and validation.
- Property tests for ragged offsets, token conservation, and deterministic reset.
- Gymnasium conformance tests.
- Tracker protocol tests using a fake tracker.
- Reference ByteTrack adapter tests behind an optional marker.
- End-to-end synthetic CLI acceptance tests.
- Opt-in D-FINE and MOT17 integration tests.
- Wheel and source-distribution installation tests.

Scientific dataset or model availability never controls whether the core test suite passes.

## 16. Checkpoint ladder after M0

| Checkpoint | Experiment | Claim gate |
|---|---|---|
| M0 | One stack, held-out scenes, fixed hardware | Squint is a valid reproducible RL benchmark |
| M1 | Held-out token rates and burst profiles | One policy interpolates across compute budgets |
| M2 | Held-out detector traces | Detector transfer |
| M3 | Held-out tracker adapter | Tracker transfer |
| M4 | CPU and second GPU cost profiles | Hardware-profile transfer |
| M5 | Detector, tracker, scene, and hardware held out together | Full-stack transfer |
| M6 | Frozen replay-trained policy in a live stream | Replay-to-live transfer |

Each checkpoint changes one variable before the combined test. Claims advance only when their corresponding held-out experiment succeeds.

## 17. Architecture and dependency budget

The core should remain understandable in one sitting:

- Four public concepts.
- NumPy and Gymnasium as the principal runtime dependencies.
- No framework-specific policy base class.
- Optional adapters convert external types at the boundary.
- Configuration and artifacts use standard serializable data.
- Any proposal that adds a service, registry, process boundary, or UI must wait until an approved checkpoint requires it.

Existing repository code is reused only when it directly satisfies this specification. Camera, GUI, execution-worker, and detector-runner architecture is not preserved merely because it already exists.

## 18. Known limitations

- Two held-out MOT17 scenes do not establish broad scene generalization.
- M0 uses one pedestrian dataset and one perception stack.
- The dense reward is custom shaping; only standard MOT metrics support claims.
- Cached inference cannot reproduce every live runtime interaction.
- A p95 admission reserve can create one-call compute debt.
- Measured latency is tied to the recorded hardware and software stack.
- ByteTrack lifecycle summaries are not calibrated uncertainty.
- MOT17 and model terms must be rechecked before distributing any derived artifact.

## 19. References

- D-FINE: https://github.com/Peterande/D-FINE
- ByteTrack: https://github.com/FoundationVision/ByteTrack
- Detector-neutral tracker package: https://github.com/roboflow/trackers
- Gymnasium environment guide: https://gymnasium.farama.org/main/tutorials/environment_creation/
- HOTA: https://www.cvlibs.net/publications/Luiten2020IJCV.pdf
- IDF1: https://arxiv.org/abs/1609.01775
- TrackEval: https://github.com/JonathonLuiten/TrackEval
- MOTChallenge: https://motchallenge.net/
- Existing PyPI squint project: https://pypi.org/project/squint/
