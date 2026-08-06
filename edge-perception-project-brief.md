# Adaptive Edge Perception — Working Project Brief

**Date:** 2026-08-05  
**Status:** Checkpoint 1 boundary approved for implementation; later Gym, RL, reward, caching, and benchmark-design decisions are deliberately deferred

## The project in one sentence

Build an open-source toolkit that helps a small edge computer decide where to spend limited object-detection compute on high-resolution video, then proves whether the chosen strategy works through reproducible benchmarks.

## The ten-year-old explanation

A 4K camera produces a giant picture. A small computer does not have enough time to inspect every part closely. We give it a limited number of close looks and test different ways of choosing where to look.

## Real-world reference use case

A fixed high-resolution camera monitors a wide scene using limited local compute. Relevant examples include:

- people or vehicles entering a warehouse, parking lot, or restricted area;
- small animals entering a wildlife-camera scene;
- objects or defects appearing on a wide conveyor;
- events or anomalies that deserve closer inspection.

The first scientific task is known-class object detection. General anomaly detection remains a future extension because it requires different labels and scoring.

## Product identity

The project is an **adaptive edge-perception toolkit**. Gymnasium is its training and benchmarking subsystem, not the entire product.

The product should eventually provide:

1. A budget-enforcing runtime for recorded and live high-resolution video.
2. An extensible strategy interface for fixed, heuristic, learned, and third-party policies.
3. A detector-neutral adapter contract.
4. A Gymnasium environment for training selection policies.
5. Reproducible accuracy, latency, freshness, memory, energy, and thermal benchmarks.
6. A deployment path that runs a selected strategy on a live camera.

## Primary falsifiable claim

On unseen, independently human-annotated high-resolution video, and under the same real-time hardware budget, an adaptive tile-selection policy should detect more small objects than fixed resizing, fixed tile rotation, and simple motion-based selection without materially increasing new-object acquisition delay.

The project does not assume reinforcement learning will win. If a simple heuristic wins, the benchmark must report that result.

## Scientific experiment boundary

### The experimental variable

The **selection strategy** is the one component changed between contestants.

Reference contestants:

- whole-frame downscaling;
- round-robin tile selection;
- motion-ranked tile selection;
- random budget-matched selection;
- learned selection policy;
- exhaustive and label-aware oracles used only as unattainable ceilings.

### Controlled components

Within one experiment, keep these fixed:

- video and frame timing;
- detector and detector configuration;
- tracker setting, if tracking is enabled;
- hardware and power configuration;
- compute/deadline budget;
- labels, metrics, and evaluation code.

Repeat the complete experiment with additional detector families later to test whether the result depends on one detector.

## What is being learned

The selection policy learns how to allocate a limited inspection budget over time.

- **Observation:** cheap overview information, motion, previous detections, tile age, uncertainty, and remaining budget.
- **Action:** choose tiles to inspect or stop spending for the current frame.
- **Reward:** timely detection value minus computation, delay, and waste.
- **Episode:** a chronological video clip.

Required generalization test: train on some cameras and evaluate on entirely different cameras and scenes. Unseen frames from the same camera are not sufficient evidence of generalization.

## Extensibility boundaries

### Strategy contract

A strategy receives only the standardized observation and returns a standardized action. It cannot access labels, rewards, or unselected high-resolution pixels.

Strategies are named, versioned, configurable plugins rather than branches inside the runtime. Users select a strategy through a namespaced identifier and serializable configuration. Built-in baselines and third-party strategies use the same loader and runtime contract.

The strategy boundary must provide:

- registry discovery and loading by strategy ID;
- declared strategy identity, version, capabilities, and configuration schema;
- an episode reset lifecycle;
- one standardized selection method;
- optional resource cleanup;
- deterministic seeding where the algorithm uses randomness;
- automatic recording of strategy identity, version, seed, and configuration in benchmark manifests;
- third-party registration through normal package/plugin discovery without editing the core repository.

A strategy chooses inspection actions but cannot invoke the detector, read benchmark labels, bypass the budget controller, or sample hidden high-resolution pixels. The runtime remains responsible for validating actions, executing inference, and constructing the next allowed observation.

The contract must accommodate:

- fixed rules;
- motion heuristics;
- contextual bandits;
- RL policies;
- custom research algorithms;
- externally hosted strategies.

Conceptually, user selection should be as simple as choosing `builtin/round-robin`, `builtin/motion-ranked`, or `my-package/my-policy`. Exact Python method names remain a later implementation-specification decision.

### Initial strategy action boundary — approved

Version 1 uses a finite, configurable catalog of inspection regions. A strategy makes one sequential decision at a time:

```text
inspect(region_id) | stop
```

After an accepted inspection, the runtime executes the detector and constructs the next permitted observation. The strategy may then inspect another region or stop. The runtime ends the decision sequence when the strategy stops or when the enforced deadline or budget is exhausted.

This single action contract supports predetermined rules, adaptive heuristics, and learned policies. It also avoids a combinatorial “choose an entire subset” action space and allows later choices to react to earlier inspection results. Arbitrary free-form crop coordinates are outside the initial action space but may be added through a future compatible capability.

### Initial strategy observation boundary — approved

Every strategy receives the same permitted information:

- one shared, fixed-size low-resolution overview of the current source frame;
- the configured region catalog and its source-frame geometry;
- prior detection results only from regions that were actually inspected;
- each region's inspection age;
- the remaining enforced compute budget and deadline;
- basic source-frame dimensions and timing.

The observation never contains benchmark labels, reward values, or uninspected high-resolution pixels. A strategy may derive additional features from the permitted observation, but its computation time and resource use remain part of the measured end-to-end system cost. The cost of creating the shared overview is also measured.

The exact overview resolution and encoding remain measurement-informed configuration choices rather than assumptions embedded in the strategy API.

### Training reward versus evaluation — approved for version 1

Training reward and final evaluation are separate systems with different responsibilities.

- A **training reward** is a configurable Gym component used only to teach or analyze a policy. It may use labels from the training split and cached detector outcomes.
- The **evaluator** scores completed strategies with fixed accuracy, latency, freshness, and hardware metrics on held-out, independently annotated footage.
- Evaluation never treats accumulated training reward as evidence of success.
- Test labels and evaluator-only information are unavailable to strategies and training code.
- Official experiments pin and record the reward implementation, version, parameters, data split, and random seeds.
- Custom reward functions may be registered for research, but they do not change the official evaluation scoreboard.

The exact initial reward equation remains open until the available labels and detector behavior are inspected.

### Detector contract

The required semantic contract is intentionally small:

```text
batch of images -> boxes, class IDs, and confidence scores per image
```

The adapter owns model loading, model-specific preprocessing, inference, and model-level postprocessing. The core owns tiling, full-frame coordinate mapping, cross-tile merging, budget enforcement, tracking, telemetry, and benchmark records.

Optional detector capabilities may advertise supported shapes, batching, dynamic inputs, device/backend information, warm-up, stage timing, or zero-copy input. A basic CPU adapter does not need to implement accelerator concepts.

### Detector registry and artifact ownership

The toolkit packages integrations, not third-party model weights.

Conceptual loading flow:

```text
namespaced detector ID
    -> integration registry
    -> upstream artifact resolver
    -> loaded Detector
```

Convenience aliases may exist for interactive use. Official benchmarks must pin the upstream revision, artifact checksum, adapter version, backend, precision, input shape, confidence threshold, and relevant postprocessing configuration.

## Canonical record boundary — approved

Each inference operation produces three linked but separate kinds of records:

1. **Detection result:** what the detector reported, including boxes, class IDs, confidence scores, and frame context.
2. **Execution record:** what work was requested and how it executed, including detector identity, inspected region, backend, input shape, and stage timings.
3. **Hardware samples:** independently timestamped measurements such as CPU utilization, RAM, GPU utilization, VRAM, power, and temperature when the platform exposes them.

The record families are joined through stable run, frame, and inference identifiers. This separates model output from benchmark cost and avoids pretending that asynchronous hardware samples are exact properties of an individual detection.

Backend-native objects such as PyTorch tensors cannot cross the detector adapter boundary. Canonical records use ordinary backend-neutral values suitable for validation and serialization.

### Canonical detection coordinates — approved

All canonical detections are expressed on the original source-frame coordinate system, regardless of whether inference inspected the entire frame or a tile.

- Boxes use continuous `x1, y1, x2, y2` pixel-edge coordinates.
- `(0, 0)` is the upper-left edge of the source frame; `(frame_width, frame_height)` is its lower-right edge.
- Fractional values are preserved until a raster operation requires rounding.
- The inspected region is recorded separately from the resulting detection boxes.
- A detector adapter reports boxes relative to the image it received; the core maps them into source-frame coordinates exactly once.
- Normalized coordinates may be offered as an export view but are not the canonical representation.

This gives full-frame and tile-based contestants a common, directly comparable output space.

## First detector — approved

**D-FINE-N**, with feasibility validated as part of Checkpoint 1.

Why it is the current recommendation:

- modern real-time detector accepted as an ICLR 2025 Spotlight;
- approximately 4 million parameters and 7 GFLOPs at 640-pixel input;
- Apache-2.0 repository;
- official pretrained checkpoints;
- documented ONNX and TensorRT export paths;
- supported directly by Hugging Face Transformers;
- small enough to be credible on the 4 GB reference GPU;
- materially different from older YOLO-style detectors, leaving YOLOX as a useful second-family integration.

The model remains upstream-owned and is never redistributed in the core package.

If the feasibility check exposes unacceptable integration or runtime problems, stop and document the finding before selecting a replacement. **YOLOX-S or YOLOX-Tiny** is the current fallback candidate because it is mature, Apache-2.0, and documented across ONNX Runtime, TensorRT, OpenVINO, and other edge runtimes.

## First execution backend — approved

Checkpoint 1 uses **Hugging Face Transformers with PyTorch** as its bootstrap execution backend.

This choice minimizes uncertainty in the first experiment:

- it loads D-FINE-N through a pinned upstream model revision;
- one implementation can execute on both CPU and CUDA;
- it avoids making model export and conversion correctness part of the first checkpoint;
- it lets the project validate its detector contract, coordinate mapping, and measurement path before optimizing the runtime.

This is an adapter-level decision, not a permanent core dependency. The detector-neutral core must not import or expose PyTorch types. ONNX Runtime is the intended next portability backend after the reference behavior is established. TensorRT remains a later NVIDIA-specific optimization.

## Reference hardware

- Dell Precision 3591
- Intel Core Ultra 7 155H, 16 cores / 22 threads
- 16 GB system RAM
- NVIDIA RTX 500 Ada Laptop GPU
- 4 GB GDDR6 VRAM
- 30 W active GPU power limit during inspection
- EMEET SmartCam Nova 4K
- Windows 11 with WSL2 Ubuntu available

The laptop is the reference edge device. Passing on this machine demonstrates constrained-device viability, not automatic scalability. Larger-device claims require separate measurements.

## Checkpoint 1 fixture recipe — approved

Checkpoint 1 uses a deliberately staged, privacy-safe recording from the EMEET camera:

- approximately 20–30 seconds at 3840 × 2160 and 30 FPS;
- fixed camera and stable lighting;
- no audio, faces, screens, addresses, or private information;
- several objects from the detector's supported label space, placed across the frame and at different apparent sizes;
- at least one object entering and leaving the scene;
- original codec and timing metadata preserved;
- exact file identified by a cryptographic checksum and distributed as a separate test asset rather than inside the Python package.

This is an engineering fixture for chronology, coordinate mapping, repeatability, and performance. It is not training data and cannot establish policy accuracy or generalization. The exact recording is replaceable until it has been inspected and frozen in the benchmark manifest.

## First checkpoint — approved: Eyes and Stopwatch

Before implementing Gymnasium, reinforcement learning, tracking, a dashboard, or multiple strategies, prove the detector boundary and measurement path end to end.

### Input

- one short, deterministic prerecorded high-resolution video;
- one pinned upstream D-FINE-N model revision;
- the reference laptop.

### Work performed

1. Resolve and load the external model by ID.
2. Decode frames in chronological order.
3. Run whole-frame resized inference.
4. Run inference on one or more explicit high-resolution tiles.
5. Convert tile-relative boxes back into full-frame coordinates.
6. Produce the same canonical detection records for both paths.
7. Measure end-to-end latency and resource use around the complete operation.
8. Save machine-readable results and an annotated output video or frame set.

### Checkpoint success criteria

- clean, documented setup on the reference machine;
- no model weights committed or redistributed;
- exact model revision and checksum recorded;
- correct full-frame coordinate mapping verified with tests;
- detector adapter contains all model-specific behavior;
- core code contains no D-FINE-specific branches;
- chronological, non-batched-lookahead execution;
- p50/p95/p99 end-to-end latency recorded;
- peak RAM and VRAM recorded where available;
- CPU and GPU capability detection reported honestly;
- structured detection and telemetry output defined and validated;
- repeated run produces equivalent detections within declared numerical tolerance.

### Explicitly excluded from this checkpoint

- Gymnasium environment;
- RL training;
- tracker integration;
- learned or motion-based selection;
- live webcam capture;
- annotation tooling;
- dashboard or polished frontend;
- multiple detector integrations;
- TensorRT-specific optimization unless required to establish basic feasibility.

## Packaging and portability principles

- Windows and Linux are first-class targets.
- Base package remains usable without CUDA.
- Accelerator support is optional and capability-detected.
- CPU CI uses deterministic synthetic or tiny video fixtures.
- No model or network download occurs at module import time.
- Generated hardware-specific artifacts are cached locally and keyed by model, runtime, device, precision, and shape.
- TensorRT engines are not treated as portable artifacts.
- Platform-specific camera, codec, and telemetry behavior remains behind adapters.

## Non-goals for the first system

- general-purpose labeling platform;
- detector training platform;
- VLM orchestration framework;
- multi-camera fleet manager;
- edge/cloud offloading scheduler;
- generalized anomaly-detection benchmark;
- simultaneous dynamic detector switching;
- proof of transfer to every detector or hardware platform.

## Still-open design decisions

These do not block Checkpoint 1 unless explicitly included in its approved scope. Implementation-level choices required by the checkpoint may be made conservatively and recorded in its plan.

1. Define the remaining exact fields, identifiers, timing units, and serialization format within the approved canonical record families.
2. Record, inspect, and freeze the exact Checkpoint 1 fixture and checksum from the approved recipe.
3. Define the exact tile coordinate system and overlap behavior.
4. Decide whether tracking is absent or fixed in the first scientific benchmark.
5. Define the initial reward equation and choose the measured reference overview configuration; the reward/evaluation separation, observation boundary, and sequential action boundary are approved.
6. Select datasets and camera-level train/tune/test splits.
7. Define the minimum supported installation profiles for Windows and Linux.
8. Finalize the strategy lifecycle, plugin discovery mechanism, and configuration schema.

## Decision log

- **2026-08-05:** Approved the adaptive edge-perception toolkit product boundary.
- **2026-08-05:** Approved detector and strategy extensibility as first-class contracts.
- **2026-08-05:** Approved upstream-owned model artifacts selected through namespaced IDs; the project does not redistribute weights.
- **2026-08-05:** Approved D-FINE-N as the first detector.
- **2026-08-05:** Approved “Eyes and Stopwatch” as Checkpoint 1.
- **2026-08-05:** Approved Hugging Face Transformers with PyTorch as Checkpoint 1's bootstrap CPU/CUDA backend; the core remains backend-neutral.
- **2026-08-05:** Approved separate linked detection, execution, and hardware record families as the canonical measurement boundary.
- **2026-08-05:** Approved continuous full-source-frame `x1, y1, x2, y2` pixel coordinates as the canonical detection space.
- **2026-08-05:** Approved a short staged, privacy-safe 4K local recording recipe for the Checkpoint 1 engineering fixture; it is not generalization evidence.
- **2026-08-05:** Approved named, versioned, configurable strategy plugins selected through a common public API; built-ins and third-party strategies follow the same contract.
- **2026-08-05:** Approved sequential `inspect(region_id) | stop` decisions over a finite configurable region catalog as the version 1 strategy action boundary.
- **2026-08-05:** Approved a shared low-resolution overview, permitted inspection history, region ages, and remaining budget as the version 1 observation boundary; hidden high-resolution pixels and evaluation data remain inaccessible.
- **2026-08-05:** Approved strict separation between configurable Gym training rewards and the fixed held-out evaluation scoreboard; the initial reward equation remains provisional.
- **2026-08-05:** Froze further future-system design and authorized implementation of the minimal Eyes and Stopwatch checkpoint.
