# Eyes and Stopwatch: CLI-First Workflow Proof

**Verified:** 2026-08-07

## Verified claim

Adaptive Edge Perception can execute an offline, source-agnostic local-video workflow through the real CLI, configuration, PyAV decode, chronological runner, canonical artifact writers, run projection, terminal inspection renderer, and semantic comparison implementation. Two runs over a deterministic 200×100, 30 FPS, exactly three-frame synthetic video each completed with 3 processed frames, 9 inferences, and 3 annotated PNGs; inspection reported `Completed`, and comparison reported `equivalent=true` with zero mismatches.

The only replaced boundary was `edge_perception.cli.load_detector`, the external model-loading seam. It returned the deterministic test `FakeDetector`. This proof does not establish detector accuracy, real-model latency, camera performance, CPU performance, or CUDA performance.

The acceptance test passed on its first Task 7 run:

```text
uv run pytest tests/test_cli_workflow_acceptance.py -q
.                                                                        [100%]
1 passed in 0.66s
```

Tasks 2–4 already existed at Task 7 entry, so this is recorded honestly as a pre-existing acceptance oracle. No artificial RED was created and the oracle was not weakened to force one.

## Generated source recipe

The acceptance test generates the source locally with no network, model, camera, display server, or public media:

```python
from fractions import Fraction

import av
import numpy as np

with av.open("three-frames.mp4", mode="w") as container:
    stream = container.add_stream("mpeg4", rate=30)
    stream.width = 200
    stream.height = 100
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 30)

    for frame_index, red_value in enumerate((20, 120, 220)):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[..., 0] = red_value
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = frame_index
        frame.time_base = Fraction(1, 30)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
```

This fixes the encoded dimensions at 200×100, the stream rate and time base at 30 FPS, and the submitted frame sequence at exactly three frames. The acceptance oracle then uses production `probe_video` to assert 200×100 and approximately 30 FPS, fully consumes production `iter_video`, and asserts decoded indices `[0, 1, 2]` with three `(100, 200, 3)` RGB images. The `--max-frames 3` run limit therefore cannot hide a fourth decoded frame.

The first focused run after adding these characterization assertions passed honestly without a fabricated RED:

```text
uv run pytest tests/test_cli_workflow_acceptance.py -q
1 passed in 0.92s
```

## Exact acceptance workflow

The test calls `edge_perception.cli.main` in-process with these command shapes:

```text
edge-perception run three-frames.mp4 \
  --output run-a \
  --detector dfine-nano-coco \
  --device cpu \
  --max-frames 3 \
  --warmup-runs 0 \
  --annotate-every 1 \
  --crop left:0,0,100,100 \
  --crop right:100,0,100,100

edge-perception inspect run-a

edge-perception run three-frames.mp4 \
  --output run-b \
  --detector dfine-nano-coco \
  --device cpu \
  --max-frames 3 \
  --warmup-runs 0 \
  --annotate-every 1 \
  --crop left:0,0,100,100 \
  --crop right:100,0,100,100

edge-perception compare run-a run-b
```

Each frame produces one full-frame inference followed by the declared `left` and `right` ROI inferences: 3 frames × 3 inference regions = 9 inferences. The fake emits one deterministic source-mappable detection per inference, so comparison reports 9 detections on each side.

The three annotated PNGs are evidence for this particular run configuration, not a required product output. Setting `--annotate-every 0` disables diagnostic PNG generation while preserving canonical detections, telemetry, inspection, and comparison.

## Fake-detector boundary

Only `edge_perception.cli.load_detector` is replaced. The fake supplies a complete detector identity, deterministic single-image predictions, deterministic timing records, and no device-memory peak. The following remain production implementations:

- argument parsing and detector-ID validation;
- explicit `RunConfig` construction and path resolution;
- local video validation and PyAV decoding;
- full-frame and source-coordinate ROI execution;
- warm-up, telemetry, cancellation callback plumbing, and chronological runner control;
- manifest, summary, inference, detection, hardware, and PNG artifact publication;
- `load_run_view` validation and projection;
- terminal inspection rendering; and
- semantic run comparison.

There are no assertions about the fake-loader call itself. Assertions target user-visible CLI output and durable production artifacts.

## Artifact tree

Both `run-a` and `run-b` contained exactly the required canonical files and exactly three direct annotated PNGs:

```text
run-a/
├── manifest.json
├── summary.json
├── inferences.jsonl
├── detections.jsonl
├── hardware.jsonl
└── annotated/
    ├── 000000.png
    ├── 000001.png
    └── 000002.png

run-b/
├── manifest.json
├── summary.json
├── inferences.jsonl
├── detections.jsonl
├── hardware.jsonl
└── annotated/
    ├── 000000.png
    ├── 000001.png
    └── 000002.png
```

Observed inspection fields included:

```text
Run
  status: Completed
Metrics
  frames processed: 3
  inference count: 9
```

Observed comparison output was:

```text
equivalent=true left=9 right=9 mismatches=0
```

## Required verification and packaging gates

The required commands and observed results were:

```text
uv run pytest tests/test_release_archives.py tests/test_cli.py -q -k \
  "release or missing_optional_extra or preserves_unrelated_import_error or pyside6_import_classifier or missing_detector_runtime or preserves_unrelated_missing_model_dependency"
20 passed, 44 deselected in 2.17s

uv run pytest -m "not model" -q
409 passed, 1 skipped, 1 deselected in 12.56s

uv run ruff check src tests scripts
All checks passed!

uv run mypy src
Success: no issues found in 27 source files

uv lock --check --offline
Resolved 69 packages in 1ms

uv build --offline
Successfully built dist\adaptive_edge_perception-0.1.0.tar.gz
Successfully built dist\adaptive_edge_perception-0.1.0-py3-none-any.whl

uv run python scripts/verify_release_archives.py \
  dist/adaptive_edge_perception-0.1.0-py3-none-any.whl \
  dist/adaptive_edge_perception-0.1.0.tar.gz
wheel inventory: 32 files; 27 package Python; 5 metadata/license
sdist inventory: 59 files; 27 package Python; 25 test Python; 1 release verifier; 1 checkpoint; 5 project/metadata
release archive policy: passed
```

The wheel contains 27 `edge_perception/**/*.py` files and exactly five `adaptive_edge_perception-0.1.0.dist-info` metadata/license files: `METADATA`, `WHEEL`, `entry_points.txt`, `licenses/LICENSE`, and `RECORD`. Its `METADATA` declares metadata version 2.4, `License-Expression: Apache-2.0`, and `License-File: LICENSE`.

The sdist contains only `LICENSE`, `README.md`, `pyproject.toml`, `uv.lock`, generated `PKG-INFO`, package Python, test Python, `scripts/verify_release_archives.py`, and Markdown under `docs/checkpoints`. Its `PKG-INFO` carries the same PEP 639 metadata. The verifier rejects every other member family, including ignored checkout state, before printing either inventory.

Together these checks prove that the wheel contains Python/package metadata only and the sdist contains only the declared public source families. Neither archive contains video, run artifacts, PNG annotations, model weights, private captures, Qt/PySide6 payloads, `.superpowers` state, or unexpected data.

Clean lazy-startup probes using installed-uv command shapes—`uv run edge-perception --help`, `uv run edge-perception run --help`, `uv run edge-perception gui --help`, and `uv run edge-perception camera --help`—all exited 0 without loading a model, opening a camera, or launching the GUI.

## Additive, not gating: cached D-FINE and CUDA

This lane is optional and does not change the source-agnostic pass/fail result. The exact pinned cache snapshot was already present:

```text
ustc-community/dfine-nano-coco
revision 066438d3d8f0da137a37b38fdf3368fd4afceced
files: config.json, model.safetensors, preprocessor_config.json
cuda_available=True
cuda_device_count=1
```

With network access disabled through `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, one generated frame was run through the real adapter on CUDA with `--max-frames 1 --warmup-runs 0 --annotate-every 0`. Observed output:

```text
output: <temporary-directory>/dfine-cuda-run
status=complete frames=1 inferences=1
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model.
```

The Transformers warning continued by explaining its future `use_fast=True` default and possible minor output differences. No model or video was downloaded. This proves only that the cached pinned adapter completed one bounded local run on the visible CUDA runtime; it is not an accuracy, determinism, throughput, latency, or memory benchmark.

## Additive, not gating: EMEET camera

This lane is optional and does not change the generated-video proof. Read-only discovery completed in 1.9 seconds and reported three devices: `EMEET SmartCam Nova 4K`, `Integrated Webcam`, and `AvStream Media Device`. For the EMEET, discovery reported NV12 and JPEG modes from 640×360 through 3840×2160, plus 640×360 and 640×480 YUYV modes.

The exact short strict command requested 640×480 at 30 FPS for 0.5 seconds. It exited 2 with:

```text
[h264_mf @ <encoder-instance omitted>] MFT name: 'H264 Encoder MFT'
error: captured FPS differs from strict request
```

The explicit destination remained unpublished. A second explicit-destination command used the same 640×480/30 FPS request for 1 second without strict validation and exited 0:

```text
Capture: <temporary-directory>/emeet.mp4
SHA-256: <private sample digest omitted>
Capture request: device=<device-id omitted> description=EMEET SmartCam Nova 4K width=640 px height=480 px fps=30 fps strict=false
Applied camera format: 640x480 30-30 fps NV12
Recorded format: width=640 px height=480 px fps=20.0008 fps container=mov,mp4,m4a,3gp,3g2,mj2 codec=h264 duration=0.799967 s audio=false size=18597 bytes
[h264_mf @ <encoder-instance omitted>] MFT name: 'H264 Encoder MFT'
```

Machine-specific device identifiers, encoder-instance values, temporary paths, and the private sample digest are deliberately omitted from this public checkpoint. The private capture and temporary directories were deleted after recording this output. This observation proves bounded discovery, strict rejection without publication, and non-strict capture/finalization on this machine. It does not establish stable frame rate, image quality, camera reliability, or hardware performance.

## Optional evidence not run

No public-video lane was run. Public media is unnecessary for the acceptance proof. Any future public-video lane must use content the researcher may lawfully obtain and must retain provider/source URL, title or asset ID, license or permission basis, retrieval date, original filename, and SHA-256 provenance without redistributing media unless permitted.

No CPU real-model lane was run because the already-cached CUDA lane was sufficient additive functional evidence. Neither real-model lane is a completion gate.

No strict-4K evidence run or second bounded CUDA pass was run. The old strict-4K, two-pass CUDA, and real-model CPU criteria are superseded as checkpoint completion gates and remain planned opt-in hardware validation.

## Limitations

- FakeDetector output cannot support detector-accuracy, model-quality, or hardware-performance claims.
- The offline acceptance proves deterministic semantic equivalence for the fake boundary, not bit-identical timing or telemetry.
- The one-frame real D-FINE/CUDA run is a functional smoke check only.
- The short strict EMEET sample did not meet the requested recorded FPS; the successful non-strict sample reported 20.0008 FPS rather than 30 FPS.
- No public provider, network downloader, public dataset, ground truth, tracking, policy, or reinforcement-learning environment was evaluated.
- Windows was exercised by this checkpoint. Linux compatibility is supported by design but was not exercised in this checkpoint. macOS remains unvalidated.

## Next scientific question

On a lawfully obtained, checksum-pinned video with ground-truth detections, how do full-frame and ROI schedules change D-FINE accuracy, end-to-end frame latency, and memory use across CPU and CUDA when the model revision, thresholds, frame order, and artifact contract are held constant?
