# Eyes and Stopwatch: CLI-First Workflow Proof

**Verified:** 2026-08-07

## Verified claim

Adaptive Edge Perception can execute an offline, source-agnostic local-video workflow through the real CLI, configuration, PyAV decode, chronological runner, canonical artifact writers, run projection, terminal inspection renderer, and semantic comparison implementation. Two runs over a deterministic 200×100, 30 FPS, exactly three-frame synthetic video each completed with 3 processed frames, 9 inferences, and 3 annotated PNGs; inspection reported `Completed`, and comparison reported `equivalent=true` with zero mismatches.

The only replaced boundary was `edge_perception.cli.load_detector`, the external model-loading seam. It returned the deterministic test `FakeDetector`. This proof does not establish detector accuracy, real-model latency, camera performance, CPU performance, or CUDA performance.

The acceptance test passed on its first Task 7 run:

```text
./.tools/uv.exe run pytest tests/test_cli_workflow_acceptance.py -q
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

This fixes the encoded dimensions at 200×100, the stream rate and time base at 30 FPS, and the submitted frame sequence at exactly three frames.

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
./.tools/uv.exe run pytest tests/test_cli_workflow_acceptance.py -q
1 passed in 0.94s (final rerun; the first Task 7 run is recorded above)

./.tools/uv.exe run pytest -m "not model" -q
396 passed, 1 skipped, 1 deselected in 10.29s

./.tools/uv.exe run ruff check src tests
All checks passed!

./.tools/uv.exe run mypy src
Success: no issues found in 27 source files

./.tools/uv.exe build
Successfully built dist\adaptive_edge_perception-0.1.0.tar.gz
Successfully built dist\adaptive_edge_perception-0.1.0-py3-none-any.whl

./.tools/uv.exe run python -m zipfile -l \
  dist/adaptive_edge_perception-0.1.0-py3-none-any.whl
```

The wheel listing contained 31 entries: 27 `edge_perception/**/*.py` files and four `adaptive_edge_perception-0.1.0.dist-info` metadata files (`METADATA`, `WHEEL`, `entry_points.txt`, and `RECORD`). The exact allowlist/denylist inventory check reported:

```text
wheel inventory: 31 entries; 27 package Python files; 4 metadata files; no blocked or unexpected data
```

It proved that the wheel contains Python/package metadata only. It contains no MP4 or other video, JSON/JSONL run artifact, PNG annotation, model weight (`.pt`, `.pth`, `.onnx`, `.safetensors`, or `.bin`), private capture, Qt binary, PySide6 payload, `.superpowers` path, test file, or unexpected data file.

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

No public-video lane was run. Public media is unnecessary for the acceptance proof and would have required externally materialized content that the user may lawfully obtain.

No CPU real-model lane was run because the already-cached CUDA lane was sufficient additive functional evidence. Neither real-model lane is a completion gate.

## Limitations

- FakeDetector output cannot support detector-accuracy, model-quality, or hardware-performance claims.
- The offline acceptance proves deterministic semantic equivalence for the fake boundary, not bit-identical timing or telemetry.
- The one-frame real D-FINE/CUDA run is a functional smoke check only.
- The short strict EMEET sample did not meet the requested recorded FPS; the successful non-strict sample reported 20.0008 FPS rather than 30 FPS.
- No public provider, network downloader, public dataset, ground truth, tracking, policy, or reinforcement-learning environment was evaluated.
- Windows and Linux are supported; macOS remains unvalidated.

## Next scientific question

On a lawfully obtained, checksum-pinned video with ground-truth detections, how do full-frame and ROI schedules change D-FINE accuracy, end-to-end frame latency, and memory use across CPU and CUDA when the model revision, thresholds, frame order, and artifact contract are held constant?
