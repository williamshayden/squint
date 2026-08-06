# Eyes and Stopwatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the smallest executable proof that a pinned upstream D-FINE-N detector can process chronological high-resolution video through whole-frame and explicit-crop paths, emit source-frame detections, and report reproducible CPU/GPU latency and memory measurements on the reference laptop.

**Architecture:** A dependency-light core defines backend-neutral records and crop geometry. A single optional Hugging Face/PyTorch adapter owns D-FINE loading, preprocessing, inference, and postprocessing. A synchronous runner decodes one frame at a time, runs the full frame followed by configured crops, maps crop detections into source coordinates, and writes three linked record streams plus a summary and annotated PNG frames.

**Tech Stack:** Python 3.12, uv 0.11.32, NumPy, Pillow, PyAV, psutil, optional NVIDIA NVML, PyTorch 2.11.0 CPU or CUDA 12.8 wheels, Transformers 4.57.6, pytest, Ruff, mypy.

## Global Constraints

- Repository working title and package name remain provisional: distribution `adaptive-edge-perception`, import package `edge_perception`, CLI `edge-perception`.
- The only model integration is `ustc-community/dfine-nano-coco` at immutable revision `066438d3d8f0da137a37b38fdf3368fd4afceced`.
- Model weights are resolved from upstream at an explicit command, hashed locally, cached outside Git, and never redistributed.
- The detector adapter owns loading, preprocessing, inference, model postprocessing, device placement, and CUDA synchronization.
- Core modules contain no D-FINE, Transformers, PyTorch, or CUDA imports.
- Canonical boxes are finite continuous `x1, y1, x2, y2` coordinates in original source-frame pixels.
- Measured frames execute chronologically with batch size one and without random access or lookahead.
- Warm-up work is excluded from measured records and declared in the manifest.
- Timing uses monotonic nanoseconds internally and serialized milliseconds with explicit `_ms` suffixes.
- Memory uses bytes with explicit `_bytes` suffixes; absent sensors serialize as `null` rather than failing a run.
- Windows and Linux are supported; CPU-only installation and tests must not require CUDA, NVML, a webcam, a GUI, or a model download.
- Generated videos, model files, run directories, caches, and hardware-specific artifacts remain ignored by Git.
- Gymnasium, RL, strategies, tracking, live capture, dashboards, detector registries, ONNX, TensorRT, and cross-tile fusion are outside this plan.

---

## File Map

```text
pyproject.toml                         dependency profiles, CLI, lint/test configuration
.python-version                       project Python selection
src/edge_perception/__init__.py       package version and public core exports
src/edge_perception/contracts.py      backend-neutral immutable data contracts
src/edge_perception/geometry.py       crop validation and source-coordinate mapping
src/edge_perception/detector.py       detector protocol
src/edge_perception/detectors/dfine.py pinned D-FINE Hugging Face adapter
src/edge_perception/video.py          chronological PyAV decoding
src/edge_perception/telemetry.py      host report and timestamped sampling
src/edge_perception/outputs.py        JSON/JSONL and annotated PNG output
src/edge_perception/runner.py         synchronous checkpoint orchestration
src/edge_perception/compare.py        timing-free repeated-detection comparison
src/edge_perception/cli.py            one CLI with run and compare commands
tests/conftest.py                      reusable fake detector and frame fixtures
tests/test_contracts.py               validation and serialization tests
tests/test_geometry.py                crop and coordinate tests
tests/test_dfine.py                   mocked adapter tests
tests/test_dfine_integration.py       opt-in real-model smoke test
tests/test_video.py                   generated-video chronological decode tests
tests/test_telemetry.py               partial-sensor and monitor tests
tests/test_outputs.py                 record-stream and rendering tests
tests/test_runner.py                  fake-detector end-to-end tests
tests/test_compare.py                 repeatability comparison tests
tests/test_cli.py                     CLI parsing and smoke tests
docs/checkpoints/eyes-and-stopwatch.md measured checkpoint report
```

## Public Interfaces

The implementation tasks use these exact signatures:

```python
# contracts.py
@dataclass(frozen=True, slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass(frozen=True, slots=True)
class Region:
    region_id: str
    x: int
    y: int
    width: int
    height: int

@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    class_id: int
    score: float
    label: str | None = None

@dataclass(frozen=True, slots=True)
class StageTiming:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float

@dataclass(frozen=True, slots=True)
class BatchPrediction:
    detections: tuple[tuple[Detection, ...], ...]
    timing: StageTiming

@dataclass(frozen=True, slots=True)
class DetectorIdentity:
    adapter: str
    model_id: str
    revision: str
    weights_sha256: str
    backend: str
    backend_version: str
    device: str
    dtype: str

# detector.py
class Detector(Protocol):
    @property
    def identity(self) -> DetectorIdentity: ...
    def warmup(self, image: np.ndarray, runs: int) -> None: ...
    def predict(self, images: Sequence[np.ndarray]) -> BatchPrediction: ...
    def peak_device_memory_bytes(self) -> int | None: ...

# video.py
@dataclass(frozen=True, slots=True)
class DecodedFrame:
    frame_index: int
    source_time_ms: float | None
    image: np.ndarray

def iter_video(path: Path) -> Iterator[DecodedFrame]: ...

# runner.py
@dataclass(frozen=True, slots=True)
class RunConfig:
    input_path: Path
    output_dir: Path
    regions: tuple[Region, ...]
    max_frames: int | None
    warmup_runs: int
    annotate_every: int

def run_checkpoint(config: RunConfig, detector: Detector) -> dict[str, object]: ...

# compare.py
def compare_runs(
    left: Path,
    right: Path,
    *,
    box_atol: float = 0.01,
    score_atol: float = 1e-4,
) -> dict[str, object]: ...
```

---

### Task 1: Reproducible project bootstrap and neutral contracts

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Modify: `.gitignore`
- Create: `src/edge_perception/__init__.py`
- Create: `src/edge_perception/contracts.py`
- Create: `tests/test_contracts.py`

**Interfaces:**
- Consumes: approved canonical coordinate and record boundaries from `edge-perception-project-brief.md`.
- Produces: `Box`, `Region`, `Detection`, `StageTiming`, `BatchPrediction`, and `DetectorIdentity` exactly as declared above.

- [ ] **Step 1: Add the project metadata and mutually exclusive CPU/CUDA extras**

Create `pyproject.toml` with a Hatchling build, Python `>=3.12,<3.13`, base dependencies `av>=16,<17`, `numpy>=2.2,<3`, `pillow>=11,<13`, and `psutil>=7,<8`. Define `cpu` and `cu128` extras that both contain `torch==2.11.0`, `transformers==4.57.6`, and `safetensors>=0.6,<1`; add `nvidia-ml-py>=13,<14` only to `cu128`. Route Torch through explicit `https://download.pytorch.org/whl/cpu` and `https://download.pytorch.org/whl/cu128` indexes using uv extra markers, and declare the two extras conflicting. Add a `dev` group containing pytest, pytest-cov, Ruff, and mypy. Register `edge-perception = "edge_perception.cli:main"`.

Create `.python-version` containing exactly:

```text
3.12
```

Add `.tools/`, `fixtures/private/`, and `docs/checkpoints/artifacts/` to `.gitignore`.

- [ ] **Step 2: Install project-local uv and resolve the CPU development lock**

Run from the repository root:

```powershell
$env:UV_UNMANAGED_INSTALL = "$PWD\.tools"
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.11.32/install.ps1 | iex"
.\.tools\uv.exe lock
.\.tools\uv.exe sync --extra cpu --group dev
```

Expected: `.tools\uv.exe --version` reports `uv 0.11.32`, `.venv` exists, and `uv.lock` records the CPU and CUDA resolutions without installing both Torch variants together.

- [ ] **Step 3: Write failing contract tests**

Create tests that assert:

```python
def test_box_rejects_non_finite_and_degenerate_values() -> None:
    with pytest.raises(ValueError):
        Box(float("nan"), 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        Box(1.0, 0.0, 1.0, 1.0)

def test_detection_is_backend_neutral_and_json_serializable() -> None:
    detection = Detection(Box(1.25, 2.5, 10.75, 20.0), 3, 0.875, "car")
    encoded = json.dumps(detection.to_dict(), sort_keys=True)
    assert "torch" not in encoded.lower()
    assert json.loads(encoded)["box"] == [1.25, 2.5, 10.75, 20.0]

def test_region_requires_positive_integer_extent() -> None:
    with pytest.raises(ValueError):
        Region("bad", 0, 0, 0, 10)
```

- [ ] **Step 4: Run the contract tests and observe the expected import failure**

Run:

```powershell
.\.tools\uv.exe run pytest tests/test_contracts.py -q
```

Expected: FAIL because `edge_perception.contracts` does not exist.

- [ ] **Step 5: Implement immutable validated contracts and deterministic `to_dict()` methods**

Use frozen slotted dataclasses. Validate with `math.isfinite`; require `x2 > x1`, `y2 > y1`, `0.0 <= score <= 1.0`, positive region extents, and non-negative timing values. `to_dict()` must return only JSON-native dictionaries, lists, strings, integers, floats, booleans, and nulls.

- [ ] **Step 6: Verify the task**

Run:

```powershell
.\.tools\uv.exe run pytest tests/test_contracts.py -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Expected: all commands succeed.

- [ ] **Step 7: Commit the task**

```powershell
git add pyproject.toml uv.lock .python-version .gitignore src/edge_perception tests/test_contracts.py
git commit -m "build: establish checkpoint contracts and environments"
```

---

### Task 2: Crop validation and source-coordinate mapping

**Files:**
- Create: `src/edge_perception/geometry.py`
- Create: `tests/test_geometry.py`

**Interfaces:**
- Consumes: `Box`, `Region`, and `Detection` from `edge_perception.contracts`.
- Produces: `full_frame_region(frame_width, frame_height)`, `validate_region(region, frame_width, frame_height)`, `crop_region(image, region)`, and `map_detection_to_source(detection, region, frame_width, frame_height)`.

- [ ] **Step 1: Write failing mapping tests**

Cover the full-frame identity path, crop offsets, fractional boxes, frame-edge clipping, invalid negative origins, regions extending past the frame, and non-contiguous array crops. The central assertion is:

```python
def test_crop_detection_maps_once_to_source_pixels() -> None:
    local = Detection(Box(10.5, 20.25, 110.5, 220.25), 1, 0.9, "person")
    region = Region("left", 960, 540, 1280, 1080)
    mapped = map_detection_to_source(local, region, 3840, 2160)
    assert mapped.box == Box(970.5, 560.25, 1070.5, 760.25)
```

- [ ] **Step 2: Run the geometry tests and observe the expected import failure**

```powershell
.\.tools\uv.exe run pytest tests/test_geometry.py -q
```

Expected: FAIL because `edge_perception.geometry` does not exist.

- [ ] **Step 3: Implement minimal geometry functions**

`crop_region` slices `image[y:y+height, x:x+width]` after validation and returns a contiguous RGB array. `map_detection_to_source` adds the crop origin exactly once and clips the resulting continuous edges to `[0, frame_width] × [0, frame_height]`; it raises if clipping produces a degenerate box.

- [ ] **Step 4: Verify the task**

```powershell
.\.tools\uv.exe run pytest tests/test_geometry.py -q
.\.tools\uv.exe run ruff check src/edge_perception/geometry.py tests/test_geometry.py
```

Expected: PASS.

- [ ] **Step 5: Commit the task**

```powershell
git add src/edge_perception/geometry.py tests/test_geometry.py
git commit -m "feat: map crop detections into source coordinates"
```

---

### Task 3: Detector protocol and pinned D-FINE adapter

**Files:**
- Create: `src/edge_perception/detector.py`
- Create: `src/edge_perception/detectors/__init__.py`
- Create: `src/edge_perception/detectors/dfine.py`
- Create: `tests/test_dfine.py`
- Create: `tests/test_dfine_integration.py`

**Interfaces:**
- Consumes: contract types from Task 1 and RGB NumPy arrays.
- Produces: `Detector` protocol and `DfineDetector.load(...)`, `.warmup(...)`, `.predict(...)`, and `.peak_device_memory_bytes()`.

- [ ] **Step 1: Write mocked adapter tests before importing Torch or Transformers**

Use injected fake processor, model, Torch facade, clock, and artifact resolver. Assert that:

```python
def test_dfine_requests_output_boxes_in_each_input_image_size(fake_runtime) -> None:
    detector = DfineDetector.from_components(fake_runtime)
    images = (
        np.zeros((1080, 1280, 3), dtype=np.uint8),
        np.zeros((2160, 3840, 3), dtype=np.uint8),
    )
    result = detector.predict(images)
    assert fake_runtime.processor.target_sizes == [(1080, 1280), (2160, 3840)]
    assert len(result.detections) == 2

def test_dfine_uses_inference_mode_and_synchronizes_cuda(fake_runtime) -> None:
    fake_runtime.device = "cuda"
    DfineDetector.from_components(fake_runtime).predict(
        (np.zeros((32, 32, 3), dtype=np.uint8),)
    )
    assert fake_runtime.torch.inference_mode_entries == 1
    assert fake_runtime.torch.cuda_synchronize_calls == 2
```

Also assert explicit CPU fallback for `device="auto"`, a clear error for unavailable requested CUDA, `model.eval()`, model and input movement to the selected device, class-label extraction, float conversion, model revision preservation, and SHA-256 validation.

- [ ] **Step 2: Run the mocked tests and observe the expected import failure**

```powershell
.\.tools\uv.exe run pytest tests/test_dfine.py -q
```

Expected: FAIL because the detector modules do not exist.

- [ ] **Step 3: Implement the dependency-neutral protocol**

Define the exact `Detector` protocol from the Public Interfaces section. Do not add loading, registries, plugin discovery, model IDs, or framework types to the protocol.

- [ ] **Step 4: Implement the single D-FINE adapter**

Use these immutable defaults:

```python
DEFAULT_MODEL_ID = "ustc-community/dfine-nano-coco"
DEFAULT_REVISION = "066438d3d8f0da137a37b38fdf3368fd4afceced"
DEFAULT_THRESHOLD = 0.3
WEIGHTS_FILENAME = "model.safetensors"
```

`load()` must lazy-import Torch, Transformers, and Hugging Face Hub; resolve and hash `model.safetensors`; call `AutoImageProcessor.from_pretrained(..., revision=...)`; call `DFineForObjectDetection.from_pretrained(..., revision=...)`; select explicit FP32 CPU/CUDA placement; and call `eval()`. `predict()` must convert RGB arrays to PIL images, time preprocessing/inference/postprocessing separately, synchronize immediately before and after timed CUDA inference, call `post_process_object_detection(outputs, target_sizes=[(height, width), ...], threshold=...)`, and return backend-neutral detections.

- [ ] **Step 5: Add an opt-in real-model integration test**

Mark it `@pytest.mark.model` and skip unless `RUN_MODEL_TESTS=1`. Load the pinned model in CPU mode, process one deterministic 640×640 generated RGB image, and assert identity metadata, SHA-256 length 64, batch cardinality one, finite timing, and backend-neutral output. This test validates loading and contracts, not accuracy.

- [ ] **Step 6: Verify mocked tests and import behavior**

```powershell
.\.tools\uv.exe run pytest tests/test_dfine.py -q
.\.tools\uv.exe run python -c "import edge_perception; assert 'torch' not in __import__('sys').modules"
.\.tools\uv.exe run ruff check src/edge_perception/detector.py src/edge_perception/detectors tests/test_dfine.py
```

Expected: all commands succeed without downloading a model.

- [ ] **Step 7: Commit the task**

```powershell
git add src/edge_perception/detector.py src/edge_perception/detectors tests/test_dfine.py tests/test_dfine_integration.py pyproject.toml
git commit -m "feat: add pinned D-FINE detector adapter"
```

---

### Task 4: Chronological video decoding and timestamped hardware telemetry

**Files:**
- Create: `src/edge_perception/video.py`
- Create: `src/edge_perception/telemetry.py`
- Create: `tests/test_video.py`
- Create: `tests/test_telemetry.py`

**Interfaces:**
- Consumes: local video paths and optional NVIDIA NVML availability.
- Produces: `DecodedFrame`, `iter_video`, `collect_host_report`, `HardwareProbe.sample()`, and `TelemetryMonitor` context manager with immutable samples.

- [ ] **Step 1: Write a failing chronological decode test**

Create a three-frame 64×48 MPEG-4 file inside pytest's temporary directory with PyAV. Encode RGB frames whose red-channel means increase substantially. Assert decoded indices `[0, 1, 2]`, nondecreasing non-null source timestamps, shape `(48, 64, 3)`, RGB dtype `uint8`, and increasing red-channel means. Add errors for a missing path and a container with no video stream.

- [ ] **Step 2: Write failing telemetry tests**

Use fake system and NVIDIA probes to assert one common record shape, nanosecond timestamps, nullable GPU fields, continued sampling when NVIDIA is absent, clean thread shutdown, and peak calculations. Assert the host report includes OS, Python, logical CPU count, total RAM, and optional GPU identity without parsing human-formatted `nvidia-smi` output.

- [ ] **Step 3: Run tests and observe expected import failures**

```powershell
.\.tools\uv.exe run pytest tests/test_video.py tests/test_telemetry.py -q
```

Expected: FAIL because the modules do not exist.

- [ ] **Step 4: Implement the video iterator**

Open the container with PyAV, select exactly the first video stream, decode in iterator order, convert each frame with `to_ndarray(format="rgb24")`, and derive `source_time_ms` from `frame.pts * frame.time_base * 1000` when both values exist. Do not seek, reorder, batch, display, or silently skip decode errors.

- [ ] **Step 5: Implement partial, non-fatal telemetry**

Use psutil for process RSS and system memory. Lazy-import `pynvml`; when initialization succeeds, record device utilization, used VRAM, power watts, and temperature; when it does not, leave those values null and include one capability message in the host report. `TelemetryMonitor` samples every 50 ms on a daemon thread using `threading.Event.wait`, records `time.perf_counter_ns()`, captures probe errors as unavailable fields, and always joins on context exit.

- [ ] **Step 6: Verify the task**

```powershell
.\.tools\uv.exe run pytest tests/test_video.py tests/test_telemetry.py -q
.\.tools\uv.exe run ruff check src/edge_perception/video.py src/edge_perception/telemetry.py tests/test_video.py tests/test_telemetry.py
```

Expected: PASS on a CPU-only host.

- [ ] **Step 7: Commit the task**

```powershell
git add src/edge_perception/video.py src/edge_perception/telemetry.py tests/test_video.py tests/test_telemetry.py
git commit -m "feat: decode chronological video and sample hardware"
```

---

### Task 5: Machine-readable records, annotated frames, and latency summaries

**Files:**
- Create: `src/edge_perception/outputs.py`
- Create: `tests/test_outputs.py`

**Interfaces:**
- Consumes: plain dictionaries, RGB frames, source-space detections, and region metadata.
- Produces: `RunOutputs`, deterministic JSON/JSONL streams, PNG annotations, and `summarize_latencies(records)`.

- [ ] **Step 1: Write failing output tests**

Assert that a temporary run directory contains:

```text
manifest.json
inferences.jsonl
detections.jsonl
hardware.jsonl
summary.json
annotated/000000.png
```

Parse every line with `json.loads`, assert `schema_version == "0.1.0"`, verify stable `run_id`, `frame_id`, and `inference_id` joins, verify source-space decimal boxes, and assert no NaN/Infinity or backend-native string representations. For latency values `[1, 2, 3, 4, 100]`, assert the summary exposes count and finite p50/p95/p99 values using NumPy's linear percentile method.

- [ ] **Step 2: Run tests and observe the expected import failure**

```powershell
.\.tools\uv.exe run pytest tests/test_outputs.py -q
```

Expected: FAIL because `edge_perception.outputs` does not exist.

- [ ] **Step 3: Implement atomic manifest/summary writes and append-only streams**

Write JSON to a sibling `.tmp` file and replace the destination only after successful serialization. Write JSONL with one `json.dumps(..., allow_nan=False, sort_keys=True)` object per line. Flush record streams after each completed source frame so an interrupted run retains valid prior rows.

- [ ] **Step 4: Implement deterministic annotation**

Copy the RGB frame, draw configured regions with one stable color per region ID, draw mapped boxes with class label and three-decimal confidence, and save lossless PNG files named by zero-padded frame index. Rendering is diagnostic output and its timing is excluded from detector latency.

- [ ] **Step 5: Verify the task**

```powershell
.\.tools\uv.exe run pytest tests/test_outputs.py -q
.\.tools\uv.exe run ruff check src/edge_perception/outputs.py tests/test_outputs.py
```

Expected: PASS.

- [ ] **Step 6: Commit the task**

```powershell
git add src/edge_perception/outputs.py tests/test_outputs.py
git commit -m "feat: write checkpoint records and annotated frames"
```

---

### Task 6: Synchronous checkpoint runner and CLI

**Files:**
- Create: `src/edge_perception/runner.py`
- Create: `src/edge_perception/compare.py`
- Create: `src/edge_perception/cli.py`
- Create: `tests/conftest.py`
- Create: `tests/test_runner.py`
- Create: `tests/test_compare.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: all interfaces from Tasks 1–5.
- Produces: `run_checkpoint`, `compare_runs`, and `edge-perception run|compare`.

- [ ] **Step 1: Write a fake-detector end-to-end test**

Use three generated frames and a fake detector that returns a known local box. Configure full-frame plus `Region("right", 100, 20, 80, 60)`. Assert exactly six inference rows, deterministic frame and inference ordering, crop detections offset into source coordinates, three annotated PNGs when `annotate_every=1`, telemetry samples, and p50/p95/p99 summary sections for full-frame, crop, and complete-frame latency.

- [ ] **Step 2: Write CLI parsing and failure tests**

Assert this command shape:

```text
edge-perception run INPUT --output RUN_DIR \
  --crop right:1920,0,1920,1080 \
  --crop lower-left:0,1080,1920,1080 \
  --device auto --threshold 0.3 --warmup-runs 2 \
  --max-frames 60 --annotate-every 10
```

Reject duplicate crop IDs, malformed coordinates, non-positive extents, thresholds outside `[0, 1]`, negative frame counts, output equal to the input path, and requested CUDA when unavailable. Assert nonzero exit and one-line actionable stderr messages.

- [ ] **Step 3: Write repeatability comparison tests**

Create two detection streams with different run IDs and timing but identical semantic keys `(frame_id, region_id, detection_index, class_id, label)` and values inside tolerances. Assert `equivalent: true`. Add failures for missing detections, class mismatch, coordinate difference above `box_atol`, score difference above `score_atol`, and different model revisions.

- [ ] **Step 4: Run the new tests and observe expected import failures**

```powershell
.\.tools\uv.exe run pytest tests/test_runner.py tests/test_compare.py tests/test_cli.py -q
```

Expected: FAIL because the orchestration modules do not exist.

- [ ] **Step 5: Implement the synchronous runner**

Open the video once to obtain the first frame, validate every configured region, warm up on the full frame and each distinct crop shape, close that iterator, then reopen from frame zero for measurement. For each measured frame, run the full-frame region first and explicit regions in declared order, batch size one. Measure decode, crop, detector, coordinate mapping, frame pipeline, serialization, and annotation separately. Start telemetry before measured decoding and stop it in `finally`. Write a manifest with configuration, source-video SHA-256, host report, detector identity, dependency versions, threshold, warm-up count, and timing definitions.

- [ ] **Step 6: Implement semantic run comparison**

Read both manifests and detection streams, require identical schema/model/revision/threshold/source checksum/regions, ignore run IDs and all timing/hardware fields, compare sorted semantic detections with declared tolerances, and return a JSON-native report listing the first mismatch plus aggregate counts.

- [ ] **Step 7: Implement the stdlib argparse CLI**

`run` loads `DfineDetector` only after parsing and validating paths and numeric values. `compare` calls `compare_runs` and exits `0` only when equivalent. `--device` accepts `auto`, `cpu`, or `cuda`. Print the resolved output directory and one compact summary; do not start a GUI or web server.

- [ ] **Step 8: Verify the full CPU test suite**

```powershell
.\.tools\uv.exe run pytest -m "not model" --cov=edge_perception --cov-report=term-missing -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
```

Expected: all commands succeed; coverage is reported but no arbitrary percentage gate is introduced.

- [ ] **Step 9: Commit the task**

```powershell
git add src/edge_perception/runner.py src/edge_perception/compare.py src/edge_perception/cli.py tests
git commit -m "feat: run and compare Eyes and Stopwatch checkpoints"
```

---

### Task 7: Reference-laptop model and pipeline proof

**Files:**
- Modify: `README.md`
- Create: `docs/checkpoints/eyes-and-stopwatch.md`
- Create locally but do not commit: `fixtures/private/checkpoint-01.mp4`
- Create locally but do not commit: `docs/checkpoints/artifacts/cpu-*`
- Create locally but do not commit: `docs/checkpoints/artifacts/cuda-*`

**Interfaces:**
- Consumes: completed CLI, pinned model, approved fixture recipe, and reference laptop.
- Produces: an evidence-backed checkpoint report plus ignored raw run artifacts.

- [ ] **Step 1: Resolve the CUDA environment and record capability evidence**

```powershell
.\.tools\uv.exe sync --extra cu128 --group dev
.\.tools\uv.exe run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device')"
```

Expected on the reference laptop: Torch imports, CUDA is true, and the RTX 500 Ada Laptop GPU is named. If CUDA is false, preserve the command output in the checkpoint report and execute the CPU path while diagnosing the environment; do not label CPU results as GPU results.

- [ ] **Step 2: Run the opt-in model-loading smoke tests on CPU and CUDA**

```powershell
$env:RUN_MODEL_TESTS = "1"
$env:MODEL_TEST_DEVICE = "cpu"
.\.tools\uv.exe run pytest tests/test_dfine_integration.py -m model -q
$env:MODEL_TEST_DEVICE = "cuda"
.\.tools\uv.exe run pytest tests/test_dfine_integration.py -m model -q
```

Expected: both supported device paths load the pinned revision, report a 64-character weight digest, and return backend-neutral results.

- [ ] **Step 3: Record and freeze the private checkpoint fixture**

Use the approved recipe: 20–30 seconds, 3840×2160, 30 FPS, fixed EMEET camera, no audio or private information, supported everyday object classes distributed across the frame, and at least one entering/leaving object. Save it to `fixtures/private/checkpoint-01.mp4`. Run the CLI once with `--max-frames 1` and confirm the manifest reports 3840×2160 before the measured runs.

- [ ] **Step 4: Run two deterministic CUDA passes**

Use the same explicit regions and output configuration twice:

```powershell
.\.tools\uv.exe run edge-perception run fixtures/private/checkpoint-01.mp4 --output docs/checkpoints/artifacts/cuda-a --crop upper-right:1920,0,1920,1080 --crop lower-left:0,1080,1920,1080 --device cuda --threshold 0.3 --warmup-runs 2 --max-frames 60 --annotate-every 10
.\.tools\uv.exe run edge-perception run fixtures/private/checkpoint-01.mp4 --output docs/checkpoints/artifacts/cuda-b --crop upper-right:1920,0,1920,1080 --crop lower-left:0,1080,1920,1080 --device cuda --threshold 0.3 --warmup-runs 2 --max-frames 60 --annotate-every 10
.\.tools\uv.exe run edge-perception compare docs/checkpoints/artifacts/cuda-a docs/checkpoints/artifacts/cuda-b --box-atol 0.01 --score-atol 0.0001
```

Expected: comparison exits zero and emits `equivalent: true`.

- [ ] **Step 5: Run a bounded CPU reference pass**

```powershell
.\.tools\uv.exe sync --extra cpu --group dev
.\.tools\uv.exe run edge-perception run fixtures/private/checkpoint-01.mp4 --output docs/checkpoints/artifacts/cpu-a --crop upper-right:1920,0,1920,1080 --crop lower-left:0,1080,1920,1080 --device cpu --threshold 0.3 --warmup-runs 1 --max-frames 10 --annotate-every 5
```

Expected: successful CPU run with honest CPU manifest; no CUDA or NVML requirement.

- [ ] **Step 6: Write the checkpoint report from generated manifests**

Document exact commands, Git commit, input SHA-256, model ID/revision/weight SHA-256, Python/dependency versions, driver/CUDA/device details, warm-up policy, region geometry, threshold, frame counts, p50/p95/p99 operation and frame latency, peak RSS, peak Torch VRAM, NVML availability, repeatability result, annotated examples, observed limitations, and whether each approved success criterion passed. Clearly label this an engineering checkpoint rather than a policy-quality benchmark.

- [ ] **Step 7: Update README quick start and verify a clean package build**

```powershell
.\.tools\uv.exe run pytest -m "not model" -q
.\.tools\uv.exe run ruff check src tests
.\.tools\uv.exe run mypy src
.\.tools\uv.exe build
.\.tools\uv.exe run python -m zipfile -l dist\adaptive_edge_perception-0.1.0-py3-none-any.whl
git status --short
```

Expected: tests/lint/types/build succeed; the wheel contains no model/video/run artifacts; ignored raw evidence is absent from `git status`.

- [ ] **Step 8: Commit the verified checkpoint**

```powershell
git add README.md docs/checkpoints/eyes-and-stopwatch.md src tests pyproject.toml uv.lock
git commit -m "docs: publish Eyes and Stopwatch checkpoint"
```

---

## Self-Review Record

- **Spec coverage:** Tasks 1–6 cover pinned upstream loading, chronological decode, full-frame and crop inference, source-coordinate mapping, three record families, latency percentiles, RAM/VRAM capability reporting, structured results, annotations, and semantic repeatability. Task 7 produces CPU/GPU evidence and documentation.
- **Scope control:** No task introduces Gymnasium, RL, strategies, tracking, live capture, a dashboard, a registry, a second detector, ONNX, TensorRT, or accuracy claims.
- **Placeholder scan:** The plan contains no `TODO`, `TBD`, unspecified implementation steps, or references to undefined interfaces.
- **Type consistency:** `Detector.predict` always accepts `Sequence[np.ndarray]` and returns `BatchPrediction`; adapter outputs remain input-image-local until `map_detection_to_source` runs once; runner and comparison inputs match declared signatures.
- **Scientific honesty:** Warm-up, optional sensors, rendering time, CPU/GPU identity, cached artifacts, and fixture limitations are explicit. This checkpoint validates engineering behavior, not learned-policy quality or generalization.
