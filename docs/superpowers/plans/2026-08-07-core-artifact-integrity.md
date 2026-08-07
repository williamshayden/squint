# Core Artifact Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every run directory an exclusively owned, recoverable experiment record and make `compare` reject runs that differ in detector weights, terminal coverage, or frame-region inference schedule.

**Architecture:** `run_checkpoint` remains the sole execution authority and the `0.1.0` artifact schema remains unchanged. `RunOutputs` owns atomic directory claiming, buffered completed-frame publication, and terminal summary ordering; a model-free preflight is shared by direct CLI and worker paths; comparison validates manifests, summaries, inference schedules, and detections while continuing to ignore hardware and timing noise.

**Tech Stack:** Python 3.12, pathlib/os JSONL artifacts, PyAV video decoding, NumPy/Pillow annotations, pytest 9, Ruff, mypy, uv.

## Global Constraints

- Preserve schema version `0.1.0`, canonical filenames, detector default, batch size one, and terminal values `complete`, `failed`, and `cancelled`.
- A run output may be absent or an existing empty directory. Exactly one process may claim it; a loser must fail without truncating, replacing, or mixing any winner artifact.
- `summary.json` is the authoritative terminal commit marker: every JSONL stream is flushed, fsynced, and closed before its atomic, durable publication. Normal successful finalization then removes the ownership marker and flushes its parent directory metadata where supported.
- No owned temporary file remains after terminal finalization. A crash after summary publication may leave a stale ownership marker; `summary.json` wins. An interrupted nonterminal run may retain an ownership marker to make incompleteness explicit.
- `frames_processed`, `inference_count`, and `annotated_frame_count` count only fully committed frames. A detector/ROI/annotation failure cannot publish a partial frame's rows or final PNG.
- Optional telemetry and detector peak-memory collection may report `null`; they must never replace the primary detector/runner exception or prevent a failed summary.
- `compare` remains timing-, host-, and device-neutral, but detector adapter/model/revision/weights, source/config, terminal status/counts, and every frame-region inference are semantic.
- Preflight is detector-free and shared by CLI and worker. Invalid source, ROI, output, capture provenance, or config must fail before detector loading.
- Existing-file sources remain validated references, not physically immutable files. A `CaptureResult` is stricter: its path and SHA-256 must agree with the bytes used by the run and with manifest source provenance.
- Default verification remains offline, model-free, camera-free, GPU-free, and browser-free.
- Every production change follows red-green-refactor and receives a fresh task review before the next task.

---

## File and Responsibility Map

- `src/edge_perception/outputs.py` — exclusive run ownership, atomic JSON/PNG publication, buffered frame commit, durable stream closure, and final summary publication.
- `src/edge_perception/runner.py` — completed-frame staging, nonfatal optional telemetry, capture-source consistency, and runner preflight defense in depth.
- `src/edge_perception/contracts.py` — producer-side rejection of empty region IDs.
- `src/edge_perception/preflight.py` — detector-free source/output/ROI/capture validation shared by CLI and worker.
- `src/edge_perception/cli.py` — preflight and controlled malformed-config errors before detector loading.
- `src/edge_perception/worker.py` — preflight before the isolated worker loads a detector.
- `src/edge_perception/run_view.py` — capture/source path and SHA-256 consistency on read.
- `src/edge_perception/compare.py` — validated terminal/schedule/detector semantic comparison.
- `tests/test_outputs.py`, `tests/test_runner.py`, `tests/test_contracts.py` — ownership, durability, complete-frame, annotation, and telemetry regressions.
- `tests/test_cli.py`, `tests/test_worker.py`, `tests/test_inspection.py` — preflight ordering and controlled user-facing failures.
- `tests/test_compare.py` — terminal coverage, zero-detection schedule, weights identity, and malformed-artifact comparison tests.

---

### Task 1: Make Run Artifacts Exclusively Owned and Frame-Complete

**Files:**
- Modify: `src/edge_perception/outputs.py`
- Modify: `src/edge_perception/runner.py`
- Modify: `src/edge_perception/contracts.py`
- Modify: `tests/test_outputs.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_contracts.py`

**Interfaces:**
- Consumes: `RunOutputs(run_dir, run_id, manifest)`, existing write methods, `run_checkpoint`, and `Detector.peak_device_memory_bytes()`.
- Produces: exclusive output ownership inside `RunOutputs`; one buffered completed-frame commit path; atomic annotation publication; terminal `write_summary()` semantics that close durable streams before publishing the summary. No schema or canonical filename changes.

- [ ] **Step 1: Write failing ownership tests**

Add real-filesystem tests that open one `RunOutputs` over an absent directory and over a pre-created empty directory, then attempt a second owner. Assert the second constructor raises a controlled `ValueError` containing `output directory is already owned`, while the first owner's manifest/run ID and open streams remain intact. Add a race probe using two threads released by one barrier; assert exactly one constructor succeeds and the loser never truncates the winner.

The production mutation these tests catch is replacing exclusive creation with `exist_ok=True`, `open("w")`, or a check-then-create sequence.

- [ ] **Step 2: Verify ownership RED**

Run:

```powershell
./.tools/uv.exe run --offline --frozen --no-sync pytest tests/test_outputs.py -k "owner or race" -q
```

Expected: FAIL because both constructors currently reuse the directory and overwrite fixed canonical paths.

- [ ] **Step 3: Implement an exclusive run claim**

Use one fixed private ownership filename and exclusive creation (`open("x")`) after ensuring the parent exists. Re-check that the claimed directory contains only the owned marker before creating artifacts. Open canonical JSONL files with exclusive mode. Temporary JSON/PNG names must include the run ID (or another owned unique suffix), not a process-global fixed `.tmp` name.

If claim/setup fails, preserve all pre-existing paths and remove only files created by that failed constructor. A completed run removes its marker after durably publishing terminal `summary.json`; a crash in that interval may leave a stale marker, and the summary remains authoritative. A nonterminal interrupted run may retain its marker. Do not add the marker to the public artifact schema.

- [ ] **Step 4: Write failing complete-frame and atomic-annotation tests**

Add a detector that succeeds for full-frame inference and fails on the first crop. Assert the raised primary exception remains `RuntimeError("crop inference failed")`, `summary.json` records `failed`, and all of these remain zero/empty: `frames_processed`, `inference_count`, `annotated_frame_count`, `inferences.jsonl`, `detections.jsonl`, and `annotated/*.png`.

Patch Pillow save so it writes bytes to its requested path and raises. Assert no canonical PNG and no owned temporary PNG remains. Add `Region("", 0, 0, 1, 1)` and assert `ValueError("region_id must be a non-empty string")`.

- [ ] **Step 5: Verify frame/annotation RED**

Run:

```powershell
./.tools/uv.exe run --offline --frozen --no-sync pytest tests/test_runner.py tests/test_outputs.py tests/test_contracts.py -k "partial or annotation or region_id" -q
```

Expected: FAIL because rows are written per region, PNGs publish directly, and `Region` accepts an empty ID.

- [ ] **Step 6: Buffer and commit only complete frames**

Stage every inference row and detection batch in memory while all execution regions run. Pre-serialize staged rows before mutating streams. Publish staged rows and a scheduled annotation only after all detector, mapping, and annotation encoding work for that frame succeeds. If frame publication raises, restore every stream to its pre-frame position and remove only that frame's owned temporary/final annotation. Increment all counters and latency lists only after commit succeeds.

Render annotations to a uniquely owned sibling temporary and atomically replace `annotated/<frame>.png` only after successful PNG encoding. Cleanup is limited to that owned temporary.

- [ ] **Step 7: Write failing finalization/primary-error tests**

Add a detector whose `predict` raises `RuntimeError("primary inference failure")` and whose `peak_device_memory_bytes` raises `RuntimeError("secondary peak failure")`. Assert `run_checkpoint` raises the primary error, failed `summary.json.error` names the primary error, and `detector_peak_device_memory_bytes` is `null`.

Wrap summary publication and assert every JSONL stream is closed and all expected hardware rows are readable before `summary.json` appears. The production mutation caught is writing summary while buffered streams remain open.

- [ ] **Step 8: Implement durable terminal finalization**

Treat peak-memory collection as optional: catch ordinary `Exception`, record `None`, and preserve any existing failure. Append telemetry, flush and `os.fsync` each JSONL stream, close them, then atomically and durably publish `summary.json`. Remove the ownership marker afterward and flush its parent directory metadata where supported. `close()` remains idempotent. Do not catch `KeyboardInterrupt` or `SystemExit` as optional telemetry errors.

- [ ] **Step 9: Verify Task 1**

Run:

```powershell
./.tools/uv.exe run --offline --frozen --no-sync pytest tests/test_outputs.py tests/test_runner.py tests/test_contracts.py tests/test_cli_workflow_acceptance.py -q -p no:cacheprovider
./.tools/uv.exe run --offline --frozen --no-sync ruff check src tests scripts
./.tools/uv.exe run --offline --frozen --no-sync mypy src
```

Expected: focused tests and acceptance pass; Ruff and mypy succeed.

- [ ] **Step 10: Commit**

```powershell
git add src/edge_perception/outputs.py src/edge_perception/runner.py src/edge_perception/contracts.py tests/test_outputs.py tests/test_runner.py tests/test_contracts.py
git commit -m "fix: make run artifacts transactionally complete"
```

---

### Task 2: Preflight Before Model Load and Enforce Capture Provenance

**Files:**
- Create: `src/edge_perception/preflight.py`
- Modify: `src/edge_perception/cli.py`
- Modify: `src/edge_perception/worker.py`
- Modify: `src/edge_perception/runner.py`
- Modify: `src/edge_perception/run_view.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_worker.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_inspection.py`

**Interfaces:**
- Consumes: `RunConfig`, `iter_video`, `validate_output_directory`, source SHA-256, `CaptureResult`, canonical manifest/summary/inference/detection files, and the detector registry loader.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class RunPreflight:
    frame_width: int
    frame_height: int
    source_sha256: str

def preflight_run(config: RunConfig) -> RunPreflight: ...
```

- [ ] **Step 1: Write failing detector-free preflight ordering tests**

For direct CLI and worker paths, use a loader spy that raises if called. Cover: missing/deleted source, decoded source with an out-of-bounds integer ROI, nonempty output, and capture SHA-256 mismatch. Assert each path returns/reports a controlled error and the detector loader call list remains empty.

Add a valid JSON config with `"regions": {}`. Direct CLI must return exit 2 with one `error:` line and no traceback; it must not broaden GUI/internal `ImportError` handling.

- [ ] **Step 2: Verify preflight RED**

Run:

```powershell
./.tools/uv.exe run --offline --frozen --no-sync pytest tests/test_cli.py tests/test_worker.py -k "preflight or out_of_bounds or capture_sha or regions_object" -q
```

Expected: FAIL because detector loading precedes decoded source/ROI validation and malformed config `TypeError` escapes direct CLI handling.

- [ ] **Step 3: Implement shared model-free preflight**

`preflight_run` must: require a regular input file; validate absent/empty output without claiming it; decode exactly the first frame and close the iterator on every path; reject empty video; validate each ROI against decoded dimensions; hash the source once; and, when capture provenance exists, require resolved capture path equality and exact SHA-256 equality. Return the literal `RunPreflight` values above.

Call it after config loading and before `load_detector` in direct CLI and worker. Keep runner defense in depth by preflighting at `run_checkpoint` entry before output claim. Convert only expected config-boundary `TypeError`/`ValueError` into controlled CLI errors; unrelated import/programming errors still propagate.

- [ ] **Step 4: Write failing run-view provenance tests**

Create a canonical manifest whose `source_video.sha256` differs from nested capture SHA-256 and one whose resolved source/capture paths differ. Assert `load_run_view` rejects each with a field-specific `ValueError`. A consistent capture continues to load.

- [ ] **Step 5: Implement provenance consistency on read**

Validate source SHA-256 format, resolved source path, and parsed `CaptureResult` path/hash before constructing `RunViewData`. Do not require the original media file to still exist and do not re-hash it during inspection.

- [ ] **Step 6: Verify Task 2**

Run:

```powershell
./.tools/uv.exe run --offline --frozen --no-sync pytest tests/test_cli.py tests/test_worker.py tests/test_runner.py tests/test_inspection.py tests/test_cli_workflow_acceptance.py -q -p no:cacheprovider
./.tools/uv.exe run --offline --frozen --no-sync ruff check src tests scripts
./.tools/uv.exe run --offline --frozen --no-sync mypy src
```

Expected: focused tests and acceptance pass; Ruff and mypy succeed; detector loaders remain untouched on invalid preflight inputs.

- [ ] **Step 7: Commit**

```powershell
git add src/edge_perception/preflight.py src/edge_perception/cli.py src/edge_perception/worker.py src/edge_perception/runner.py src/edge_perception/run_view.py tests/test_cli.py tests/test_worker.py tests/test_runner.py tests/test_inspection.py
git commit -m "fix: preflight runs before model loading"
```

---

### Task 3: Compare Complete Experiment Coverage

**Files:**
- Modify: `src/edge_perception/compare.py`
- Modify: `tests/test_compare.py`
- Modify: `tests/test_cli_workflow_acceptance.py`

**Interfaces:**
- Consumes: canonical `manifest.json`, `summary.json`, `inferences.jsonl`, and `detections.jsonl` from Task 1's terminal artifact contract.
- Produces: the existing `compare_runs(left, right, *, box_atol=0.01, score_atol=1e-4) -> dict[str, object]` report keys, with manifest, terminal-summary, inference-schedule, and detection mismatches all contributing to `mismatch_count` and deterministic `first_mismatch`.

- [ ] **Step 1: Write failing comparison coverage tests**

Extend fixtures to write `summary.json` and `inferences.jsonl`. Add literal tests for:

- same detections but `complete` versus `cancelled` status;
- same status but different `frames_processed` or `inference_count`;
- a missing frame-region inference with zero detections;
- same model ID/revision but different `weights_sha256`;
- different detector `adapter`;
- identical semantic experiment with different run IDs, host, hardware, latency, detector device/backend/dtype, and inference timing.

The first five must report non-equivalence with deterministic `first_mismatch`; the last must remain equivalent.

- [ ] **Step 2: Verify comparison RED**

Run:

```powershell
./.tools/uv.exe run --offline --frozen --no-sync pytest tests/test_compare.py -q
```

Expected: FAIL because comparison currently reads only manifest/detections and omits adapter/weights.

- [ ] **Step 3: Compare terminal coverage and inference schedules**

Validate both summaries and inference streams as JSON objects/JSONL with matching schema/run IDs. Compare detector `adapter`, `model_id`, `revision`, and 64-hex `weights_sha256`; source SHA-256, threshold, and ordered ROIs; summary `status`, `frames_processed`, and `inference_count`; and the sorted set of frame-region inference keys `(frame_index, frame_id, region_id, region, input_shape, source_time_ms)`. Include zero-detection inferences because schedule comparison is independent of detection rows.

Continue ignoring run IDs as values, host/hardware, all latency/timing fields, detector device/backend/backend version/dtype, and row order. Preserve box/score tolerances and all existing report keys.

- [ ] **Step 4: Strengthen acceptance without changing the workflow**

In `tests/test_cli_workflow_acceptance.py`, parse both runs' summary/inference streams and assert the repeated run schedules contain the same nine `(frame_index, region_id)` pairs before invoking the real compare command. Keep the fake detector only at the external loader seam.

- [ ] **Step 5: Verify Task 3 and full branch**

Run:

```powershell
./.tools/uv.exe run --offline --frozen --no-sync pytest tests/test_compare.py tests/test_cli_workflow_acceptance.py -q -p no:cacheprovider
QT_QPA_PLATFORM=offscreen ./.tools/uv.exe run --offline --frozen --no-sync pytest -m "not model" -q -p no:cacheprovider
./.tools/uv.exe run --offline --frozen --no-sync ruff check src tests scripts
./.tools/uv.exe run --offline --frozen --no-sync mypy src
./.tools/uv.exe lock --check --offline
```

Expected: all model-free tests pass, one model test remains deselected, Ruff/mypy/lock succeed, and acceptance reports equivalent repeated runs with complete schedules.

- [ ] **Step 6: Commit**

```powershell
git add src/edge_perception/compare.py tests/test_compare.py tests/test_cli_workflow_acceptance.py
git commit -m "fix: compare complete experiment coverage"
```

---

## Plan Self-Review Checklist

- Every whole-branch core finding maps to one task: output race, partial frame, annotation atomicity, final-summary ordering, peak-memory masking, empty region ID (Task 1); model-free preflight, malformed config, and capture consistency (Task 2); detector weights, terminal coverage, and zero-detection inference schedules (Task 3).
- No task changes the schema, filenames, detector default, device-neutral comparison intent, or public CLI command shape.
- Output ownership, preflight/provenance, and scientific comparison are separate reviewable boundaries; Tasks 2 and 3 consume Task 1's truthful counters and terminal marker.
- Each production mutation has a real behavioral test that is run RED before implementation.
- Default gates remain offline and require neither camera nor model runtime.
