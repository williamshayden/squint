# Adaptive Edge Perception

Adaptive Edge Perception is a CLI-first, detector-neutral research tool for running chronological object-detection checkpoints on local video. The optional native GUI uses the same capture, run-configuration, runner, artifact, and result-projection contracts as the CLI.

## Install from a repository checkout

Python 3.12 is required. No package-index release or ownership of the `adaptive-edge-perception` package-index name has been verified, so `pip install adaptive-edge-perception` is not a current installation path. From the repository root, install the local checkout with pip:

```text
python -m pip install .
python -m pip install ".[camera]"
python -m pip install ".[gui]"
```

The base dependencies support `inspect`, `compare`, and non-model CLI/library surfaces without Qt. They do not include Torch, Transformers, Safetensors, a detector model, or model weights. D-FINE is currently the only registered detector, so a real `run` requires one of the detector-runtime procedures below plus access to the pinned model files.

### Real D-FINE runtime with uv

uv reads this checkout's explicit PyTorch indexes and selects the matching Torch build. Choose CPU or CUDA 12.8; do not combine the conflicting `cpu` and `cu128` extras:

| Need | Source-checkout command |
| --- | --- |
| Base inspect/compare surfaces | `uv sync` |
| D-FINE on CPU | `uv sync --extra cpu` |
| D-FINE on CUDA 12.8 | `uv sync --extra cu128` |
| Headless camera only | `uv sync --extra camera` |
| CPU D-FINE plus camera | `uv sync --extra cpu --extra camera` |
| CPU D-FINE plus native GUI | `uv sync --extra cpu --extra gui` |
| CUDA D-FINE plus native GUI | `uv sync --extra cu128 --extra gui` |

Run checkout commands through that environment with `uv run edge-perception ...`. The `camera` extra adds headless camera discovery/capture; the `gui` extra adds the Qt Widgets application and installs the same compatible PySide6 runtime.

### Real D-FINE runtime with pip

pip does not read `[tool.uv.sources]`, so `python -m pip install ".[cpu]"` or `".[cu128]"` alone does not select the intended PyTorch wheel index. Install the exact Torch build from the matching PyTorch index first, then install the local extra:

```text
# CPU
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[cpu]"

# CUDA 12.8
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ".[cu128]"
```

After either runtime, install `".[camera]"` or `".[gui]"` as needed. Detector model files and weights are never bundled; the pinned model must already be cached or be materialized by the detector runtime under the user's network and licensing policy.

Windows was exercised by this checkpoint. Linux compatibility is supported by design but was not exercised in this checkpoint. macOS remains unvalidated.

## Local-video workflow

The examples below use the recommended uv environment. When the checkout was installed with pip into the active environment, omit the `uv run` prefix. Start with a local video and two source-frame ROIs:

```text
uv run edge-perception run videos/reference.mp4 \
  --output runs/reference-a \
  --detector dfine-nano-coco \
  --device cpu \
  --max-frames 3 \
  --warmup-runs 0 \
  --annotate-every 1 \
  --crop left:0,0,100,100 \
  --crop right:100,0,100,100

uv run edge-perception inspect runs/reference-a

uv run edge-perception run videos/reference.mp4 \
  --output runs/reference-b \
  --detector dfine-nano-coco \
  --device cpu \
  --max-frames 3 \
  --warmup-runs 0 \
  --annotate-every 1 \
  --crop left:0,0,100,100 \
  --crop right:100,0,100,100

uv run edge-perception compare runs/reference-a runs/reference-b
```

The output directory must be empty. The full frame is always evaluated before the declared ROIs. `inspect` renders the canonical run projection; `compare` checks semantic detections and the reproducibility-relevant manifest fields while ignoring run IDs, timing, and hardware telemetry.

On the first `Ctrl+C`, a direct run cooperatively finalizes canonical `cancelled` artifacts and exits with status 130. A second interrupt uses normal immediate interrupt behavior.

## Camera acquisition

Camera support is optional and does not open a window:

```text
uv run edge-perception camera list
uv run edge-perception camera capture \
  --device CAMERA_ID \
  --duration 5 \
  --output captures/reference.mp4 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --strict
```

Omit `--output` for a private, unique application-managed destination. Existing destinations are never overwritten. A finalized capture is validated, hashed, atomically published, and can be passed directly to `edge-perception run`.

Physical-camera evidence is optional and never gates the generated-video workflow.

## Native GUI

Launch the optional native application with:

```text
uv run edge-perception gui
uv run edge-perception gui --run runs/reference-a
```

The GUI can select an existing video or acquire a camera source, define source-pixel ROIs, publish a Run configuration, execute or cancel the shared runner through its isolated worker, and inspect canonical results.

## Public videos

This project intentionally has no provider downloader. If a public video is useful for research, materialize it outside the tool using only content you may lawfully obtain, retain any required provenance, and pass the resulting stable local file to `edge-perception run`. Network access and public-video evidence are optional.

## Canonical run artifacts

Every initialized run directory uses this contract:

```text
run/
├── manifest.json
├── summary.json
├── inferences.jsonl
├── detections.jsonl
├── hardware.jsonl
└── annotated/
    └── 000000.png
```

`manifest.json` records configuration, detector identity, source provenance, and timing definitions. `summary.json` records the terminal status and aggregate metrics. The JSONL files retain inference, source-space detection, and sampled hardware records. Scheduled diagnostic annotations are lossless PNGs. Failed and cancelled runs preserve the canonical files that could be finalized safely.

The package wheel contains Python modules and package metadata only—no videos, run artifacts, annotations, private captures, model weights, tests, Qt binaries, or SDD working files.

See [the Eyes and Stopwatch checkpoint](docs/checkpoints/eyes-and-stopwatch.md) for the reproducible offline proof and its exact verification evidence.
