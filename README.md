# Adaptive Edge Perception

Adaptive Edge Perception is a CLI-first, detector-neutral research tool for running chronological object-detection checkpoints on local video. The optional native GUI uses the same capture, run-configuration, runner, artifact, and result-projection contracts as the CLI.

## Install

Python 3.12 is required. Install only the surface you need:

```text
pip install adaptive-edge-perception
pip install adaptive-edge-perception[camera]
pip install adaptive-edge-perception[gui]
```

The base install supports local-video `run`, `inspect`, and `compare` workflows without Qt. The `camera` extra adds headless camera discovery and timed capture. The `gui` extra adds the native Qt Widgets application and camera acquisition.

Detector model files and weights are not bundled. Running the D-FINE adapter requires a compatible detector runtime and externally materialized model weights; the optional `cpu` and `cu128` extras provide the corresponding pinned Python runtimes. Model and CUDA validation are additive evidence, not requirements for the offline acceptance proof.

Windows and Linux are supported. macOS has not been validated.

## Local-video workflow

Start with a local video and two source-frame ROIs:

```text
edge-perception run videos/reference.mp4 \
  --output runs/reference-a \
  --detector dfine-nano-coco \
  --device cpu \
  --max-frames 3 \
  --warmup-runs 0 \
  --annotate-every 1 \
  --crop left:0,0,100,100 \
  --crop right:100,0,100,100

edge-perception inspect runs/reference-a

edge-perception run videos/reference.mp4 \
  --output runs/reference-b \
  --detector dfine-nano-coco \
  --device cpu \
  --max-frames 3 \
  --warmup-runs 0 \
  --annotate-every 1 \
  --crop left:0,0,100,100 \
  --crop right:100,0,100,100

edge-perception compare runs/reference-a runs/reference-b
```

The output directory must be empty. The full frame is always evaluated before the declared ROIs. `inspect` renders the canonical run projection; `compare` checks semantic detections and the reproducibility-relevant manifest fields while ignoring run IDs, timing, and hardware telemetry.

On the first `Ctrl+C`, a direct run cooperatively finalizes canonical `cancelled` artifacts and exits with status 130. A second interrupt uses normal immediate interrupt behavior.

## Camera acquisition

Camera support is optional and does not open a window:

```text
edge-perception camera list
edge-perception camera capture \
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
edge-perception gui
edge-perception gui --run runs/reference-a
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
