# Adaptive Edge Perception

Adaptive Edge Perception is a CLI-first, detector-neutral research tool for running chronological object-detection checkpoints on local video. The optional native GUI uses the same capture, run-configuration, runner, artifact, and result-projection contracts as the CLI.

## Install from a repository checkout

Python 3.12 is required. No package-index release or ownership of the `adaptive-edge-perception` package-index name has been verified, so `pip install adaptive-edge-perception` is not a current installation path. Use an installed `uv` from a clean repository checkout, or install the local checkout into an activated virtual environment with pip:

```text
uv sync

python -m venv .venv-base
# Activate .venv-base using the command for your shell, then:
python -m pip install -e .
python -m pip install -e ".[camera]"
python -m pip install -e ".[gui]"
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

After selecting a detector backend, verify what Python actually imported:

```text
# After `uv sync --extra cpu`
uv run python -c "import torch, transformers, huggingface_hub, safetensors; print(torch.__version__, torch.version.cuda); assert torch.version.cuda is None"

# After `uv sync --extra cu128`
uv run python -c "import torch, transformers, huggingface_hub, safetensors; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.version.cuda == '12.8' and torch.cuda.is_available()"
```

### Real D-FINE runtime with pip

pip does not read `[tool.uv.sources]`, so `python -m pip install -e ".[cpu]"` or `".[cu128]"` alone does not select the intended PyTorch wheel index. Use a separate fresh virtual environment for each backend, install the exact Torch build from the matching PyTorch index first, install the local extra second, and verify the imported backend. Do not convert an existing CPU environment into a CUDA environment or vice versa.

```text
# CPU
python -m venv .venv-cpu
# Activate .venv-cpu using the command for your shell, then:
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[cpu]"
python -c "import torch, transformers, huggingface_hub, safetensors; print(torch.__version__, torch.version.cuda); assert torch.version.cuda is None"

# CUDA 12.8
python -m venv .venv-cu128
# Activate .venv-cu128 using the command for your shell, then:
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[cu128]"
python -c "import torch, transformers, huggingface_hub, safetensors; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.version.cuda == '12.8' and torch.cuda.is_available()"
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

Diagnostic annotations are optional. `--annotate-every N` writes one lossless PNG every `N` frames; `--annotate-every 0` disables them without disabling detections, telemetry, inspection, or comparison.

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

This project intentionally has no provider downloader. If a public video is useful for research, materialize it outside the tool only when you may lawfully obtain and use the content, then pass the stable local file to `edge-perception run`. Keep a provenance record beside the run containing the provider and source URL, title or asset ID, license or other permission basis, retrieval date, original filename, and SHA-256 digest. Do not redistribute or commit the media unless its terms permit that. Network access and public-video evidence are optional.

## Development and release checks

From a clean checkout, use the installed `uv`. The complete non-model test suite includes native GUI tests, so synchronize the `gui` extra first; detector-runtime extras and model files are not required for this gate.

```text
uv sync --extra gui
uv run pytest -m "not model"
uv run ruff check src tests scripts
uv run mypy src
uv lock --check
uv build
uv run python scripts/verify_release_archives.py \
  dist/adaptive_edge_perception-0.1.0-py3-none-any.whl \
  dist/adaptive_edge_perception-0.1.0.tar.gz
```

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

The package wheel contains Python modules and package metadata only—no videos, run artifacts, annotations, private captures, model weights, tests, Qt binaries, or SDD working files. The source archive is independently allowlisted to the license, project metadata, source, tests, release verifier, and checkpoint documentation.

The verified checkpoint is the generated 200×100 source-agnostic workflow plus separately identified optional hardware smoke evidence. Earlier strict-4K, two-pass CUDA, and real-model CPU criteria are not proven by this checkpoint; they are superseded as completion gates and remain planned, opt-in hardware validation.

See [the Eyes and Stopwatch checkpoint](docs/checkpoints/eyes-and-stopwatch.md) for the reproducible offline proof and its exact verification evidence.
