# Architecture Review Artifact Implementation Plan

## Goal

Create a canonical Markdown architecture review and a polished, self-contained HTML companion for Adaptive Edge Perception. The pair must explain what the repository proves today, the target research architecture, the scientific evaluation boundary, the Windows/WSL/Linux support position, and the smallest credible path to a mildly shippable v0.1.

## Global Constraints

- Work only in the `docs/architecture-review` branch and its isolated worktree.
- The Markdown file is canonical. The HTML companion must agree with it on every architectural claim, metric, risk, and recommendation.
- Produce exactly two tracked deliverables: `docs/architecture-review.md` and `docs/architecture-review.html`.
- The HTML must be one self-contained file with inline CSS and, only if necessary, minimal inline JavaScript. It must open directly from disk without a server, package manager, build command, CDN, remote font, or other runtime dependency.
- This HTML is repository documentation, not the product GUI. Do not introduce a browser-based application or change the native PySide6 product direction.
- Preserve scientific honesty. Clearly separate current evidence, target architecture, recommendations, and unimplemented future work.
- Treat recorded video as the deterministic replay/training/evaluation lane and live camera as the deployment/transfer lane. A live trial must record its source stream, actions, predictions, and timings for offline scoring; it must never fabricate online reward.
- Gymnasium is the sole proposed public environment contract. Stable-Baselines3, RLlib, TorchRL, Minari, EnvPool, and PettingZoo are comparisons or future adapters, not core dependencies.
- The scheduling problem is single-agent unless independently acting cameras or devices are introduced. Detector, tracker, and strategy implementations are components, not agents.
- Detector implementations and model weights are not packaged. The detector adapter remains neutral; benchmark evidence may use industry-standard external detectors.
- Use only evidence verified in the current branch or directly cited primary sources. Do not invent files, commands, performance numbers, APIs, or support claims.
- Include the verified non-model baseline `492 passed, 1 skipped, 1 deselected`; identify it as test evidence, not model-accuracy evidence.
- Include the existing real CUDA reference evidence: 10 frames, 30 inferences, 259 detection records, 5 annotations, complete-frame p50 259.507 ms, and semantic CPU/GPU comparison with 259 detections on each side and zero mismatches. Label this as a checkpoint/reference run, not a generalized performance result.
- Present the two code-audit P0 findings prominently: ordinary file inputs are not pinned to the exact bytes later inferred, and the shared completed-run reader can accept missing primary data streams.
- The platform matrix must say: Windows 11 native is the Tier-1/reference live-device path; Ubuntu under WSL2 is a developer-preview Linux user-space/headless/CUDA-smoke path with no camera or native-Linux claim; Ubuntu 24.04 native is a CI portability target until physical GUI/GPU/camera validation exists.
- Use portable repository-relative links in Markdown. Use normal HTTPS links for external primary sources.
- Cite claims near the text they support. Prefer official framework/platform documentation and original papers or repositories.
- Visual direction for HTML: true-white background; near-black ink; cool-slate rules; restrained cobalt for navigation/architecture; muted safety orange only for risks; sober sans-serif plus compact monospace; fixed narrow desktop navigation; open page bands, ruled tables, and technical diagrams; almost no rounded corners; no shadows, gradients, glows, stock imagery, fake charts, marketing CTA, decorative badge, or card-grid layout.
- HTML must be semantic, keyboard navigable, responsive down to a 390px viewport, printable, and legible with `prefers-reduced-motion` and forced-colors/high-contrast settings.
- Do not modify runtime code, tests, packaging configuration, or the open hardening PR.

## Task 1: Author the canonical Markdown architecture review

Create `docs/architecture-review.md` as the complete evidence-backed source of truth.

Required structure:

1. Executive verdict and ten-year-old explanation.
2. What the current checkpoint proves and does not prove.
3. Clean-install and first-run workflows for Windows native and WSL2 headless, including the recommendation to keep WSL workloads on the Linux filesystem rather than `/mnt/c`.
4. Current repository/system map with real module links and a Mermaid diagram.
5. Recorded replay/training workflow and live deployment/transfer workflow.
6. Target system boundaries: `PreparedSource`/`FrameSource`, `Strategy`, `DetectorAdapter`, `Tracker`, `InspectionRuntime`/`PerceptionEngine`, `ActionCatalog`, `BudgetLedger`, replay backend, live backend, `RewardOracle`, evaluator, and canonical artifact writer/reader.
7. Gymnasium research contract, simple observation/action/info schemas, termination semantics, train/evaluation separation, counterfactual-cache requirement, framework comparison, and conventions to defer.
8. Scientific proof and benchmark plan, including baselines, held-out splits, quality-versus-compute results, live time-to-detection protocol, latency distributions, dropped/stale frames, budget violations, and uncertainty across seeds.
9. Platform support matrix and evidence required to promote native Linux support.
10. Architecture review: strengths, P0 blockers, P1 risks, P2/deferred work, and explicit current/target distinction.
11. `NOW / NEXT / THEN / LATER` roadmap.
12. Review questions and decision log.
13. Primary-source references, including Gymnasium, SB3, RLlib, TorchRL, Minari, EnvPool, PettingZoo, Microsoft WSL documentation, NVIDIA CUDA-on-WSL documentation, Qt platform/media documentation, and overlapping work such as SmartTBD, DorT, and Chanakya.

Acceptance criteria:

- Every present-tense code claim has a portable link to the relevant repository file or test.
- Every external convention or platform claim has a nearby primary-source link.
- The document explicitly states that detector-versus-tracker scheduling is not novel by itself and identifies the open-source contribution as the reproducible environment/runtime/evaluation boundary.
- The document is readable by William as a learning artifact: introduce each technical boundary in plain English before using its formal name.
- No tracked file other than `docs/architecture-review.md` is changed in this task.
- Run a link/path sanity check and report the result; run the full non-model test suite once before committing.

## Task 2: Implement the self-contained HTML companion

Create `docs/architecture-review.html` from the canonical Markdown and the approved visual references.

Visual references, in order:

- `/mnt/c/Users/William/.codex/generated_images/019fd313-d455-7340-9f91-1f5d0d11a17a/exec-4459d492-d75e-4e16-9aa0-3ba48629956b.png`
- `/mnt/c/Users/William/.codex/generated_images/019fd313-d455-7340-9f91-1f5d0d11a17a/exec-ec5d29d9-2e98-4b9c-860f-3ea3646309f6.png`
- `/mnt/c/Users/William/.codex/generated_images/019fd313-d455-7340-9f91-1f5d0d11a17a/exec-9b6c4b9e-a920-4e43-8dac-5a3e46c7bf61.png`
- `/mnt/c/Users/William/.codex/generated_images/019fd313-d455-7340-9f91-1f5d0d11a17a/exec-57be4454-a933-4100-9edb-f53c796a239a.png`

Required page sections and navigation anchors:

- Overview
- Evidence
- Install
- Architecture
- Workflows
- RL contract
- Benchmarks
- Platform support
- Review
- Roadmap
- Decisions and references

Required behavior and composition:

- Preserve the approved editorial engineering-manual visual system and section rhythm.
- Use a fixed left rail on wide screens and a compact sticky horizontal rail on narrow screens.
- Recreate diagrams with semantic HTML/CSS and production-quality inline SVG only where a connector cannot be expressed clearly in layout. Do not embed the concept screenshots as UI.
- Include copyable commands as real selectable text, but do not add an inert “copy” control.
- All external links open normally and show their destination host in the link text or adjacent copy.
- Provide visible focus styles, skip navigation, meaningful headings, table captions, diagram descriptions, and print styles.
- Any inline JavaScript must be optional enhancement only; all content and navigation must work with JavaScript disabled.
- Do not add claims or decisions that are absent from the Markdown source.
- No tracked file other than `docs/architecture-review.html` is changed in this task.

Acceptance criteria:

- Opening the file with a `file:///` URL renders the complete document without network requests or console errors.
- At 1536×1024, the first viewport and the Architecture, RL contract, and Review sections closely match their corresponding approved concepts in hierarchy, spacing, palette, typography, rules, and container model.
- At 390×844, there is no horizontal page overflow, clipped text, unusable table, or obscured anchor target.
- Print preview is readable and excludes navigation chrome.
- All required Markdown headings, evidence values, P0 findings, support statuses, roadmap stages, and primary links appear in the HTML.
- Run an automated local HTML/link/accessibility sanity check plus the full non-model test suite once before committing.
