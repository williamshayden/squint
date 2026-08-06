# Adaptive Edge Perception

> Working title and early-stage repository.

An open-source, detector-neutral toolkit for testing how a constrained edge device should spend limited object-detection compute on high-resolution video.

The first checkpoint is **Eyes and Stopwatch**: load a pinned upstream D-FINE-N model, process a deterministic 4K video chronologically through whole-frame and explicit crop paths, map every detection into source-frame coordinates, and publish reproducible latency and memory measurements.

The first checkpoint deliberately excludes Gymnasium, reinforcement learning, tracking, live cameras, dashboards, and additional detector integrations.

See [edge-perception-project-brief.md](edge-perception-project-brief.md) for the approved system boundary and decision log.

