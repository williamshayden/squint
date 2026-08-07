"""Presentation-only workflow states for the native research client."""

from enum import StrEnum


class SourceState(StrEnum):
    """Whether a validated source reference is active."""

    NO_SOURCE = "No source"
    READY = "Ready"


class AcquisitionState(StrEnum):
    """State of the optional camera acquisition operation."""

    IDLE = "Idle"
    PREVIEWING = "Previewing"
    RECORDING = "Recording"
    FINALIZING = "Finalizing"
    FINALIZED = "Finalized"
    FAILED = "Failed"


class RunState(StrEnum):
    """Presentation state of the current or most recent run."""

    NOT_STARTED = "Not started"
    RUNNING = "Running"
    CANCELLING = "Cancelling"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
