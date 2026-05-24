"""Terminal progress reporting for VLM batch runs."""

from __future__ import annotations

import sys

# stderr is typically unbuffered; avoids conda run stdout capture hiding logs
_LOG = sys.stderr


def configure_stdio() -> None:
    """Line-buffer stdout/stderr so progress appears immediately in terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True, write_through=True)
            except (OSError, ValueError):
                pass


class RunProgress:
    """Numbered phase progress for end-to-end run_batch visibility."""

    def __init__(self, ref_id: str, total_phases: int = 6) -> None:
        self.ref_id = ref_id
        self.total_phases = total_phases
        self.phase_num = 0
        self.phase_label = ""

    def begin_phase(self, label: str) -> None:
        self.phase_num += 1
        self.phase_label = label
        print(f"[{self.phase_num}/{self.total_phases}] {label}", file=_LOG, flush=True)

    def step(self, current: int, total: int, extra: str = "") -> None:
        suffix = f" ({extra})" if extra else ""
        print(
            f"[{self.phase_num}/{self.total_phases}] {self.phase_label}: "
            f"{current}/{total}{suffix}",
            file=_LOG,
            flush=True,
        )

    def note(self, message: str) -> None:
        print(
            f"[{self.phase_num}/{self.total_phases}] {self.phase_label}: {message}",
            file=_LOG,
            flush=True,
        )

    def banner(self, message: str) -> None:
        print(message, file=_LOG, flush=True)
