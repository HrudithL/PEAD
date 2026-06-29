"""End-to-end orchestration for the drift-ML study.

Wires the stages together: build (or load) the event-feature-label table, fit
the Fama-MacBeth baseline and the LightGBM models under purged walk-forward CV,
run the attribution / consistency protocol, and render the PDF report. Kept thin
so each stage stays independently testable.
"""

from __future__ import annotations

from .config import DriftMLConfig


def _log(msg: str) -> None:
    print(f"[drift-ml] {msg}", flush=True)


def run(cfg: DriftMLConfig) -> str:
    """Run the full pipeline and return the path to the generated PDF report."""
    raise NotImplementedError
