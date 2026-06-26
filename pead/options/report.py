"""Write options-PEAD summary artifacts (CSV + bar plot of IV drift by bucket)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def write_summary(summary: pd.DataFrame, output_dir, tag: str = "") -> dict:
    """Persist the bucket summary as CSV + PNG; return the written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = (f"_{tag}" if tag else "") + f"_{stamp}"

    csv_path = output_dir / f"options_pead_buckets{suffix}.csv"
    summary.to_csv(csv_path, index=False)

    png_path = output_dir / f"options_pead_drift{suffix}.png"
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(summary["bucket"].astype(str), summary["mean_iv_drift"], color="#3b7dd8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Surprise bucket (low -> high)")
        ax.set_ylabel("Mean ATM IV drift (post - pre)")
        ax.set_title("Options-market PEAD: implied-vol drift by earnings surprise")
        fig.tight_layout()
        fig.savefig(png_path, dpi=130)
        plt.close(fig)

    return {"csv": csv_path, "png": png_path if not summary.empty else None}
