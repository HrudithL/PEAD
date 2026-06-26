"""Compute stage: turn the per-option event panel into per-event features.

Prefers the native C++/CUDA engine (``native/build/pead_engine``) which reads the
panel via Arrow and runs on the GPU. When that binary is not built, a vectorized
pandas fallback produces identical columns so the pipeline always runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from ..io import resolver
from . import panel as panel_io

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NATIVE_BIN_CANDIDATES = [
    _REPO_ROOT / "native" / "build" / "pead_engine.exe",
    _REPO_ROOT / "native" / "build" / "pead_engine",
    _REPO_ROOT / "native" / "build" / "Release" / "pead_engine.exe",
]

# At-the-money band on |delta| around 0.5.
_ATM_DELTA = 0.5
_ATM_BAND = 0.1


def native_binary() -> Optional[Path]:
    for c in _NATIVE_BIN_CANDIDATES:
        if c.is_file():
            return c
    return None


def run(panel_path, out_path, use_native: bool = True) -> Path:
    """Compute per-event features from a derived panel.

    Returns the path to a parquet of per-event results.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bin_path = native_binary() if use_native else None
    if bin_path is not None:
        subprocess.run(
            [str(bin_path), "--panel", str(panel_path), "--out", str(out_path)],
            check=True,
        )
        return out_path

    _run_python(panel_path, out_path)
    return out_path


def _run_python(panel_path, out_path) -> None:
    """Vectorized pandas fallback for the native compute kernels."""
    df = panel_io.read_panel_df(panel_path)
    if df.empty:
        panel_io.write_panel(pd.DataFrame(columns=_RESULT_COLUMNS), out_path)
        return

    # ATM call options only, by |delta| proximity to 0.5.
    calls = df[df["cp_flag"].astype("string").str.upper() == "C"].copy()
    calls["atm_dist"] = (calls["delta"].abs() - _ATM_DELTA).abs()
    atm = calls[calls["atm_dist"] <= _ATM_BAND]

    pre = atm[atm["rel_day"] < 0]
    post = atm[atm["rel_day"] > 0]

    keys = ["secid", "ann_date"]
    pre_iv = pre.groupby(keys)["impl_volatility"].mean().rename("atm_iv_pre")
    post_iv = post.groupby(keys)["impl_volatility"].mean().rename("atm_iv_post")
    n_pre = pre.groupby(keys).size().rename("n_pre")
    n_post = post.groupby(keys).size().rename("n_post")
    vol = df.groupby(keys)["volume"].sum().rename("total_volume")

    res = pd.concat([pre_iv, post_iv, n_pre, n_post, vol], axis=1).reset_index()
    res["iv_drift"] = res["atm_iv_post"] - res["atm_iv_pre"]
    res = res[_RESULT_COLUMNS]
    panel_io.write_panel(res, out_path)


_RESULT_COLUMNS = [
    "secid", "ann_date",
    "atm_iv_pre", "atm_iv_post", "iv_drift",
    "n_pre", "n_post", "total_volume",
]
