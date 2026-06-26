"""Matplotlib figures for the PEAD report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_BLUE = "#1f4e79"
_RED = "#c0392b"
_GREEN = "#1e8449"


def _bucket_colors(buckets):
    cmap = matplotlib.colormaps["RdYlGn"]
    n = len(buckets)
    return {b: cmap(i / max(n - 1, 1)) for i, b in enumerate(buckets)}


def fig_drift_fan(agg: dict, cfg: Config):
    offsets, buckets = agg["offsets"], agg["buckets"]
    colors = _bucket_colors(buckets)
    fig, ax = plt.subplots(figsize=(9, 5))
    for b in buckets:
        ax.plot(offsets, agg["car_by_bucket"][b] * 100, color=colors[b],
                lw=1.8, label=f"{b}")
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Trading days relative to announcement (day 0 = first post-announcement session)")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.set_title(f"Post-Earnings Announcement Drift by surprise {_bucket_word(cfg)}")
    leg = ax.legend(title=f"Surprise {_bucket_word(cfg)}\n(1=lowest, {len(buckets)}=highest)",
                    fontsize=7, title_fontsize=7, ncol=2, loc="upper left",
                    frameon=False)
    fig.tight_layout()
    return fig


def fig_terminal_bar(agg: dict, cfg: Config):
    buckets = agg["buckets"]
    colors = _bucket_colors(buckets)
    iend = len(agg["offsets"]) - 1
    vals = [agg["car_by_bucket"][b][iend] * 100 for b in buckets]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([str(b) for b in buckets], vals, color=[colors[b] for b in buckets])
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel(f"Surprise {_bucket_word(cfg)} (1=lowest, {len(buckets)}=highest)")
    ax.set_ylabel(f"CAR at +{cfg.window_post} days (%)")
    ax.set_title("Terminal drift is monotonic in earnings surprise")
    fig.tight_layout()
    return fig


def fig_long_short(agg: dict, cfg: Config):
    offsets = agg["offsets"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(offsets, agg["car_ls"] * 100, color=_BLUE, lw=2)
    ax.fill_between(offsets, agg["car_ls"] * 100, 0,
                    where=(agg["car_ls"] >= 0), color=_BLUE, alpha=0.12)
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("Trading days relative to announcement")
    ax.set_ylabel("CAR spread: top minus bottom (%)")
    ttl = (f"Long top / short bottom {_bucket_word(cfg, singular=True)}  "
           f"|  +{cfg.window_post}d spread = {agg['ls_mean']*100:.2f}%  "
           f"(t = {agg['ls_t']:.1f})")
    ax.set_title(ttl, fontsize=10)
    fig.tight_layout()
    return fig


def fig_counts_surprise(ev: pd.DataFrame, agg: dict, cfg: Config):
    buckets = agg["buckets"]
    colors = _bucket_colors(buckets)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ns = [agg["n_by_bucket"][b] for b in buckets]
    ax1.bar([str(b) for b in buckets], ns, color=[colors[b] for b in buckets])
    ax1.set_title("Events per bucket")
    ax1.set_xlabel(f"Surprise {_bucket_word(cfg)}")
    ax1.set_ylabel("Number of events")

    means = [float(np.nanmean(ev.loc[ev["bucket"] == b, "surprise"])) for b in buckets]
    ax2.bar([str(b) for b in buckets], means, color=[colors[b] for b in buckets])
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_title("Mean surprise per bucket")
    ax2.set_xlabel(f"Surprise {_bucket_word(cfg)}")
    ax2.set_ylabel("Mean surprise")
    fig.tight_layout()
    return fig


def fig_surprise_hist(ev: pd.DataFrame, cfg: Config):
    fig, ax = plt.subplots(figsize=(9, 4))
    data = ev["surprise"].dropna()
    ax.hist(data, bins=80, color=_BLUE, alpha=0.8)
    ax.axvline(0, color=_RED, ls="--", lw=1)
    ax.set_title(f"Distribution of {cfg.label_measure().split('  ')[0].lower()}")
    ax.set_xlabel("Surprise (winsorized 1/99%)")
    ax.set_ylabel("Number of events")
    fig.tight_layout()
    return fig


def fig_annual_stability(ev: pd.DataFrame, agg: dict, cfg: Config):
    """Long/short terminal CAR by announcement year."""
    iend = len(agg["offsets"]) - 1
    car_term = agg["car_event"][:, iend]
    work = ev.copy()
    work["term"] = car_term
    work["year"] = work["anndats"].dt.year
    top, bottom = agg["buckets"][-1], agg["buckets"][0]

    years, spreads = [], []
    for y, g in work.groupby("year"):
        t = g.loc[g["bucket"] == top, "term"].mean()
        b = g.loc[g["bucket"] == bottom, "term"].mean()
        if np.isfinite(t) and np.isfinite(b):
            years.append(int(y))
            spreads.append((t - b) * 100)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [_GREEN if s >= 0 else _RED for s in spreads]
    ax.bar([str(y) for y in years], spreads, color=colors)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title(f"Long/short +{cfg.window_post}-day drift by announcement year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Top-minus-bottom CAR (%)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig


def fig_announcement_zoom(agg: dict, cfg: Config):
    """Zoom on the immediate announcement reaction for extreme buckets."""
    offsets = agg["offsets"]
    buckets = agg["buckets"]
    colors = _bucket_colors(buckets)
    show = [buckets[0], buckets[len(buckets) // 2], buckets[-1]]
    mask = (offsets >= -cfg.window_pre) & (offsets <= min(10, cfg.window_post))
    fig, ax = plt.subplots(figsize=(9, 4))
    for b in show:
        ax.plot(offsets[mask], agg["car_by_bucket"][b][mask] * 100,
                color=colors[b], lw=2, marker="o", ms=3,
                label=f"Bucket {b}")
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title("Immediate reaction around the announcement (extreme buckets)")
    ax.set_xlabel("Trading days relative to announcement")
    ax.set_ylabel("CAR (%)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def _bucket_word(cfg: Config, singular: bool = False) -> str:
    n = cfg.buckets
    name = {10: "decile", 5: "quintile", 4: "quartile", 3: "tercile"}.get(n, "bucket")
    return name if singular else name + "s"
