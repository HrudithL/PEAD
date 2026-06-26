"""End-to-end PEAD pipeline orchestration."""

from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd

from ..config import Config
from .. import ticker_groups
from . import data_loader, surprise, event_study, report


def _log(msg: str) -> None:
    print(f"[pead] {msg}", flush=True)


def _warn_if_stale(cfg: Config, px: pd.DataFrame) -> None:
    """Warn (never fail) when the price data looks behind today's date.

    Uses the data already loaded (the benchmark trades every session, so its
    last date tracks the dataset's latest date) so this costs no extra I/O.
    Refreshing is left as an explicit manual step because
    master_stock_update.py is interactive and makes a network call per ticker.
    """
    try:
        latest = pd.to_datetime(px["date"], errors="coerce").max()
        if pd.isna(latest):
            return
        latest_d = latest.date()
        behind = (date.today() - latest_d).days
        if behind > 4:
            updater = os.path.join(os.path.dirname(os.path.abspath(cfg.stock_path)),
                                   "master_stock_update.py")
            _log(f"NOTE: price data ends {latest_d} ({behind} days behind today). "
                 f"To refresh, run:  python \"{updater}\"")
    except Exception:
        pass


def run(cfg: Config) -> str:
    if cfg.ticker_spec:
        spec_desc = ticker_groups.describe_spec(cfg.ticker_spec)
        if spec_desc:
            _log(f"Universe referenced by group name: {spec_desc} "
                 f"-> {len(cfg.tickers):,} unique tickers (deduped).")
    _log(f"Loading IBES events {cfg.start_year}-{cfg.end_year} ...")
    events = data_loader.load_events(cfg)
    if events.empty:
        raise SystemExit("No earnings events matched the filters.")
    _log(f"  {len(events):,} quarterly EPS announcements after consensus dedup.")

    events = surprise.attach_anchor_date(events)

    needed = set(events["oftic"].unique())
    _log(f"Loading prices for {len(needed):,} tickers (+benchmark) ...")
    px = data_loader.load_prices(cfg, needed)
    if px.empty:
        raise SystemExit("No matching price data found.")
    _warn_if_stale(cfg, px)

    _log("Building return panel ...")
    rets, calendar, bench_ret, prices_wide = data_loader.build_return_panel(px, cfg)
    _log(f"  {len(calendar):,} trading days, {rets.shape[1]:,} tickers in panel.")

    _log("Locating events on the trading calendar ...")
    ev = event_study.locate_events(events, rets, calendar, cfg)
    pre_price = event_study.attach_pre_price(ev, prices_wide, calendar)
    ev = surprise.compute_surprise(ev, pre_price, cfg)

    located = len(ev)
    ev = surprise.assign_buckets(ev, cfg).reset_index(drop=True)
    _log(f"  {len(ev):,} usable events across {ev['oftic'].nunique():,} firms.")

    if ev.empty:
        raise SystemExit(
            f"No events survived bucketing. PEAD sorts the cross-section of firms "
            f"into {cfg.buckets} buckets every calendar quarter, which needs at least "
            f"{cfg.buckets} announcements per quarter. This run located {located} "
            f"event(s) but no quarter had enough firms to form {cfg.buckets} buckets "
            f"(e.g. a single-ticker run reports once per quarter). Fix: include more "
            f"tickers (drop --tickers for the full universe) or lower --buckets."
        )

    _log("Computing abnormal-return paths ...")
    offsets, ar_mat = event_study.compute_ar_matrix(ev, rets, bench_ret, cfg)
    agg = event_study.aggregate_by_bucket(ev, offsets, ar_mat, cfg)

    _log(f"  Long/short +{cfg.window_post}d spread: "
         f"{agg['ls_mean']*100:.2f}%  (t = {agg['ls_t']:.2f})")

    _log("Building PDF report ...")
    out = report.build_pdf(cfg, ev, agg)
    _log(f"Done -> {out}")
    return out
