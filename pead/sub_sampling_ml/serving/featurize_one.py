"""Build the point-in-time feature row for one ``(ticker, anndate)`` event.

The research pipeline (:mod:`pead.sub_sampling_ml.dataset`) builds features in
batch over a whole date range. Live serving needs the same output shape for a
single event without paying the full-universe cost, so this module wires up a
minimally scoped equities panel (prices for just the ticker + benchmark,
returns, calendar, AR matrix) and calls the family builders directly.

The ``ev`` row is constructed either from the IBES summary cache (auto-lookup
mode) or from user-supplied overrides for `actual` / `meanest` / `stdev` etc.
(useful for scoring an announcement that isn't in the cache yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

from ...equities import data_loader, surprise, event_study
from .. import features as features_mod
from .. import labels as labels_mod
from ..config import DriftMLConfig
from .artifact import DriftModel
from . import wrds_incremental


# ------------------------------------------------------------ user inputs


@dataclass
class EventInputs:
    """Everything the caller supplies (or overrides) for a single event.

    ``anndate`` is the announcement date (any parseable form). Surprise fields
    are optional; when missing we try the IBES cache first (see
    :func:`find_in_ibes_cache`).
    """

    ticker: str
    anndate: str                    # 'YYYY-MM-DD' or Timestamp
    anntime: Optional[str] = None   # 'HH:MM:SS'; None -> assume after close
    fpedats: Optional[str] = None   # fiscal-period-end date
    statpers: Optional[str] = None  # last consensus snapshot date
    cname: Optional[str] = None
    actual: Optional[float] = None
    meanest: Optional[float] = None
    medest: Optional[float] = None
    stdev: Optional[float] = None
    numest: Optional[int] = None
    numup: Optional[int] = None
    numdown: Optional[int] = None


# ------------------------------------------------------------ IBES cache lookup


def find_in_ibes_cache(ticker: str, anndate: str,
                       cfg: DriftMLConfig) -> Optional[dict]:
    """Return the IBES-summary row matching ``(ticker, anndate)``, if present.

    Uses ``data_loader.load_events`` scoped to the single ticker so the result
    has the exact schema the downstream pipeline expects. Match is against
    ``anndats`` (date equality). Returns ``None`` when nothing is found.
    """
    tkr = ticker.upper()
    ann = pd.Timestamp(anndate).normalize()
    # Widen the year window around the requested announcement so tickers
    # active outside cfg.start_year/end_year are still findable.
    scoped = DriftMLConfig(
        ibes_path=cfg.ibes_path,
        stock_path=cfg.stock_path,
        start_year=min(cfg.start_year, ann.year - 1),
        end_year=max(cfg.end_year, ann.year + 1),
        tickers=[tkr],
        min_numest=1,
        benchmark=cfg.benchmark,
        horizons=cfg.horizons,
        window_pre=cfg.window_pre,
        n_deciles=cfg.n_deciles,
        use_wrds=False,
    )
    try:
        events = data_loader.load_events(scoped.to_equities_config())
    except Exception:
        return None
    if events.empty:
        return None
    hit = events[events["anndats"].dt.normalize() == ann]
    if hit.empty:
        return None
    row = hit.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def merge_inputs(user: EventInputs, cached: Optional[dict],
                 *, prefer_cache: bool) -> dict:
    """Combine user overrides with cached IBES fields.

    ``prefer_cache=True`` uses the cached value whenever the user didn't
    explicitly supply one. ``prefer_cache=False`` is the CLI's
    ``--override-cache`` contract ("ignore cached IBES values, use
    CLI-supplied only") -- the cache is not consulted at all, so any field
    the caller omitted stays unset rather than silently falling back to a
    stale cached row.
    """
    fields = ["anntime", "fpedats", "statpers", "cname", "actual", "meanest",
              "medest", "stdev", "numest", "numup", "numdown"]
    resolved = {"oftic": user.ticker.upper(), "anndats": user.anndate}
    cache_map = {
        "anntime": "anntims", "fpedats": "fpedats", "statpers": "statpers",
        "cname": "cname", "actual": "actual", "meanest": "meanest",
        "medest": "medest", "stdev": "stdev", "numest": "numest",
        "numup": "numup", "numdown": "numdown",
    }
    for name in fields:
        user_val = getattr(user, name)
        cached_val = cached.get(cache_map[name]) if (cached and prefer_cache) else None
        if prefer_cache and cached_val is not None:
            chosen = cached_val if user_val is None else user_val
        else:
            chosen = user_val
        resolved[cache_map[name]] = chosen
    return resolved


# ------------------------------------------------------------ ev row assembly


def _build_ev_row(fields: dict) -> pd.DataFrame:
    """Turn the resolved-field dict into the one-row events DataFrame ``load_events``
    would have produced."""
    row = {
        "oftic": str(fields["oftic"]).upper(),
        "cname": fields.get("cname") or fields["oftic"],
        "statpers": pd.to_datetime(fields.get("statpers"), errors="coerce"),
        "fpedats": pd.to_datetime(fields.get("fpedats"), errors="coerce"),
        "anndats": pd.to_datetime(fields["anndats"], errors="coerce"),
        "anntims": fields.get("anntims"),
        "actual": _to_float(fields.get("actual")),
        "meanest": _to_float(fields.get("meanest")),
        "medest": _to_float(fields.get("medest")),
        "stdev": _to_float(fields.get("stdev")),
        "numest": _to_float(fields.get("numest")),
        "numup": _to_float(fields.get("numup")),
        "numdown": _to_float(fields.get("numdown")),
    }
    if pd.isna(row["anndats"]):
        raise ValueError(f"Invalid anndats: {fields['anndats']!r}")
    if row["actual"] is None or row["meanest"] is None:
        raise ValueError(
            "actual and meanest are required to build surprise features. "
            "Supply them explicitly or ensure the IBES cache covers this event.")
    return pd.DataFrame([row])


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------ panel assembly


def _build_single_event_panel(ev_row: pd.DataFrame,
                              cfg: DriftMLConfig) -> dict:
    """Scoped equities pipeline for one event."""
    eq = cfg.to_equities_config()
    ticker = ev_row["oftic"].iloc[0]

    events = surprise.attach_anchor_date(ev_row)
    px = data_loader.load_prices(eq, {ticker})
    if px.empty:
        raise SystemExit(
            f"No price data for {ticker} in {cfg.stock_path}; cannot build features.")

    rets, calendar, bench_ret, prices_wide = data_loader.build_return_panel(px, eq)

    # Serving only needs point-in-time features (window_pre history), not the
    # drift_raw_h{H} label -- unlike the batch research pipeline, nothing here
    # consumes the AR matrix. Locate against window_post=0 so a current or
    # recent announcement doesn't get dropped for lacking H future trading
    # days on the panel.
    locate_cfg = replace(eq, window_post=0)
    ev = event_study.locate_events(events, rets, calendar, locate_cfg)
    if ev.empty:
        raise SystemExit(
            f"Event {ticker} {ev_row['anndats'].iloc[0].date()} could not be "
            "located on the trading calendar (window would run off the panel).")

    pre_price = event_study.attach_pre_price(ev, prices_wide, calendar)
    ev = surprise.compute_surprise(ev, pre_price, eq).reset_index(drop=True)

    return {
        "ev": ev, "rets": rets, "prices_wide": prices_wide,
        "calendar": calendar, "bench_ret": bench_ret,
    }


# ------------------------------------------------------------ industry drift


def industry_drift_from_history(history: pd.DataFrame, ff12,
                                anndate: pd.Timestamp) -> float:
    """Trailing-mean drift of prior closed events in the same industry.

    ``history`` has columns ``ff12``, ``close_date``, ``drift_raw`` and comes
    from the frozen artifact. Only rows whose window closed on/before
    ``anndate`` count (strictly causal).
    """
    if history is None or history.empty or ff12 is None or pd.isna(ff12):
        return float("nan")
    sub = history[history["ff12"].astype(str) == str(ff12)]
    if sub.empty:
        return float("nan")
    prior = sub[pd.to_datetime(sub["close_date"]) <= pd.Timestamp(anndate)]
    if prior.empty:
        return float("nan")
    return float(prior["drift_raw"].mean())


# ------------------------------------------------------------ public API


@dataclass
class FeaturizeResult:
    """Feature row for one event plus context needed by the caller."""

    features: pd.DataFrame  # 1-row DataFrame, columns = feature_cols
    ev: pd.DataFrame        # located event row (has pos0, pre_price, surprise)
    used_cache: bool        # True if IBES cache supplied surprise fields
    cache_row: Optional[dict] = field(default=None)  # what we found in cache


def featurize_one(user: EventInputs, cfg: DriftMLConfig, *,
                  model: Optional[DriftModel] = None,
                  prefer_cache: bool = True) -> FeaturizeResult:
    """Build the single-event feature row aligned to the training schema.

    * ``model`` (optional): if provided, its persisted industry history is used
      to fill ``industry_drift_base``.
    * ``prefer_cache``: when True, cached IBES values win over user-supplied
      ones (except where the user explicitly overrides).
    """
    cached = find_in_ibes_cache(user.ticker, user.anndate, cfg)
    fields = merge_inputs(user, cached, prefer_cache=prefer_cache)
    ev_row = _build_ev_row(fields)

    panel = _build_single_event_panel(ev_row, cfg)

    wrds_panels = None
    if cfg.use_wrds:
        wrds_panels = wrds_incremental.ensure_wrds_for_ticker(user.ticker, cfg)

    feats = features_mod.build_features(
        panel["ev"], panel["rets"], panel["prices_wide"], panel["calendar"],
        panel["bench_ret"], cfg, wrds_panels=wrds_panels or None,
    )

    # Overlay industry_drift_base from the model's persisted history so the
    # feature matches what training saw.
    if model is not None and "industry_drift_base" in feats.columns \
            and "ff12" in feats.columns:
        ff12_val = feats["ff12"].iloc[0]
        anndate = pd.Timestamp(fields["anndats"])
        feats.loc[feats.index[0], "industry_drift_base"] = \
            industry_drift_from_history(model.industry_history, ff12_val, anndate)

    return FeaturizeResult(
        features=feats.reset_index(drop=True),
        ev=panel["ev"],
        used_cache=prefer_cache and cached is not None,
        cache_row=cached,
    )


# ------------------------------------------------------------ realized outcome


def realized_drift_one(ticker: str, anndate, cfg: DriftMLConfig,
                       horizon: Optional[int] = None) -> Optional[float]:
    """Realized CAR[+1, +horizon] for one event, or ``None`` if not observable.

    Unlike :func:`featurize_one` (which locates against ``window_post=0`` so
    a just-announced event can still be scored), this needs the *actual*
    trailing price history -- it is used after the fact to check a
    prediction against what happened, not to build inference features. Returns
    ``None`` both when the ticker/date can't be located at all and when the
    price panel doesn't yet reach ``horizon`` trading days past the
    announcement (event hasn't finished playing out).
    """
    horizon = int(horizon or cfg.primary_horizon)
    eq = cfg.to_equities_config()
    ticker = str(ticker).upper()

    cached = find_in_ibes_cache(ticker, str(anndate), cfg)
    anntims = cached.get("anntims") if cached else None
    events = pd.DataFrame({"oftic": [ticker], "anndats": [pd.Timestamp(anndate)],
                          "anntims": [anntims]})
    events = surprise.attach_anchor_date(events)

    px = data_loader.load_prices(eq, {ticker})
    if px.empty:
        return None
    rets, calendar, bench_ret, prices_wide = data_loader.build_return_panel(px, eq)

    ev = event_study.locate_events(events, rets, calendar, eq)
    if ev.empty:
        return None

    offsets, ar_mat = event_study.compute_ar_matrix(ev, rets, bench_ret, eq)
    car = labels_mod.car_window(offsets, ar_mat, start=1, end=horizon)
    val = float(car[0])
    return val if np.isfinite(val) else None
