"""Configuration, CLI parsing, and credential wiring for the drift-ML pipeline.

This config is deliberately separate from :class:`pead.config.Config` (the
equities event-study knobs) but knows how to *project down* to one via
:meth:`DriftMLConfig.to_equities_config`, so event construction and abnormal
returns are reused verbatim from ``pead.equities``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

from ..config import Config, DEFAULT_IBES, DEFAULT_STOCK
from .. import ticker_groups
from ..io import resolver

DEFAULT_OUTPUT_DIR = Config.output_dir  # type: ignore[assignment]
# Resolve via the same machinery the equities pipeline uses.
_DEFAULT = Config()
DEFAULT_OUTPUT_DIR = _DEFAULT.output_dir

# Horizons (trading days) for the CAR[+1, +H] drift labels. The first entry is
# the primary modelling horizon; the rest are robustness checks (Section 3).
DEFAULT_HORIZONS: tuple[int, ...] = (60, 20, 5)


def _load_dotenv() -> None:
    """Best-effort load of a repo-root ``.env`` so WRDS creds reach os.environ.

    Uses python-dotenv when installed; otherwise parses ``KEY=VALUE`` lines so
    the pipeline still works on a bare environment. Never raises.
    """
    repo_root = resolver._REPO_ROOT  # noqa: SLF001 - single source of truth
    env_path = repo_root / ".env"
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_path)
        return
    except Exception:
        pass
    try:
        if not env_path.is_file():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
    except Exception:
        pass


@dataclass
class DriftMLConfig:
    """All knobs for one drift-attribution run."""

    # --- data location (reuses the equities resolver) ---
    ibes_path: str = DEFAULT_IBES
    stock_path: str = DEFAULT_STOCK
    output_dir: str = DEFAULT_OUTPUT_DIR

    # --- universe / sample (Section 2) ---
    start_year: int = 2015
    end_year: int = 2024
    tickers: Optional[list[str]] = None
    ticker_spec: Optional[str] = None
    min_numest: int = 1
    benchmark: str = "spy"

    # --- label (Section 3) ---
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    window_pre: int = 5
    n_deciles: int = 10  # per-quarter decile cut for drift_decile / drift_class

    # --- features (Section 5) ---
    use_wrds: bool = True  # pull CRSP/Compustat families; False -> repo-only features

    # --- validation / model (Section 8) ---
    embargo_months: int = 3
    min_train_quarters: int = 8
    random_state: int = 7
    fit_classifier: bool = True

    # --- WRDS access (Section 6) ---
    wrds_username: Optional[str] = None
    wrds_cache_dir: Optional[str] = None  # None -> resolver default (Data Source)
    refresh_wrds: bool = False  # ignore cache and re-pull

    # Reuse a previously built event_features.parquet instead of rebuilding the
    # whole event x feature x label table on every run.
    use_cache: bool = True

    def __post_init__(self) -> None:
        _load_dotenv()
        if self.wrds_username is None:
            self.wrds_username = os.environ.get("WRDS_USERNAME")
        if self.wrds_cache_dir is None:
            self.wrds_cache_dir = str(resolver.wrds_cache_dir())

    @property
    def primary_horizon(self) -> int:
        return self.horizons[0]

    @property
    def window_post(self) -> int:
        """The AR matrix must reach the longest label horizon."""
        return max(self.horizons)

    def to_equities_config(self) -> Config:
        """Project to an equities :class:`~pead.config.Config` for event/AR reuse."""
        return Config(
            ibes_path=self.ibes_path,
            stock_path=self.stock_path,
            output_dir=self.output_dir,
            start_year=self.start_year,
            end_year=self.end_year,
            tickers=self.tickers,
            ticker_spec=self.ticker_spec,
            window_pre=self.window_pre,
            window_post=self.window_post,
            buckets=self.n_deciles,
            min_numest=self.min_numest,
            benchmark=self.benchmark,
            measure="std",
        )

    def derived_path(self, name: str = "event_features.parquet") -> str:
        return str(resolver.DERIVED_DIR / name)

    def as_cli_command(self) -> str:
        parts = [
            "python run_drift_ml.py",
            f"--start-year {self.start_year}",
            f"--end-year {self.end_year}",
            f"--horizons {','.join(str(h) for h in self.horizons)}",
            f"--benchmark {self.benchmark}",
            f"--embargo-months {self.embargo_months}",
        ]
        if not self.use_wrds:
            parts.append("--no-wrds")
        if self.ticker_spec:
            parts.append("--tickers " + self.ticker_spec)
        return " ".join(parts)


def _parse_tickers(raw: Optional[str]) -> Optional[list[str]]:
    if not raw:
        return None
    if os.path.isfile(raw):
        with open(raw) as fh:
            items = [ln.strip() for ln in fh if ln.strip()]
    else:
        items = [t.strip() for t in raw.split(",") if t.strip()]
    items = [t.upper() for t in items]
    return ticker_groups.expand(items) or None


def _parse_horizons(raw: str) -> tuple[int, ...]:
    hs = tuple(int(x) for x in str(raw).replace(" ", "").split(",") if x)
    if not hs:
        raise argparse.ArgumentTypeError("--horizons needs at least one integer")
    return hs


def parse_args(argv: Optional[list[str]] = None) -> DriftMLConfig:
    """Build a :class:`DriftMLConfig` from CLI flags."""
    p = build_parser()
    args = p.parse_args(argv)
    return config_from_args(args)


def build_parser(*, prog: str = "run_drift_ml",
                 description: str = "Sub-sample drift attribution "
                                    "(feature -> realized PEAD).") -> argparse.ArgumentParser:
    """Return the shared DriftMLConfig argument parser.

    Serving CLIs (``run_train_drift_model.py`` / ``run_predict_drift.py`` /
    ``run_backtest_drift_model.py``) build this parser and then attach their
    own flags on top, so ``--help`` shows every option in one place.
    """
    p = argparse.ArgumentParser(
        prog=prog, description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ibes", dest="ibes_path", default=DEFAULT_IBES)
    p.add_argument("--stock", dest="stock_path", default=DEFAULT_STOCK)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--tickers", default=None,
                   help="Comma list / group name / file path. Default: full universe.")
    p.add_argument("--horizons", type=_parse_horizons, default=DEFAULT_HORIZONS,
                   help="CAR[+1,+H] horizons; first is primary. e.g. 60,20,5")
    p.add_argument("--window-pre", type=int, default=5)
    p.add_argument("--n-deciles", type=int, default=10)
    p.add_argument("--min-numest", type=int, default=1)
    p.add_argument("--benchmark", choices=["spy", "raw"], default="spy")
    p.add_argument("--embargo-months", type=int, default=3)
    p.add_argument("--min-train-quarters", type=int, default=8)
    p.add_argument("--no-wrds", action="store_true",
                   help="Skip CRSP/Compustat families; use repo-only features.")
    p.add_argument("--refresh-wrds", action="store_true",
                   help="Ignore the cached WRDS CSVs and re-pull from the API.")
    p.add_argument("--no-cache", action="store_true",
                   help="Rebuild event_features.parquet instead of reusing the cache.")
    p.add_argument("--no-classifier", action="store_true")
    p.add_argument("--wrds-username", default=None)
    return p


def config_from_args(args: argparse.Namespace) -> DriftMLConfig:
    """Turn a parsed ``argparse.Namespace`` from :func:`build_parser` into a config."""
    tickers = _parse_tickers(args.tickers)
    ticker_spec = args.tickers if (args.tickers and tickers) else None
    return DriftMLConfig(
        ibes_path=args.ibes_path,
        stock_path=args.stock_path,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        tickers=tickers,
        ticker_spec=ticker_spec,
        horizons=tuple(args.horizons),
        window_pre=abs(args.window_pre),
        n_deciles=args.n_deciles,
        min_numest=args.min_numest,
        benchmark=args.benchmark,
        embargo_months=args.embargo_months,
        min_train_quarters=args.min_train_quarters,
        use_wrds=not args.no_wrds,
        refresh_wrds=args.refresh_wrds,
        use_cache=not args.no_cache,
        fit_classifier=not args.no_classifier,
        wrds_username=args.wrds_username,
    )
