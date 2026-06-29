"""Scaffold-level tests: the subpackage imports and the config contract holds.

Implementation lives in later PRs; here we only pin the public surface so the
stacked branches build against a stable contract.
"""

import importlib

import pytest

from pead.sub_sampling_ml import DriftMLConfig
from pead.sub_sampling_ml import config as ml_config


def test_subpackage_modules_import():
    for mod in ("labels", "features", "wrds_extract", "dataset", "model",
                "attribution", "report", "pipeline"):
        importlib.import_module(f"pead.sub_sampling_ml.{mod}")


def test_default_config_horizons_and_window():
    cfg = DriftMLConfig()
    assert cfg.primary_horizon == cfg.horizons[0]
    # The AR matrix must reach the longest label horizon.
    assert cfg.window_post == max(cfg.horizons)
    assert cfg.window_pre >= 1


def test_to_equities_config_projects_window_and_universe():
    cfg = DriftMLConfig(start_year=2019, end_year=2021, horizons=(60, 20, 5))
    eq = cfg.to_equities_config()
    assert eq.window_post == 60
    assert eq.start_year == 2019 and eq.end_year == 2021
    assert eq.benchmark == cfg.benchmark


def test_parse_args_horizons_and_no_wrds():
    cfg = ml_config.parse_args(
        ["--start-year", "2018", "--horizons", "40,10", "--no-wrds"]
    )
    assert cfg.horizons == (40, 10)
    assert cfg.primary_horizon == 40
    assert cfg.use_wrds is False


def test_parse_args_ticker_group_expands():
    cfg = ml_config.parse_args(["--tickers", "MAG7"])
    assert cfg.tickers and len(cfg.tickers) >= 5
    assert cfg.ticker_spec == "MAG7"


def test_derived_path_points_into_repo_derived_dir():
    cfg = DriftMLConfig()
    assert cfg.derived_path().replace("\\", "/").endswith(
        "data/derived/event_features.parquet"
    )
