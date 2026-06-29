"""Unit tests for the WRDS extraction layer.

These run WITHOUT a live WRDS connection or the ``wrds`` package installed:
every test either exercises pure caching logic or monkeypatches the connection.
"""

from __future__ import annotations

import pandas as pd

from pead.sub_sampling_ml import wrds_extract as we
from pead.sub_sampling_ml.config import DriftMLConfig

def _cfg(tmp_path, **kwargs) -> DriftMLConfig:
    cfg = DriftMLConfig(**kwargs)
    cfg.wrds_cache_dir = str(tmp_path)
    return cfg

def test_cache_path_builds_expected_path(tmp_path):
    cfg = _cfg(tmp_path)
    assert we.cache_path("crsp_daily", cfg) == tmp_path / "crsp_daily.csv"

def test_write_then_read_cache_round_trips(tmp_path):
    cfg = _cfg(tmp_path)
    df = pd.DataFrame(
        {
            "permno": [10001, 10002],
            "date": ["2020-01-02", "2020-01-03"],
            "ret": [0.01, -0.02],
        }
    )
    we._write_cache(df, "crsp_daily", cfg)

    assert we.cache_path("crsp_daily", cfg).is_file()
    out = we._read_cache("crsp_daily", cfg)
    assert out is not None
    assert list(out["permno"]) == [10001, 10002]
    # date-like columns are parsed back to datetime on read
    assert pd.api.types.is_datetime64_any_dtype(out["date"])

def test_read_cache_missing_returns_none(tmp_path):
    cfg = _cfg(tmp_path)
    assert we._read_cache("does_not_exist", cfg) is None

def test_refresh_wrds_bypasses_existing_cache(tmp_path):
    cfg = _cfg(tmp_path)
    we._write_cache(pd.DataFrame({"a": [1]}), "thing", cfg)
    assert we._read_cache("thing", cfg) is not None

    cfg.refresh_wrds = True
    assert we._read_cache("thing", cfg) is None

def test_build_panels_returns_empty_when_wrds_disabled(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, use_wrds=False)

    def _boom(_cfg):
        raise AssertionError("connection must not be attempted when use_wrds=False")

    monkeypatch.setattr(we, "get_connection", _boom)
    ev = pd.DataFrame({"oftic": ["AAPL", "MSFT"]})
    assert we.build_wrds_panels(ev, cfg) == {}

def test_build_panels_returns_empty_on_connection_failure(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, use_wrds=True)

    def _no_creds(_cfg):
        raise RuntimeError("no WRDS credentials / package missing")

    monkeypatch.setattr(we, "get_connection", _no_creds)
    ev = pd.DataFrame({"oftic": ["AAPL", "MSFT"]})
    assert we.build_wrds_panels(ev, cfg) == {}
