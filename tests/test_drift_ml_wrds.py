"""Unit tests for the WRDS extraction layer.

These run WITHOUT a live WRDS connection or the ``wrds`` package installed:
every test either exercises pure caching logic or monkeypatches the connection.
"""

from __future__ import annotations

import re

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


# ---------------------------------------------------------------------------
# Checkpointed bulk extraction (Section 7.2): incremental, resumable pulls.
#
# ``_FakeConn`` stands in for ``wrds.Connection``: it records every SQL string
# it is asked to run and returns a synthetic frame for whichever ids appear in
# that query's IN-list, so tests can assert on exactly which ids were (or
# weren't) queried without touching a live WRDS connection.
# ---------------------------------------------------------------------------

def _queried_ids(sql: str) -> list[str]:
    """Pull the IN-list values out of one generated WRDS SQL string."""
    match = re.search(r"in \(([^)]*)\)", sql)
    assert match, f"expected an IN-list in generated SQL: {sql}"
    return [v.strip().strip("'") for v in match.group(1).split(",")]

class _FakeConn:
    """Records queries; returns synthetic rows for the ids in each IN-list."""

    def __init__(self):
        self.queries: list[str] = []

    def raw_sql(self, sql: str) -> pd.DataFrame:
        self.queries.append(sql)
        ids = _queried_ids(sql)
        if "crsp.dsf" in sql:
            return pd.DataFrame(
                {
                    "permno": [int(i) for i in ids],
                    "date": ["2020-01-02"] * len(ids),
                    "ret": [0.01] * len(ids),
                    "prc": [10.0] * len(ids),
                    "vol": [1000] * len(ids),
                    "shrout": [5000] * len(ids),
                }
            )
        if "comp.fundq" in sql:
            data = {
                "gvkey": ids,
                "datadate": ["2020-03-31"] * len(ids),
                "rdq": ["2020-04-20"] * len(ids),
                "fqtr": [1] * len(ids),
                "fyearq": [2020] * len(ids),
                "atq": [100.0] * len(ids),
                "ceqq": [50.0] * len(ids),
                "niq": [5.0] * len(ids),
                "revtq": [40.0] * len(ids),
                "cogsq": [20.0] * len(ids),
                "saleq": [40.0] * len(ids),
                "dlttq": [10.0] * len(ids),
                "dlcq": [1.0] * len(ids),
                "xrdq": [2.0] * len(ids),
                "actq": [30.0] * len(ids),
                "lctq": [15.0] * len(ids),
                "cheq": [8.0] * len(ids),
                "txpq": [1.0] * len(ids),
                "dpq": [3.0] * len(ids),
            }
            # Mirror the production emp/no-emp select-list toggle exactly.
            if sql.split(" from comp.fundq")[0].strip().endswith("emp"):
                data["emp"] = [1.5] * len(ids)
            return pd.DataFrame(data)
        if "comp.company" in sql:
            return pd.DataFrame(
                {
                    "gvkey": ids,
                    "gsector": ["45"] * len(ids),
                    "sic": ["7372"] * len(ids),
                }
            )
        raise AssertionError(f"FakeConn got unexpected SQL: {sql}")

def _seed_crsp(cfg, permnos: list[int]) -> None:
    we._append_cache(
        pd.DataFrame(
            {
                "permno": permnos,
                "date": ["2020-01-02"] * len(permnos),
                "ret": [0.01] * len(permnos),
                "prc": [10.0] * len(permnos),
                "vol": [1000] * len(permnos),
                "shrout": [5000] * len(permnos),
            }
        ),
        "crsp_daily",
        cfg,
    )

def test_extract_crsp_daily_cold_cache_pulls_all_and_writes(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    fake = _FakeConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake)

    out = we.extract_crsp_daily([10003, 10001, 10002], cfg)

    assert len(fake.queries) == 1  # single chunk under the default _PERMNO_CHUNK
    assert sorted(out["permno"].unique().tolist()) == [10001, 10002, 10003]
    on_disk = pd.read_csv(we.cache_path("crsp_daily", cfg))
    assert sorted(on_disk["permno"].unique().tolist()) == [10001, 10002, 10003]

def test_extract_crsp_daily_warm_cache_queries_only_missing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _seed_crsp(cfg, [10001, 10002])

    fake = _FakeConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake)

    out = we.extract_crsp_daily([10001, 10002, 10003, 10004], cfg)

    assert len(fake.queries) == 1
    assert sorted(_queried_ids(fake.queries[0])) == ["10003", "10004"]
    assert sorted(out["permno"].unique().tolist()) == [10001, 10002, 10003, 10004]
    on_disk = pd.read_csv(we.cache_path("crsp_daily", cfg))
    assert sorted(on_disk["permno"].unique().tolist()) == [10001, 10002, 10003, 10004]

def test_extract_crsp_daily_appends_one_chunk_at_a_time(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(we, "_PERMNO_CHUNK", 1)
    fake = _FakeConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake)

    permnos = [10001, 10002, 10003]
    out = we.extract_crsp_daily(permnos, cfg)

    assert len(fake.queries) == len(permnos)  # one raw_sql call per chunk
    on_disk = pd.read_csv(we.cache_path("crsp_daily", cfg))
    assert sorted(on_disk["permno"].unique().tolist()) == permnos
    assert sorted(out["permno"].unique().tolist()) == permnos

def test_extract_crsp_daily_all_cached_never_opens_connection(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _seed_crsp(cfg, [10001, 10002])

    def _boom(_cfg):
        raise AssertionError("get_connection must not be called when nothing is missing")

    monkeypatch.setattr(we, "get_connection", _boom)

    out = we.extract_crsp_daily([10001, 10002], cfg)
    assert sorted(out["permno"].unique().tolist()) == [10001, 10002]

def test_extract_crsp_daily_empty_request_skips_cache_entirely(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)

    def _boom(_cfg):
        raise AssertionError("get_connection must not be called for an empty request")

    monkeypatch.setattr(we, "get_connection", _boom)

    out = we.extract_crsp_daily([], cfg)
    assert out.empty
    assert list(out.columns) == ["permno", "date", "ret", "prc", "vol", "shrout"]
    assert not we.cache_path("crsp_daily", cfg).is_file()

def test_extract_crsp_daily_refresh_wrds_forces_full_repull(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _seed_crsp(cfg, [10001])

    cfg.refresh_wrds = True
    fake = _FakeConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake)

    out = we.extract_crsp_daily([10001, 10002], cfg)

    assert len(fake.queries) == 1
    assert sorted(_queried_ids(fake.queries[0])) == ["10001", "10002"]  # re-pulled, not skipped
    assert sorted(out["permno"].unique().tolist()) == [10001, 10002]

def test_extract_company_warm_cache_queries_only_missing_gvkeys(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    we._append_cache(
        pd.DataFrame({"gvkey": ["1000"], "gsector": ["45"], "sic": ["7372"]}),
        "compustat_company",
        cfg,
    )

    fake = _FakeConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake)

    out = we.extract_company(["1000", "2000"], cfg)

    assert len(fake.queries) == 1
    assert _queried_ids(fake.queries[0]) == ["2000"]
    assert sorted(out["gvkey"].astype(str).unique().tolist()) == ["1000", "2000"]

def test_extract_company_all_cached_never_opens_connection(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    we._append_cache(
        pd.DataFrame(
            {"gvkey": ["1000", "2000"], "gsector": ["45", "20"], "sic": ["7372", "3711"]}
        ),
        "compustat_company",
        cfg,
    )

    def _boom(_cfg):
        raise AssertionError("get_connection must not be called when nothing is missing")

    monkeypatch.setattr(we, "get_connection", _boom)

    out = we.extract_company(["1000", "2000"], cfg)
    assert sorted(out["gvkey"].astype(str).unique().tolist()) == ["1000", "2000"]

def test_extract_compustat_fundq_skips_cached_gvkeys_and_appends_per_chunk(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(we, "_GVKEY_CHUNK", 1)
    fake = _FakeConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake)

    first = we.extract_compustat_fundq(["1000"], cfg)
    assert len(fake.queries) == 1
    assert list(first["gvkey"].astype(str)) == ["1000"]
    assert "emp" in first.columns

    second = we.extract_compustat_fundq(["1000", "2000", "3000"], cfg)

    # "1000" was already cached; the two new gvkeys are pulled one per chunk.
    assert len(fake.queries) == 3
    newly_queried = _queried_ids(fake.queries[-2]) + _queried_ids(fake.queries[-1])
    assert sorted(newly_queried) == ["2000", "3000"]
    assert sorted(second["gvkey"].astype(str).unique().tolist()) == ["1000", "2000", "3000"]


# ---------------------------------------------------------------------------
# Codex review follow-ups on PR #15:
#   (1) gvkey zero-padding must survive a CSV cache round-trip.
#   (2) a failed/quota-limited cache write must not drop pulled rows from the
#       return value (the extractors must not depend on reading their own
#       write back from disk within the same run).
#   (3) the fundq cache's ``emp`` column must stay uniform across appends.
# ---------------------------------------------------------------------------

def _seed_fundq_row(cfg, gvkey: str) -> None:
    """Append one full ``compustat_fundq`` row (incl. ``emp``) for ``gvkey``."""
    we._append_cache(
        pd.DataFrame(
            {
                "gvkey": [gvkey],
                "datadate": ["2020-03-31"],
                "rdq": ["2020-04-20"],
                "fqtr": [1],
                "fyearq": [2020],
                "atq": [100.0],
                "ceqq": [50.0],
                "niq": [5.0],
                "revtq": [40.0],
                "cogsq": [20.0],
                "saleq": [40.0],
                "dlttq": [10.0],
                "dlcq": [1.0],
                "xrdq": [2.0],
                "actq": [30.0],
                "lctq": [15.0],
                "cheq": [8.0],
                "txpq": [1.0],
                "dpq": [3.0],
                "emp": [1.5],
            }
        ),
        "compustat_fundq",
        cfg,
    )

def test_norm_gvkey_canonicalizes_int_float_and_padded_forms():
    assert we._norm_gvkey("001690") == "001690"
    assert we._norm_gvkey("1690") == "001690"
    assert we._norm_gvkey(1690) == "001690"
    assert we._norm_gvkey(1690.0) == "001690"

def test_extract_company_gvkey_padding_survives_cache_round_trip(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    we._append_cache(
        pd.DataFrame({"gvkey": ["001690"], "gsector": ["45"], "sic": ["7372"]}),
        "compustat_company",
        cfg,
    )
    # A pandas round-trip through CSV coerces an all-numeric gvkey column to
    # int64, dropping the zero-padding -- confirm that actually happened so
    # this test is exercising the real bug, not a no-op.
    on_disk = we._read_cache("compustat_company", cfg)
    assert str(on_disk["gvkey"].iloc[0]) != "001690"

    def _boom(_cfg):
        raise AssertionError("get_connection must not be called: padded gvkey should hit cache")

    monkeypatch.setattr(we, "get_connection", _boom)

    padded = we.extract_company(["001690"], cfg)
    assert not padded.empty
    assert padded["gvkey"].map(we._norm_gvkey).tolist() == ["001690"]

    unpadded = we.extract_company(["1690"], cfg)
    assert not unpadded.empty
    assert unpadded["gvkey"].map(we._norm_gvkey).tolist() == ["001690"]

def test_extract_compustat_fundq_gvkey_padding_survives_cache_round_trip(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    _seed_fundq_row(cfg, "001690")
    on_disk = we._read_cache("compustat_fundq", cfg)
    assert str(on_disk["gvkey"].iloc[0]) != "001690"

    def _boom(_cfg):
        raise AssertionError("get_connection must not be called: padded gvkey should hit cache")

    monkeypatch.setattr(we, "get_connection", _boom)

    padded = we.extract_compustat_fundq(["001690"], cfg)
    assert not padded.empty
    assert padded["gvkey"].map(we._norm_gvkey).tolist() == ["001690"]

    unpadded = we.extract_compustat_fundq(["1690"], cfg)
    assert not unpadded.empty
    assert unpadded["gvkey"].map(we._norm_gvkey).tolist() == ["001690"]

def test_extract_crsp_daily_returns_pulled_rows_when_cache_write_fails(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    fake = _FakeConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake)
    # Simulate every cache write failing (e.g. a quota-limited Drive mount);
    # _append_cache is best-effort in production, so make it a true no-op.
    monkeypatch.setattr(we, "_append_cache", lambda *a, **k: None)

    out = we.extract_crsp_daily([10001, 10002], cfg)

    assert sorted(out["permno"].unique().tolist()) == [10001, 10002]
    assert not we.cache_path("crsp_daily", cfg).is_file()

def test_extract_compustat_fundq_returns_pulled_rows_when_cache_write_fails(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    fake = _FakeConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake)
    monkeypatch.setattr(we, "_append_cache", lambda *a, **k: None)

    out = we.extract_compustat_fundq(["1000", "2000"], cfg)

    assert sorted(out["gvkey"].astype(str).unique().tolist()) == ["1000", "2000"]
    assert "emp" in out.columns
    assert not we.cache_path("compustat_fundq", cfg).is_file()

def test_extract_compustat_fundq_reindexes_emp_uniformly_across_chunks(tmp_path, monkeypatch):
    """A chunk pulled via the emp fallback still lands with an emp column (NaN)."""
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(we, "_GVKEY_CHUNK", 1)

    class _NoEmpConn(_FakeConn):
        def raw_sql(self, sql: str) -> pd.DataFrame:
            if "comp.fundq" in sql and sql.split(" from comp.fundq")[0].strip().endswith("emp"):
                raise RuntimeError("emp column not available on this fundq vintage")
            return super().raw_sql(sql)

    fake = _NoEmpConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake)

    first = we.extract_compustat_fundq(["1000"], cfg)
    assert "emp" in first.columns
    assert pd.isna(first["emp"].iloc[0])

    on_disk = pd.read_csv(we.cache_path("compustat_fundq", cfg))
    assert "emp" in on_disk.columns

    # A second, emp-bearing chunk appended afterwards must not make the CSV
    # ragged: both rows should read back cleanly with a uniform column set.
    fake2 = _FakeConn()
    monkeypatch.setattr(we, "get_connection", lambda _cfg: fake2)
    second = we.extract_compustat_fundq(["1000", "2000"], cfg)
    assert sorted(second["gvkey"].astype(str).unique().tolist()) == ["1000", "2000"]
    assert "emp" in second.columns
