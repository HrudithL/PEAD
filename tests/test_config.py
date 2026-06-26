"""Tests for configuration parsing."""

from pead.config import Config, parse_args, _parse_tickers
from pead import ticker_groups


def test_defaults_are_sane():
    cfg = Config()
    assert cfg.start_year <= cfg.end_year
    assert cfg.buckets >= 2
    assert cfg.window_pre >= 0 and cfg.window_post > 0
    assert cfg.benchmark in ("spy", "raw")
    assert cfg.measure in ("std", "price")


def test_parse_tickers_comma_list():
    assert _parse_tickers("aapl, msft ,NVDA") == ["AAPL", "MSFT", "NVDA"]
    assert _parse_tickers("") is None
    assert _parse_tickers(None) is None


def test_parse_tickers_from_file(tmp_path):
    f = tmp_path / "tk.txt"
    f.write_text("aapl\nmsft\n\nnvda\n")
    assert _parse_tickers(str(f)) == ["AAPL", "MSFT", "NVDA"]


def test_parse_args_flags():
    cfg = parse_args([
        "--start-year", "2018", "--end-year", "2022",
        "--buckets", "5", "--benchmark", "raw", "--measure", "price",
        "--window-pre", "3", "--window-post", "45", "--tickers", "AAPL,MSFT",
    ])
    assert cfg.start_year == 2018 and cfg.end_year == 2022
    assert cfg.buckets == 5
    assert cfg.benchmark == "raw"
    assert cfg.measure == "price"
    assert cfg.window_pre == 3 and cfg.window_post == 45
    assert cfg.tickers == ["AAPL", "MSFT"]


def test_labels_render():
    cfg = Config()
    assert "STDEV" in cfg.label_measure()
    assert "SPY" in cfg.label_benchmark()


def test_group_expands_and_is_recognized():
    assert ticker_groups.is_group("mag7")
    assert ticker_groups.is_group("FAANG")
    assert not ticker_groups.is_group("AAPL")
    faang = ticker_groups.resolve_group("faang")
    assert faang and "AAPL" in faang and "NFLX" in faang


def test_parse_tickers_expands_group():
    out = _parse_tickers("mag7")
    assert out and "AAPL" in out and "TSLA" in out
    # Plain tickers still pass through unchanged.
    assert _parse_tickers("aapl,msft") == ["AAPL", "MSFT"]


def test_parse_tickers_dedupes_overlapping_groups_and_tickers():
    # AAPL is in MAG7, FAANG and listed explicitly -> appears exactly once.
    out = _parse_tickers("MAG7,FAANG,AAPL")
    assert out.count("AAPL") == 1
    assert len(out) == len(set(out))


def test_ticker_spec_preserved_for_reproduce_command():
    cfg = parse_args(["--tickers", "MAG7,AAPL"])
    assert cfg.ticker_spec == "MAG7,AAPL"
    assert "--tickers MAG7,AAPL" in cfg.as_cli_command()
    assert cfg.tickers.count("AAPL") == 1
