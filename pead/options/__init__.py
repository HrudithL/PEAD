"""Options-market PEAD.

Pipeline stages:
    events.py   build the earnings-event table (ticker, secid, date) from IBES
    extract.py  DuckDB reduction of OptionMetrics parquet -> compact event panel
    panel.py    derived-panel schema + Arrow/parquet IO (hand-off to native)
    engine.py   compute stage (native C++/CUDA engine, with a pandas fallback)
    analysis.py bucket per-event drift by earnings surprise
    report.py   write summary tables / plots
"""
