"""Named ticker groups (S&P 500, Nasdaq-100, Dow 30, ...).

A "group" is just a convenient alias for a list of tickers. Anywhere a ticker
is accepted (e.g. ``--tickers``), a group name can be used instead and it
expands to its member tickers. Groups and individual tickers can be freely
mixed, and any ticker that appears more than once -- whether typed twice or
shared across overlapping groups -- is counted exactly once.

Group membership is sourced two ways:

* Large indices that already have a constituent file in ``Data Source/`` are
  read straight from that CSV (no dataset is modified -- the files are only
  read). This is the case for SP500 and the Russell 2000 / 3000.
* Small, well-known baskets (MAG7, FAANG, DOW30, NASDAQ100) are defined inline.

Member tickers are matched against the price/IBES data as-is; any member that
is not present in the data simply contributes no events, so an occasionally
stale constituent has no effect beyond being ignored.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Optional

from .io import resolver

# ---------------------------------------------------------------------------
# Group definitions
# ---------------------------------------------------------------------------

# Curated baskets defined inline (small and stable).
_CURATED: dict[str, list[str]] = {
    "MAG7": ["AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA"],
    "FAANG": ["META", "AMZN", "AAPL", "NFLX", "GOOGL"],
    "DOW30": [
        "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
        "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
        "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
    ],
    "NASDAQ100": [
        "ADBE", "AMD", "ABNB", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", "ADI",
        "ANSS", "AAPL", "AMAT", "APP", "ARM", "ASML", "AZN", "TEAM", "ADSK",
        "ADP", "AXON", "BKR", "BIIB", "BKNG", "AVGO", "CDNS", "CDW", "CHTR",
        "CTAS", "CSCO", "CCEP", "CTSH", "CMCSA", "CEG", "CPRT", "CSGP", "COST",
        "CRWD", "CSX", "DDOG", "DXCM", "FANG", "DLTR", "EA", "EXC", "FAST",
        "FTNT", "GEHC", "GILD", "GFS", "HON", "IDXX", "INTC", "INTU", "ISRG",
        "KDP", "KLAC", "KHC", "LRCX", "LIN", "LULU", "MAR", "MRVL", "MELI",
        "META", "MCHP", "MU", "MSFT", "MSTR", "MDLZ", "MDB", "MNST", "NFLX",
        "NVDA", "NXPI", "ORLY", "ODFL", "ON", "PCAR", "PLTR", "PANW", "PAYX",
        "PYPL", "PDD", "PEP", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS",
        "TTWO", "TMUS", "TSLA", "TXN", "TTD", "VRSK", "VRTX", "WBD", "WDAY",
        "XEL", "ZS",
    ],
}

# Groups whose members are read from an existing constituent CSV in Data Source.
_CSV_GROUPS: dict[str, str] = {
    "SP500": "S&P 500 Companies.csv",
    "RUSSELL2000": "Russell_2000_companies.csv",
    "RUSSELL3000": "Russell_3000_companies.csv",
}

# Accepted spellings -> canonical group name.
_ALIAS_RAW: dict[str, list[str]] = {
    "MAG7": ["MAG7", "MAGNIFICENT7", "MAGNIFICENTSEVEN"],
    "FAANG": ["FAANG"],
    "DOW30": ["DOW30", "DOW", "DJIA", "DJI", "DOWJONES"],
    "NASDAQ100": ["NASDAQ100", "NDX", "NQ100"],
    "SP500": ["SP500", "SANDP500", "SPX"],
    "RUSSELL2000": ["RUSSELL2000", "R2000", "RUT"],
    "RUSSELL3000": ["RUSSELL3000", "R3000"],
}

CANONICAL_NAMES: list[str] = sorted(set(_CURATED) | set(_CSV_GROUPS))


def _norm(s: str) -> str:
    """Normalize a name for lookup: keep only A-Z/0-9, uppercase."""
    return re.sub(r"[^A-Z0-9]", "", s.upper())


_LOOKUP: dict[str, str] = {}
for _canon in CANONICAL_NAMES:
    _LOOKUP[_norm(_canon)] = _canon
for _canon, _names in _ALIAS_RAW.items():
    for _nm in _names:
        _LOOKUP[_norm(_nm)] = _canon

_members_cache: dict[str, list[str]] = {}


def _load_csv_tickers(filename: str) -> list[str]:
    """Read the ticker column from a constituent CSV (in-repo reference first)."""
    path = str(resolver.constituent_csv(filename))
    out: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        idx = 0
        if header:
            low = [h.strip().lower() for h in header]
            for cand in ("tic", "ticker", "symbol"):
                if cand in low:
                    idx = low.index(cand)
                    break
        for row in reader:
            if not row or idx >= len(row):
                continue
            tic = row[idx].strip().upper()
            if tic:
                out.append(tic)
    return out


def _members(canon: str) -> list[str]:
    if canon not in _members_cache:
        if canon in _CURATED:
            raw = _CURATED[canon]
        else:
            raw = _load_csv_tickers(_CSV_GROUPS[canon])
        seen: set[str] = set()
        members: list[str] = []
        for t in raw:
            t = t.strip().upper()
            if t and t not in seen:
                seen.add(t)
                members.append(t)
        _members_cache[canon] = members
    return _members_cache[canon]


def is_group(name: str) -> bool:
    """True if ``name`` (any accepted spelling) refers to a known group."""
    return _norm(name) in _LOOKUP


def resolve_group(name: str) -> Optional[list[str]]:
    """Return the member tickers for ``name``, or None if it is not a group."""
    canon = _LOOKUP.get(_norm(name))
    return list(_members(canon)) if canon is not None else None


def expand(items: list[str]) -> list[str]:
    """Expand a mixed list of tickers/group names into deduped tickers.

    Group names are replaced by their members; plain tickers pass through. The
    result preserves first-seen order and contains each ticker exactly once,
    even when groups overlap or a ticker is listed alongside a group it belongs
    to.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item or not item.strip():
            continue
        members = resolve_group(item)
        tokens = members if members is not None else [item.strip().upper()]
        for tok in tokens:
            if tok and tok not in seen:
                seen.add(tok)
                out.append(tok)
    return out


def classify_spec(raw: str) -> tuple[list[str], list[str]]:
    """Split a raw --tickers spec into (canonical group names, plain tickers).

    Order is preserved and each entry is de-duplicated. A spec that is a file
    path (not a comma list of names) yields no groups.
    """
    groups: list[str] = []
    plains: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        canon = _LOOKUP.get(_norm(item))
        if canon is not None:
            if canon not in groups:
                groups.append(canon)
        else:
            tok = item.upper()
            if tok not in plains:
                plains.append(tok)
    return groups, plains


def describe_spec(raw: str) -> Optional[str]:
    """Human summary of a spec that uses at least one group, else None.

    e.g. "SP500 (503 members), MAG7 (8 members) + 1 individual ticker".
    """
    groups, plains = classify_spec(raw)
    if not groups:
        return None
    parts = [f"{g} ({len(_members(g))} members)" for g in groups]
    text = ", ".join(parts)
    if plains:
        text += f" + {len(plains)} individual ticker" + ("s" if len(plains) != 1 else "")
    return text


def describe_groups() -> str:
    """One-line-per-group human summary for help text."""
    lines = []
    for canon in CANONICAL_NAMES:
        try:
            n = len(_members(canon))
            lines.append(f"{canon} ({n})")
        except FileNotFoundError:
            lines.append(f"{canon} (constituent file missing)")
    return ", ".join(lines)
