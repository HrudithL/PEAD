"""Expected OptionMetrics dataset description plus a light presence validator.

Lets the options pipeline fail fast with a precise message when the mounted
drive is missing or only partially converted, instead of producing silently
wrong results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import om_schema

# Coverage actually converted to parquet on the workstation drive.
DEFAULT_START_YEAR = 1996
DEFAULT_END_YEAR = 2013


@dataclass
class Manifest:
    start_year: int = DEFAULT_START_YEAR
    end_year: int = DEFAULT_END_YEAR
    required_tables: list[str] = field(
        default_factory=lambda: [om_schema.SECNMD, om_schema.ZEROCD]
    )

    def expected_partitions(self) -> list[str]:
        names = []
        for y in range(self.start_year, self.end_year + 1):
            names.append(om_schema.partitioned_path(Path(""), om_schema.OPPRCD_STEM, y).name)
        return names

    def missing(self, om_dir) -> list[str]:
        """Return names of required files absent from ``om_dir`` (empty == OK)."""
        om_dir = Path(om_dir)
        miss: list[str] = []
        for table in self.required_tables:
            if not (om_dir / table).exists():
                miss.append(table)
        for y in range(self.start_year, self.end_year + 1):
            p = om_schema.partitioned_path(om_dir, om_schema.OPPRCD_STEM, y)
            if not p.exists():
                miss.append(p.name)
        return miss

    def validate(self, om_dir) -> None:
        miss = self.missing(om_dir)
        if miss:
            shown = ", ".join(miss[:8]) + ("..." if len(miss) > 8 else "")
            raise FileNotFoundError(
                f"OptionMetrics dataset incomplete at '{om_dir}': "
                f"missing {len(miss)} file(s): {shown}"
            )
