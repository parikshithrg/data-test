"""One-time build: parse every already-fetched Axis Mutual Fund per-scheme
portfolio workbook (`data/amc_portfolios/axis/`, from `scripts/
fetch_amc_portfolios.py`) into one consolidated table of per-stock EQUITY
holdings - Tier 1 item 4's actual point. NOT a live/network call - pure
local file parsing, same category as `dtest/data/amc_holdings.py`'s own
scope (see that module's docstring for why equity-only, and for the two
real file formats both named `.xls`).

    python scripts/build_amc_equity_holdings.py

PROCESSES: every manifest row with `scheme_name` set (the per-scheme
files - 3,625 of them) PLUS every real scheme sheet inside the ONE
consolidated workbook (2021-09-30) that predates per-scheme coverage
(`amc_portfolios.py`'s own module docstring: this is the single month
per-scheme files don't reach). The other 46 consolidated workbooks are
SKIPPED - their content is already covered by per-scheme files for the
same months, and processing both would double-count every holding.

Cross-checks each file's OWN embedded "as on" date (read from inside the
workbook) against the manifest's filename-derived `period_end` - real
disagreements are counted and reported, not silently trusted either way.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import openpyxl
import pandas as pd
import xlrd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.amc_holdings import SCHEMA, parse_axis_scheme_sheet

CONSOLIDATED_ONLY_PERIOD = pd.Timestamp("2021-09-30")  # the one month per-scheme files don't cover


def _read_rows_by_sheet(path: Path) -> dict[str, list[tuple]]:
    """Real files are either modern XLSX-zip content or legacy BIFF - the
    format must be sniffed from content, never the file's own extension
    (confirmed live: `openpyxl.load_workbook` rejects a real ZIP/XLSX file
    outright if its NAME ends in `.xls`, regardless of actual content - a
    real trap, worked around by always reading bytes first and handing
    `openpyxl` a nameless `io.BytesIO` buffer instead of the path)."""
    data = path.read_bytes()
    if data[:2] == b"PK":
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        return {name: list(wb[name].iter_rows(values_only=True)) for name in wb.sheetnames}
    wb = xlrd.open_workbook(file_contents=data)
    return {sh.name: [sh.row_values(r) for r in range(sh.nrows)] for sh in wb.sheets()}


def main() -> int:
    cfg = load_config()
    amc_dir = Path(cfg.paths.amc_portfolios_dir)
    manifest = pd.read_csv(amc_dir / "manifest.csv", parse_dates=["period_end", "filing_date"])
    axis = manifest[manifest["amc_name"] == "Axis Mutual Fund"]

    per_scheme = axis[axis["scheme_name"].notna()].copy()
    # REAL DEDUP, NOT OPTIONAL: `amc_portfolios.py`'s own module docstring
    # already documents 13 (scheme, period_end) collisions where two
    # DIFFERENT real filing-metadata entries (different title casing, e.g.
    # "Axis Nifty Smallcap 50 Index Fund" vs "AXIS NIFTY SMALLCAP 50 INDEX
    # FUND") resolve to the SAME local filename - Windows filesystems are
    # case-insensitive, so both manifest rows point at one physical file on
    # disk. `fetch_amc_portfolios.py` correctly downloads it only once, but
    # never deduped the MANIFEST rows themselves - left unfixed here, this
    # loop would parse that one file twice under two scheme_name labels,
    # silently doubling every holding for those 13 (scheme, month) pairs
    # (caught live: several schemes' own pct_to_nav summed to ~2.0 instead
    # of the normal ~0.85-1.0 before this fix).
    n_before = len(per_scheme)
    per_scheme["_dedup_key"] = per_scheme["local_filename"].str.lower()
    per_scheme = per_scheme.drop_duplicates(subset="_dedup_key", keep="first")
    print(f"deduped {n_before - len(per_scheme)} manifest rows sharing a "
          f"case-insensitive-identical local_filename with another row")

    consolidated_only_row = axis[(axis["scheme_name"].isna())
                                  & (axis["period_end"] == CONSOLIDATED_ONLY_PERIOD)]

    all_rows = []
    n_files, n_ok, n_failed, n_date_mismatch = 0, 0, 0, 0

    for _, filing in per_scheme.iterrows():
        n_files += 1
        path = amc_dir / filing["local_filename"]
        try:
            sheets = _read_rows_by_sheet(path)
            rows = next(iter(sheets.values()))
        except Exception as e:
            print(f"  FAILED {filing['local_filename']}: {e}")
            n_failed += 1
            continue
        holdings = parse_axis_scheme_sheet(rows)
        n_ok += 1
        for h in holdings:
            if h.sheet_as_on_date is not None and h.sheet_as_on_date != filing["period_end"]:
                n_date_mismatch += 1
            all_rows.append({
                "amc_name": "Axis Mutual Fund", "scheme_name": filing["scheme_name"],
                "scheme_code": h.scheme_code, "period_end": filing["period_end"],
                "instrument_code": h.instrument_code, "instrument_name": h.instrument_name,
                "isin": h.isin, "industry": h.industry, "quantity": h.quantity,
                "market_value_lakhs": h.market_value_lakhs, "pct_to_nav": h.pct_to_nav,
            })
        if n_ok % 500 == 0:
            print(f"  {n_ok}/{len(per_scheme)} per-scheme files processed, {len(all_rows)} equity rows so far")

    for _, filing in consolidated_only_row.iterrows():
        path = amc_dir / filing["local_filename"]
        sheets = _read_rows_by_sheet(path)
        for sheet_name, rows in sheets.items():
            if sheet_name.lower() == "index":
                continue
            holdings = parse_axis_scheme_sheet(rows)
            for h in holdings:
                all_rows.append({
                    "amc_name": "Axis Mutual Fund", "scheme_name": h.scheme_name,
                    "scheme_code": h.scheme_code, "period_end": filing["period_end"],
                    "instrument_code": h.instrument_code, "instrument_name": h.instrument_name,
                    "isin": h.isin, "industry": h.industry, "quantity": h.quantity,
                    "market_value_lakhs": h.market_value_lakhs, "pct_to_nav": h.pct_to_nav,
                })
        print(f"  consolidated {filing['local_filename']}: {len(sheets)-1} scheme sheets processed")

    print(f"\n{n_ok}/{n_files} per-scheme files parsed ({n_failed} failed), "
          f"{n_date_mismatch} rows with a sheet-vs-filename date mismatch")

    df = pd.DataFrame(all_rows)
    df = df.astype({k: v for k, v in SCHEMA.items() if k in df.columns and k != "sheet_as_on_date"})
    df = df.sort_values(["scheme_name", "period_end", "pct_to_nav"],
                         ascending=[True, True, False]).reset_index(drop=True)
    out_path = amc_dir / "axis_equity_holdings.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} equity-holding rows ({df['scheme_name'].nunique()} schemes, "
          f"{df['period_end'].nunique()} distinct months) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
