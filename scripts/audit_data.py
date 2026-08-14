"""Audit the source price data. Run this before trusting any backtest.

    python scripts/audit_data.py

Writes a manifest + CSV reports under runs/<run_id>/ and prints a summary.
Nothing here is a repair: the audit reports, the universe rule decides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.prices import build_panels, source_inventory
from dtest.data.quality import audit
from dtest.data.symbols import load_symbol_map
from dtest.determinism import RunManifest

RUN_ID = "audit_data"


def main() -> int:
    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 30)

    cfg = load_config()
    cfg.paths.check_readable()
    set_seeds()

    available = sorted(source_inventory(cfg))
    smap = load_symbol_map(cfg, available)
    print(f"Symbols on disk: {len(available)}  ->  "
          f"{len(smap.stocks)} stocks, {len(smap.indices)} indices")
    if smap.missing_prices:
        print(f"  listed in nifty500 but no price file: {len(smap.missing_prices)} "
              f"({', '.join(smap.missing_prices[:6])}{' ...' if len(smap.missing_prices) > 6 else ''})")

    manifest = RunManifest(run_id=RUN_ID, as_of=str(cfg.as_of), config=cfg.as_dict())

    # Calendar comes from STOCKS only. Indices and NAV series publish on their
    # own calendars and would inject days the market never traded.
    print(f"\nLoading panels as_of {cfg.as_of} (calendar from stocks) ...")
    panels = build_panels(cfg, manifest=manifest, calendar_from=smap.stocks)
    rep = panels.report
    print(f"  {len(panels.symbols)} symbols x {len(panels.dates)} sessions "
          f"({panels.dates[0].date()} .. {panels.dates[-1].date()})")
    if rep.union_days:
        print(f"  calendar: {rep.calendar_days} trading days kept, "
              f"{rep.dropped_dates} dropped from a {rep.union_days}-day union")
        print(f"  files with a time component : {len(rep.timed_rows)}")
        print(f"  files with same-day dupes   : {len(rep.duplicate_days)} "
              f"(resolved: prefer real-range row, else last)")

    # The tradeable audit is the one that matters. Indices are reported apart.
    stock_panels = panels.subset(smap.stocks)
    print(f"\n--- auditing {len(stock_panels.symbols)} STOCKS ---")
    report = audit(stock_panels)

    print("\n=== ISSUE SUMMARY (stocks) ===")
    summary = report.summary()
    print(summary.to_string(index=False) if not summary.empty else "  none")

    print("\n=== CORPORATE ACTIONS (stocks) ===")
    for n in report.notes:
        print(f"  {n}")
    flagged = report.split_candidates
    if not flagged.empty:
        likely = flagged[flagged["likely_unadjusted_action"]]
        by_year = likely.assign(year=likely["date"].dt.year).groupby("year").size()
        print(f"\n  {len(likely)} suspected unadjusted actions, by year:")
        print("   " + "  ".join(f"{y}:{n}" for y, n in by_year.items()))
        cols = ["symbol", "date", "prev_close", "close", "pct_move",
                "matched_fraction", "volume_ratio_vs_20d"]
        print("\n  sample:")
        print(likely.head(12)[cols].to_string(index=False))

    print("\n=== COVERAGE (stocks) ===")
    cov = report.coverage
    print(f"  median history (rows) : {cov['rows'].median():.0f}")
    print(f"  symbols < 252 rows    : {int((cov['rows'] < 252).sum())}")
    print(f"  median gap rate       : {cov['gap_rate'].median():.4f}")
    print(f"  symbols gap rate > 5% : {int((cov['gap_rate'] > 0.05).sum())}")

    # Sub-Rs-5 prices: the cost model is a fixed percentage and stops being
    # meaningful against a Rs 0.05 tick. Also the signature of over-adjusted
    # early history (ASHOKLEY prints Rs 0.05 in 2003).
    c = stock_panels.close
    cheap = (c < cfg.universe.min_price) & c.notna()
    print(f"  rows below min_price {cfg.universe.min_price}: {int(cheap.to_numpy().sum())} "
          f"across {int((cheap.sum() > 0).sum())} symbols")

    out = Path(cfg.paths.runs) / RUN_ID
    out.mkdir(parents=True, exist_ok=True)
    report.issues.to_csv(out / "issues.csv", index=False)
    report.split_candidates.to_csv(out / "split_candidates.csv", index=False)
    report.coverage.to_csv(out / "coverage.csv", index=False)
    rep.frame().to_csv(out / "load_report.csv", index=False)
    smap.industry_frame().to_csv(out / "stock_industries.csv", index=False)

    manifest.record_frame("close_stocks", stock_panels.close)
    manifest.record_frame("volume_stocks", stock_panels.volume)
    manifest.record_frame("coverage", report.coverage)
    manifest.notes = (
        f"{len(smap.stocks)} stocks, {len(smap.indices)} indices, "
        f"{rep.calendar_days} trading days"
    )
    manifest.finish()
    manifest.write(out)
    print(f"\nWrote {out} (manifest.json + 5 CSVs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
