"""How much of the price history is destroyed by 2-decimal rounding?

Source prices are stored to 2 decimals AFTER back-adjustment for splits and
bonuses. For a name that has since split several times the adjusted early price
collapses toward zero, and the smallest representable move - one paisa - becomes
an enormous percentage. BAJFINANCE prints `1.00 -> 0.50` on six separate dates
in 2003-2009; that is not six splits, it is a one-tick move in a series that has
run out of precision.

This matters because every return, every volatility estimate and every stop
level in the project is computed from these numbers. A quantised return series
does not merely add noise, it adds noise that LOOKS like signal: huge apparent
moves clustered in exactly the low-priced names a momentum ranking will select.

The quantum at price P is 0.01/P. This script measures how much of each year's
data sits above the tolerable threshold, which is what decides the earliest
defensible start date for the project. That start date is then a MEASURED
choice, not the round number 2004 the predecessor project happened to use.

    python scripts/price_precision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config, set_seeds
from dtest.data.prices import build_panels, source_inventory
from dtest.data.symbols import load_symbol_map

# A one-paisa tick should be a small fraction of a typical daily move. NSE daily
# vol is ~2%, so a quantum above 0.20% (a tenth of that) starts to dominate the
# smallest real moves. Price >= Rs 5 satisfies it: 0.01/5 = 0.20%.
MAX_TOLERABLE_QUANTUM_PCT = 0.20


def main() -> int:
    pd.set_option("display.width", 150)
    cfg = load_config()
    set_seeds()

    smap = load_symbol_map(cfg, sorted(source_inventory(cfg)))
    panels = build_panels(cfg, calendar_from=smap.stocks).subset(smap.stocks)
    c = panels.close

    quantum_pct = (0.01 / c) * 100.0        # one tick as a % of price
    ok = quantum_pct <= MAX_TOLERABLE_QUANTUM_PCT
    live = c.notna()

    years = c.index.year
    rows = []
    for y in sorted(set(years)):
        m = years == y
        n_live = int(live[m].to_numpy().sum())
        if not n_live:
            continue
        n_ok = int((ok & live)[m].to_numpy().sum())
        px = c[m].to_numpy()
        px = px[np.isfinite(px)]
        rows.append({
            "year": y,
            "stock_days": n_live,
            "symbols": int(live[m].any().sum()),
            "pct_usable": 100.0 * n_ok / n_live,
            "median_price": float(np.median(px)),
            "p05_price": float(np.percentile(px, 5)),
            "median_quantum_pct": float(np.median(0.01 / px * 100)),
        })
    tab = pd.DataFrame(rows)

    print(f"Precision adequacy by year (tolerable quantum <= {MAX_TOLERABLE_QUANTUM_PCT}% "
          f"of price, i.e. price >= Rs {0.01/MAX_TOLERABLE_QUANTUM_PCT*100:.2f})\n")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

    # Earliest year from which the data is usable at three strictness levels.
    print("\nEarliest year where every subsequent year clears a usability bar:")
    for bar in (90.0, 95.0, 99.0):
        good = tab[tab["pct_usable"] >= bar]["year"].to_numpy()
        start = None
        for y in tab["year"]:
            tail = tab[tab["year"] >= y]
            if (tail["pct_usable"] >= bar).all():
                start = y
                break
        print(f"  >= {bar:4.0f}% of stock-days usable : {start}")

    print("\nWorst offenders (most sub-threshold stock-days):")
    bad = (~ok & live).sum().sort_values(ascending=False, kind="stable").head(10)
    for sym, n in bad.items():
        s = c[sym].dropna()
        print(f"  {sym:14s} {int(n):5d} rows below threshold; "
              f"first={s.index[0].date()} min={s.min():.2f} last={s.iloc[-1]:.2f}")

    out = Path(cfg.paths.runs) / "price_precision"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "by_year.csv", index=False)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
