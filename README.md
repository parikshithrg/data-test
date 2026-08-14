# Data test

A deterministic research harness for Indian cash-equity strategies.

This is a rebuild. Its predecessor (`market_gate`) produced a dashboard whose
numbers could not be trusted, for reasons that were only discovered at the end:
same-bar fills, no costs until session three, a survivorship-biased universe, no
benchmark comparison, and no count of how many hypotheses had been tried. This
project inverts that order — the harness is built to be honest first, and no
strategy is written until it is.

## The five rules

Each is enforced in code, not by discipline, and each exists because it was
learned the hard way. The reasoning lives next to the code that enforces it.

1. **Deterministic.** Same inputs, byte-identical outputs, provable via run
   manifests. Nothing reads the wall clock — enforced by an AST scan in
   `tests/test_determinism.py`.
2. **Point-in-time.** No datum reaches a decision before it existed. The
   universe is a recomputable rule, never today's index membership.
3. **Executable.** A signal from bar T's close fills at bar T+1's **open**,
   with costs and a participation cap. `config.execution.fill_at` refuses any
   other value. A price you could not have traded is not a result.
4. **Benchmark-relative.** Every headline is excess over NIFTY50, on capital,
   after costs. Absolute return is never the headline.
5. **Counted.** Every hypothesis tested is logged, so the significance bar can
   rise with the size of the search.

## Why the data was rebuilt

The predecessor backtested 2004–2026 against 294 CSVs that are *today's*
Nifty 500 members. Measured on the one window where a comparison is possible
(2019+, `cash_delivery_daily`, 4,619 symbols): **22.5% of EQ symbols trading
before 2020 are gone by 2026** — HDFC, MINDTREE, LTI, CADILAHC, PVR and 366
others. A survivor-only dataset shows 0% there.

Auditing those CSVs also turned up defects nobody had looked for:

| Finding | Detail |
|---|---|
| Two series merged in one file | 25 files carry a flat synthetic row *and* a real-OHLC row at a different price level on the same day |
| Wrong trading calendar | The union of all files gives 8,458 "days" over 26 years vs NSE's ~6,650 |
| Rounding destroys early returns | BAJFINANCE prints `1.00 → 0.50` on six separate dates — one tick, not six splits |
| Literal zero prices | ASHOKLEY and VEDL contain ₹0.00 closes |
| Impossible cross-section | 4 symbols exist in 2000, 120 in 2003 — a top-200 universe is unbuildable before 2015 |

So price history is rebuilt from **NSE daily cash bhavcopy archives** (every
traded symbol, no hindsight, reachable back to 1995). Start date is
**2004-01-01** on structural grounds: STT was introduced in Oct 2004 and rolling
T+2 settlement replaced badla around 2001–02, so applying a modern cost model to
earlier data would be measuring a market that no longer exists.

The bhavcopy format also carries **`prev_close`, which NSE adjusts for corporate
actions** — so `prev_close[t] != close[t-1]` is an exchange-published split or
bonus marker with an exact ratio, rather than a ratio guessed from a suspicious
price jump.

## Layout

```
config/config.toml     every constant; if a number is in code, that is a bug
dtest/
  determinism.py       seeds, content hashes, run manifests
  config.py            typed loader + validation (refuses same-bar fills)
  data/
    prices.py          CSV -> aligned panels, content-addressed cache
    bhavcopy.py        NSE archive ingestion (fetch and parse separated)
    quality.py         data + corporate-action audit
    symbols.py         stock vs index classification
  engine/
    costs.py           statutory Indian delivery schedule
scripts/               runnable entry points; each writes a run manifest
runs/                  manifests + result CSVs (committed — the audit trail)
artifacts/             derived + raw cache (gitignored, regenerable)
```

## Usage

```bash
python scripts/audit_data.py                                    # data audit
python scripts/price_precision.py                               # rounding check
python scripts/ingest_bhavcopy.py --start 2004-01-01 --end 2026-08-13
python -m pytest
```

`ingest_bhavcopy.py` is resumable and safe to re-run: cached days and known
holidays cost no request.

## Costs

Delivery round trip is **0.222% of position** statutory, **0.322%** with the
default 5 bps/side slippage. Rates are looked up, not fitted; slippage is the
single named assumption and is meant to be swept.

Note the units trap, pinned by a test: charges divide by *turnover* (buy+sell),
P&L divides by *position* (buy). A turnover-quoted rate is half the same
position-quoted rate, and mixing them makes costs look survivable when they are
not.
