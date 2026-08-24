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

## Status (2026-08-24)

**31 hypotheses tested, all with the same rigor, `runs/hypothesis_log.csv`:
25 rejected outright, 6 accepted on train but none survived val
confirmation. Zero for 31.** Technical signals (mean reversion, momentum,
delivery/OI/participant-flow breakouts, volatility squeeze, price-action,
pairs trading — both correlation-screened and plain same-sector), three
generalization attempts on the best-performing constructions, and three
fundamentals signals (PEAD, value, quality) have all been tried. The one
consistent pattern across every rejection: real, honestly-measured effects
that do not survive contact with real execution costs, real fills, and an
honest placebo comparison — never a coding bug, always the same
entry-timing/no-edge-after-costs shape. See `runs/hypothesis_log.csv` for
the full, append-only record and each signal's own module docstring for
its specific story and result.

**Currently in a data-collection phase, not actively testing.** After the
technical and fundamentals lines were exhausted, the project pivoted to
building genuinely different data sources — see "Data sources" below —
before writing any new signal. Nothing new has been tested against any of
the sources added since 2026-08-24 yet; that is deliberate, not an
oversight. See "Next steps" at the end of this file.

## Data sources

Beyond the core price/F&O data below, this project has sourced several
category-specific datasets itself, each read-only for the deterministic
harness and each fetched via its own `scripts/fetch_*.py` (never a live
call from inside a signal):

| Source | Module | Coverage | Real caveat |
|---|---|---|---|
| Quarterly financial results | `dtest/data/financial_results.py` | ~2007–2026, 706/926 symbols | No balance sheet in quarterly filings — a value factor must lean on P/E, not P/B |
| Shareholding pattern | `dtest/data/shareholding.py` | ~2021–2026, 597/926 symbols | Real coverage floor is recent — NSE's own XBRL disclosure system, not a scrape limitation |
| Insider trading (SEBI PIT) | `dtest/data/insider_trading.py` | 2015–2026, 584/926 symbols, 251,933 disclosures | 2015 is a real regulatory floor (SEBI PIT Regulations, 2015), not a data gap |
| Index reconstitution calendar | `dtest/data/index_reconstitution.py` | 2010–2026, 18,391 events, 303 indices | Real floor is ~2010 — pre-2010 NSE press releases have no ticker-symbol column at all |
| Macro cross-asset stress | `dtest/data/` (macro fetch) | varies by series | US VIX, USD/INR, DXY, gold |

Two more datasets were investigated and are explicitly **not** built:
- **Bulk/block deal bulletins** — NSE's historical-deals API currently
  returns a genuine server-side 503 on every parameter combination tried
  (confirmed reaching NSE's real backend, not a firewall block) — likely a
  temporary outage, worth retrying later.
- **AMC monthly portfolio disclosures** — no central AMFI aggregator
  exists; real coverage means scraping ~40 individual fund-house websites,
  most of them JavaScript-rendered with no clean underlying API found on
  the ones checked (HDFC, Axis, UTI, Mirae). Needs real browser-automation
  tooling this project doesn't have yet — scoped, deliberately not started.

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
    fno_oi.py           continuous front-month OI stitcher
    fno_price.py         continuous front-month futures price
    financial_results.py quarterly P&L filings (NSE corporates-financial-results)
    shareholding.py       promoter/MF/FII/DII/pledge % (NSE shareholding XBRL)
    insider_trading.py    SEBI PIT buy/sell/pledge disclosures (NSE corporates-pit)
    index_reconstitution.py  NIFTY index add/exclude events (NSE Indices press releases)
  features/             point-in-time feature layer (technical, fundamentals, pairs, regime)
  signals/               one file per hypothesis (11 built, see Status)
  engine/
    costs.py            statutory Indian delivery schedule
    futures_costs.py     statutory Indian F&O schedule (STT sell-side only)
    simulate.py          long-only single-leg simulator, T+1-open fills
    pairs_simulate.py    two-leg long+short simulator, rollover-aware
  evaluate/              metrics, 30-seed placebo, append-only hypothesis_log.csv
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

## Next steps

**Collection phase, in priority order (see `hypothesis_log.csv` and the
"Status" section above for why testing is paused until this completes):**

1. ~~Shareholding pattern~~ — done
2. Bulk & block deal bulletins — **blocked**, NSE's historical API is
   currently down server-side; retry later
3. ~~Insider trading (SEBI PIT)~~ — done
4. AMC monthly portfolio disclosures — **scoped, not started**; needs a
   real decision on investing in browser-automation tooling for ~40
   individual fund-house sites, or dropping it from scope
5. ~~Index reconstitution calendar~~ — done
6. **ETF flow / AUM data (AMFI)** — next up, not yet started
7. Per-stock options chain (strike-level OI/volume/IV, ~180-200 F&O names)
8. Per-stock participant-wise OI (index-wide version already tried and
   rejected as `participant_tilt` — per-stock breakdown untested)
9. Credit rating actions (CRISIL/ICRA/CARE/India Ratings)
10. G-Sec yield curve (RBI) — macro regime context, not a standalone signal
11. Corporate announcement feed (M&A, contract wins, dividends/buybacks)
12. Earnings call transcripts — richer than existing RSS headlines
13. Macro series (RBI repo/M3/forex, MOSPI IIP/CPI/GDP, GST collections)
14. Global cross-asset (FRED rates, crude, EM-FX, global equity indices)

**Once collection is declared done**, the honest next analytical step is a
genuinely new construction on the data already in hand, not another
parameter retune of PEAD/value/quality/mean-reversion/momentum — those
have each had their own real, disciplined test. Candidates worth
considering first: a fundamentals trigger combined with the entry-timing
delay already found real for technical signals; a cross-sectional
fundamentals ranking rather than an absolute-threshold trigger; or a
signal built directly on the new shareholding/insider-trading/
reconstitution data (e.g. promoter-pledge trend, insider net-buying
streaks, pre-effective-date positioning ahead of a known index add) —
none of these have been tried yet.
