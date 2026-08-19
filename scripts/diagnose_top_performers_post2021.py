"""DIAGNOSTIC ONLY - post-2021 counterpart to `diagnose_top_performers.py`,
for the 6 stocks that repeated across multiple strategies' best-5 lists in
the post-2021 breakdown (TATAELXSI, IRCTC, ABFRL, MPHASIS, ADANIPOWER,
BAJFINANCE - see `analyze_stock_performance_post2021.py`). Same method:
market-regime backdrop at each winning trade, sector-peer/NIFTY excess over
the identical window, and a full buy-and-hold check over the span the
trades occurred in.

    python scripts/diagnose_top_performers_post2021.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dtest import load_config
from dtest.data.bhav_store import build_store, load_long, to_panel
from dtest.data.bhavcopy import COLUMNS as BHAV_COLUMNS
from dtest.features.regime import trailing_return
from dtest.features.stress import cross_asset_stress_composite
from dtest.universe import build_universe

RUNS = Path(__file__).resolve().parent.parent / "runs"
CUTOFF = pd.Timestamp("2021-01-01")

# (symbol, strategy, trades file, kind, date_col, exit_col)
CELLS = [
    ("TATAELXSI", "delivery_breakout", "delivery_breakout_delivery_train/trades.csv",
     "single", "entry_date", "exit_date"),
    ("IRCTC", "delivery_breakout", "delivery_breakout_delivery_train/trades.csv",
     "single", "entry_date", "exit_date"),
    ("ABFRL", "delivery_breakout", "delivery_breakout_delivery_train/trades.csv",
     "single", "entry_date", "exit_date"),
    ("IRCTC", "oi_momentum", "oi_momentum_delivery_train/trades.csv",
     "single", "entry_date", "exit_date"),
    ("MPHASIS", "participant_tilt_stress_gated",
     "participant_tilt_stress_gated_delivery_train/trades.csv",
     "single", "entry_date", "exit_date"),
    ("TATAELXSI", "vol_squeeze_breakout", "vol_squeeze_breakout_delivery_train/trades.csv",
     "single", "entry_date", "exit_date"),
    ("ABFRL", "vol_squeeze_breakout", "vol_squeeze_breakout_delivery_train/trades.csv",
     "single", "entry_date", "exit_date"),
    ("ADANIPOWER", "price_action_long", "price_action_long_delivery_train/trades.csv",
     "single", "entry_date", "exit_date"),
    ("ADANIPOWER", "momentum", "momentum_delivery_train/trades.csv",
     "single", "entry_date", "exit_date"),
    ("BAJFINANCE", "pairs_reversion", "pairs_reversion_honest/real_trades_delivery_train.csv",
     "pairs", "entry_fill_date", "exit_fill_date"),
    ("BAJFINANCE", "same_sector_pairing", "same_sector_pairing/random_delivery_train.csv",
     "pairs", "entry_fill_date", "exit_fill_date"),
    ("TATAELXSI", "same_sector_pairing", "same_sector_pairing/random_delivery_train.csv",
     "pairs", "entry_fill_date", "exit_fill_date"),
    ("MPHASIS", "same_sector_pairing", "same_sector_pairing/random_delivery_train.csv",
     "pairs", "entry_fill_date", "exit_fill_date"),
]

REPEAT_SYMBOLS = ["TATAELXSI", "IRCTC", "ABFRL", "MPHASIS", "ADANIPOWER", "BAJFINANCE"]


def _load_price_panels(cfg):
    cache = Path(cfg.paths.artifacts) / "bhav"
    build_store(cache, years=None)
    long_df = load_long(cache, columns=list(BHAV_COLUMNS))
    stocks = sorted(long_df["symbol"].unique())
    panels = {f: to_panel(long_df, f) for f in ("open", "high", "low", "close",
                                                 "volume", "turnover")}
    return panels, stocks


def _pairs_rows_for_symbol(df: pd.DataFrame, symbol: str, date_col: str, exit_col: str) -> pd.DataFrame:
    long_rows = df[(df["long_symbol"] == symbol) & df["long_net_pct"].notna()].copy()
    long_rows["own_return_pct"] = long_rows["long_net_pct"]
    long_rows["leg"] = "long"
    short_rows = df[(df["short_symbol"] == symbol) & df["short_net_pct"].notna()].copy()
    short_rows["own_return_pct"] = short_rows["short_net_pct"]
    short_rows["leg"] = "short"
    out = pd.concat([long_rows, short_rows], ignore_index=True)
    return out[[date_col, exit_col, "own_return_pct", "leg"]]


def main() -> int:
    pd.set_option("display.width", 180)
    cfg = load_config()

    print("loading full price panels ...")
    panels, stocks = _load_price_panels(cfg)
    close, turnover = panels["close"], panels["turnover"]
    stock_close, stock_turnover = close[stocks], turnover[stocks]
    dates = stock_close.index

    print("building point-in-time universe ...")
    uni = build_universe(stock_close, stock_turnover, cfg)
    eligible = uni.membership

    sector = pd.read_csv(cfg.paths.industry_map)
    sector_map = dict(zip(sector["symbol"].astype(str).str.strip(),
                           sector["industry"].astype(str).str.strip()))

    print("loading NIFTY50 ...")
    nifty = pd.read_csv(cfg.paths.price_dir / "NIFTY50_DAILY.csv",
                         parse_dates=["date"]).set_index("date").sort_index()
    nifty = nifty.loc[nifty.index <= pd.Timestamp(cfg.as_of)]["close"].reindex(dates).ffill()
    nifty_trail = trailing_return(nifty, lookback=63)

    print("loading macro stress series and building the composite ...")
    india_vix = pd.read_csv(cfg.paths.price_dir / "INDIAVIX_DAILY.csv",
                             parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    us_vix = pd.read_csv(cfg.paths.macro_dir / "VIX_DAILY.csv",
                          parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    usdinr = pd.read_csv(cfg.paths.macro_dir / "USDINR_DAILY.csv",
                          parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    dxy = pd.read_csv(cfg.paths.macro_dir / "DXY_DAILY.csv",
                       parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    gold = pd.read_csv(cfg.paths.macro_dir / "GOLD_DAILY.csv",
                        parse_dates=["date"]).set_index("date")["close"].reindex(dates).ffill()
    sma200 = stock_close.rolling(200, min_periods=200).mean()
    above = (stock_close > sma200) & eligible
    denom = eligible.sum(axis=1).astype(float).replace(0.0, np.nan)
    breadth_pct = (above.sum(axis=1).astype(float) / denom * 100.0)
    dims = cross_asset_stress_composite(india_vix, us_vix, breadth_pct, usdinr, dxy, gold)

    def _sector_peer_return(symbol: str, entry, exit_) -> float:
        sec = sector_map.get(symbol)
        if sec is None or pd.isna(entry) or pd.isna(exit_):
            return np.nan
        peers = [s for s, sc in sector_map.items() if sc == sec and s != symbol
                 and s in stock_close.columns]
        if not peers:
            return np.nan
        try:
            p0 = stock_close.loc[entry, peers]
            p1 = stock_close.loc[exit_, peers]
        except KeyError:
            return np.nan
        rets = (p1 / p0 - 1.0) * 100.0
        return float(rets.dropna().mean()) if rets.notna().any() else np.nan

    rows = []
    for symbol, strategy, rel_path, kind, date_col, exit_col in CELLS:
        df = pd.read_csv(RUNS / rel_path, parse_dates=[date_col, exit_col])
        if kind == "single":
            sub = df[(df["symbol"] == symbol) & df["net_pnl_pct"].notna()].copy()
            sub["own_return_pct"] = sub["net_pnl_pct"]
            sub["leg"] = "long"
            sub = sub[[date_col, exit_col, "own_return_pct", "leg"]]
        else:
            sub = _pairs_rows_for_symbol(df, symbol, date_col, exit_col)
        sub = sub[sub[date_col] >= CUTOFF]

        for _, r in sub.iterrows():
            entry, exit_, own_ret, leg = r[date_col], r[exit_col], float(r["own_return_pct"]), r["leg"]
            nifty_window_ret = (
                (nifty.reindex([exit_]).iloc[0] / nifty.reindex([entry]).iloc[0] - 1.0) * 100.0
                if pd.notna(entry) and pd.notna(exit_)
                and entry in nifty.index and exit_ in nifty.index else np.nan
            )
            sector_peer_ret = _sector_peer_return(symbol, entry, exit_)
            trail = nifty_trail.reindex([entry]).iloc[0] if entry in nifty_trail.index else np.nan
            stress = dims["composite"].reindex([entry]).iloc[0] if entry in dims.index else np.nan

            rows.append({
                "symbol": symbol, "strategy": strategy, "leg": leg,
                "entry_date": entry, "exit_date": exit_,
                "own_return_pct": own_ret, "nifty_window_return_pct": nifty_window_ret,
                "sector_peer_return_pct": sector_peer_ret,
                "excess_vs_nifty_pct": own_ret - nifty_window_ret if pd.notna(nifty_window_ret) else np.nan,
                "excess_vs_sector_pct": own_ret - sector_peer_ret if pd.notna(sector_peer_ret) else np.nan,
                "nifty_trailing_63d_return_pct": trail * 100.0 if pd.notna(trail) else np.nan,
                "stress_composite_at_entry": stress,
            })

    out = pd.DataFrame(rows)
    out_dir = RUNS / "top_performer_diagnosis_post2021"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "trade_level_regime.csv", index=False)

    print("\n=== PER-TRADE DETAIL ===")
    print(out[["symbol", "strategy", "leg", "entry_date", "own_return_pct", "excess_vs_nifty_pct",
               "excess_vs_sector_pct", "nifty_trailing_63d_return_pct",
               "stress_composite_at_entry"]].to_string(index=False))

    print("\n\n=== PER-SYMBOL SUMMARY ===")
    summary = out.groupby("symbol").agg(
        n_trades=("own_return_pct", "size"),
        mean_own_return_pct=("own_return_pct", "mean"),
        mean_excess_vs_nifty_pct=("excess_vs_nifty_pct", "mean"),
        mean_excess_vs_sector_pct=("excess_vs_sector_pct", "mean"),
        mean_nifty_trailing_63d_pct=("nifty_trailing_63d_return_pct", "mean"),
        mean_stress_composite=("stress_composite_at_entry", "mean"),
        pct_beat_nifty=("excess_vs_nifty_pct", lambda s: 100 * (s > 0).mean()),
    ).reset_index()
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "per_symbol_summary.csv", index=False)

    print("\n\n=== FULL-HISTORY BUY-AND-HOLD CHECK (own stock vs NIFTY50, "
          "over the span its post-2021 winning trades occurred in) ===")
    bh_rows = []
    for symbol in REPEAT_SYMBOLS:
        sym_trades = out[out["symbol"] == symbol]
        if sym_trades.empty or symbol not in stock_close.columns:
            continue
        span_start, span_end = sym_trades["entry_date"].min(), sym_trades["exit_date"].max()
        s = stock_close[symbol].reindex(dates)
        p0, p1 = s.reindex([span_start]).iloc[0], s.reindex([span_end]).iloc[0]
        n0, n1 = nifty.reindex([span_start]).iloc[0], nifty.reindex([span_end]).iloc[0]
        stock_bh = (p1 / p0 - 1.0) * 100.0 if pd.notna(p0) and pd.notna(p1) and p0 > 0 else np.nan
        nifty_bh = (n1 / n0 - 1.0) * 100.0 if pd.notna(n0) and pd.notna(n1) and n0 > 0 else np.nan
        bh_rows.append({"symbol": symbol, "span_start": span_start.date(), "span_end": span_end.date(),
                         "stock_buy_and_hold_pct": stock_bh, "nifty_buy_and_hold_pct": nifty_bh,
                         "stock_minus_nifty_pct": stock_bh - nifty_bh if pd.notna(nifty_bh) else np.nan})
    bh = pd.DataFrame(bh_rows)
    print(bh.to_string(index=False))
    bh.to_csv(out_dir / "buy_and_hold_check.csv", index=False)

    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
