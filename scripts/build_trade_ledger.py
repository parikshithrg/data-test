"""DIAGNOSTIC / REPORTING ONLY - cross-strategy trade-level analytics across
every hypothesis logged in `runs/hypothesis_log.csv`, built from the REAL
trades.csv files already saved under runs/<name>/ - nothing here is
estimated or fabricated, and nothing is written back to hypothesis_log.csv.

    python scripts/build_trade_ledger.py

Feeds the published "Data Test Ledger" artifact (first built 2026-08-26):
per-strategy stock/trade counts, win/loss streaks, and simulated brokerage
cost, plus a most-traded-stocks-across-everything view.

MANIFEST keyed by hypothesis_id (hypothesis_log.csv's own identifier, not
row position) so it survives the log being re-sorted or re-filtered. This is
a separate, purpose-built manifest from `monte_carlo_hypotheses.py`'s own
MANIFEST: that one only covers single-leg, weekly-bucketable trades for a
bootstrap and skips both pairs-trading rows and the 2026-08-19+ additions
(sector-pair variants, style factors, MF-holdings hypotheses) entirely -
this one covers all 34 logged rows, including two-leg pairs trades. Every
single-leg row's resolved-trade count is asserted against hypothesis_log's
own logged n_trades before being trusted, same discipline that file's own
MANIFEST comment describes.

PAIRS-TRADE DOUBLE-COUNTING, DELIBERATE - a pair position touches two
symbols at once, so each leg becomes its own row here: a stock's trade/cost
count includes every pair it was ever a member of on either side. This
means a pair hypothesis's logged n_trades (one row per pair-trade) is
exactly HALF this script's per-leg trade count for that hypothesis - by
design, not a mismatch. net_pnl_pct is the pair's combined return, attached
identically to both legs; it is not separably additive per leg.

hypothesis_id cdd796d6e171 (log_index 6, pairs_reversion pre-rollforward-fix)
has NO surviving raw trades file: the 2026-08-18 rollforward-timing fix
re-ran and overwrote runs/pairs_reversion_honest/ in place. hypothesis_id
0b11b017cef9 (log_index 8) is the corrected re-run of the same construction
- hypothesis_log.csv's own `supersedes` column is blank for both (a gap in
how that column was populated at the time, not evidence they're unrelated).
This script attaches 0b11b017cef9's surviving file to BOTH rows so log_index
6 isn't silently dropped from the ledger, and skips the n_trades assertion
for it alone since its logged 197 reflects data that no longer exists on
disk (0b11b017cef9's own logged 216 is asserted normally).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"

# hypothesis_id -> (kind, path relative to runs/)
# kind: "single" (symbol/net_pnl_pct/cost_pct) or "pair" (two-leg)
MANIFEST: dict[str, tuple[str, str]] = {
    "5d7650b1840e": ("single", "mean_reversion_primary_train/trades.csv"),
    "61fcd3023c4b": ("single", "delivery_breakout_delivery_train/trades.csv"),
    "80aff81a749e": ("single", "oi_momentum_primary_train/trades.csv"),
    "ee447825d3a7": ("single", "participant_tilt_delivery_train/trades.csv"),
    "663cebb9f054": ("single", "vol_squeeze_breakout_primary_train/trades.csv"),
    "e39de46a0f24": ("single", "vol_squeeze_breakout_delay2_primary_train/trades.csv"),
    "cdd796d6e171": ("pair", "pairs_reversion_honest/real_trades_primary_train.csv"),  # see module docstring - superseded, file borrowed from 0b11b017cef9
    "ddc14822fb70": ("single", "price_action_long_primary_train/trades.csv"),
    "0b11b017cef9": ("pair", "pairs_reversion_honest/real_trades_primary_train.csv"),
    "c55019a896e7": ("pair", "same_sector_pairing/random_primary_train.csv"),
    "1d82fec2bbbc": ("pair", "same_sector_pairing/liquidity_primary_train.csv"),
    "5dbcd00310b3": ("pair", "same_sector_pairing/random_primary_val.csv"),
    "a7f9414d3392": ("pair", "same_sector_pairing/liquidity_primary_val.csv"),
    "71357c1af8cd": ("pair", "same_sector_pairing/random_delivery_train.csv"),
    "99a2610cabee": ("pair", "same_sector_pairing/liquidity_delivery_train.csv"),
    "fac149cb6b0b": ("single", "vol_squeeze_breakout_delivery_train/trades.csv"),
    "423c548cc0d3": ("single", "momentum_primary_train/trades.csv"),
    "0d8f2ac002c2": ("single", "mean_reversion_delivery_train/trades.csv"),
    "c9d3ab9ffc36": ("single", "oi_momentum_delivery_train/trades.csv"),
    "fc8303956df7": ("single", "price_action_long_delivery_train/trades.csv"),
    "0151b95c725e": ("single", "momentum_delivery_train/trades.csv"),
    "f68079c5b0b8": ("pair", "pairs_reversion_honest/real_trades_delivery_train.csv"),
    "ee89fa192a3e": ("single", "momentum_delivery_val/trades.csv"),
    "dc18a3964319": ("single", "participant_tilt_stress_gated_delivery_train/trades.csv"),
    "180ea073ac1a": ("pair", "sector_pairing_oilgas/random_primary_train.csv"),
    "0e0d594da875": ("pair", "sector_pairing_oilgas/liquidity_primary_train.csv"),
    "12b05d12a039": ("pair", "sector_pairing_oilgas/random_primary_val.csv"),
    "f3eb92550553": ("pair", "sector_pairing_oilgas/liquidity_primary_val.csv"),
    "b3c0793a9edd": ("single", "earnings_surprise_primary_train/trades.csv"),
    "e39e744cc436": ("single", "value_primary_train/trades.csv"),
    "f5263108087e": ("single", "quality_primary_train/trades.csv"),
    "7facf033cb36": ("single", "mf_accumulation_delivery_train/trades.csv"),
    "270dd119a8fb": ("single", "mf_new_entrant_delivery_train/trades.csv"),
    "345373baf942": ("single", "mf_breadth_delivery_train/trades.csv"),
}

# hypothesis_ids whose n_trades assertion is skipped, with the reason - see
# module docstring for cdd796d6e171.
SKIP_TRADE_COUNT_CHECK = {"cdd796d6e171"}

TARGET_VALUE_PER_TRADE = 10_000.0  # this project's own trade-level sizing target
TOP_N_STOCKS = 25


def load_normalized(kind: str, path: Path) -> pd.DataFrame:
    """Returns a normalized frame: symbol, exit_date, net_pct, cost_pct,
    position_value (approx, Rs). For pair trades, each leg becomes its own
    row (see module docstring on double-counting)."""
    df = pd.read_csv(path)
    resolved = df["exit_reason"].notna() & ~df["exit_reason"].isin(["no_fill", "unresolved"]) & df["net_pnl_pct"].notna()
    if kind == "single":
        out = pd.DataFrame({
            "symbol": df["symbol"],
            "exit_date": pd.to_datetime(df["exit_date"], errors="coerce"),
            "net_pct": df["net_pnl_pct"],
            "cost_pct": df["cost_pct"],
            "position_value": df["entry_price"] * df["shares"],
        })
        return out[resolved].copy()
    else:  # pair
        base = df[resolved].copy()
        rows = []
        for _, r in base.iterrows():
            exit_date = pd.to_datetime(r.get("exit_fill_date") or r.get("signal_exit_date"), errors="coerce")
            for leg_symbol, leg_cost in (
                (r.get("long_symbol"), r.get("long_cost_pct")),
                (r.get("short_symbol"), r.get("short_cost_pct")),
            ):
                rows.append({
                    "symbol": leg_symbol,
                    "exit_date": exit_date,
                    "net_pct": r["net_pnl_pct"],
                    "cost_pct": leg_cost,
                    "position_value": TARGET_VALUE_PER_TRADE,  # pairs sizing differs; approximate
                })
        return pd.DataFrame(rows)


def streaks(sorted_net_pct: list[float]) -> dict:
    """Consecutive win/loss streak lengths from a chronologically-ordered
    list of net_pct values (win = net_pct > 0)."""
    win_streaks, loss_streaks = [], []
    cur_sign, cur_len = None, 0
    for v in sorted_net_pct:
        sign = "win" if v > 0 else "loss"
        if sign == cur_sign:
            cur_len += 1
        else:
            if cur_sign == "win":
                win_streaks.append(cur_len)
            elif cur_sign == "loss":
                loss_streaks.append(cur_len)
            cur_sign, cur_len = sign, 1
    if cur_sign == "win":
        win_streaks.append(cur_len)
    elif cur_sign == "loss":
        loss_streaks.append(cur_len)
    return {
        "max_win_streak": max(win_streaks) if win_streaks else 0,
        "avg_win_streak": sum(win_streaks) / len(win_streaks) if win_streaks else 0.0,
        "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
        "avg_loss_streak": sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0.0,
    }


def main():
    log = pd.read_csv(RUNS / "hypothesis_log.csv")
    strategies = []
    per_stock_all = []
    missing = []

    for idx, log_row in log.iterrows():
        hyp_id = log_row["hypothesis_id"]
        if hyp_id not in MANIFEST:
            missing.append((idx, hyp_id))
            continue
        kind, relpath = MANIFEST[hyp_id]
        path = RUNS / relpath
        if not path.exists():
            missing.append((idx, relpath))
            continue

        norm = load_normalized(kind, path)
        norm = norm.dropna(subset=["net_pct", "exit_date"])
        norm = norm.sort_values("exit_date", kind="stable")

        n_trades = len(norm)
        if hyp_id not in SKIP_TRADE_COUNT_CHECK:
            expected = log_row["n_trades"] * (2 if kind == "pair" else 1)
            assert n_trades == expected, (
                f"log_index {idx} ({hyp_id}): resolved trade count {n_trades} != "
                f"expected {expected} (logged n_trades={log_row['n_trades']}, kind={kind})"
            )

        label = f"{log_row['title']} [{log_row['split']}/{log_row['window']}]"
        n_stocks = norm["symbol"].nunique()
        s = streaks(norm["net_pct"].tolist())
        total_cost_rs = (norm["cost_pct"] / 100.0 * norm["position_value"]).sum()
        mean_net_pct = norm["net_pct"].mean() if n_trades else float("nan")

        strategies.append({
            "log_index": int(idx),
            "hypothesis_id": hyp_id,
            "strategy": label,
            "decision": log_row["decision"],
            "n_stocks_traded": int(n_stocks),
            "n_trades": int(n_trades),
            "avg_trades_per_stock": round(n_trades / n_stocks, 3) if n_stocks else 0.0,
            "mean_net_pct": round(mean_net_pct, 3),
            "logged_real_value": float(log_row["real_value"]),
            "max_win_streak": s["max_win_streak"],
            "avg_win_streak": round(s["avg_win_streak"], 3),
            "max_loss_streak": s["max_loss_streak"],
            "avg_loss_streak": round(s["avg_loss_streak"], 3),
            "total_cost_rs": round(float(total_cost_rs), 2),
            "avg_cost_pct": round(norm["cost_pct"].mean(), 3) if n_trades else float("nan"),
        })

        for sym, g in norm.groupby("symbol"):
            per_stock_all.append({
                "strategy": label, "symbol": sym, "n_trades": len(g),
                "mean_net_pct": g["net_pct"].mean(), "total_net_pct": g["net_pct"].sum(),
            })

    if missing:
        raise RuntimeError(f"missing manifest entries or files: {missing}")

    summary = pd.DataFrame(strategies)
    per_stock = pd.DataFrame(per_stock_all)

    top_stocks_df = (
        per_stock.groupby("symbol")
        .agg(n_trades=("n_trades", "sum"),
             strategies=("strategy", "nunique"),
             mean_net_pct=("mean_net_pct", "mean"))
        .sort_values("n_trades", ascending=False)
        .head(TOP_N_STOCKS)
        .reset_index()
    )
    top_stocks_df["mean_net_pct"] = top_stocks_df["mean_net_pct"].round(3)
    top_stocks = top_stocks_df.to_dict(orient="records")

    all_stocks = set(per_stock["symbol"].unique()) if len(per_stock) else set()
    grand = {
        "n_strategies": len(summary),
        "n_distinct_stocks_all": len(all_stocks),
        "n_total_trades": int(summary["n_trades"].sum()),
        "total_cost_rs_all": float(summary["total_cost_rs"].sum()),
        "avg_cost_pct_all": float((summary["avg_cost_pct"] * summary["n_trades"]).sum() / summary["n_trades"].sum()),
    }

    print(f"strategies: {len(summary)}, distinct stocks: {grand['n_distinct_stocks_all']}, "
          f"total trades: {grand['n_total_trades']}")

    summary.to_csv(RUNS / "trade_ledger_summary.csv", index=False)
    per_stock.to_csv(RUNS / "trade_ledger_per_stock.csv", index=False)
    top_stocks_df.to_csv(RUNS / "trade_ledger_top_stocks.csv", index=False)
    with open(RUNS / "trade_ledger_report_data.json", "w") as f:
        json.dump({"grand": grand, "strategies": strategies, "top_stocks": top_stocks}, f, indent=2)


if __name__ == "__main__":
    main()
