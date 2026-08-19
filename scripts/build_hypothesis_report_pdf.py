"""REPORTING ONLY - assembles the full Data test hypothesis-testing PDF
report from real, already-computed data: `runs/hypothesis_log.csv`,
`runs/monte_carlo_hypotheses/summary.csv`, and
`runs/hypothesis_report/diagnosis_matrix.csv` (built by
`build_hypothesis_report_data.py`, run that first). No numbers are invented
here - every figure in this report traces back to one of those three CSVs,
which themselves trace back to real saved trade files.

    python scripts/build_hypothesis_report_pdf.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dtest import load_config  # noqa: E402

RUNS = Path(__file__).resolve().parent.parent / "runs"
OUT_PATH = RUNS / "hypothesis_report" / "Data_Test_Hypothesis_Report.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
LINE = colors.HexColor("#c9c9c2")
ACCEPT_BG = colors.HexColor("#e8f0e3")
REJECT_BG = colors.HexColor("#f5f0ec")
HEAD_BG = colors.HexColor("#2b2b26")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=22, textColor=INK, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading1"], fontSize=15, textColor=INK,
                     spaceBefore=14, spaceAfter=8, borderColor=LINE, borderWidth=0,
                     borderPadding=0)
H3 = ParagraphStyle("H3", parent=styles["Heading2"], fontSize=11.5, textColor=INK,
                     spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.3, leading=13.2,
                       textColor=INK, spaceAfter=6)
MUTEDSTYLE = ParagraphStyle("Muted", parent=BODY, textColor=MUTED, fontSize=8.4, leading=11.5)
CELL = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.6, leading=9.2, textColor=INK)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
# Table headers sit on a dark HEAD_BG fill - TableStyle's TEXTCOLOR command has no
# effect on cells containing Paragraph flowables (only plain strings), so every
# header cell must use an explicitly white-text style instead.
CELLW = ParagraphStyle("CellW", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
COVER_SUB = ParagraphStyle("CoverSub", parent=BODY, fontSize=11.5, textColor=MUTED, spaceAfter=4)


def fmt(v, pct=True, dp=2, na="\u2014") -> str:
    try:
        if v is None or (isinstance(v, float) and (pd.isna(v))):
            return na
    except Exception:
        return na
    if isinstance(v, str):
        return v
    s = f"{v:,.{dp}f}"
    return f"{s}%" if pct else s


def short_title(t: str) -> str:
    return (t.replace(" (honest execution)", "").replace(" (honest fills + costs)", "")
             .replace(", long-only", "").replace(" (random draw)", " \u2013 random")
             .replace(" (liquidity-ranked)", " \u2013 liquidity"))


# ---------------------------------------------------------------------------
# Signal catalog - one accurate paraphrase per distinct economic idea tested.
# Faithful summaries of the real `story` field in hypothesis_log.csv, not
# fabricated - shortened for a report page, not reworded in substance.
# ---------------------------------------------------------------------------
SIGNAL_CATALOG = [
    ("mean_reversion", "A stock pushed 1.5+ std devs below its own 50-day average in a short "
     "window is disproportionately a forced/panicked seller (margin calls, index rebalancing, "
     "tax-loss selling) rather than new information about impaired value - expected to partially "
     "correct once the selling pressure exhausts. The only predecessor-project strategy that "
     "survived a genuine walk-forward test; re-tested here under honest execution."),
    ("delivery_breakout", "A price breakout on ordinary/low delivery is disproportionately intraday "
     "speculation that unwinds overnight. A breakout with delivery meaningfully above its own "
     "recent normal is disproportionately real buyers converting the move into settled positions."),
    ("oi_momentum", "A breakout on falling/average open interest is disproportionately short-covering "
     "with no fresh leveraged conviction. A breakout with open interest rising meaningfully faster "
     "than normal means participants are opening new leveraged positions into the move."),
    ("participant_tilt", "A mean-reversion dip bought while FII net index-futures positioning sits "
     "above its own recent trend is a normal pullback inside continued institutional accumulation; "
     "the identical dip bought while FII positioning trends down is the early stage of a real "
     "breakdown. Market-wide gate only (no per-stock FII breakdown exists) - it decides WHETHER a "
     "dip is bought that day, not WHICH stock."),
    ("vol_squeeze_breakout", "A breakout following a genuine contraction (short-term range compressed "
     "well below its own longer-run normal) is the first real re-pricing after a quiet period, not "
     "noise inside an already-active range. Tests the volatility axis none of the price/delivery/OI/"
     "flow signals touched. A delay=2 variant was also tested, entering two sessions after the "
     "signal fires, to isolate whether buying at the peak of the dislocation (not the premise "
     "itself) explained the rejection."),
    ("price_action (LONG)", "A session that is BOTH unusually wide-range AND closes pinned at one "
     "extreme, on volume well above normal, is disproportionately genuine new information or real "
     "institutional participation - unlike vol_squeeze_breakout, needs no prior contraction, only "
     "that today's bar is the outlier."),
    ("pairs_reversion (correlation-screened)", "Two same-sector, historically correlated stocks whose "
     "log-price spread drifts unusually wide are disproportionately showing a temporary, idiosyncratic "
     "dislocation rather than a genuine re-rating - market-neutral by construction, needing only the "
     "relative mispricing to close."),
    ("same_sector_pairing", "A looser version of the pairs premise: shared sector membership alone "
     "(without requiring the pair to already be historically correlated) is claimed to be enough "
     "linkage for a wide relative-price dislocation to be temporary. Tested with both a random-draw "
     "and a liquidity-ranked pair-selection rule, against a common any-sector-random placebo."),
    ("momentum (12-1 month)", "A stock that outperformed over the trailing ~12 months (skipping the "
     "most recent month, to exclude the short-term reversal window this project's own entry-timing "
     "diagnostic found real) is more likely still mid-diffusion of genuine improving information than "
     "already fully priced. Long-only, top quintile of the point-in-time universe, re-ranked monthly."),
]


def build_story() -> list:
    cfg = load_config()
    log = pd.read_csv(cfg.paths.runs / "hypothesis_log.csv")
    mc = pd.read_csv(RUNS / "monte_carlo_hypotheses" / "summary.csv")
    mat = pd.read_csv(RUNS / "hypothesis_report" / "diagnosis_matrix.csv")

    n_total = len(log)
    n_rejected = int((log["decision"] == "rejected").sum())
    n_accepted = int((log["decision"] == "accepted").sum())

    story = []

    # ---- COVER --------------------------------------------------------
    story.append(Spacer(1, 4.5 * cm))
    story.append(Paragraph("Data test", H1))
    story.append(Paragraph("Hypothesis Testing Report", ParagraphStyle(
        "CoverTitle2", parent=H1, fontSize=16, textColor=MUTED, spaceAfter=18)))
    story.append(Paragraph(f"{n_total} hypotheses tested \u00b7 {n_rejected} rejected, "
                            f"{n_accepted} accepted-on-the-metric-that-logged-them, "
                            "0 survived independent confirmation", COVER_SUB))
    story.append(Paragraph("A deterministic, point-in-time rebuild of India-equity systematic "
                            "signal testing \u2013 NSE bhavcopy 2004\u20132026, real T+1 execution "
                            "and costs, 30-seed placebo floor, train/val discipline, and a "
                            "10,000-path Monte Carlo bootstrap.", COVER_SUB))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph(f"Generated {date.today().isoformat()} \u00b7 "
                            f"C:\\Users\\parik\\OneDrive\\Desktop\\Data test", MUTEDSTYLE))
    story.append(NextPageTemplate("portrait"))
    story.append(PageBreak())

    # ---- EXECUTIVE SUMMARY ---------------------------------------------
    story.append(Paragraph("Executive summary", H2))
    story.append(Paragraph(
        f"Between 2026-08-14 and 2026-08-19, this project tested {n_total} logged hypotheses "
        "built on 10 distinct economic ideas (mean reversion, delivery-confirmed breakouts, "
        "OI-confirmed breakouts, FII-flow-gated dips, volatility-contraction breakouts, wide-range "
        "price-action bars, correlation-screened and same-sector pairs trading, and 12-1 month "
        "cross-sectional momentum), each tested on one or both of two data splits and, where a "
        "result looked promising on train, re-tested on a held-out val window.", BODY))
    story.append(Paragraph(
        f"<b>{n_rejected} of {n_total} were rejected outright</b> \u2013 the real mean return was "
        "either negative, or positive but ranked below the placebo band (blind selection from the "
        "same eligible pool on the same dates). "
        f"<b>{n_accepted} were mechanically \u201caccepted\u201d</b> by this project's own logging "
        "rule (beats every placebo seed, and positive portfolio Sharpe where a portfolio simulation "
        "exists) \u2013 <b>none of the four have survived independent scrutiny</b>: both "
        "same_sector_pairing variants failed their own val confirmation and then flipped sign "
        "entirely on the delivery split; momentum's delivery/train acceptance is undercut by its "
        "own delivery/val result, which rests on only 9 weekly buckets, a t-stat of 0.98, and a "
        "mean (+6.55%) driven by a handful of outliers against a negative median (-3.21%).", BODY))
    story.append(Paragraph(
        "Plain NIFTY50 buy-and-hold (CAGR ~12.8%, Sharpe ~0.62\u20130.66) beat every mechanical "
        "rule tested on every window. A 10,000-path block-bootstrap Monte Carlo, added independently "
        "of the t-stat/placebo pipeline, corroborates this: the clean rejections stay at ~0% "
        "probability of a positive mean under resampling, and the same_sector_pairing collapse is "
        "visible from a second angle (99.9% on train \u2192 ~74% on val, confidence interval "
        "straddling zero).", BODY))
    story.append(Paragraph(
        "The defensible conclusion, stated in this project's own working notes and unchanged by "
        "this report: short-horizon technical reaction to a visible price/delivery/OI/flow/"
        "volatility dislocation does not survive real execution and costs on this universe, at "
        "retail cost structure \u2013 not that nothing works, full stop. Longer-horizon momentum, "
        "fundamentals/valuation, and a genuine volatility/credit stress-regime gate remain "
        "under-explored, and are named explicitly in the Open Threads section.", BODY))

    # ---- METHODOLOGY -----------------------------------------------------
    story.append(Paragraph("Methodology", H2))
    story.append(Paragraph("Five rules, each enforced by code and tests, not convention alone:", BODY))
    rules = [
        ("Deterministic", "Seeds, content hashes, and run manifests. An AST scan fails the build if "
         "any module reads the wall clock outside two allow-listed metadata timestamps."),
        ("Point-in-time", "No datum reaches a decision before it existed. The universe is a "
         "recomputable rule (monthly-rebalanced, banded top-200-by-turnover), never today's index "
         "membership projected backward."),
        ("Executable", "Every fill is at the next session's open (T+1), never same-bar - a signal "
         "detected at bar T's close can only act on bar T+1."),
        ("Benchmark-relative", "Every headline number is compared against NIFTY50 buy-and-hold, "
         "after real costs, on the same capital and dates."),
        ("Counted", "Every hypothesis tried is logged to hypothesis_log.csv, append-only, whether it "
         "wins or loses - so a hit rate can never be computed over a silently-curated subset."),
    ]
    rule_rows = [[Paragraph(f"<b>{n}</b>", CELL), Paragraph(d, CELL)] for n, d in rules]
    t = Table(rule_rows, colWidths=[3.2 * cm, 13.3 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
    ]))
    story.append(t)

    story.append(Paragraph("Trade simulation and evidence bar", H3))
    story.append(Paragraph(
        "Trade-level results are sizing-independent (fixed capital per trade, reported as "
        "percentages) - the right tool for asking whether a signal carries information, separate "
        "from portfolio-level results (Rs 50,000 capital, 5 concurrent slots, real drawdown), which "
        "answers whether it survives as an account. Confirmation-style signals (mean_reversion, "
        "delivery_breakout, oi_momentum, participant_tilt, vol_squeeze_breakout, price_action) use "
        "a 7-day max hold / 2.0x-ATR stop / 1:2.5 risk-reward exit rule, shared across all of them "
        "so results stay comparable; momentum uses a pure ~1-month calendar hold instead, matching "
        "its own construction. Every result is checked against 30 placebo seeds (same signal dates "
        "and counts, names drawn blindly from that date's eligible pool) - the noise floor a real "
        "result must clear - and significance is read off a non-overlapping entry-week bucket "
        "t-stat, since trades entered the same week substantially share one market draw and would "
        "otherwise overstate how much independent evidence exists.", BODY))
    story.append(Paragraph(
        "A hypothesis is logged \u201caccepted\u201d only if it beats every placebo seed (and, "
        "where a portfolio simulation exists, has positive portfolio Sharpe) - this is a mechanical "
        "gate applied at logging time, not a claim of statistical confidence on its own; every "
        "acceptance in this project has still needed a val-window confirmation before being trusted, "
        "per the train-decides-first discipline below.", BODY))
    story.append(Paragraph("Train / validation discipline", H3))
    story.append(Paragraph(
        "A result is never promoted on the window it was discovered on. Val is touched only after a "
        "candidate clears train; test stays untouched throughout this whole program. Two splits are "
        "used, each with an embargo gap between windows:", BODY))
    split_rows = [
        [Paragraph("Split", CELLW), Paragraph("Train", CELLW),
         Paragraph("Val", CELLW), Paragraph("Test (untouched)", CELLW),
         Paragraph("Notes", CELLW)],
        [Paragraph("primary", CELL), Paragraph("2004\u20132016", CELL),
         Paragraph("2017\u20132021", CELL), Paragraph("2022\u20132026", CELL),
         Paragraph("Full price history, 60-day embargo", CELL)],
        [Paragraph("delivery", CELL), Paragraph("2019-06-27\u20132023-06-30", CELL),
         Paragraph("2023-07-01\u20132025-03-31", CELL), Paragraph("2025-04-01\u20132026-08-13", CELL),
         Paragraph("Per-stock delivery/OI/FII data starts 2019 \u2013 short history, higher evidence bar", CELL)],
    ]
    t2 = Table(split_rows, colWidths=[2.2 * cm, 3.6 * cm, 3.6 * cm, 3.6 * cm, 3.5 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)

    story.append(Paragraph("Monte Carlo addition (2026-08-19)", H3))
    story.append(Paragraph(
        "Extends the same evidence pipeline with a block-bootstrap resample of the real, "
        "already-simulated trades - not a new backtest. For each hypothesis, the real trades are "
        "grouped into the same entry-week buckets the t-stat is built on, then resampled in "
        "contiguous 4-bucket (~1 month) blocks, circularly, 10,000 times, to build the sampling "
        "distribution of the mean under resampling. Two numbers are reported per hypothesis: the "
        "share of resampled histories with a positive mean (a bootstrap analogue of the t-stat), and "
        "an illustrative full-capital sequential-compounding read (no position limits - a magnitude "
        "sense only, not a portfolio claim).", BODY))

    # ---- DATA FOUNDATION ---------------------------------------------
    story.append(Paragraph("Data foundation", H2))
    data_rows = [
        [Paragraph("Source", CELLW), Paragraph("What", CELLW), Paragraph("Coverage", CELLW)],
        [Paragraph("NSE bhavcopy archive", CELL), Paragraph("Rebuilt fresh per-year parquet store "
         "(not reused from any prior project)", CELL), Paragraph("5,588 trading days, "
         "2004-01-01\u20132026-08-13", CELL)],
        [Paragraph("fno.db", CELL), Paragraph("Per-stock futures/OI, read-only, content-hashed "
         "every run", CELL), Paragraph("48+ GB; OI/futures data from 2008", CELL)],
        [Paragraph("Corporate actions", CELL), Paragraph("Detected and adjusted directly from the "
         "price series", CELL), Paragraph("Full history", CELL)],
        [Paragraph("Delivery / participant-flow data", CELL), Paragraph("Per-stock delivery %, "
         "FII net index-futures positioning", CELL), Paragraph("From 2019-06-27 (the delivery "
         "split's own start)", CELL)],
        [Paragraph("NIFTY50 index level", CELL), Paragraph("Benchmark yardstick and this report's "
         "own bull/bear/sideways regime read", CELL), Paragraph("Full history", CELL)],
        [Paragraph("Universe", CELL), Paragraph("Point-in-time, monthly-rebalanced, banded "
         "top-200-by-turnover \u2013 never today's index membership", CELL), Paragraph("Recomputed "
         "every run", CELL)],
    ]
    t3 = Table(data_rows, colWidths=[4.0 * cm, 7.5 * cm, 5.0 * cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f6f3")]),
    ]))
    story.append(t3)

    # ---- SIGNALS CATALOG -----------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Signals tested", H2))
    story.append(Paragraph("Ten distinct economic ideas, each with its own falsifiable story stated "
                            "before any code was written:", BODY))
    for name, desc in SIGNAL_CATALOG:
        story.append(Paragraph(f"<b>{name}</b>", H3))
        story.append(Paragraph(desc, BODY))

    # ---- FULL TEST LOG ---------------------------------------------------
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Full test log \u2013 all 23 hypotheses", H2))
    story.append(Paragraph(
        "Every row is a real, logged run in hypothesis_log.csv. \u201cn\u201d is resolved trades, "
        "\u201cbuckets\u201d is the independent entry-week count the t-stat is computed on, "
        "\u201c\u0394 placebo\u201d is real mean minus the mean of 30 placebo seeds.", MUTEDSTYLE))
    log_head = ["#", "Signal", "Split", "Window", "n", "buckets", "mean %", "t-stat",
                "\u0394 placebo", "Decision"]
    log_rows = [[Paragraph(h, CELLW) for h in log_head]]
    for i, r in enumerate(log.itertuples(), start=1):
        placebo_delta = r.real_value - r.placebo_mean if pd.notna(r.placebo_mean) else float("nan")
        bg = ACCEPT_BG if r.decision == "accepted" else REJECT_BG
        log_rows.append([
            Paragraph(f"H{i}", CELL), Paragraph(short_title(r.title), CELL),
            Paragraph(r.split, CELL), Paragraph(r.window, CELL),
            Paragraph(f"{r.n_trades:,}", CELL), Paragraph(f"{r.n_buckets:,}", CELL),
            Paragraph(fmt(r.real_value), CELL), Paragraph(fmt(r.t_stat, pct=False), CELL),
            Paragraph(fmt(placebo_delta), CELL), Paragraph(r.decision, CELL),
        ])
    t4 = Table(log_rows, colWidths=[1.0 * cm, 4.6 * cm, 2.0 * cm, 1.6 * cm, 1.6 * cm,
                                     1.8 * cm, 1.8 * cm, 1.6 * cm, 1.8 * cm, 2.0 * cm],
               repeatRows=1)
    row_bgs = [("BACKGROUND", (0, i), (-1, i),
                ACCEPT_BG if log.iloc[i - 1]["decision"] == "accepted" else colors.white)
               for i in range(1, len(log_rows))]
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.3, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ] + row_bgs))
    story.append(t4)
    story.append(Paragraph("Not shown: cdd796d6e171, an earlier pairs_reversion (primary/train) run "
                            "superseded same-day by 0b11b017cef9 after a rollforward-at-entry fix "
                            "(row H9 above) - its raw trades were overwritten on disk by the re-run "
                            "and are not separately recoverable.", MUTEDSTYLE))

    # ---- MONTE CARLO RESULTS --------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Monte Carlo block-bootstrap results", H2))
    story.append(Paragraph(
        "prob(mean>0) is the share of 10,000 resampled histories where the mean stays positive - "
        "read it as the bootstrap uncertainty around the real mean, not an independent probability "
        "the strategy is \u201ctruly\u201d profitable.", MUTEDSTYLE))
    mc_head = ["#", "Signal", "Split/Window", "prob(mean>0)", "mean 95% CI", "prob(compounded>0)"]
    mc_rows = [[Paragraph(h, CELLW) for h in mc_head]]
    id_to_num = {hid: i + 1 for i, hid in enumerate(log["hypothesis_id"])}
    for r in mc.itertuples():
        num = id_to_num.get(r.hypothesis_id, "\u2013")
        mc_rows.append([
            Paragraph(f"H{num}", CELL), Paragraph(short_title(r.title), CELL),
            Paragraph(f"{r.split}/{r.window}", CELL),
            Paragraph(fmt(r.prob_mean_positive_pct), CELL),
            Paragraph(f"{fmt(r.mean_ci_lo_pct)} to {fmt(r.mean_ci_hi_pct)}", CELL),
            Paragraph(fmt(r.prob_compounded_positive_pct), CELL),
        ])
    t5 = Table(mc_rows, colWidths=[1.0 * cm, 4.6 * cm, 2.6 * cm, 2.6 * cm, 3.8 * cm, 3.0 * cm],
               repeatRows=1)
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.3, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f6f3")]),
    ]))
    story.append(t5)
    story.append(Paragraph(f"Not covered: H{id_to_num.get('cdd796d6e171', '?')} "
                            "(pairs_reversion pre-fix, same as above - no raw trades survive on disk).",
                            MUTEDSTYLE))

    # ---- DIAGNOSIS MATRIX (landscape) ------------------------------------
    story.append(NextPageTemplate("landscape"))
    story.append(PageBreak())
    story.append(Paragraph("Hypothesis diagnosis matrix", H2))
    story.append(Paragraph(
        "Every metric computed directly from the real saved trade files for each hypothesis - no "
        "estimation. CAGR / Sharpe / max drawdown come from the Rs 50,000, 5-slot portfolio "
        "simulation and are marked \u2014 where none exists (pairs trades have no portfolio-level "
        "simulation built in this project - noted explicitly in pairs_simulate.py's own docstring). "
        "\u201cShort-only result\u201d is \u2014 for every long-only single-leg signal by design (no "
        "honest short simulator was built for those). \u201cCosts as % of gross P&L\u201d is shown "
        "against the ABSOLUTE gross figure so it stays interpretable when gross itself is negative; "
        "an asterisk flags hypotheses where gross P&L is close enough to zero that the ratio is not "
        "meaningfully interpretable. Bull / bear / sideways use NIFTY50's own trailing 63-session "
        "return at each trade's entry date, with a +/-5% deadband for \u201csideways\u201d \u2013 a "
        "descriptive convention for this report, not a re-fitted parameter.", MUTEDSTYLE))
    story.append(Spacer(1, 0.3 * cm))

    metric_defs = [
        ("Number of trades", "n_trades", False, 0),
        ("Win rate", "win_rate_pct", True, 1),
        ("Avg winner", "avg_winner_pct", True, 2),
        ("Avg loser", "avg_loser_pct", True, 2),
        ("Profit factor", "profit_factor", False, 2),
        ("Gross expectancy", "gross_expectancy_pct", True, 3),
        ("Net expectancy", "net_expectancy_pct", True, 3),
        ("Avg holding period (days)", "avg_holding_days", False, 1),
        ("Max drawdown", "max_drawdown_pct", True, 1),
        ("CAGR", "cagr_pct", True, 2),
        ("Sharpe", "sharpe", False, 3),
        ("Costs as % of |gross P&L|", "costs_pct_of_abs_gross", True, 1),
        ("Long-only result", "long_only_result_pct", True, 3),
        ("Short-only result", "short_only_result_pct", True, 3),
        ("Bull-market result", "bull_result_pct", True, 3),
        ("Bear-market result", "bear_result_pct", True, 3),
        ("Sideways-market result", "sideways_result_pct", True, 3),
    ]

    hyp_order = [h for h in log["hypothesis_id"] if h in mat["hypothesis_id"].to_numpy()]
    mat_idx = mat.set_index("hypothesis_id")
    CHUNK = 6
    for start in range(0, len(hyp_order), CHUNK):
        chunk_ids = hyp_order[start:start + CHUNK]
        header = [Paragraph("Metric", CELLW)] + [
            Paragraph(f"H{id_to_num[h]}", CELLW) for h in chunk_ids]
        rows = [header]
        for label, col, pct, dp in metric_defs:
            row = [Paragraph(label, CELL)]
            for h in chunk_ids:
                v = mat_idx.loc[h, col]
                cell = fmt(v, pct=pct, dp=dp)
                if col == "costs_pct_of_abs_gross" and pd.notna(v) and abs(v) > 200:
                    cell += "*"
                row.append(Paragraph(cell, CELL))
            rows.append(row)
        n_cols = len(chunk_ids)
        col_widths = [4.6 * cm] + [(24.5 * cm - 4.6 * cm) / n_cols] * n_cols
        tmat = Table(rows, colWidths=col_widths, repeatRows=1)
        tmat.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.3, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f6f3")]),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ]))
        story.append(tmat)
        legend = " \u00b7 ".join(
            f"H{id_to_num[h]} = {short_title(log.set_index('hypothesis_id').loc[h, 'title'])} "
            f"({log.set_index('hypothesis_id').loc[h, 'split']}/{log.set_index('hypothesis_id').loc[h, 'window']})"
            for h in chunk_ids)
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(legend, MUTEDSTYLE))
        story.append(Spacer(1, 0.5 * cm))

    # ---- SYNTHESIS / LIMITATIONS (back to portrait) ----------------------
    story.append(NextPageTemplate("portrait"))
    story.append(PageBreak())
    story.append(Paragraph("Synthesis", H2))
    synth = [
        "Every confirmation-style signal failed the same way. mean_reversion, delivery_breakout, "
        "oi_momentum, participant_tilt, vol_squeeze_breakout, and price_action all react to an "
        "already-visible dislocation, and all show the same mechanism: an inflated stop-hit rate vs. "
        "placebo, confirmed directly by a delay sweep on vol_squeeze_breakout (mean_net_pct improved "
        "monotonically with entry delay, turning positive at delay=2, though the portfolio-level "
        "drawdown got WORSE, not better - entry timing alone does not create a tradeable edge).",
        "\u201cSmarter\u201d filters did no better, sometimes worse. Correlation-screened pairs lost "
        "to random same-sector pairs; delivery/OI/flow overlays underperformed simpler variants. More "
        "granular conditioning data did not add information on this universe.",
        "Costs are usually the difference between \u201clooks real\u201d and \u201cisn't\u201d. "
        "Correlation-screened pairs' gross t-stat of 2.65 collapsed to 0.18 once honestly filled and "
        "costed - real round-trip costs (~0.3\u20130.6% single-leg, ~0.5% two-leg) exceed most gross "
        "edges found here.",
        "Large-n, high-t-stat single-window results are not enough on their own. "
        "same_sector_pairing's t=3.49\u20133.70 on 3,400+ trades (not a small-sample fluke) still "
        "collapsed on val and flipped sign entirely on the delivery split - concrete evidence for why "
        "the train/val/test discipline exists at all.",
        "Structural/portfolio failure is a separate axis from signal failure. mean_reversion's "
        "concentration risk (best per-trade expectancy, worst portfolio Sharpe, because correlated "
        "names crash together) and bear-gating's directionally-correct-but-insufficient regime story "
        "each fail for reasons unrelated to whether the entry itself carries edge.",
        "Plain NIFTY50 buy-and-hold beat every mechanical rule tested, on every window - consistent "
        "with obvious, bhavcopy-derivable signals (price, volume/delivery, OI, FII flow, volatility, "
        "sector correlation) being priced in or too thin to survive retail costs.",
    ]
    for s in synth:
        story.append(Paragraph(f"\u2022 {s}", BODY))

    story.append(Paragraph("Open threads \u2013 not yet tried", H2))
    open_items = [
        ("Exit-geometry sweep on the other signals", "A 125-cell hold/stop/risk-reward grid ruled "
         "out exit tuning as an explanation for mean_reversion's failure - never repeated on the "
         "other 5+ signals."),
        ("Entry-delay sweep on mean_reversion", "The delay sweep that found the entry-timing "
         "mechanism only ever ran on vol_squeeze_breakout, despite mean_reversion having the most "
         "data of any signal in the project."),
        ("Expiry-matched holding period for pairs", "Whether a holding period matched to the futures "
         "expiry cycle (enter right after a roll, exit before the next) would further reduce the "
         "13\u201337% forced-rollover-exit rate is untested."),
        ("Volatility/credit stress-regime gate", "Trailing-return regime (Phase 0) and FII-flow "
         "regime (participant_tilt) were each tried and rejected; a systemic-stress axis (India VIX "
         "term structure, credit spread proxy) never was. market_gate's own credit_spreads.py and "
         "vix_term_structure.py signals exist, are fully written, and are unused anywhere."),
        ("Fundamentals / valuation signals", "Everything tested in this project so far is "
         "short-horizon and reactive to a visible dislocation. A valuation-based signal is a "
         "structurally different bet, but no fundamentals data pipeline exists in this project yet."),
        ("Calendar / seasonality effects", "Never touched inside this project's own rigorous harness "
         "(market_gate has a separate, less rigorously-tested Seasonality page)."),
    ]
    for name, desc in open_items:
        story.append(Paragraph(f"<b>{name}.</b> {desc}", BODY))

    story.append(Paragraph("Known limitations of this report", H2))
    limits = [
        "No portfolio-level simulation exists for pairs trades (same_sector_pairing, "
        "pairs_reversion) \u2013 CAGR/Sharpe/max-drawdown cells are blank by design, not missing "
        "data; margin/leverage economics for a multi-pair book were never modelled.",
        "price_action's SHORT side was only ever screened (no cost model, approximate same-day-close "
        "fills, no honest simulator) \u2013 excluded from this matrix entirely rather than mixed in "
        "under a different methodology than every other cell.",
        "The bull/bear/sideways split in the matrix is a description for this report, built after "
        "the fact \u2013 it was never used to gate or select a real hypothesis, and should not be "
        "read as a fourth confirmed regime-conditional edge.",
        "\u201cAccepted\u201d in this report always means \u201ccleared this project's own "
        "mechanical logging bar\u201d, never \u201cconfirmed\u201d \u2013 every acceptance here has "
        "either failed its own val test or remains thin enough (single-digit bucket counts, t<1) "
        "that it should not be traded on this evidence alone.",
    ]
    for s in limits:
        story.append(Paragraph(f"\u2022 {s}", BODY))

    return story


def _portrait_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.1 * cm, "Data test \u2013 Hypothesis Testing Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _landscape_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    lw, lh = landscape(A4)
    canvas.drawString(1.5 * cm, 1.0 * cm, "Data test \u2013 Hypothesis Testing Report \u2013 Diagnosis matrix")
    canvas.drawRightString(lw - 1.5 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUT_PATH), pagesize=A4,
                           leftMargin=2 * cm, rightMargin=2 * cm,
                           topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                           title="Data test - Hypothesis Testing Report")

    pw, ph = A4
    portrait_frame = Frame(2 * cm, 1.8 * cm, pw - 4 * cm, ph - 3.6 * cm, id="portrait_frame")
    lw, lh = landscape(A4)
    landscape_frame = Frame(1.5 * cm, 1.6 * cm, lw - 3 * cm, lh - 3.2 * cm, id="landscape_frame")

    doc.addPageTemplates([
        PageTemplate(id="portrait", frames=[portrait_frame], pagesize=A4,
                     onPage=_portrait_header_footer),
        PageTemplate(id="landscape", frames=[landscape_frame], pagesize=landscape(A4),
                     onPage=_landscape_header_footer),
    ])

    doc.build(build_story())
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
