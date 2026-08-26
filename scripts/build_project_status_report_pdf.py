"""REPORTING ONLY - assembles a project-status PDF report covering what
this differs from `build_hypothesis_report_pdf.py` (which is scoped to
signal-test *results* only): this report additionally covers the full
data inventory, the deterministic/backtesting architecture itself, and
open next steps - a status/methodology document, not a results document.

No numbers are invented - every figure traces back to `runs/
hypothesis_log.csv`, `config/config.toml`, or the real dataset sizes
already recorded in `README.md`.

    python scripts/build_project_status_report_pdf.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dtest import load_config  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "runs" / "project_status_report" / "Data_Test_Status_Report.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
LINE = colors.HexColor("#c9c9c2")
ACCEPT_BG = colors.HexColor("#e8f0e3")
REJECT_BG = colors.HexColor("#f5f0ec")
HEAD_BG = colors.HexColor("#2b2b26")
DONE_BG = colors.HexColor("#eef1ec")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=22, textColor=INK, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading1"], fontSize=15, textColor=INK,
                     spaceBefore=14, spaceAfter=8)
H3 = ParagraphStyle("H3", parent=styles["Heading2"], fontSize=11.5, textColor=INK,
                     spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.3, leading=13.2,
                       textColor=INK, spaceAfter=6)
MUTEDSTYLE = ParagraphStyle("Muted", parent=BODY, textColor=MUTED, fontSize=8.4, leading=11.5)
CELL = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7.6, leading=9.8, textColor=INK)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
CELLW = ParagraphStyle("CellW", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
COVER_SUB = ParagraphStyle("CoverSub", parent=BODY, fontSize=11.5, textColor=MUTED, spaceAfter=4)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=12, bulletIndent=0, spaceAfter=4)


def P(text, style=BODY):
    return Paragraph(text, style)


def styled_table(data, col_widths, header_rows=1, row_bg=None):
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), HEAD_BG),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if row_bg:
        for r, bg in row_bg.items():
            cmds.append(("BACKGROUND", (0, r), (-1, r), bg))
    t.setStyle(TableStyle(cmds))
    return t


# ---------------------------------------------------------------------------
# Data inventory - faithful to README.md's own table, condensed for a report
# page. Every row traces to a real dtest/data/*.py module or scripts/fetch_*.py.
# ---------------------------------------------------------------------------
DATA_ROWS = [
    ("Core", "NSE cash bhavcopy (rebuilt)", "2004\u20132026, every traded EQ symbol",
     "Replaces predecessor's 294 survivorship-biased CSVs; 22.5% of pre-2020 symbols are gone by 2026"),
    ("Core", "F&O bhavcopy (fno.db, read-only)", "2008\u20132026", "Pre-existing 48GB db; instrument-code taxonomy changed 2024-06, filter on asset_class+contract_type"),
    ("Fundamentals", "Quarterly financial results", "~2007\u20132026, 706/926 symbols", "Standalone only \u2013 no balance sheet in quarterly filings"),
    ("Fundamentals", "Shareholding pattern", "~2021\u20132026, 597/926 symbols", "Real NSE XBRL coverage floor, not a scrape limit"),
    ("Flow", "Insider trading (SEBI PIT)", "2015\u20132026, 584/926 symbols, 251,933 disclosures", "2015 = real regulatory floor"),
    ("Flow", "Index reconstitution calendar", "2010\u20132026, 18,391 events, 303 indices", "Pre-2010 NSE releases have no ticker column"),
    ("Flow", "ETF / Index-Fund AUM", "2006\u20132026, 24,836 rows, 1,719 schemes", "filing_date is an assumed 15-day lag"),
    ("Flow", "AMC portfolio disclosures (SBI + Axis)", "SBI 2013\u20132026 (68 files); Axis 2021\u20132026 (3,672 files)", "Raw workbooks; ICICI/HDFC scoped but need real browser automation"),
    ("Flow", "SBI + Axis per-stock equity holdings", "Axis full range; SBI 2023\u20132026 only \u2013 243,224 rows, 199 schemes", "SBI's 2013-2016 files use a 6-col template not yet parsed"),
    ("Derivatives", "Per-stock options chain", "2008\u20132026, 460 symbols, strike-level", "Already in fno.db \u2013 raw only, IV/PCR/max-pain not built"),
    ("Derivatives", "Continuous front-month OI / futures price", "OI from 2008; futures price from 2008", "Two different stitching rules \u2013 OI can bridge a roll, price cannot"),
    ("Credit/Rates", "Credit rating actions", "Apr 2023\u20132026, 1,080 actions, 6 agencies", "Only 64.5% resolve to a real symbol"),
    ("Credit/Rates", "India G-Sec yields", "2011\u20132026, 2 tenors (10Y, 3M)", "Full curve needs an obfuscated RBI download, not built"),
    ("Credit/Rates", "Repo-rate proxy & bank credit", "Repo 1968\u20132026; credit 1951\u20132025", "Proxies, not the real RBI series \u2013 M3 dropped entirely, dead everywhere checked"),
    ("Disclosure", "Corporate announcements", "2004\u20132026, 117,121 rows, 3,043 symbols", "Best-behaved NSE feed found \u2013 both symbol and date genuinely filter"),
    ("Disclosure", "Earnings call transcripts", "2010\u20132026, 19,045 full-text files, 1,436 symbols", "~90% of volume is 2022+; 832MB text corpus"),
    ("Macro", "National macro series (CPI/IIP/WPI/GDP/PLFS/forex)", "Varies, 1990/1994\u20132026", "MOSPI REST family; M3/repo/bank-credit not on this API"),
    ("Macro", "GST monthly collections", "Jul 2017\u2013Mar 2026, 105 months", "Parsed from a GSTN retrospective PDF"),
    ("Macro", "Global cross-asset (VIX, crude, EM-FX, global equities)", "Varies by series, through 2026-08", "12 yfinance series; raw material, not wired into any signal"),
]

NOT_BUILT_ROWS = [
    ("Bulk & block deal bulletins", "Blocked \u2013 NSE's historical API returns a genuine server-side 503 on every parameter combination, confirmed reaching the real backend"),
    ("Per-stock participant-wise OI", "Closed \u2013 does not exist. NSE's own report is category \u00d7 date only; no per-stock breakdown has ever been published"),
    ("ICICI Prudential AMC portfolios", "Deferred \u2013 confirmed to need real click-driven browser automation, each file URL only exists inside a React click handler"),
    ("HDFC AMC portfolios", "Deferred \u2013 data itself is trivial static links, but the site sits behind a real Akamai bot wall blocking all non-browser traffic"),
    ("SBI legacy holdings template (2013\u20132016)", "Scoped, unstarted \u2013 a genuinely different 6-column table, not yet parsed"),
]

SIGNAL_SUMMARY = [
    ("mean_reversion", "50-day mean-reversion dip-buy", 2, "Only predecessor strategy that survived a walk-forward test \u2013 failed honest re-test on both splits"),
    ("delivery_breakout", "Price breakout confirmed by delivery %", 1, "Barely positive gross, below every placebo"),
    ("oi_momentum", "Price breakout confirmed by rising OI", 2, "Worse than every placebo on both splits"),
    ("participant_tilt", "Mean-reversion gated by FII flow regime", 2, "Directionally sensible, still below placebo"),
    ("vol_squeeze_breakout", "Breakout after a volatility contraction", 3, "Worst t-stat of any signal; an entry-delay variant improved but didn't save it"),
    ("price_action (LONG)", "Wide-range, volume-confirmed conviction bar", 2, "Confirms the entry-timing mechanism at its sharpest"),
    ("pairs_reversion", "Same-sector spread mean-reversion, market-neutral", 3, "Positive gross; collapsed once honestly costed"),
    ("same_sector_pairing", "Pure same-sector pairing, no correlation filter", 6, "Only construction to clear t>2 on train \u2013 failed val both times"),
    ("momentum (12-1 month)", "Trailing 12-month momentum, skip-month", 3, "Accepted on delivery/train and val \u2013 never re-tested past val"),
    ("earnings_surprise", "SUE/PEAD quarterly earnings surprise", 1, "Positive gross, not significant"),
    ("value", "Trailing P/E vs own history", 1, "Positive t-stat, still below placebo"),
    ("quality", "TTM margin trend", 1, "No real signal either direction"),
    ("mf_accumulation", "Combined Axis+SBI MF holdings growth (2-AMC scope)", 1, "First non-price-derived mechanism tested \u2013 same entry-timing failure shape"),
]


def _fmt_pct(v):
    try:
        return f"{v:.2f}%"
    except (TypeError, ValueError):
        return "\u2014"


def _fmt_t(v):
    try:
        return f"{v:.2f}"
    except (TypeError, ValueError):
        return "\u2014"


def build_story() -> list:
    cfg = load_config()
    log = pd.read_csv(cfg.paths.runs / "hypothesis_log.csv")
    n_total = len(log)
    n_rejected = int((log["decision"] == "rejected").sum())
    n_accepted = int((log["decision"] == "accepted").sum())
    n_beats_placebo = int(log["beats_best_placebo"].sum())

    story = []

    # ---- COVER ----------------------------------------------------------
    story.append(Spacer(1, 4.2 * cm))
    story.append(P("Data test", H1))
    story.append(P("Status &amp; Methodology Report", ParagraphStyle(
        "CoverTitle2", parent=H1, fontSize=16, textColor=MUTED, spaceAfter=18)))
    story.append(P(f"What data exists, how it is tested, what has been found "
                    f"({n_total} hypotheses, 0 survived), and what is open next.", COVER_SUB))
    story.append(P("A deterministic, point-in-time research harness for Indian cash-equity "
                    "strategies \u2013 rebuilt from scratch after its predecessor (market_gate) "
                    "produced numbers that could not be trusted.", COVER_SUB))
    story.append(Spacer(1, 1.2 * cm))
    story.append(P(f"Generated {date.today().isoformat()} \u00b7 "
                    "C:\\Users\\parik\\OneDrive\\Desktop\\Data test", MUTEDSTYLE))
    story.append(PageBreak())

    # ---- EXECUTIVE SUMMARY ------------------------------------------------
    story.append(P("Executive summary", H2))
    story.append(P(
        "Data test is a from-scratch rebuild of the market_gate research project, built to correct "
        "specific, measured failures in its predecessor: same-bar fills, no costs until the third "
        "session of work, a survivorship-biased universe, no benchmark comparison, and no running "
        "count of how many hypotheses had been tried. Five rules \u2013 deterministic, point-in-time, "
        "executable, benchmark-relative, and counted \u2013 are enforced in code, not by discipline "
        "alone.", BODY))
    story.append(P(
        f"As of today, <b>{n_total} hypotheses have been tested with the same rigor</b>: "
        f"{n_rejected} rejected outright, {n_accepted} accepted on their own training window but "
        f"none survived independent (val) confirmation. <b>Zero for {n_total}.</b> "
        f"{n_beats_placebo} of {n_total} beat their own placebo floor at all, and every one of "
        "those failed the very next check (a val re-test, or an honest-fills/cost re-simulation). "
        "The consistent pattern is not a coding bug \u2013 it is real, honestly measured effects "
        "that do not survive contact with real execution costs, real fills, and an honest placebo "
        "comparison, and specifically an entry-timing signature (buying right as a dislocation is "
        "detected, not before it).", BODY))
    story.append(P(
        "The project is currently between phases: a large data-collection push (2026-08-23 to "
        "2026-08-26) added roughly a dozen new category-specific sources \u2013 shareholding, "
        "insider trading, index reconstitution, ETF AUM, options chain, credit ratings, "
        "macro series, and, most recently, real per-stock mutual fund holdings from two AMCs "
        "(SBI and Axis) \u2013 and one new hypothesis (mf_accumulation) has been tested against "
        "the newest of these. This report inventories everything collected, explains the testing "
        "machinery precisely, summarizes what has been found, and lays out the open decision "
        "points for what comes next.", BODY))

    # ---- SECTION 1: WHAT THIS PROJECT IS ----------------------------------
    story.append(P("1. What this project is, and why the data was rebuilt", H2))
    story.append(P(
        "The predecessor project backtested 2004\u20132026 against 294 CSVs that are today's "
        "Nifty 500 members \u2013 a survivorship-biased universe. Measured on the one window where "
        "a comparison is possible (2019+), 22.5% of EQ symbols trading before 2020 are gone by "
        "2026 (HDFC, MINDTREE, LTI, CADILAHC, PVR, and 366 others); a survivor-only dataset shows "
        "0% attrition there \u2013 the exact blind spot that matters. Auditing those CSVs also found "
        "two merged price series in 25 files, a trading calendar with 8,458 \u201cdays\u201d over 26 "
        "years versus NSE's real ~6,650, rounding that destroyed six years of one stock's early "
        "returns, and literal Rs 0.00 closes.", BODY))
    story.append(P(
        "Price history is therefore rebuilt from NSE's own daily cash bhavcopy archives \u2013 every "
        "traded symbol, no hindsight, back to 1995 \u2013 starting the usable window at 2004-01-01 on "
        "structural grounds (STT introduced Oct 2004; T+2 settlement replaced badla around "
        "2001\u201302). The bhavcopy format also carries NSE's own corporate-action-adjusted "
        "<i>prev_close</i>, so a split or bonus is read from an exchange-published marker with an "
        "exact ratio, never guessed from a suspicious price jump.", BODY))
    story.append(P("The five rules, each enforced in code:", H3))
    for label, text in [
        ("Deterministic", "Same inputs, byte-identical outputs, provable via run manifests. An AST scan "
         "(<i>tests/test_determinism.py</i>) fails the build if any module reads the wall clock outside "
         "two allowlisted metadata timestamps."),
        ("Point-in-time", "No datum reaches a decision before it existed. The universe is a recomputable "
         "rule (monthly rebalance, banded top-200-by-turnover), never today's index membership."),
        ("Executable", "A signal from bar T's close fills at bar T+1's open, with real statutory costs "
         "and a participation cap. <i>config.execution.fill_at</i> raises on anything else."),
        ("Benchmark-relative", "Every headline is excess return over NIFTY50, on capital, after costs "
         "\u2013 absolute return is never the reported number."),
        ("Counted", "Every hypothesis tried is logged append-only to <i>hypothesis_log.csv</i>, win or "
         "lose, so the significance bar can rise honestly with the size of the search."),
    ]:
        story.append(P(f"<b>{label}.</b> {text}", BULLET))

    story.append(PageBreak())

    # ---- SECTION 2: DATA INVENTORY ----------------------------------------
    story.append(P("2. Data inventory \u2013 what exists today", H2))
    story.append(P(
        "Beyond the core price/F&amp;O rebuild, this project has sourced 17 category-specific "
        "datasets itself, each read-only for the deterministic harness and each fetched via its "
        "own <i>scripts/fetch_*.py</i> \u2013 never a live call from inside a signal. Every source "
        "below was verified live before being trusted (real probes, not assumed availability), and "
        "every real limitation is stated plainly rather than smoothed over.", BODY))

    header = [P("Category", CELLW), P("Source", CELLW), P("Coverage", CELLW), P("Real caveat", CELLW)]
    rows = [header]
    for cat, name, cov, caveat in DATA_ROWS:
        rows.append([P(cat, CELL), P(name, CELLB), P(cov, CELL), P(caveat, CELL)])
    story.append(styled_table(rows, col_widths=[2.1 * cm, 4.0 * cm, 4.3 * cm, 6.3 * cm]))

    story.append(Spacer(1, 10))
    story.append(P("Investigated and explicitly not built", H3))
    header2 = [P("Item", CELLW), P("Why", CELLW)]
    rows2 = [header2]
    for name, why in NOT_BUILT_ROWS:
        rows2.append([P(name, CELLB), P(why, CELL)])
    story.append(styled_table(rows2, col_widths=[5.0 * cm, 11.7 * cm]))
    story.append(P(
        "A recurring, useful finding across this whole collection effort: the cost of getting a "
        "source rarely correlates with how important or large the underlying institution is. The "
        "largest mutual fund by AUM (SBI) and a much smaller one (Axis) both needed no browser "
        "automation at all; the 2nd- and 3rd-largest (ICICI, HDFC) were each genuinely expensive, "
        "for two completely different reasons (a JS-only click-driven UI, and a real bot wall). "
        "Each source was checked live rather than assumed from its category.", MUTEDSTYLE))

    story.append(PageBreak())

    # ---- SECTION 3: ANALYSIS / BACKTESTING MECHANISM ----------------------
    story.append(P("3. Analysis and backtesting mechanism", H2))
    story.append(P("3.1 Architecture", H3))
    story.append(P(
        "<font face='Courier' size=7.6>"
        "config/config.toml&nbsp;&nbsp;&nbsp;every constant \u2013 a number in code is a bug<br/>"
        "dtest/determinism.py&nbsp;&nbsp;&nbsp;seeds, content hashes, run manifests<br/>"
        "dtest/config.py&nbsp;&nbsp;&nbsp;typed loader, validates on load (refuses same-bar fills)<br/>"
        "dtest/universe.py&nbsp;&nbsp;&nbsp;point-in-time monthly rebalanced universe<br/>"
        "dtest/data/&nbsp;&nbsp;&nbsp;17 source-specific fetch+parse modules (Section 2)<br/>"
        "dtest/features/&nbsp;&nbsp;&nbsp;point-in-time feature layer (technical, fundamentals, "
        "pairs, regime, MF holdings)<br/>"
        "dtest/signals/&nbsp;&nbsp;&nbsp;one file per hypothesis \u2013 12 built<br/>"
        "dtest/engine/&nbsp;&nbsp;&nbsp;costs.py, futures_costs.py, simulate.py (long-only, "
        "T+1-open), pairs_simulate.py (two-leg), portfolio.py<br/>"
        "dtest/evaluate/&nbsp;&nbsp;&nbsp;metrics.py, placebo.py (30-seed), hypothesis_log.py "
        "(append-only)<br/>"
        "scripts/&nbsp;&nbsp;&nbsp;runnable entry points; every run writes a manifest<br/>"
        "runs/&nbsp;&nbsp;&nbsp;manifests + result CSVs, committed \u2013 the audit trail"
        "</font>", BODY))

    story.append(P("3.2 The point-in-time universe", H3))
    story.append(P(
        "Rebalanced monthly (last trading day), 200 names by trailing 63-day turnover, with a "
        "250-name band so an incumbent isn't dropped for a marginal rank slip (real turnover cost "
        "under this project's cost model). A name needs 252 days of history and to have traded "
        "within the last 5 sessions to be eligible at all. This is a recomputable rule applied "
        "identically to every historical month \u2013 never today's actual index membership "
        "projected backward.", BODY))

    story.append(P("3.3 Execution and costs", H3))
    story.append(P(
        "Every fill happens at the NEXT session's open after a signal closes \u2013 never same-bar. "
        "Delivery-equity round trip is <b>0.222% of position</b> on pure statutory rates (STT "
        "0.1% each side, exchange transaction 0.00297% per side, stamp duty 0.015% buy-side, SEBI "
        "fee, 18% GST on brokerage+exchange+SEBI, zero brokerage \u2013 matches Zerodha delivery "
        "pricing), <b>0.322%</b> including the project's one named assumption, 5 bps/side slippage. "
        "Futures carry a separate, structurally different schedule (STT sell-side only, real "
        "non-zero brokerage). Rates are looked up from the real statutory schedule, not fitted.", BODY))
    story.append(P(
        "Exit rule, shared across every signal so results stay comparable: a maximum 7-session "
        "hold, a 2.0\u00d7-ATR stop, and a 1:2.5 risk\u2013reward target \u2013 whichever triggers "
        "first. Position sizing targets Rs 10,000 per trade at the trade level; the portfolio "
        "simulation separately runs a real Rs 50,000 account, 5 concurrent positions, equal-weight "
        "sizing, and a 40% max-sector-weight cap.", BODY))

    story.append(P("3.4 Statistical discipline", H3))
    story.append(P(
        "Two splits exist. <b>primary</b> (train 2004\u20132016, val 2017\u20132021, test "
        "2022\u20132026-08) is the default, used by anything price-derived. <b>delivery</b> (train "
        "2019-06-27\u20132023-06-30, val 2023-07\u20132025-03, test 2025-04\u20132026-08) is used "
        "by anything needing per-stock delivery, OI, participant-flow, or (as of this month) MF "
        "holdings data, which doesn't exist before 2019 \u2013 a real, stated higher evidence bar "
        "given roughly a third of primary's history. A 60-trading-day embargo separates every "
        "adjacent window.", BODY))
    story.append(P(
        "Every real result is compared against <b>30 placebo seeds</b> \u2013 the identical signal "
        "count and timing, but firing on randomly chosen eligible names instead of the real "
        "selection rule \u2013 to establish a noise floor (6 seeds gave an empirical p of only "
        "~0.14; 30 is the project's stated floor for a real claim). Significance is measured via a "
        "<b>non-overlapping (bucketed) t-statistic</b>, not a naive per-trade t-test, since trades "
        "from the same signal cluster in time and are not independent draws.", BODY))
    story.append(P(
        "The rule that matters most for interpreting the results table below: <b>train decides, "
        "val confirms</b>. A candidate that looks good on train is never reported as a result until "
        "it has been re-tested, unmodified, on the held-out val window. Two constructions in this "
        "project's own history (same_sector_pairing, and the Oil &amp; Gas-only cut of it) cleared "
        "t&gt;2 with real statistical significance on train and were logged as \u201caccepted\u201d "
        "there \u2013 both then failed val outright. That is the discipline working as designed, "
        "not a failure worth second-guessing.", BODY))

    story.append(P("3.5 The hypothesis log", H3))
    story.append(P(
        "Every variant tried is appended to <i>runs/hypothesis_log.csv</i>, win or lose, and never "
        "edited or deleted \u2013 a later re-test of the same idea is a new row, not a rewrite. Each "
        "row requires a non-empty <i>story</i> field (the economic mechanism, stated before running "
        "anything) \u2013 the project's explicit defense against a parameter search masquerading as "
        "a hypothesis test. Rejections are logged with the same weight as acceptances; the file is "
        "committed to git like everything else, so \u201chow many things have been tried\u201d is "
        "answerable by reading a file, not by scrolling chat history.", BODY))

    story.append(PageBreak())

    # ---- SECTION 4: RESULTS SO FAR -----------------------------------------
    story.append(P("4. Results so far", H2))
    story.append(P(
        f"<b>{n_total} logged hypotheses, {n_rejected} rejected, {n_accepted} accepted-on-train-"
        f"but-not-confirmed. Zero for {n_total}.</b> Thirteen distinct economic constructions have "
        "been tried, several across both splits and several parameter variants:", BODY))

    header3 = [P("Construction", CELLW), P("Idea", CELLW), P("Log entries", CELLW), P("Outcome", CELLW)]
    rows3 = [header3]
    for name, idea, n, outcome in SIGNAL_SUMMARY:
        rows3.append([P(name, CELLB), P(idea, CELL), P(str(n), CELL), P(outcome, CELL)])
    story.append(styled_table(rows3, col_widths=[3.4 * cm, 4.3 * cm, 1.6 * cm, 7.4 * cm]))

    story.append(Spacer(1, 10))
    story.append(P(
        "<b>The one consistent pattern across every rejection</b> is not a coding bug \u2013 it is "
        "real, honestly-measured effects that do not survive contact with real execution costs, "
        "real fills, and an honest placebo comparison. A dedicated diagnostic (2026-08-15/17) "
        "confirmed the mechanism directly: every signal's real trades show a stop-hit rate "
        "meaningfully above their own placebo's, with target-hit rates flat or worse \u2013 each "
        "one buys right as a 1\u20132-day dislocation is detected, and the dislocation tends to "
        "continue for a session or two right into the fresh position before reverting. Delaying "
        "entry by 2 sessions measurably closed that gap on vol_squeeze_breakout (mean_net_pct "
        "\u2212ve to marginally +ve) without producing a tradeable portfolio result \u2013 it removes "
        "the wrong reason to lose without yet producing a right reason to win.", BODY))
    story.append(P(
        "The most recent entry, <b>mf_accumulation</b>, is the first test against a data source "
        "with no relationship to price, open interest, delivery, or participant flow \u2013 real "
        "per-stock mutual fund positioning (Axis + SBI's own combined holdings, 2 of ~50 AMCs by "
        "explicit scope decision). It failed the same way (t=\u22122.94, below its own placebo), "
        "which is itself informative: the entry-timing pattern generalizes beyond price-derived "
        "constructions, suggesting it is closer to a structural property of \u201cbuy right when a "
        "dislocation becomes visible\u201d than an artifact of any one data transformation.", BODY))

    story.append(PageBreak())

    # ---- SECTION 5: FURTHER STEPS -------------------------------------------
    story.append(P("5. Further steps \u2013 open decision points", H2))
    story.append(P(
        "Nothing below has been decided yet; each is a real, live option rather than a queued "
        "task.", BODY))
    for label, text in [
        ("Extend the entry-timing-delay diagnostic to mf_accumulation",
         "The one variant (delayed entry) that measurably helped a prior signal has not yet been "
         "tried on the newest one \u2013 a cheap, natural next check before concluding anything "
         "further about MF-holdings data."),
        ("Test the other two MF-holdings hypotheses scoped but not chosen",
         "MF new-entrant (a scheme opening a brand-new position, a more discrete event than a "
         "gradual increase) and MF ownership breadth (a cross-sectional stability/quality read "
         "rather than a momentum-style trigger) were both scoped on 2026-08-26 and set aside in "
         "favor of mf_accumulation."),
        ("Build the SBI legacy-template holdings parser (2013\u20132016)",
         "A real, scoped, unstarted job \u2013 a genuinely different 6-column table that would "
         "extend SBI's usable per-stock holdings history back from 2023 to 2013, subject to the "
         "already-known 2016\u20132023 filing gap."),
        ("Decide on HDFC / ICICI Prudential AMC data",
         "Both confirmed to need real added investment (a bot-wall bypass, or real click-driven "
         "browser automation respectively) \u2013 worth it only if more AMC breadth is judged "
         "likely to change the mf_accumulation-style result materially."),
        ("Build a genuinely new construction on data already in hand",
         "Not another parameter retune of a rejected construction. Real, unstarted candidates: a "
         "fundamentals trigger combined with the entry-timing delay already found real for "
         "technical signals; a cross-sectional fundamentals ranking rather than an absolute "
         "threshold; a signal built directly on shareholding/insider-trading/index-reconstitution "
         "data (promoter-pledge trend, insider net-buying streaks, pre-effective-date positioning "
         "ahead of a known index add) \u2013 none of these have been tried yet."),
        ("Retry the two blocked/deprioritized data items",
         "Bulk & block deal bulletins (a real NSE-side 503, may have cleared); the 8 lower-priority "
         "macro series were already picked up 2026-08-26 and closed out."),
    ]:
        story.append(KeepTogether([
            P(f"<b>{label}</b>", H3),
            P(text, BODY),
        ]))

    story.append(Spacer(1, 10))
    story.append(P(
        "This report is a snapshot, not a living document \u2013 regenerate it "
        "(<i>python scripts/build_project_status_report_pdf.py</i>) after any further data or "
        "hypothesis work to keep it current.", MUTEDSTYLE))

    return story


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.1 * cm, "Data test \u2013 Status & Methodology Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUT_PATH), pagesize=A4,
                           leftMargin=2 * cm, rightMargin=2 * cm,
                           topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                           title="Data test - Status & Methodology Report")
    pw, ph = A4
    frame = Frame(2 * cm, 1.8 * cm, pw - 4 * cm, ph - 3.6 * cm, id="frame")
    doc.addPageTemplates([PageTemplate(id="portrait", frames=[frame], onPage=_header_footer)])
    doc.build(build_story())
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
