"""Which symbols are tradeable stocks and which are indices.

The distinction is load-bearing, not cosmetic. Indices in this dataset carry
zero volume for almost all history, are sometimes two different series merged
into one file, and cannot be bought in the cash segment. The predecessor project
ran a full-universe backtest over all 430 symbols and hit a divide-by-zero on
`NIFTY50DIVPOINT` - a dividend-points series that was never tradeable and had no
business being in a stock backtest at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dtest.config import Config


@dataclass(frozen=True)
class SymbolMap:
    stocks: list[str]            # tradeable, present in both the price dir and nifty500
    indices: list[str]           # everything else in the price dir
    industry: dict[str, str]     # stock -> industry label
    missing_prices: list[str]    # listed in nifty500 but no price file

    def industry_of(self, symbol: str) -> str:
        return self.industry.get(symbol, "Unknown")

    def industry_frame(self) -> pd.DataFrame:
        return (
            pd.DataFrame({"symbol": self.stocks})
            .assign(industry=lambda d: d["symbol"].map(self.industry).fillna("Unknown"))
            .sort_values("symbol", kind="stable")
            .reset_index(drop=True)
        )


def load_symbol_map(cfg: Config, available: list[str]) -> SymbolMap:
    """Split `available` price symbols into stocks vs indices.

    `nifty500.csv` is a CURRENT membership snapshot with no dates, so it cannot
    be used for point-in-time universe construction - that is exactly the
    survivorship hole in the predecessor project. It is used here only for what
    it can honestly support: telling a stock apart from an index, and supplying
    an industry label. The universe RULE (universe.py) never consults it.
    """
    ref = pd.read_csv(cfg.paths.industry_map)
    ref["symbol"] = ref["symbol"].astype(str).str.strip()
    ref["industry"] = ref["industry"].astype(str).str.strip()

    listed = set(ref["symbol"])
    have = set(available)

    stocks = sorted(listed & have)
    indices = sorted(have - listed)
    return SymbolMap(
        stocks=stocks,
        indices=indices,
        industry=dict(zip(ref["symbol"], ref["industry"])),
        missing_prices=sorted(listed - have),
    )
