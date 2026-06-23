from __future__ import annotations

from dataclasses import dataclass, field


RETURN_PERIODS: dict[str, int] = {
    "5 days ret": 5,
    "15 Days Returns": 15,
    "1M Return": 21,
    "2 months Ret.": 42,
    "3M Return": 63,
    "4 Months Ret.": 84,
    "6M Return": 126,
    "1Y Return": 252,
    "3Y R": 756,
    "5Y Return": 1260,
}

DEFAULT_MOMENTUM_WEIGHTS: dict[str, float] = {
    "5 days ret": 0.15,
    "15 Days Returns": 0.15,
    "1M Return": 0.20,
    "2 months Ret.": 0.20,
    "3M Return": 0.20,
    "6M Return": 0.10,
}

DEFAULT_POSITIVE_RETURN_FILTERS = ("5 days ret", "15 Days Returns", "1M Return")

BENCHMARKS: dict[str, list[str]] = {
    "Nifty 50": ["^NSEI"],
    "Nifty Midcap": ["^NSEMDCP50", "^CNXMDCP", "NIFTY_MIDCAP_100.NS"],
}


@dataclass(frozen=True)
class FundamentalThresholds:
    min_market_cap_cr: float = 1500.0
    min_quarterly_revenue_growth_pct: float = 10.0
    min_annual_revenue_growth_pct: float = 15.0
    max_promoter_holding_change_pct: float = 5.0


@dataclass(frozen=True)
class ScreeningConfig:
    momentum_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_MOMENTUM_WEIGHTS.copy())
    positive_return_filters: tuple[str, ...] = DEFAULT_POSITIVE_RETURN_FILTERS
    fundamental_thresholds: FundamentalThresholds = field(default_factory=FundamentalThresholds)
    top_momentum_for_fundamentals: int = 100
    final_count: int = 100
    price_batch_size: int = 80
    backtest_months: int = 6
