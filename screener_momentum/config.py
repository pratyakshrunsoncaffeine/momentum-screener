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

POST_EARNINGS_STOCK_RETURN_PERIODS: dict[str, int] = {
    "Earnings 2D Return": 2,
    "Earnings 5D Return": 5,
    "Earnings 10D Return": 10,
}

DEFAULT_POST_EARNINGS_STOCK_RETURN_WEIGHTS: dict[str, float] = {
    "Earnings 2D Return": 0.20,
    "Earnings 5D Return": 0.30,
    "Earnings 10D Return": 0.50,
}

BENCHMARKS: dict[str, list[str]] = {
    "Nifty 50": ["^NSEI"],
    "Nifty Midcap": ["^NSEMDCP50", "^CNXMDCP", "NIFTY_MIDCAP_100.NS"],
}

NSE_SECTOR_INDICES: tuple[str, ...] = (
    "Nifty Auto",
    "Nifty Bank",
    "Nifty Capital Goods",
    "Nifty Cement",
    "Nifty Chemicals",
    "Nifty Commercial & Transport Services",
    "Nifty Construction",
    "Nifty Consumer Durables",
    "Nifty Consumer Services",
    "Nifty Financial Services",
    "Nifty Financial Services 25/50",
    "Nifty Financial Services Ex Bank",
    "Nifty FMCG",
    "Nifty Healthcare",
    "Nifty Hospitals",
    "Nifty Housing Finance",
    "Nifty Insurance",
    "Nifty IT",
    "Nifty Media",
    "Nifty Metal",
    "Nifty NBFC",
    "Nifty Oil & Gas",
    "Nifty Pharma",
    "Nifty Power",
    "Nifty Private Bank",
    "Nifty PSU Bank",
    "Nifty Realty",
    "Nifty REITs & Realty",
    "Nifty Retail",
    "Nifty Telecommunications",
    "Nifty500 Healthcare",
    "Nifty MidSmall Financial Services",
    "Nifty MidSmall Healthcare",
    "Nifty MidSmall IT & Telecom",
)

DEFAULT_NSE_SECTORS: tuple[str, ...] = (
    "Nifty Auto",
    "Nifty Bank",
    "Nifty Financial Services",
    "Nifty FMCG",
    "Nifty Healthcare",
    "Nifty IT",
    "Nifty Media",
    "Nifty Metal",
    "Nifty Oil & Gas",
    "Nifty Pharma",
    "Nifty PSU Bank",
    "Nifty Private Bank",
    "Nifty Realty",
)


@dataclass(frozen=True)
class FundamentalThresholds:
    min_market_cap_cr: float = 1500.0
    min_quarterly_revenue_growth_pct: float = 10.0
    min_annual_revenue_growth_pct: float = 15.0
    max_promoter_holding_change_pct: float = 5.0


@dataclass(frozen=True)
class Sma200ScanConfig:
    window_days: int = 200
    min_distance_pct: float = 0.0
    max_distance_pct: float = 10.0
    slope_lookback_days: int = 20
    price_mode: str = "Near-live shortlist"
    near_live_buffer_pct: float = 5.0
    price_batch_size: int = 80
    backtest_months: int = 6


@dataclass(frozen=True)
class DerivativesSignalConfig:
    min_underlying_return_pct: float = 2.0
    min_call_return_pct: float = 8.0
    min_call_price: float = 5.0
    min_volume_contracts: int = 100
    min_open_interest_contracts: int = 100
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 45
    corporate_action_return_pct: float = 20.0
    result_count: int = 100
    call_return_weight: float = 0.35
    underlying_return_weight: float = 0.25
    call_oi_change_weight: float = 0.15
    call_volume_ratio_weight: float = 0.15
    futures_return_weight: float = 0.10

    def score_weights(self) -> dict[str, float]:
        return {
            "Call Return %": self.call_return_weight,
            "Underlying Return %": self.underlying_return_weight,
            "Call OI Change %": self.call_oi_change_weight,
            "Call Volume Ratio": self.call_volume_ratio_weight,
            "Futures Return %": self.futures_return_weight,
        }


@dataclass(frozen=True)
class ScreeningConfig:
    momentum_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_MOMENTUM_WEIGHTS.copy())
    positive_return_filters: tuple[str, ...] = DEFAULT_POSITIVE_RETURN_FILTERS
    fundamental_thresholds: FundamentalThresholds = field(default_factory=FundamentalThresholds)
    top_momentum_for_fundamentals: int = 100
    final_count: int = 100
    price_batch_size: int = 80
    backtest_months: int = 6
