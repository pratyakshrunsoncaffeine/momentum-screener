from __future__ import annotations

import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory

from screener_momentum.correlation import (
    FACTOR_CATALOG,
    NseHistoricalEquityProvider,
    add_ridge_regression,
    calculate_correlations,
    effective_frequency,
    factor_changes,
    normalized_factor_stock_performance,
    parse_nse_equity_history,
    select_correlation_leaders,
)


def _universe(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ticker": list(tickers),
            "Name": [f"{ticker} Ltd" for ticker in tickers],
            "Industry": ["Test"] * len(tickers),
            "YFinance Ticker": [f"{ticker}.NS" for ticker in tickers],
        }
    )


def _levels_from_returns(returns_pct: np.ndarray, start: float = 100.0) -> np.ndarray:
    values = [start]
    for value in returns_pct:
        values.append(values[-1] * (1.0 + value / 100.0))
    return np.asarray(values)


def test_price_and_yield_changes_use_correct_units() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    price = pd.Series([100.0, 102.0, 101.0, 106.05], index=dates)
    yields = pd.Series([6.50, 6.55, 6.45, 6.70], index=dates)

    price_result = factor_changes(price, FACTOR_CATALOG["Gold"], "Daily")
    yield_result = factor_changes(yields, FACTOR_CATALOG["India 10Y Yield"], "Daily")

    assert np.isclose(price_result.iloc[1], 2.0)
    assert np.isclose(price_result.iloc[3], 5.0)
    assert np.isclose(yield_result.iloc[1], 5.0)
    assert np.isclose(yield_result.iloc[2], -10.0)
    assert np.isclose(yield_result.iloc[3], 25.0)


def test_known_positive_and_inverse_relationships_rank_correctly() -> None:
    dates = pd.bdate_range("2025-01-01", periods=41)
    factor_return = np.tile(np.array([-2.0, -1.0, 0.5, 1.5, 2.5]), 8)
    factor_levels = pd.Series(_levels_from_returns(factor_return), index=dates)
    positive_prices = _levels_from_returns(factor_return * 1.2)
    inverse_prices = _levels_from_returns(factor_return * -0.8)
    noise_prices = _levels_from_returns(np.sin(np.arange(40)) * 0.4)
    prices = pd.DataFrame(
        {
            "POS.NS": positive_prices,
            "INV.NS": inverse_prices,
            "NOISE.NS": noise_prices,
        },
        index=dates,
    )

    results, _ = calculate_correlations(
        _universe("POS", "INV", "NOISE"),
        prices,
        {"Brent Crude": factor_levels},
        requested_frequency="Daily",
        min_observations=20,
    )
    positive, inverse = select_correlation_leaders(results, top_n=1)

    assert positive.iloc[0]["Ticker"] == "POS"
    assert positive.iloc[0]["Correlation"] > 0.99
    assert inverse.iloc[0]["Ticker"] == "INV"
    assert inverse.iloc[0]["Correlation"] < -0.99
    assert inverse.iloc[0]["Avg Return When Factor Falls %"] > 0


def test_next_period_relation_aligns_factor_before_stock_return() -> None:
    dates = pd.bdate_range("2026-01-01", periods=31)
    factor_return = np.array(
        [-1.4, 0.7, 1.9, -0.3, 2.2, -1.8, 0.4, 1.1, -0.9, 2.6] * 3,
        dtype=float,
    )
    stock_return = np.empty(30)
    stock_return[0] = 0.25
    stock_return[1:] = factor_return[:-1]
    prices = pd.DataFrame(
        {"LEAD.NS": _levels_from_returns(stock_return)},
        index=dates,
    )
    factor_levels = pd.Series(_levels_from_returns(factor_return), index=dates)

    next_results, _ = calculate_correlations(
        _universe("LEAD"),
        prices,
        {"Gold": factor_levels},
        requested_frequency="Daily",
        relation="Next stock period",
        min_observations=20,
    )
    same_results, _ = calculate_correlations(
        _universe("LEAD"),
        prices,
        {"Gold": factor_levels},
        requested_frequency="Daily",
        relation="Same period",
        min_observations=20,
    )

    assert next_results.iloc[0]["Correlation"] > 0.99
    assert same_results.iloc[0]["Correlation"] < 0.5
    assert next_results.iloc[0]["Data End"] < dates[-1].date().isoformat()


def test_minimum_observations_excludes_short_history() -> None:
    dates = pd.bdate_range("2026-01-01", periods=8)
    factor_return = np.arange(1.0, 8.0)
    prices = pd.DataFrame({"SHORT.NS": _levels_from_returns(factor_return)}, index=dates)
    levels = pd.Series(_levels_from_returns(factor_return), index=dates)

    results, _ = calculate_correlations(
        _universe("SHORT"),
        prices,
        {"USD/INR": levels},
        requested_frequency="Daily",
        min_observations=10,
    )

    assert results.empty


def test_india_yield_uses_monthly_effective_frequency() -> None:
    assert effective_frequency("India 10Y Yield", "Daily") == "Monthly"
    assert effective_frequency("India 10Y Yield", "Weekly") == "Monthly"
    assert effective_frequency("India 10Y Yield", "Monthly") == "Monthly"
    assert effective_frequency("US 10Y Yield", "Weekly") == "Weekly"


def test_nse_history_parser_handles_official_columns_and_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "CH_TIMESTAMP": ["2026-01-02", "2026-01-02", "2026-01-05"],
            "CH_CLOSING_PRICE": ["101.25", "102.00", "103.50"],
        }
    )

    result = parse_nse_equity_history(frame)

    assert result.shape[0] == 2
    assert result.loc[pd.Timestamp("2026-01-02")] == 102.0
    assert result.loc[pd.Timestamp("2026-01-05")] == 103.5


def test_nse_archive_fallback_accepts_sme_and_prefers_eq_series() -> None:
    with TemporaryDirectory() as directory:
        provider = NseHistoricalEquityProvider(directory)
        reports = provider.cache_dir / "reports"
        reports.mkdir(parents=True)
        pd.DataFrame(
            {
                "SYMBOL": ["MAIN", "MAIN", "SME"],
                "SERIES": ["BE", "EQ", "SM"],
                "CLOSE_PRICE": [99.0, 101.0, 205.5],
            }
        ).to_csv(reports / "20260105.csv", index=False)

        result = provider._load_archive_date(pd.Timestamp("2026-01-05").date())

        assert result.set_index("Ticker").loc["MAIN", "Close"] == 101.0
        assert result.set_index("Ticker").loc["SME", "Close"] == 205.5


def test_multivariate_ridge_recovers_positive_and_negative_partial_relationships() -> None:
    dates = pd.bdate_range("2025-01-01", periods=101)
    factor_one = np.sin(np.arange(100) / 4.0) * 1.2
    factor_two = np.cos(np.arange(100) / 7.0) * 0.8
    stock_return = 1.5 * factor_one - 0.9 * factor_two
    prices = pd.DataFrame({"MODEL.NS": _levels_from_returns(stock_return)}, index=dates)
    factor_history = pd.concat(
        [
            pd.DataFrame(
                {
                    "Date": dates[1:],
                    "Factor": "Brent Crude",
                    "Level": 100.0,
                    "Change": factor_one,
                    "Effective Frequency": "Daily",
                }
            ),
            pd.DataFrame(
                {
                    "Date": dates[1:],
                    "Factor": "Gold",
                    "Level": 100.0,
                    "Change": factor_two,
                    "Effective Frequency": "Daily",
                }
            ),
        ],
        ignore_index=True,
    )
    results = pd.DataFrame(
        {
            "Factor": ["Brent Crude", "Gold"],
            "YFinance Ticker": ["MODEL.NS", "MODEL.NS"],
            "Ticker": ["MODEL", "MODEL"],
            "Correlation": [0.8, -0.4],
        }
    )

    modeled = add_ridge_regression(
        results,
        prices,
        factor_history,
        relation="Same period",
        alpha=1.0,
        min_observations=50,
    ).set_index("Factor")

    assert modeled.loc["Brent Crude", "Ridge Coefficient"] > 0
    assert modeled.loc["Gold", "Ridge Coefficient"] < 0
    assert modeled.loc["Brent Crude", "Ridge Model R Squared"] > 0.95
    assert modeled.loc["Brent Crude", "Ridge Factors"] == 2


def test_normalized_factor_stock_chart_starts_every_series_at_one_hundred() -> None:
    dates = pd.bdate_range("2026-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "Date": dates,
            "AAA.NS": np.arange(100.0, 108.0),
            "BBB.NS": np.arange(200.0, 216.0, 2.0),
        }
    )
    history = pd.DataFrame(
        {
            "Date": dates,
            "Factor": "Brent Crude",
            "Level": np.arange(70.0, 78.0),
            "Change": [np.nan, *np.ones(7)],
            "Effective Frequency": "Daily",
        }
    )

    chart = normalized_factor_stock_performance(prices, history, "Brent Crude", ["AAA", "BBB"])
    starts = chart.sort_values("Date").groupby("Series")["Normalized Value"].first()

    assert set(starts.index) == {"AAA", "BBB", "Brent Crude"}
    assert np.allclose(starts.to_numpy(), 100.0)
