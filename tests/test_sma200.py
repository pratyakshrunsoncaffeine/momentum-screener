from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from screener_momentum.config import FundamentalThresholds, Sma200ScanConfig
from screener_momentum.fundamentals import screen_fundamentals
from screener_momentum.sma200 import (
    calculate_sma200_snapshot,
    score_sma200_at_date,
    walk_forward_sma200_backtest,
)


class Sma200CalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.bdate_range("2025-01-01", periods=230)
        self.universe = pd.DataFrame(
            {
                "Ticker": ["BELOW", "NEAR", "FAR"],
                "Name": ["Below", "Near", "Far"],
                "YFinance Ticker": ["BELOW.NS", "NEAR.NS", "FAR.NS"],
            }
        )

    def test_positive_proximity_accepts_only_prices_above_and_within_limit(self) -> None:
        prices = pd.DataFrame(100.0, index=self.dates, columns=self.universe["YFinance Ticker"])
        prices.loc[self.dates[-1], "BELOW.NS"] = 95.0
        prices.loc[self.dates[-1], "NEAR.NS"] = 105.0
        prices.loc[self.dates[-1], "FAR.NS"] = 120.0

        snapshot = calculate_sma200_snapshot(
            self.universe,
            prices,
            Sma200ScanConfig(max_distance_pct=10.0),
        ).set_index("Ticker")

        self.assertFalse(bool(snapshot.loc["BELOW", "Proximity Pass"]))
        self.assertTrue(bool(snapshot.loc["NEAR", "Proximity Pass"]))
        self.assertFalse(bool(snapshot.loc["FAR", "Proximity Pass"]))
        self.assertLess(snapshot.loc["BELOW", "Distance Above 200DMA %"], 0)
        self.assertIn("below", snapshot.loc["BELOW", "Rejection Notes"])

    def test_missing_200_closes_is_explicit(self) -> None:
        short_dates = pd.bdate_range("2026-01-01", periods=120)
        prices = pd.DataFrame({"NEAR.NS": np.linspace(90, 100, len(short_dates))}, index=short_dates)

        snapshot = calculate_sma200_snapshot(
            self.universe.iloc[[1]],
            prices,
            Sma200ScanConfig(),
        )

        self.assertFalse(bool(snapshot.iloc[0]["Proximity Pass"]))
        self.assertIn("200 are required", snapshot.iloc[0]["Rejection Notes"])

    def test_near_live_quote_replaces_only_the_distance_numerator(self) -> None:
        prices = pd.DataFrame({"NEAR.NS": 100.0}, index=self.dates)
        quotes = pd.DataFrame(
            {
                "YFinance Ticker": ["NEAR.NS"],
                "Quote Price": [106.0],
                "Quote Time": ["2026-07-19T10:00:00+05:30"],
            }
        )

        snapshot = calculate_sma200_snapshot(
            self.universe.iloc[[1]],
            prices,
            Sma200ScanConfig(),
            latest_quotes=quotes,
        )

        self.assertEqual(snapshot.iloc[0]["CMP Rs."], 106.0)
        self.assertEqual(snapshot.iloc[0]["200DMA"], 100.0)
        self.assertAlmostEqual(snapshot.iloc[0]["Distance Above 200DMA %"], 6.0, places=3)
        self.assertIn("delayed/intraday", snapshot.iloc[0]["Price Basis"])

    def test_historical_score_ignores_prices_after_signal_date(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=260)
        values = pd.Series(100.0, index=dates)
        signal_date = dates[229]
        values.loc[dates[230:]] = 180.0
        prices = pd.DataFrame({"NEAR.NS": values})
        config = Sma200ScanConfig(max_distance_pct=10.0)

        from_full_history = score_sma200_at_date(prices, signal_date, config)
        from_truncated_history = score_sma200_at_date(prices.loc[:signal_date], signal_date, config)

        pd.testing.assert_frame_equal(from_full_history, from_truncated_history)
        self.assertEqual(from_full_history["YFinance Ticker"].tolist(), ["NEAR.NS"])


class Sma200BacktestTests(unittest.TestCase):
    def test_monthly_walk_forward_compounds_and_signals_before_entry(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=420)
        positions = np.arange(len(dates), dtype=float)
        prices = pd.DataFrame(
            {
                "AAA.NS": 100.0 * np.power(1.00035, positions),
                "BBB.NS": 80.0 * np.power(1.00025, positions),
            },
            index=dates,
        )
        eligible = pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB"],
                "Name": ["Alpha", "Beta"],
                "YFinance Ticker": ["AAA.NS", "BBB.NS"],
                "CMP Rs.": [prices["AAA.NS"].iloc[-1], prices["BBB.NS"].iloc[-1]],
                "Distance Above 200DMA %": [3.0, 2.0],
            }
        )
        config = Sma200ScanConfig(max_distance_pct=10.0, backtest_months=3)

        curves, normalized, periods, allocation = walk_forward_sma200_backtest(
            eligible,
            prices,
            config,
            initial_capital=10000.0,
            add_benchmark_data=False,
        )

        self.assertEqual(len(periods), 3)
        self.assertFalse(curves.empty)
        self.assertAlmostEqual(float(normalized.iloc[0, 0]), 100.0, places=6)
        self.assertGreater(float(curves.iloc[-1, 0]), 10000.0)
        self.assertAlmostEqual(
            float(periods.iloc[1]["Starting Capital"]),
            float(periods.iloc[0]["Ending Capital"]),
            places=2,
        )
        self.assertTrue(
            all(
                pd.Timestamp(signal) < pd.Timestamp(entry)
                for signal, entry in zip(periods["Signal Date"], periods["Rebalance Date"])
            )
        )
        self.assertAlmostEqual(float(allocation["Weight"].sum()), 1.0, places=6)

    def test_no_qualifying_month_stays_in_cash(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=320)
        prices = pd.DataFrame({"AAA.NS": np.linspace(100, 40, len(dates))}, index=dates)
        eligible = pd.DataFrame(
            {
                "Ticker": ["AAA"],
                "YFinance Ticker": ["AAA.NS"],
                "CMP Rs.": [40.0],
                "Distance Above 200DMA %": [1.0],
            }
        )

        curves, _, periods, _ = walk_forward_sma200_backtest(
            eligible,
            prices,
            Sma200ScanConfig(backtest_months=2),
            initial_capital=7500.0,
            add_benchmark_data=False,
        )

        self.assertTrue((curves["200DMA Strategy"] == 7500.0).all())
        self.assertTrue((periods["Position Count"] == 0).all())


class Sma200CheckpointTests(unittest.TestCase):
    def test_fundamental_resume_reuses_metrics_and_refreshes_candidate_fields(self) -> None:
        candidates = pd.DataFrame(
            {
                "Ticker": ["AAA", "BBB"],
                "YFinance Ticker": ["AAA.NS", "BBB.NS"],
                "Distance Above 200DMA %": [1.0, 2.0],
            }
        )
        saved = pd.DataFrame(
            {
                "Ticker": ["AAA"],
                "YFinance Ticker": ["AAA.NS"],
                "Distance Above 200DMA %": [9.0],
                "Market Cap Cr": [5000.0],
                "Quarterly Revenue Growth %": [20.0],
                "Annual Revenue Growth %": [25.0],
                "Promoter Holding Change %": [1.0],
                "Fundamental Pass": [True],
                "Fundamental Notes": ["passed"],
            }
        )
        fetched_metrics = {
            "Market Cap Cr": 6000.0,
            "Quarterly Revenue Growth %": 22.0,
            "Annual Revenue Growth %": 30.0,
            "Promoter Holding Change %": 2.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "partial.csv"
            saved.to_csv(checkpoint, index=False)
            with (
                patch(
                    "screener_momentum.fundamentals.fetch_company_fundamentals",
                    return_value=fetched_metrics,
                ) as fetch,
                patch("screener_momentum.fundamentals.time.sleep"),
            ):
                result = screen_fundamentals(
                    candidates,
                    FundamentalThresholds(),
                    checkpoint_path=checkpoint,
                    resume=True,
                )

        fetch.assert_called_once()
        self.assertEqual(fetch.call_args.args[0], "BBB")
        self.assertEqual(result.set_index("Ticker").loc["AAA", "Distance Above 200DMA %"], 1.0)
        self.assertTrue(result["Fundamental Pass"].all())


if __name__ == "__main__":
    unittest.main()
