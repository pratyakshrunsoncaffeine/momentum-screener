from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from screener_momentum.index_momentum import (
    DEFAULT_INDEX_MOMENTUM_WEIGHTS,
    NSE_INDEX_CATALOGUE,
    calculate_index_momentum,
    normalized_index_performance,
)
from screener_momentum.universe import load_ticker_universe, normalize_ticker_frame


class IndexMomentumTests(unittest.TestCase):
    def test_shared_catalogue_contains_all_categories_and_unique_indices(self) -> None:
        categories = {category for category, _index in NSE_INDEX_CATALOGUE}
        indices = [index for _category, index in NSE_INDEX_CATALOGUE]

        self.assertEqual(
            categories,
            {"Derivatives Eligible", "Broad Market", "Sectoral", "Strategy", "Thematic"},
        )
        self.assertEqual(len(indices), 128)
        self.assertEqual(len(indices), len(set(indices)))

    def test_recent_weighted_score_ranks_stronger_near_term_index_first(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=90)
        recent = np.full(len(dates), 100.0)
        recent[-11:] = np.linspace(100.0, 120.0, 11)
        stale = np.linspace(80.0, 110.0, len(dates))
        stale[-11:] = np.linspace(stale[-11], 108.0, 11)
        prices = self._long_prices(dates, {"Nifty IT": recent, "Nifty Auto": stale})

        ranking = calculate_index_momentum(
            prices,
            ["Nifty IT", "Nifty Auto"],
            DEFAULT_INDEX_MOMENTUM_WEIGHTS,
        )

        self.assertEqual(ranking.iloc[0]["Index"], "Nifty IT")
        self.assertGreater(ranking.iloc[0]["10D Return %"], ranking.iloc[1]["10D Return %"])
        self.assertEqual(ranking["Momentum Rank"].tolist(), [1, 2])

    def test_custom_weights_change_the_score_without_changing_prices(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=90)
        fast = np.concatenate([np.linspace(80, 120, 80), np.linspace(120, 130, 10)])
        reversal = np.concatenate([np.linspace(100, 80, 80), np.linspace(80, 100, 10)])
        prices = self._long_prices(dates, {"Nifty IT": fast, "Nifty Auto": reversal})

        long_weighted = calculate_index_momentum(
            prices,
            ["Nifty IT", "Nifty Auto"],
            {label: 1.0 if label == "3M Return %" else 0.0 for label in DEFAULT_INDEX_MOMENTUM_WEIGHTS},
        )
        short_weighted = calculate_index_momentum(
            prices,
            ["Nifty IT", "Nifty Auto"],
            {label: 1.0 if label == "5D Return %" else 0.0 for label in DEFAULT_INDEX_MOMENTUM_WEIGHTS},
        )

        self.assertEqual(long_weighted.iloc[0]["Index"], "Nifty IT")
        self.assertEqual(short_weighted.iloc[0]["Index"], "Nifty Auto")

    def test_indices_do_not_need_a_common_latest_date(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=90)
        first = self._long_prices(dates, {"Nifty IT": np.linspace(100, 120, len(dates))})
        second = self._long_prices(dates[:-3], {"Nifty Auto": np.linspace(90, 110, len(dates) - 3)})

        ranking = calculate_index_momentum(
            pd.concat([first, second], ignore_index=True),
            ["Nifty IT", "Nifty Auto"],
        )

        self.assertEqual(set(ranking["Index"]), {"Nifty IT", "Nifty Auto"})
        data_dates = ranking.set_index("Index")["Data Date"]
        self.assertNotEqual(data_dates["Nifty IT"], data_dates["Nifty Auto"])

    def test_normalized_chart_starts_each_index_at_one_hundred(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=90)
        prices = self._long_prices(
            dates,
            {"Nifty IT": np.linspace(100, 130, len(dates)), "Nifty Auto": np.linspace(80, 100, len(dates))},
        )

        normalized = normalized_index_performance(prices, ["Nifty IT", "Nifty Auto"], trading_days=60)

        starts = normalized.sort_values("Date").groupby("Index").first()["Normalized Value"]
        self.assertTrue(np.allclose(starts.to_numpy(), 100.0))

    @staticmethod
    def _long_prices(dates: pd.DatetimeIndex, values: dict[str, np.ndarray]) -> pd.DataFrame:
        rows = []
        for index_name, closes in values.items():
            rows.extend(
                {"Index": index_name, "Date": day, "Close": close}
                for day, close in zip(dates, closes)
            )
        return pd.DataFrame(rows)


class UploadedTickerTests(unittest.TestCase):
    def test_ticker_name_alias_and_ns_suffix_are_normalized(self) -> None:
        frame = normalize_ticker_frame(pd.DataFrame({"Ticker Name": ["RELIANCE.NS", " tcs ", "TCS"]}))

        self.assertEqual(frame["Ticker"].tolist(), ["RELIANCE", "TCS"])

    def test_single_column_csv_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constituents.csv"
            pd.DataFrame({"Companies": ["INFY", "HDFCBANK"]}).to_csv(path, index=False)

            universe = load_ticker_universe(path)

        self.assertEqual(universe["YFinance Ticker"].tolist(), ["INFY.NS", "HDFCBANK.NS"])


if __name__ == "__main__":
    unittest.main()
