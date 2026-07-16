import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from screener_momentum.sector_rotation import (
    NseSectorIndexProvider,
    calculate_sector_rotation,
    normalized_performance,
    parse_nse_historical_response,
)


class SectorRotationTests(unittest.TestCase):
    def test_parses_wrapped_nse_response_and_removes_duplicate_dates(self) -> None:
        payload = {
            "d": '[{"HistoricalDate":"14-Jul-2026","CLOSE":"24,900.50"},'
            '{"HistoricalDate":"15-Jul-2026","CLOSE":"25,000.00"},'
            '{"HistoricalDate":"15-Jul-2026","CLOSE":"25,010.00"}]'
        }

        frame = parse_nse_historical_response(payload, requested_index="Nifty Auto")

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[-1]["Close"], 25010.0)
        self.assertEqual(frame.iloc[-1]["Index"], "Nifty Auto")

    def test_requested_index_name_normalizes_nse_casing_changes(self) -> None:
        payload = [
            {"HistoricalDate": "14-Jul-2026", "CLOSE": "16,000", "INDEX_NAME": "Nifty Healthcare"},
            {"HistoricalDate": "15-Jul-2026", "CLOSE": "16,100", "INDEX_NAME": "NIFTY HEALTHCARE"},
        ]

        frame = parse_nse_historical_response(payload, requested_index="Nifty Healthcare")

        self.assertEqual(frame["Index"].unique().tolist(), ["Nifty Healthcare"])

    def test_rotation_uses_latest_common_date_and_classifies_all_quadrants(self) -> None:
        prices, dates = self._synthetic_prices(drop_last_for="Nifty IT")

        snapshot = calculate_sector_rotation(
            prices,
            ["Nifty Auto", "Nifty Bank", "Nifty IT", "Nifty Media"],
        )

        statuses = snapshot.set_index("Sector")["Rotation Status"].to_dict()
        self.assertEqual(statuses["Nifty Auto"], "Leading")
        self.assertEqual(statuses["Nifty Bank"], "Improving")
        self.assertEqual(statuses["Nifty IT"], "Weakening")
        self.assertEqual(statuses["Nifty Media"], "Lagging")
        self.assertEqual(pd.Timestamp(snapshot.iloc[0]["Data Date"]), dates[-2])
        self.assertEqual(snapshot["Rotation Rank"].tolist(), list(range(1, len(snapshot) + 1)))
        self.assertTrue(snapshot["Rotation Score"].is_monotonic_decreasing)
        self.assertTrue(snapshot["Rank Change"].notna().all())

    def test_return_and_excess_return_are_calculated_from_trading_rows(self) -> None:
        prices, _ = self._synthetic_prices()

        snapshot = calculate_sector_rotation(prices, ["Nifty Auto"])
        auto = snapshot.iloc[0]

        self.assertGreater(auto["1M Return %"], 0)
        self.assertAlmostEqual(auto["1M Return %"], auto["1M Excess vs Nifty 50 %"], places=6)
        self.assertEqual(auto["Relative Strength 3M %"], auto["3M Excess vs Nifty 50 %"])

    def test_insufficient_history_is_explicit(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=30)
        rows = []
        for index_name in ("Nifty 50", "Nifty Auto"):
            rows.extend({"Index": index_name, "Date": day, "Close": 100 + position} for position, day in enumerate(dates))

        snapshot = calculate_sector_rotation(pd.DataFrame(rows), ["Nifty Auto"])

        self.assertEqual(snapshot.iloc[0]["Rotation Status"], "Insufficient History")
        self.assertTrue(pd.isna(snapshot.iloc[0]["3M Return %"]))

    def test_normalized_chart_starts_at_one_hundred(self) -> None:
        prices, _ = self._synthetic_prices()

        chart = normalized_performance(prices, ["Nifty Auto"], trading_days=63)
        first_values = chart.sort_values("Date").groupby("Index").first()["Normalized Value"]

        self.assertTrue(first_values.eq(100.0).all())

    def test_fetch_many_recovers_from_per_index_nse_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "nifty_auto.csv"
            pd.DataFrame(
                [
                    {"Index": "Nifty Auto", "Date": "2026-07-14", "Close": 25000},
                    {"Index": "Nifty Auto", "Date": "2026-07-15", "Close": 25100},
                ]
            ).to_csv(cache_path, index=False)
            provider = NseSectorIndexProvider(temp_dir, retries=1)

            with patch.object(provider, "fetch_index", side_effect=RuntimeError("NSE unavailable")):
                prices, health = provider.fetch_many(
                    ["Nifty Auto"],
                    start_date=date(2026, 7, 1),
                    end_date=date(2026, 7, 15),
                )

            self.assertEqual(len(prices), 2)
            self.assertEqual(health.iloc[0]["Status"], "cached fallback")

    def test_missing_benchmark_is_rejected(self) -> None:
        frame = pd.DataFrame([{"Index": "Nifty Auto", "Date": "2026-07-15", "Close": 25000}])
        with self.assertRaisesRegex(ValueError, "Nifty 50"):
            calculate_sector_rotation(frame, ["Nifty Auto"])

    @staticmethod
    def _synthetic_prices(drop_last_for: str | None = None) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        dates = pd.bdate_range("2025-11-17", periods=170)

        def piecewise(midpoint: float, endpoint: float) -> list[float]:
            values = [100.0] * 107
            values.extend(
                100.0 + (midpoint - 100.0) * step / 41.0
                for step in range(1, 42)
            )
            values.extend(
                midpoint + (endpoint - midpoint) * step / 21.0
                for step in range(1, 22)
            )
            return values

        series = {
            "Nifty 50": [100.0] * len(dates),
            "Nifty Auto": piecewise(104.0, 112.0),
            "Nifty Bank": piecewise(85.0, 95.0),
            "Nifty IT": piecewise(112.0, 110.0),
            "Nifty Media": piecewise(90.0, 80.0),
        }
        rows = []
        for index_name, closes in series.items():
            for day, close in zip(dates, closes):
                if index_name == drop_last_for and day == dates[-1]:
                    continue
                rows.append({"Index": index_name, "Date": day, "Close": close})
        return pd.DataFrame(rows), dates


if __name__ == "__main__":
    unittest.main()
