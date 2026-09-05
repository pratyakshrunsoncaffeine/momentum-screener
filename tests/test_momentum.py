from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from screener_momentum.config import ScreeningConfig
from screener_momentum.momentum import download_adjusted_close, score_momentum
from screener_momentum.pipeline import output_paths, run_price_returns


class MomentumTests(unittest.TestCase):
    def test_download_retries_missing_batch_tickers(self) -> None:
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        first = pd.DataFrame(
            range(100, 130),
            index=dates,
            columns=pd.MultiIndex.from_tuples([("AAA.NS", "Close")]),
        )
        retry = pd.DataFrame({"Close": range(200, 230)}, index=dates)
        responses = iter((first, retry))
        calls: list[dict[str, object]] = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return next(responses)

        with patch("screener_momentum.momentum.yf.download", side_effect=fake_download):
            prices = download_adjusted_close(["AAA.NS", "BBB.NS"], batch_size=2, period="1y")

        self.assertEqual(list(prices.columns), ["AAA.NS", "BBB.NS"])
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["threads"], True)
        self.assertEqual(calls[1]["tickers"], ["BBB.NS"])
        self.assertIs(calls[1]["threads"], False)

    def test_empty_refresh_keeps_previous_saved_returns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ticker_csv = root / "tickers.csv"
            ticker_csv.write_text("Ticker\nAAA\n", encoding="utf-8")
            paths = output_paths(root / "latest")
            paths["returns"].parent.mkdir(parents=True, exist_ok=True)
            original = "Ticker,CMP Rs.\nOLD,100\n"
            paths["returns"].write_text(original, encoding="utf-8")

            with patch(
                "screener_momentum.pipeline.download_adjusted_close",
                return_value=pd.DataFrame(),
            ):
                with self.assertRaisesRegex(RuntimeError, "no usable prices"):
                    run_price_returns(
                        str(ticker_csv),
                        ScreeningConfig(),
                        output_dir=root / "latest",
                    )

            self.assertEqual(paths["returns"].read_text(encoding="utf-8"), original)

    def test_positive_filter_and_score_ordering(self) -> None:
        returns = pd.DataFrame(
            {
                "Ticker": ["PASS", "FAIL"],
                "5 days ret": [5.0, -1.0],
                "15 Days Returns": [4.0, 4.0],
                "1M Return": [3.0, 3.0],
                "2 months Ret.": [2.0, 2.0],
                "3M Return": [1.0, 1.0],
                "6M Return": [1.0, 1.0],
            }
        )

        result = score_momentum(
            returns,
            ScreeningConfig().momentum_weights,
            ScreeningConfig().positive_return_filters,
        )

        self.assertEqual(result["Ticker"].tolist(), ["PASS"])
        self.assertEqual(result["Momentum Rank"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
