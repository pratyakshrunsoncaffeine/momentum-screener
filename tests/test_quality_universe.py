from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from screener_momentum.quality_universe import rank_quality_universe


class QualityUniverseTests(unittest.TestCase):
    def test_bundled_quality_universe_has_2000_ranked_nse_tickers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        selected = pd.read_csv(root / "quality_momentum_universe.csv")
        source = pd.read_csv(root / "ticker.csv")
        self.assertEqual(len(selected), 2000)
        self.assertEqual(selected["Ticker"].nunique(), 2000)
        source_tickers = set(source["Ticker"].astype(str).str.strip().str.upper())
        self.assertTrue(set(selected["Ticker"]).issubset(source_tickers))
        self.assertTrue(selected["Market Cap Cr"].is_monotonic_decreasing)
        self.assertEqual(selected["Market Cap Rank"].tolist(), list(range(1, 2001)))
        self.assertTrue(selected["Market Cap Source"].notna().all())

    def test_rank_uses_market_cap_and_excludes_unknowns(self) -> None:
        universe = pd.DataFrame({"Ticker": ["A", "B", "C", "D"], "Name": ["A", "B", "C", "D"]})
        caps = pd.DataFrame({
            "Ticker": ["A", "B", "C", "D"],
            "Market Cap Cr": [100, 300, None, 200],
            "Market Cap Source": ["saved"] * 4,
            "Market Cap As Of": ["2026-09-23"] * 4,
        })
        selected = rank_quality_universe(universe, caps, limit=2)
        self.assertEqual(selected["Ticker"].tolist(), ["B", "D"])
        self.assertEqual(selected["Market Cap Rank"].tolist(), [1, 2])

    def test_insufficient_known_caps_is_explicit(self) -> None:
        universe = pd.DataFrame({"Ticker": ["A", "B"]})
        caps = pd.DataFrame({
            "Ticker": ["A", "B"],
            "Market Cap Cr": [100, None],
            "Market Cap Source": ["saved"] * 2,
            "Market Cap As Of": ["2026-09-23"] * 2,
        })
        with self.assertRaisesRegex(ValueError, "known positive market caps"):
            rank_quality_universe(universe, caps, limit=2)


if __name__ == "__main__":
    unittest.main()
