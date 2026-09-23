from __future__ import annotations

import unittest

from screener_momentum.config import FundamentalThresholds
from screener_momentum.fundamentals import passes_fundamental_filters


class FundamentalFilterTests(unittest.TestCase):
    def test_zero_promoter_change_is_valid(self) -> None:
        metrics = {
            "Market Cap Cr": 2000.0,
            "Quarterly Revenue Growth %": 11.0,
            "Annual Revenue Growth %": 16.0,
            "Promoter Holding Change %": 0.0,
        }
        passed, reasons = passes_fundamental_filters(metrics, FundamentalThresholds())
        self.assertTrue(passed, reasons)

    def test_missing_promoter_change_is_not_treated_as_zero(self) -> None:
        metrics = {
            "Market Cap Cr": 2000.0,
            "Quarterly Revenue Growth %": 11.0,
            "Annual Revenue Growth %": 16.0,
            "Promoter Holding Change %": None,
        }
        passed, reasons = passes_fundamental_filters(metrics, FundamentalThresholds())
        self.assertFalse(passed)
        self.assertIn("missing promoter holding change", reasons)


if __name__ == "__main__":
    unittest.main()
