from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from screener_momentum.pipeline import load_saved_correlation, output_paths


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_NAME = "NSE Nifty 500 (04-Sep-2026)"


class CorrelationUniverseTests(unittest.TestCase):
    def test_supplied_index_files_form_a_clean_nifty_500_universe(self) -> None:
        universe = pd.read_csv(ROOT / "correlation_universe.csv")

        self.assertEqual(len(universe), 500)
        self.assertEqual(universe["Ticker"].nunique(), 500)
        self.assertFalse(universe["Name"].fillna("").eq("").any())
        self.assertEqual(int(universe["Nifty 500"].sum()), 500)
        self.assertEqual(int(universe["Nifty MidSmallcap 400"].sum()), 400)
        self.assertEqual(int(universe["Nifty Smallcap 250"].sum()), 250)
        self.assertEqual(int(universe["Nifty Midcap 150"].sum()), 150)

    def test_legacy_saved_correlation_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = output_paths(Path(temp_dir) / "latest")
            paths["root"].mkdir(parents=True)
            pd.DataFrame(
                [{"Factor": "Gold", "Ticker": "AAA", "Correlation": 0.5}]
            ).to_csv(paths["correlation_all"], index=False)
            pd.DataFrame(
                [{"Universe Name": "Legacy saved universe"}]
            ).to_csv(paths["correlation_run_metadata"], index=False)

            with self.assertRaisesRegex(FileNotFoundError, "previous stock universe"):
                load_saved_correlation(
                    paths["root"],
                    expected_universe_name=UNIVERSE_NAME,
                )


if __name__ == "__main__":
    unittest.main()
