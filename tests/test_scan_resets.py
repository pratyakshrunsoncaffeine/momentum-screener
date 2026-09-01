from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from screener_momentum.pipeline import (
    output_paths,
    reset_derivatives_eod_data,
    reset_dii_momentum_scan,
    reset_fii_momentum_scan,
    reset_index_momentum_scan,
    reset_sma200_scan,
)


class ScanResetTests(unittest.TestCase):
    def test_fii_and_dii_resets_do_not_cross_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "latest"
            paths = output_paths(root)
            fii_keys = ("fii_partial", "fii_marketcap_partial", "fii_all", "fii_top", "fii_momentum", "fii_final")
            dii_keys = ("dii_partial", "dii_marketcap_partial", "dii_all", "dii_top", "dii_momentum", "dii_final")
            for key in (*fii_keys, *dii_keys):
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_text("saved", encoding="utf-8")

            reset_fii_momentum_scan(root)

            self.assertTrue(all(not paths[key].exists() for key in fii_keys))
            self.assertTrue(all(paths[key].exists() for key in dii_keys))

            reset_dii_momentum_scan(root)

            self.assertTrue(all(not paths[key].exists() for key in dii_keys))

    def test_derivatives_restart_removes_only_selected_range_and_derived_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "latest"
            paths = output_paths(root)
            cache = paths["derivatives_cache"]
            inside = date(2026, 7, 15)
            outside = date(2026, 7, 10)
            for trade_date in (inside, outside):
                day_name = trade_date.strftime("%Y%m%d")
                for parent in (cache / "raw", cache / "normalized"):
                    target = parent / day_name
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "marker.csv").write_text("saved", encoding="utf-8")
            pd.DataFrame(
                {
                    "Trade Date": [outside.isoformat(), inside.isoformat()],
                    "Status": ["cached", "cached"],
                }
            ).to_csv(cache / "manifest.csv", index=False)
            paths["derivatives_signals"].parent.mkdir(parents=True, exist_ok=True)
            paths["derivatives_signals"].write_text("stale", encoding="utf-8")
            paths["momentum"].write_text("keep", encoding="utf-8")

            reset_derivatives_eod_data(inside, inside, output_dir=root)

            self.assertFalse((cache / "raw" / inside.strftime("%Y%m%d")).exists())
            self.assertFalse((cache / "normalized" / inside.strftime("%Y%m%d")).exists())
            self.assertTrue((cache / "raw" / outside.strftime("%Y%m%d")).exists())
            self.assertTrue((cache / "normalized" / outside.strftime("%Y%m%d")).exists())
            manifest = pd.read_csv(cache / "manifest.csv")
            self.assertEqual(manifest["Trade Date"].tolist(), [outside.isoformat()])
            self.assertFalse(paths["derivatives_signals"].exists())
            self.assertTrue(paths["momentum"].exists())

    def test_sma200_restart_removes_only_sma200_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "latest"
            paths = output_paths(root)
            sma_keys = (
                "sma200_prices",
                "sma200_universe",
                "sma200_candidates",
                "sma200_fundamentals_partial",
                "sma200_final",
                "sma200_backtest_curve",
            )
            for key in sma_keys:
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_text("saved", encoding="utf-8")
            paths["sma200_cache"].mkdir(parents=True, exist_ok=True)
            (paths["sma200_cache"] / "cached.csv").write_text("saved", encoding="utf-8")
            paths["momentum"].write_text("keep", encoding="utf-8")
            paths["fii_partial"].write_text("keep", encoding="utf-8")

            reset_sma200_scan(root)

            self.assertTrue(all(not paths[key].exists() for key in sma_keys))
            self.assertFalse(paths["sma200_cache"].exists())
            self.assertTrue(paths["momentum"].exists())
            self.assertTrue(paths["fii_partial"].exists())

    def test_index_momentum_restart_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "latest"
            paths = output_paths(root)
            index_keys = (
                "index_momentum_prices",
                "index_momentum_ranking",
                "index_momentum_health",
                "index_momentum_metadata",
            )
            for key in index_keys:
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_text("saved", encoding="utf-8")
            paths["index_momentum_cache"].mkdir(parents=True, exist_ok=True)
            (paths["index_momentum_cache"] / "cached.csv").write_text("saved", encoding="utf-8")
            paths["momentum"].write_text("keep", encoding="utf-8")

            reset_index_momentum_scan(root)

            self.assertTrue(all(not paths[key].exists() for key in index_keys))
            self.assertFalse(paths["index_momentum_cache"].exists())
            self.assertTrue(paths["momentum"].exists())


if __name__ == "__main__":
    unittest.main()
