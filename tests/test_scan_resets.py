from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from screener_momentum.pipeline import (
    output_paths,
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
