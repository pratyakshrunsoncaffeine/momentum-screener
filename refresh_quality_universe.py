from __future__ import annotations

from pathlib import Path
import logging
import warnings

from screener_momentum.quality_universe import build_quality_universe


ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    selected, unresolved = build_quality_universe(
        ROOT / "ticker.csv",
        ROOT / "output/latest/fii_all.csv",
        ROOT / "quality_momentum_universe.csv",
        ROOT / "output/latest/quality_market_caps_checkpoint.csv",
        progress=lambda done, total: print(f"Market caps checked: {done:,}/{total:,}", end="\r", flush=True),
    )
    print(f"\nSaved {len(selected):,} ranked companies; {unresolved:,} still lack a market cap.")
