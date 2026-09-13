"""Run the macro scanner independently of a Streamlit session."""
import argparse
from dataclasses import asdict
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from screener_momentum.macro_analysis import MacroCorrelationConfig, analyze
from screener_momentum.macro_data import MacroStore, PublicMacroProvider, save_csv
from screener_momentum.macro_dashboard import load_prices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", nargs="+", default=["brent_crude", "gold", "us_10y_yield", "usd_inr"])
    parser.add_argument("--assets", nargs="+", default=["Nifty FMCG", "Nifty Bank", "RELIANCE.NS"])
    parser.add_argument("--output", default="output/macro_smoke")
    args = parser.parse_args()
    store = MacroStore(Path(args.output))
    end = date.today()
    start = (pd.Timestamp(end)-pd.DateOffset(years=16)).date()
    progress = lambda done, total, message: print(f"{done}/{total} {message}", flush=True)
    print(PublicMacroProvider(store).refresh(args.factors, start, end, progress).to_string(index=False))
    prices = load_prices(store, args.assets, start, end, progress)
    obs = store.get()
    config = MacroCorrelationConfig()
    result, _ = analyze(obs, prices, args.factors, config, end, progress,
                        checkpoint=lambda f: save_csv(f, store.root / "macro_relationships_partial.csv"))
    if result.empty or result.Observations.max() == 0:
        raise SystemExit("No aligned data; the previous saved run is retained.")
    save_csv(result, store.root / "macro_relationships.csv")
    save_csv(pd.DataFrame([{**asdict(config), "as_of": str(end), "assets": json.dumps(args.assets),
        "identifiers": json.dumps(args.factors), "version": 1,
        "input_hash": hashlib.sha256(obs.to_csv(index=False).encode()).hexdigest()}]), store.root / "macro_manifest.csv")
    print(result[["Indicator", "Asset", "Observations", "Correlation", "Status"]].to_string(index=False))
