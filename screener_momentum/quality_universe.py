from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd
import yfinance as yf

from .universe import load_ticker_universe


def rank_quality_universe(universe: pd.DataFrame, caps: pd.DataFrame, limit: int = 2000) -> pd.DataFrame:
    """Select the largest known-cap NSE symbols; never rank an unknown cap as zero."""
    if limit < 1:
        raise ValueError("Universe size must be positive.")
    required = {"Ticker", "Market Cap Cr", "Market Cap Source", "Market Cap As Of"}
    if not required.issubset(caps.columns):
        raise ValueError(f"Market-cap data needs: {', '.join(sorted(required))}")
    cap_rows = caps[["Ticker", "Market Cap Cr", "Market Cap Source", "Market Cap As Of"]].copy()
    cap_rows["Ticker"] = cap_rows["Ticker"].astype(str).str.strip().str.upper()
    cap_rows["Market Cap Cr"] = pd.to_numeric(cap_rows["Market Cap Cr"], errors="coerce")
    cap_rows = cap_rows[cap_rows["Market Cap Cr"].gt(0)].sort_values("Market Cap As Of").drop_duplicates("Ticker", keep="last")
    if len(cap_rows) < limit:
        raise ValueError(f"Only {len(cap_rows):,} tickers have known positive market caps; {limit:,} required.")
    selected = universe.drop(columns=[column for column in required - {"Ticker"} if column in universe]).merge(
        cap_rows, on="Ticker", how="inner", validate="one_to_one"
    )
    if len(selected) < limit:
        raise ValueError(f"Only {len(selected):,} universe tickers have known market caps; {limit:,} required.")
    selected = selected.sort_values(["Market Cap Cr", "Ticker"], ascending=[False, True]).head(limit).copy()
    selected.insert(0, "Market Cap Rank", range(1, len(selected) + 1))
    return selected.reset_index(drop=True)


def yahoo_market_cap_cr(ticker: str) -> float | None:
    try:
        value = yf.Ticker(f"{ticker}.NS").fast_info.market_cap
        cap = float(value) / 10_000_000 if value is not None else None
        return cap if cap is not None and cap > 0 else None
    except Exception:
        return None


def build_quality_universe(
    ticker_csv: str | Path,
    saved_cap_csv: str | Path,
    output_csv: str | Path,
    checkpoint_csv: str | Path,
    limit: int = 2000,
    workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, int]:
    universe = load_ticker_universe(ticker_csv)
    saved_path = Path(saved_cap_csv)
    saved = pd.read_csv(saved_path)
    saved["Market Cap Cr"] = pd.to_numeric(saved["Market Cap Cr"], errors="coerce")
    saved = saved[saved["Market Cap Cr"].gt(0)].drop_duplicates("Ticker", keep="last")
    snapshot_date = date.fromtimestamp(saved_path.stat().st_mtime).isoformat()
    caps = saved[["Ticker", "Market Cap Cr"]].copy()
    caps["Market Cap Source"] = "Saved Screener.in FII scan"
    caps["Market Cap As Of"] = snapshot_date

    checkpoint = Path(checkpoint_csv)
    checkpoint_rows: list[dict[str, object]] = []
    if checkpoint.exists() and checkpoint.stat().st_size:
        prior = pd.read_csv(checkpoint)
        if {"Ticker", "Market Cap Cr", "Market Cap Source", "Market Cap As Of"}.issubset(prior):
            checkpoint_rows = prior.to_dict("records")
            caps = pd.concat([caps, prior], ignore_index=True).sort_values("Market Cap As Of").drop_duplicates("Ticker", keep="last")

    known = set(caps.loc[pd.to_numeric(caps["Market Cap Cr"], errors="coerce").gt(0), "Ticker"])
    missing = [ticker for ticker in universe["Ticker"] if ticker not in known]
    today = date.today().isoformat()
    additions: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 12))) as executor:
        futures = {executor.submit(yahoo_market_cap_cr, ticker): ticker for ticker in missing}
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            value = future.result()
            if value is not None:
                additions.append({
                    "Ticker": ticker,
                    "Market Cap Cr": round(value, 2),
                    "Market Cap Source": "Yahoo Finance",
                    "Market Cap As Of": today,
                })
            if completed % 25 == 0 or completed == len(missing):
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(checkpoint_rows + additions, columns=["Ticker", "Market Cap Cr", "Market Cap Source", "Market Cap As Of"]).drop_duplicates("Ticker", keep="last").to_csv(
                    checkpoint, index=False
                )
            if progress:
                progress(completed, len(missing))

    caps = pd.concat([caps, pd.DataFrame(additions)], ignore_index=True).sort_values("Market Cap As Of").drop_duplicates("Ticker", keep="last")
    selected = rank_quality_universe(universe, caps, limit)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    unresolved = len(universe) - len(set(universe["Ticker"]) & set(caps.loc[pd.to_numeric(caps["Market Cap Cr"], errors="coerce").gt(0), "Ticker"]))
    return selected, unresolved
