from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ticker_universe(csv_path: str | Path) -> pd.DataFrame:
    """Load the user ticker file and normalize NSE symbols for Yahoo Finance."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    frame = normalize_ticker_frame(pd.read_csv(path))

    frame["YFinance Ticker"] = frame["Ticker"].apply(to_yfinance_ticker)
    frame["Screener URL"] = frame["Ticker"].apply(lambda ticker: f"https://www.screener.in/company/{ticker}/consolidated/")
    return frame


def normalize_ticker_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a ticker-only upload while preserving optional metadata columns."""
    if frame.empty and len(frame.columns) == 0:
        raise ValueError("The uploaded CSV is empty.")
    result = frame.copy()
    aliases = {str(column).strip().lower(): column for column in result.columns}
    source_column = next(
        (aliases[name] for name in ("ticker", "ticker name", "symbol", "nse symbol") if name in aliases),
        None,
    )
    if source_column is None and len(result.columns) == 1:
        source_column = result.columns[0]
    if source_column is None:
        raise ValueError("The CSV must contain a Ticker, Ticker Name, Symbol, or NSE Symbol column.")
    if source_column != "Ticker":
        result = result.rename(columns={source_column: "Ticker"})

    result["Ticker"] = result["Ticker"].astype(str).str.strip().str.upper().str.removesuffix(".NS")
    result = result[result["Ticker"].ne("") & result["Ticker"].ne("NAN")]
    return result.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)


def to_yfinance_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    return ticker if ticker.endswith(".NS") else f"{ticker}.NS"
