from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ticker_universe(csv_path: str | Path) -> pd.DataFrame:
    """Load the user ticker file and normalize NSE symbols for Yahoo Finance."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")

    frame = pd.read_csv(path)
    if "Ticker" not in frame.columns:
        raise ValueError("ticker.csv must contain a 'Ticker' column.")

    frame = frame.copy()
    frame["Ticker"] = frame["Ticker"].astype(str).str.strip().str.upper()
    frame = frame[frame["Ticker"].ne("") & frame["Ticker"].ne("NAN")]
    frame = frame.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
    frame["YFinance Ticker"] = frame["Ticker"].apply(to_yfinance_ticker)
    frame["Screener URL"] = frame["Ticker"].apply(lambda ticker: f"https://www.screener.in/company/{ticker}/consolidated/")
    return frame


def to_yfinance_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    return ticker if ticker.endswith(".NS") else f"{ticker}.NS"
