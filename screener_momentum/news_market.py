from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from .momentum import chunked
from .universe import to_yfinance_ticker


ProgressCallback = Callable[[int, int, str], None]


def download_constituent_activity_prices(
    tickers: list[str],
    start_date: date,
    end_date: date,
    batch_size: int = 80,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    normalized = list(dict.fromkeys(str(item).strip().upper().removesuffix(".NS") for item in tickers if str(item).strip()))
    yahoo_lookup = {to_yfinance_ticker(ticker): ticker for ticker in normalized}
    rows: list[pd.DataFrame] = []
    completed = 0
    for batch in chunked(list(yahoo_lookup), batch_size):
        data = yf.download(
            tickers=batch,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        for yahoo_ticker in batch:
            ticker_frame = _ticker_ohlcv(data, yahoo_ticker, len(batch))
            if ticker_frame.empty:
                continue
            ticker_frame = ticker_frame.rename_axis("Date").reset_index()
            ticker_frame["Ticker"] = yahoo_lookup[yahoo_ticker]
            rows.append(ticker_frame[["Date", "Ticker", "Close", "Volume"]])
        completed += len(batch)
        if progress_callback:
            progress_callback(completed, len(yahoo_lookup), f"Downloaded constituent activity for {completed:,} tickers")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Date", "Ticker", "Close", "Volume"])


def calculate_constituent_daily_activity(
    prices: pd.DataFrame,
    constituents: pd.DataFrame,
) -> pd.DataFrame:
    if prices.empty or constituents.empty:
        return pd.DataFrame()
    market = prices.copy()
    market["Date"] = pd.to_datetime(market["Date"], errors="coerce").dt.normalize()
    market["Ticker"] = market["Ticker"].astype(str).str.upper().str.removesuffix(".NS")
    market["Close"] = pd.to_numeric(market["Close"], errors="coerce")
    market["Volume"] = pd.to_numeric(market["Volume"], errors="coerce")
    market = market.dropna(subset=["Date", "Ticker", "Close"]).sort_values(["Ticker", "Date"])
    grouped = market.groupby("Ticker", observed=True)
    market["Return"] = grouped["Close"].pct_change()
    market["Turnover"] = market["Close"] * market["Volume"]
    market["Prior Volume Median"] = grouped["Volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).median()
    )
    market["Prior Turnover Median"] = grouped["Turnover"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).median()
    )
    market["volume_abnormal"] = market["Volume"] / market["Prior Volume Median"].replace(0, np.nan) - 1.0
    market["turnover_abnormal"] = market["Turnover"] / market["Prior Turnover Median"].replace(0, np.nan) - 1.0

    membership = constituents.copy()
    membership["Ticker"] = membership["Ticker"].astype(str).str.upper().str.removesuffix(".NS")
    for column in ("Valid From", "Valid To"):
        if column not in membership:
            membership[column] = pd.NaT
        membership[column] = pd.to_datetime(membership[column], errors="coerce").dt.normalize()
    tagged: list[pd.DataFrame] = []
    for item in membership.to_dict("records"):
        ticker_rows = market[market["Ticker"].eq(item["Ticker"])].copy()
        if pd.notna(item.get("Valid From")):
            ticker_rows = ticker_rows[ticker_rows["Date"].ge(item["Valid From"])]
        if pd.notna(item.get("Valid To")):
            ticker_rows = ticker_rows[ticker_rows["Date"].le(item["Valid To"])]
        if ticker_rows.empty:
            continue
        ticker_rows["Index"] = item["Index"]
        tagged.append(ticker_rows)
    if not tagged:
        return pd.DataFrame()
    panel = pd.concat(tagged, ignore_index=True)
    result = panel.groupby(["Date", "Index"], observed=True).agg(
        volume_abnormal=("volume_abnormal", "median"),
        turnover_abnormal=("turnover_abnormal", "median"),
        breadth_positive=("Return", lambda values: float((values > 0).mean())),
        participation_rate=("Volume", lambda values: float(values.gt(0).mean())),
        constituent_count=("Ticker", "nunique"),
    ).reset_index()
    result["Source"] = "Yahoo Finance constituents with point-in-time membership"
    return result


def _ticker_ohlcv(data: pd.DataFrame, ticker: str, batch_length: int) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            frame = data[ticker]
        elif ticker in data.columns.get_level_values(-1):
            frame = data.xs(ticker, axis=1, level=-1)
        else:
            return pd.DataFrame()
    elif batch_length == 1:
        frame = data
    else:
        return pd.DataFrame()
    if "Close" not in frame or "Volume" not in frame:
        return pd.DataFrame()
    return frame[["Close", "Volume"]].dropna(how="all")
