from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from .config import RETURN_PERIODS


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def download_adjusted_close(
    yahoo_tickers: list[str],
    batch_size: int = 80,
    period: str = "6y",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Download adjusted close series in batches and return one column per ticker."""
    closes: list[pd.DataFrame] = []
    total = len(yahoo_tickers)
    completed = 0
    for batch in chunked(yahoo_tickers, batch_size):
        if progress_callback:
            progress_callback(completed, total, f"Downloading prices for {batch[0]} to {batch[-1]}")
        data = yf.download(
            tickers=batch,
            period=period,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        close = _extract_close(data, batch)
        if not close.empty:
            closes.append(close)
        completed = min(completed + len(batch), total)
        if progress_callback:
            progress_callback(completed, total, f"Downloaded {completed:,} of {total:,} tickers")

    if not closes:
        return pd.DataFrame()

    merged = pd.concat(closes, axis=1)
    return merged.loc[:, ~merged.columns.duplicated()].sort_index()


def _extract_close(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(-1):
            close = data.xs("Close", axis=1, level=-1)
        elif "Close" in data.columns.get_level_values(0):
            close = data.xs("Close", axis=1, level=0)
        else:
            return pd.DataFrame()
    elif len(tickers) == 1 and "Close" in data.columns:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    elif "Close" in data.columns:
        close = data[["Close"]]
    else:
        return pd.DataFrame()

    close.columns = [str(column).upper() for column in close.columns]
    return close


def calculate_returns(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    return_periods: dict[str, int] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Calculate CMP and trailing trading-day returns for each ticker."""
    periods = return_periods or RETURN_PERIODS
    rows: list[dict[str, object]] = []
    records = universe.to_dict("records")
    total = len(records)

    for index, item in enumerate(records, start=1):
        yahoo_ticker = str(item["YFinance Ticker"]).upper()
        if yahoo_ticker not in prices.columns:
            rows.append(_empty_row(item))
            if progress_callback:
                progress_callback(index, total, f"Calculated returns for {index:,} of {total:,} tickers")
            continue

        series = prices[yahoo_ticker].dropna()
        row = {
            **item,
            "CMP Rs.": np.nan,
            "Data Points": int(series.shape[0]),
            "Price Error": "",
        }
        if series.empty:
            row["Price Error"] = "No price history from Yahoo Finance"
            rows.append(row)
            continue

        row["CMP Rs."] = round(float(series.iloc[-1]), 2)
        for label, days in periods.items():
            row[label] = _period_return(series, days)
        rows.append(row)
        if progress_callback and (index == total or index % 100 == 0):
            progress_callback(index, total, f"Calculated returns for {index:,} of {total:,} tickers")

    return pd.DataFrame(rows)


def _empty_row(item: dict[str, object]) -> dict[str, object]:
    row = {
        **item,
        "CMP Rs.": np.nan,
        "Data Points": 0,
        "Price Error": "Ticker missing from Yahoo Finance response",
    }
    for label in RETURN_PERIODS:
        row[label] = np.nan
    return row


def _period_return(series: pd.Series, trading_days: int) -> float:
    if series.shape[0] <= trading_days:
        return np.nan
    current = float(series.iloc[-1])
    previous = float(series.iloc[-trading_days - 1])
    if previous == 0:
        return np.nan
    return round(((current / previous) - 1.0) * 100.0, 2)


def score_momentum(
    returns: pd.DataFrame,
    weights: dict[str, float],
    positive_filters: Iterable[str],
) -> pd.DataFrame:
    """Apply the weighted momentum score and short-term positive-return gate."""
    frame = returns.copy()
    available_weights = {key: value for key, value in weights.items() if key in frame.columns}
    total_weight = sum(available_weights.values())
    if total_weight <= 0:
        raise ValueError("Momentum weights must add to more than zero.")

    weighted = np.zeros(len(frame), dtype=float)
    for period, weight in available_weights.items():
        weighted += frame[period].fillna(0).astype(float) * (weight / total_weight)

    frame["Momentum Score"] = np.round(weighted, 2)
    frame["Momentum Pass"] = True
    for period in positive_filters:
        frame["Momentum Pass"] &= frame[period].astype(float) > 0

    frame = frame[frame["Momentum Pass"]].sort_values("Momentum Score", ascending=False)
    return frame.reset_index(drop=True)
