from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from .config import BENCHMARKS


def portfolio_weights(tickers: list[str]) -> dict[str, float]:
    top = tickers[:5]
    tail = tickers[5:10]
    weights: dict[str, float] = {}
    if top:
        for ticker in top:
            weights[ticker] = 0.70 / len(top)
    if tail:
        for ticker in tail:
            weights[ticker] = 0.30 / len(tail)
    return weights


def backtest_top10(
    selected: pd.DataFrame,
    months: int = 6,
    initial_capital: float = 100000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest top 10 selected names against configured benchmarks."""
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    tickers = selected["YFinance Ticker"].head(10).astype(str).tolist()
    weights = portfolio_weights(tickers)
    period = f"{max(months, 1)}mo"
    prices = download_close(tickers, period=period)
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    normalized = prices.ffill().dropna(how="all")
    normalized = normalized.loc[:, [ticker for ticker in tickers if ticker in normalized.columns]]
    normalized = normalized.dropna(axis=1, how="all")
    if normalized.empty:
        return pd.DataFrame(), pd.DataFrame()

    first_prices = normalized.apply(lambda column: column.dropna().iloc[0] if column.dropna().shape[0] else np.nan)
    weighted_curve = pd.Series(0.0, index=normalized.index, name="Strategy")
    holdings_rows = []

    for ticker in normalized.columns:
        weight = weights.get(ticker, 0)
        capital = initial_capital * weight
        shares = capital / first_prices[ticker] if first_prices[ticker] else 0
        weighted_curve += normalized[ticker].ffill() * shares
        holdings_rows.append(
            {
                "YFinance Ticker": ticker,
                "Weight": weight,
                "Initial Allocation": round(capital, 2),
                "Entry Price": round(float(first_prices[ticker]), 2),
                "Shares": round(float(shares), 4),
            }
        )

    curves = pd.DataFrame({"Strategy": weighted_curve})
    for label, candidates in BENCHMARKS.items():
        benchmark = first_available_benchmark(candidates, period=period)
        if benchmark is None or benchmark.empty:
            continue
        benchmark = benchmark.reindex(curves.index).ffill().dropna()
        if benchmark.empty:
            continue
        curves[label] = (benchmark / benchmark.iloc[0]) * initial_capital

    curves = curves.dropna(how="all")
    return curves, pd.DataFrame(holdings_rows)


def download_close(tickers: list[str], period: str) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(-1):
            close = data.xs("Close", axis=1, level=-1)
        else:
            return pd.DataFrame()
    elif len(tickers) == 1 and "Close" in data.columns:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        return pd.DataFrame()
    close.columns = [str(column).upper() for column in close.columns]
    return close


def first_available_benchmark(candidates: list[str], period: str) -> pd.Series | None:
    for symbol in candidates:
        data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if not data.empty and "Close" in data.columns:
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            series = close.dropna()
            if not series.empty:
                return series.rename(symbol)
    return None


def performance_summary(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    rows = []
    for column in curves.columns:
        series = curves[column].dropna()
        if series.empty:
            continue
        total_return = ((series.iloc[-1] / series.iloc[0]) - 1.0) * 100.0
        running_max = series.cummax()
        drawdown = ((series / running_max) - 1.0) * 100.0
        rows.append(
            {
                "Series": column,
                "Start": date.fromisoformat(str(series.index[0].date())),
                "End": date.fromisoformat(str(series.index[-1].date())),
                "Start Value": round(float(series.iloc[0]), 2),
                "End Value": round(float(series.iloc[-1]), 2),
                "Return %": round(float(total_return), 2),
                "Max Drawdown %": round(float(drawdown.min()), 2),
            }
        )
    return pd.DataFrame(rows)
