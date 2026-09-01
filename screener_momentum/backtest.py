from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

from .config import BENCHMARKS, DEFAULT_MOMENTUM_WEIGHTS, DEFAULT_POSITIVE_RETURN_FILTERS, RETURN_PERIODS


def portfolio_weights(tickers: list[str]) -> dict[str, float]:
    top = tickers[:5]
    tail = tickers[5:10]
    weights: dict[str, float] = {}
    if not tail:
        return {ticker: 1.0 / len(top) for ticker in top} if top else {}
    if top:
        for ticker in top:
            weights[ticker] = 0.70 / len(top)
    for ticker in tail:
        weights[ticker] = 0.30 / len(tail)
    return weights


def current_allocation(selected: pd.DataFrame, capital: float = 100000.0) -> pd.DataFrame:
    """Build the current allocation table for the latest selected portfolio."""
    if selected.empty:
        return pd.DataFrame()

    tickers = selected["YFinance Ticker"].head(10).astype(str).str.upper().tolist()
    weights = portfolio_weights(tickers)
    rows = []
    for rank, item in enumerate(selected.head(10).to_dict("records"), start=1):
        ticker = str(item["YFinance Ticker"]).upper()
        weight = weights.get(ticker, 0.0)
        raw_cmp = item.get("CMP Rs.")
        cmp_value = float(raw_cmp) if raw_cmp is not None and not pd.isna(raw_cmp) else np.nan
        allocation = capital * weight
        rows.append(
            {
                "Rank": rank,
                "Ticker": item.get("Ticker", ticker.replace(".NS", "")),
                "Name": item.get("Name", ""),
                "YFinance Ticker": ticker,
                "Weight": round(weight, 4),
                "CMP Rs.": round(cmp_value, 2) if not pd.isna(cmp_value) and cmp_value > 0 else np.nan,
                "Amount Allocated": round(allocation, 2),
                "Approx Shares": round(allocation / cmp_value, 4) if not pd.isna(cmp_value) and cmp_value > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def backtest_top10(
    selected: pd.DataFrame,
    months: int = 6,
    initial_capital: float = 100000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper for the walk-forward backtest."""
    curves, _normalized, periods, allocation = walk_forward_backtest(
        selected,
        months=months,
        initial_capital=initial_capital,
    )
    if allocation.empty:
        allocation = current_allocation(selected, capital=initial_capital)
    return curves, allocation


def walk_forward_backtest(
    eligible: pd.DataFrame,
    months: int = 6,
    initial_capital: float = 100000.0,
    weights: dict[str, float] | None = None,
    positive_filters: tuple[str, ...] = DEFAULT_POSITIVE_RETURN_FILTERS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Monthly walk-forward momentum backtest.

    Signals use prices available before each rebalance entry date. The universe
    is whatever the caller passes in: fundamentals-passed stocks, or the
    momentum-only list when fundamentals are skipped.
    """
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    score_weights = weights or DEFAULT_MOMENTUM_WEIGHTS
    months = max(int(months), 1)
    capital = float(initial_capital)
    tickers = eligible["YFinance Ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    if not tickers:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    period_months = months + 9
    prices = download_close(tickers, period=f"{period_months}mo")
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_allocation(eligible, capital=initial_capital)

    prices = prices.ffill().dropna(how="all").sort_index()
    prices = prices.loc[:, [ticker for ticker in tickers if ticker in prices.columns]]
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_allocation(eligible, capital=initial_capital)

    schedule = monthly_rebalance_schedule(prices.index, months)
    if len(schedule) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_allocation(eligible, capital=initial_capital)

    metadata = eligible.drop_duplicates(subset=["YFinance Ticker"]).copy()
    metadata["_TickerKey"] = metadata["YFinance Ticker"].astype(str).str.upper()
    metadata = metadata.set_index("_TickerKey")
    curve_parts: list[pd.Series] = []
    period_rows: list[dict[str, object]] = []

    for entry_date, exit_date in zip(schedule[:-1], schedule[1:]):
        signal_dates = prices.index[prices.index < entry_date]
        if signal_dates.empty:
            continue
        signal_date = signal_dates[-1]
        scored = score_at_date(prices, signal_date, score_weights, positive_filters)
        if scored.empty:
            continue

        selected = scored.head(10).copy()
        selected_tickers = selected["YFinance Ticker"].tolist()
        period_prices = prices.loc[entry_date:exit_date, selected_tickers].dropna(axis=1, how="all")
        if period_prices.empty or period_prices.shape[0] < 2:
            continue

        active_tickers = [ticker for ticker in selected_tickers if ticker in period_prices.columns]
        selected = selected[selected["YFinance Ticker"].isin(active_tickers)].copy()
        if selected.empty:
            continue

        p_weights = portfolio_weights(selected["YFinance Ticker"].tolist())
        period_curve = pd.Series(0.0, index=period_prices.index)
        entry_prices = period_prices.ffill().iloc[0]
        entry_capital = capital

        for ticker, weight in p_weights.items():
            if ticker not in period_prices.columns or pd.isna(entry_prices.get(ticker)):
                continue
            allocation = capital * weight
            shares = allocation / float(entry_prices[ticker])
            period_curve += period_prices[ticker].ffill() * shares

        period_curve = period_curve[period_curve > 0]
        if period_curve.empty:
            continue

        capital = float(period_curve.iloc[-1])
        curve_parts.append(period_curve)
        period_return = ((capital / entry_capital) - 1.0) * 100.0 if entry_capital else np.nan
        top_names = []
        for ticker in selected["YFinance Ticker"].tolist():
            name = metadata["Name"].get(ticker, "") if "Name" in metadata.columns and ticker in metadata.index else ""
            top_names.append(f"{ticker.replace('.NS', '')}{f' ({name})' if name else ''}")

        period_rows.append(
            {
                "Rebalance Date": entry_date.date().isoformat(),
                "Signal Date": signal_date.date().isoformat(),
                "Exit Date": exit_date.date().isoformat(),
                "Selected Tickers": ", ".join(ticker.replace(".NS", "") for ticker in selected["YFinance Ticker"]),
                "Selected Names": ", ".join(top_names),
                "Starting Capital": round(entry_capital, 2),
                "Ending Capital": round(capital, 2),
                "Monthly Return %": round(period_return, 2),
            }
        )

    if not curve_parts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_allocation(eligible, capital=initial_capital)

    strategy = pd.concat(curve_parts)
    strategy = strategy[~strategy.index.duplicated(keep="last")].sort_index()
    curves = pd.DataFrame({"Strategy": strategy})
    curves = add_benchmarks(curves, initial_capital=initial_capital)
    normalized = (curves / float(initial_capital)) * 100.0
    allocation_source = eligible.sort_values("Momentum Score", ascending=False) if "Momentum Score" in eligible.columns else eligible
    allocation = current_allocation(allocation_source, capital=initial_capital)
    return curves, normalized, pd.DataFrame(period_rows), allocation


def monthly_rebalance_schedule(index: pd.DatetimeIndex, months: int) -> list[pd.Timestamp]:
    """Return month-spaced trading dates, including the final exit date."""
    dates = pd.DatetimeIndex(index).sort_values()
    end_anchor = dates.max().normalize()
    start_anchor = end_anchor - pd.DateOffset(months=months)
    anchors = [start_anchor + pd.DateOffset(months=offset) for offset in range(months + 1)]
    schedule: list[pd.Timestamp] = []
    for anchor in anchors:
        candidates = dates[dates >= anchor]
        if not candidates.empty:
            schedule.append(candidates[0])
    return list(dict.fromkeys(schedule))


def score_at_date(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    weights: dict[str, float],
    positive_filters: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    available_weights = {label: weight for label, weight in weights.items() if label in RETURN_PERIODS}
    total_weight = sum(available_weights.values())
    if total_weight <= 0:
        return pd.DataFrame()

    for ticker in prices.columns:
        series = prices.loc[:signal_date, ticker].dropna()
        if series.shape[0] <= max(RETURN_PERIODS[label] for label in available_weights):
            continue
        row = {"YFinance Ticker": ticker}
        for label in available_weights:
            row[label] = _period_return(series, RETURN_PERIODS[label])
        if any(pd.isna(row.get(label, np.nan)) or row.get(label, np.nan) <= 0 for label in positive_filters):
            continue
        score = sum(row[label] * (weight / total_weight) for label, weight in available_weights.items())
        if pd.isna(score):
            continue
        row["Momentum Score"] = round(float(score), 2)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Momentum Score", ascending=False).reset_index(drop=True)


def _period_return(series: pd.Series, trading_days: int) -> float:
    current = float(series.iloc[-1])
    previous = float(series.iloc[-trading_days - 1])
    if previous == 0:
        return np.nan
    return ((current / previous) - 1.0) * 100.0


def add_benchmarks(curves: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if curves.empty:
        return curves

    start = curves.index.min()
    end = curves.index.max()
    period = f"{max(((end.year - start.year) * 12 + end.month - start.month + 2), 2)}mo"
    for label, candidates in BENCHMARKS.items():
        benchmark = first_available_benchmark(candidates, period=period)
        if benchmark is None or benchmark.empty:
            continue
        benchmark = benchmark.reindex(curves.index).ffill().dropna()
        if benchmark.empty:
            continue
        curves[label] = (benchmark / benchmark.iloc[0]) * initial_capital
    return curves


def legacy_current_pick_backtest(
    selected: pd.DataFrame,
    months: int = 6,
    initial_capital: float = 100000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Old current-picks backtest kept for reference only; do not use in UI."""
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
