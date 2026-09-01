from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import yfinance as yf

from .backtest import add_benchmarks, current_allocation, monthly_rebalance_schedule, portfolio_weights
from .config import Sma200ScanConfig
from .momentum import _extract_close, chunked


ProgressCallback = Callable[[int, int, str], None]


def calculate_sma200_snapshot(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    config: Sma200ScanConfig,
    price_sources: dict[str, str] | None = None,
    latest_quotes: pd.DataFrame | None = None,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Calculate the latest 200DMA setup for every stock in the supplied universe."""
    if universe.empty:
        return pd.DataFrame()

    sources = {str(key).upper(): value for key, value in (price_sources or {}).items()}
    quote_lookup: dict[str, dict[str, object]] = {}
    if latest_quotes is not None and not latest_quotes.empty and "YFinance Ticker" in latest_quotes:
        quote_lookup = (
            latest_quotes.drop_duplicates("YFinance Ticker", keep="last")
            .assign(**{"YFinance Ticker": lambda frame: frame["YFinance Ticker"].astype(str).str.upper()})
            .set_index("YFinance Ticker")
            .to_dict("index")
        )

    rows: list[dict[str, object]] = []
    records = universe.to_dict("records")
    total = len(records)
    for position, item in enumerate(records, start=1):
        yahoo_ticker = str(item["YFinance Ticker"]).upper()
        series = _clean_price_series(prices[yahoo_ticker]) if yahoo_ticker in prices.columns else pd.Series(dtype=float)
        row: dict[str, object] = {
            **item,
            "CMP Rs.": np.nan,
            "200DMA": np.nan,
            "Distance Above 200DMA %": np.nan,
            f"200DMA {config.slope_lookback_days}D Slope %": np.nan,
            "SMA Trend": "Unavailable",
            "Proximity Pass": False,
            "Price Source": sources.get(yahoo_ticker, "Unavailable"),
            "Price Basis": "",
            "Price Date": "",
            "SMA Date": "",
            "Data Points": int(series.shape[0]),
            "Rejection Notes": "",
        }

        if series.shape[0] < int(config.window_days):
            row["Rejection Notes"] = (
                f"Only {series.shape[0]} valid closes; {config.window_days} are required"
                if not series.empty
                else "No usable price history"
            )
            rows.append(row)
            _report_calculation_progress(progress_callback, position, total)
            continue

        rolling = series.rolling(int(config.window_days), min_periods=int(config.window_days)).mean().dropna()
        sma_value = float(rolling.iloc[-1])
        completed_close = float(series.iloc[-1])
        price_value = completed_close
        price_date = pd.Timestamp(series.index[-1]).isoformat()
        price_basis = "Latest completed adjusted close"
        quote = quote_lookup.get(yahoo_ticker)
        if quote is not None:
            quote_price = pd.to_numeric(pd.Series([quote.get("Quote Price")]), errors="coerce").iloc[0]
            if pd.notna(quote_price) and float(quote_price) > 0:
                price_value = float(quote_price)
                price_date = str(quote.get("Quote Time") or price_date)
                price_basis = "Latest available delayed/intraday Yahoo quote"

        distance = ((price_value / sma_value) - 1.0) * 100.0 if sma_value else np.nan
        slope = _rolling_slope(rolling, int(config.slope_lookback_days))
        qualifies = bool(
            pd.notna(distance)
            and float(config.min_distance_pct) <= float(distance) <= float(config.max_distance_pct)
        )
        if qualifies:
            notes = "passed"
        elif pd.isna(distance):
            notes = "Could not calculate distance from 200DMA"
        elif float(distance) < float(config.min_distance_pct):
            notes = f"Price is {abs(float(distance)):.2f}% below the 200DMA"
        else:
            notes = f"Price is more than {float(config.max_distance_pct):.2f}% above the 200DMA"

        row.update(
            {
                "CMP Rs.": round(price_value, 2),
                "200DMA": round(sma_value, 2),
                "Distance Above 200DMA %": round(float(distance), 3) if pd.notna(distance) else np.nan,
                f"200DMA {config.slope_lookback_days}D Slope %": round(float(slope), 3)
                if pd.notna(slope)
                else np.nan,
                "SMA Trend": _trend_label(slope),
                "Proximity Pass": qualifies,
                "Price Basis": price_basis,
                "Price Date": price_date,
                "SMA Date": pd.Timestamp(series.index[-1]).date().isoformat(),
                "Rejection Notes": notes,
            }
        )
        rows.append(row)
        _report_calculation_progress(progress_callback, position, total)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_PassOrder"] = (~result["Proximity Pass"].astype(bool)).astype(int)
    result["_DistanceOrder"] = pd.to_numeric(result["Distance Above 200DMA %"], errors="coerce")
    return (
        result.sort_values(["_PassOrder", "_DistanceOrder", "Ticker"], na_position="last")
        .drop(columns=["_PassOrder", "_DistanceOrder"])
        .reset_index(drop=True)
    )


def preliminary_quote_tickers(snapshot: pd.DataFrame, config: Sma200ScanConfig) -> list[str]:
    """Return a buffered shortlist so near-live prices can catch stocks crossing into the final band."""
    if snapshot.empty:
        return []
    distance = pd.to_numeric(snapshot.get("Distance Above 200DMA %"), errors="coerce")
    lower = float(config.min_distance_pct) - float(config.near_live_buffer_pct)
    upper = float(config.max_distance_pct) + float(config.near_live_buffer_pct)
    mask = distance.between(lower, upper, inclusive="both")
    return snapshot.loc[mask, "YFinance Ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()


def download_latest_quotes(
    tickers: list[str],
    batch_size: int = 50,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch best-effort delayed/intraday Yahoo quotes for a small buffered shortlist."""
    requested = list(dict.fromkeys(str(ticker).upper() for ticker in tickers))
    rows: list[dict[str, object]] = []
    health_rows: list[dict[str, object]] = []
    total = len(requested)
    completed = 0

    for batch in chunked(requested, max(int(batch_size), 1)):
        if progress_callback:
            progress_callback(completed, total, f"Refreshing quotes for {batch[0]} to {batch[-1]}")
        try:
            data = yf.download(
                tickers=batch,
                period="5d",
                interval="5m",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                prepost=False,
                progress=False,
            )
            close = _extract_close(data, batch)
        except Exception as exc:
            close = pd.DataFrame()
            batch_error = str(exc)
        else:
            batch_error = ""

        for ticker in batch:
            series = _clean_price_series(close[ticker]) if ticker in close.columns else pd.Series(dtype=float)
            if series.empty:
                health_rows.append(
                    {
                        "Ticker": ticker.removesuffix(".NS"),
                        "YFinance Ticker": ticker,
                        "Stage": "Near-live quote",
                        "Status": "fallback",
                        "Rows": 0,
                        "Message": batch_error or "No intraday quote returned; latest completed close will be used",
                    }
                )
            else:
                rows.append(
                    {
                        "YFinance Ticker": ticker,
                        "Quote Price": float(series.iloc[-1]),
                        "Quote Time": pd.Timestamp(series.index[-1]).isoformat(),
                    }
                )
                health_rows.append(
                    {
                        "Ticker": ticker.removesuffix(".NS"),
                        "YFinance Ticker": ticker,
                        "Stage": "Near-live quote",
                        "Status": "available",
                        "Rows": int(series.shape[0]),
                        "Message": "",
                    }
                )
        completed += len(batch)
        if progress_callback:
            progress_callback(completed, total, f"Refreshed {completed:,} of {total:,} quotes")

    return pd.DataFrame(rows), pd.DataFrame(health_rows)


def insufficient_history_tickers(prices: pd.DataFrame, tickers: list[str], required_rows: int) -> list[str]:
    missing: list[str] = []
    for ticker in tickers:
        key = str(ticker).upper()
        rows = _clean_price_series(prices[key]).shape[0] if key in prices.columns else 0
        if rows < int(required_rows):
            missing.append(key)
    return missing


def combine_price_history(
    yahoo_prices: pd.DataFrame,
    fallback_prices: pd.DataFrame,
    tickers: list[str],
    required_rows: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Choose adjusted Yahoo history where sufficient, otherwise use the official NSE fallback."""
    selected: list[pd.Series] = []
    sources: dict[str, str] = {}
    for ticker in list(dict.fromkeys(str(item).upper() for item in tickers)):
        yahoo = _clean_price_series(yahoo_prices[ticker]) if ticker in yahoo_prices.columns else pd.Series(dtype=float)
        fallback = (
            _clean_price_series(fallback_prices[ticker]) if ticker in fallback_prices.columns else pd.Series(dtype=float)
        )
        if yahoo.shape[0] >= int(required_rows) or fallback.shape[0] <= yahoo.shape[0]:
            chosen = yahoo
            source = "Yahoo Finance"
        else:
            chosen = fallback
            source = "NSE India"
        if not chosen.empty:
            chosen = chosen.rename(ticker)
            selected.append(chosen)
            sources[ticker] = source
        else:
            sources[ticker] = "Unavailable"
    prices = pd.concat(selected, axis=1).sort_index() if selected else pd.DataFrame()
    return prices, sources


def score_sma200_at_date(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    config: Sma200ScanConfig,
) -> pd.DataFrame:
    """Rank historical positive-proximity setups using only data through signal_date."""
    rows: list[dict[str, object]] = []
    for ticker in prices.columns:
        series = _clean_price_series(prices.loc[:signal_date, ticker])
        if series.shape[0] < int(config.window_days):
            continue
        rolling = series.rolling(int(config.window_days), min_periods=int(config.window_days)).mean().dropna()
        if rolling.empty:
            continue
        price_value = float(series.iloc[-1])
        sma_value = float(rolling.iloc[-1])
        if not sma_value:
            continue
        distance = ((price_value / sma_value) - 1.0) * 100.0
        if not float(config.min_distance_pct) <= distance <= float(config.max_distance_pct):
            continue
        slope = _rolling_slope(rolling, int(config.slope_lookback_days))
        rows.append(
            {
                "YFinance Ticker": str(ticker).upper(),
                "Signal Price": price_value,
                "200DMA": sma_value,
                "Distance Above 200DMA %": distance,
                f"200DMA {config.slope_lookback_days}D Slope %": slope,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["Distance Above 200DMA %", "YFinance Ticker"],
        ascending=[True, True],
    ).reset_index(drop=True)


def walk_forward_sma200_backtest(
    eligible: pd.DataFrame,
    prices: pd.DataFrame,
    config: Sma200ScanConfig,
    initial_capital: float = 100000.0,
    add_benchmark_data: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a monthly, no-price-lookahead 200DMA proximity portfolio backtest."""
    if eligible.empty or prices.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_allocation(eligible, initial_capital)

    tickers = eligible["YFinance Ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    available = [ticker for ticker in tickers if ticker in prices.columns]
    if not available:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_allocation(eligible, initial_capital)

    panel = prices[available].apply(pd.to_numeric, errors="coerce").sort_index()
    panel.index = pd.to_datetime(panel.index, errors="coerce")
    panel = panel.loc[~panel.index.isna()]
    if panel.index.tz is not None:
        panel.index = panel.index.tz_localize(None)
    panel.index = panel.index.normalize()
    panel = panel.loc[~panel.index.duplicated(keep="last")].sort_index()
    panel = panel.dropna(how="all")
    schedule = monthly_rebalance_schedule(panel.index, max(int(config.backtest_months), 1))
    if len(schedule) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_allocation(eligible, initial_capital)

    metadata = eligible.drop_duplicates("YFinance Ticker").copy()
    metadata["_TickerKey"] = metadata["YFinance Ticker"].astype(str).str.upper()
    metadata = metadata.set_index("_TickerKey")
    capital = float(initial_capital)
    curve_parts: list[pd.Series] = []
    period_rows: list[dict[str, object]] = []

    for entry_date, exit_date in zip(schedule[:-1], schedule[1:]):
        signal_dates = panel.index[panel.index < entry_date]
        if signal_dates.empty:
            continue
        signal_date = signal_dates[-1]
        scored = score_sma200_at_date(panel, signal_date, config)
        selected = scored.head(10).copy()
        entry_capital = capital
        period_index = panel.loc[entry_date:exit_date].index
        if period_index.empty:
            continue

        if selected.empty:
            period_curve = pd.Series(capital, index=period_index, dtype=float)
            active = selected
        else:
            entry_row = panel.reindex([entry_date]).iloc[0]
            active = selected[
                selected["YFinance Ticker"].map(
                    lambda ticker: ticker in entry_row.index and pd.notna(entry_row.get(ticker))
                )
            ].copy()
            if active.empty:
                period_curve = pd.Series(capital, index=period_index, dtype=float)
            else:
                active_tickers = active["YFinance Ticker"].tolist()
                weights = portfolio_weights(active_tickers)
                period_prices = panel.loc[entry_date:exit_date, active_tickers].ffill()
                period_curve = pd.Series(0.0, index=period_prices.index, dtype=float)
                for ticker, weight in weights.items():
                    entry_price = float(period_prices[ticker].iloc[0])
                    shares = (capital * weight) / entry_price
                    period_curve += period_prices[ticker] * shares

        capital = float(period_curve.iloc[-1])
        curve_parts.append(period_curve)
        names: list[str] = []
        for ticker in active.get("YFinance Ticker", pd.Series(dtype=str)).astype(str):
            name = metadata["Name"].get(ticker, "") if "Name" in metadata.columns and ticker in metadata.index else ""
            names.append(f"{ticker.removesuffix('.NS')}{f' ({name})' if name else ''}")
        distances = active.get("Distance Above 200DMA %", pd.Series(dtype=float))
        period_rows.append(
            {
                "Rebalance Date": entry_date.date().isoformat(),
                "Signal Date": signal_date.date().isoformat(),
                "Exit Date": exit_date.date().isoformat(),
                "Selected Tickers": ", ".join(
                    active.get("YFinance Ticker", pd.Series(dtype=str)).astype(str).str.removesuffix(".NS")
                ),
                "Selected Names": ", ".join(names),
                "Signal Distances %": ", ".join(f"{float(value):.2f}" for value in distances),
                "Starting Capital": round(entry_capital, 2),
                "Ending Capital": round(capital, 2),
                "Monthly Return %": round(((capital / entry_capital) - 1.0) * 100.0, 2)
                if entry_capital
                else np.nan,
                "Position Count": int(len(active)),
            }
        )

    if not curve_parts:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), current_allocation(eligible, initial_capital)

    strategy = pd.concat(curve_parts)
    strategy = strategy.loc[~strategy.index.duplicated(keep="last")].sort_index()
    curves = pd.DataFrame({"200DMA Strategy": strategy})
    if add_benchmark_data:
        try:
            curves = add_benchmarks(curves, initial_capital=float(initial_capital))
        except Exception:
            # Preserve the strategy result when a benchmark quote is temporarily unavailable.
            pass
    normalized = (curves / float(initial_capital)) * 100.0
    allocation_source = eligible.sort_values("Distance Above 200DMA %", ascending=True)
    allocation = current_allocation(allocation_source, capital=float(initial_capital))
    return curves, normalized, pd.DataFrame(period_rows), allocation


def sma200_chart_data(prices: pd.DataFrame, ticker: str, window_days: int = 200) -> pd.DataFrame:
    key = str(ticker).upper()
    if key not in prices.columns:
        return pd.DataFrame()
    series = _clean_price_series(prices[key])
    if series.empty:
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "Date": series.index,
            "Price": series.to_numpy(),
            f"{int(window_days)}DMA": series.rolling(int(window_days), min_periods=int(window_days)).mean().to_numpy(),
        }
    ).tail(max(int(window_days) + 80, 300))
    return frame.melt("Date", var_name="Series", value_name="Value").dropna(subset=["Value"])


def _rolling_slope(rolling: pd.Series, lookback_days: int) -> float:
    if rolling.shape[0] <= int(lookback_days):
        return np.nan
    previous = float(rolling.iloc[-int(lookback_days) - 1])
    current = float(rolling.iloc[-1])
    if previous == 0:
        return np.nan
    return ((current / previous) - 1.0) * 100.0


def _trend_label(slope: float) -> str:
    if pd.isna(slope):
        return "Insufficient slope history"
    if slope > 0:
        return "Rising"
    if slope < 0:
        return "Falling"
    return "Flat"


def _clean_price_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    values = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(),
        index=pd.to_datetime(series.index, errors="coerce"),
        name=series.name,
    )
    values = values.loc[~values.index.isna()].dropna()
    if values.index.tz is not None:
        values.index = values.index.tz_localize(None)
    values.index = values.index.normalize()
    return values.loc[~values.index.duplicated(keep="last")].sort_index()


def _report_calculation_progress(
    progress_callback: ProgressCallback | None,
    position: int,
    total: int,
) -> None:
    if progress_callback and (position == total or position % 100 == 0):
        progress_callback(position, total, f"Calculated 200DMA for {position:,} of {total:,} stocks")
