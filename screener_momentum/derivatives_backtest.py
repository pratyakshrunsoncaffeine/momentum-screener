from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from .config import DEFAULT_MOMENTUM_WEIGHTS, DerivativesSignalConfig
from .derivatives import (
    NseEodDerivativesProvider,
    build_volume_history,
    scan_derivatives_momentum,
)


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class DerivativesBacktestResult:
    events: pd.DataFrame
    curve: pd.DataFrame
    summary: pd.DataFrame


def derivatives_backtest(
    provider: NseEodDerivativesProvider,
    universe: pd.DataFrame,
    config: DerivativesSignalConfig,
    start_date: date,
    end_date: date,
    holding_days: int = 5,
    top_n: int = 10,
    round_trip_cost_pct: float = 0.30,
    progress_callback: ProgressCallback | None = None,
) -> DerivativesBacktestResult:
    """Walk forward through cached NSE reports and enter equity positions next session."""
    if holding_days < 1:
        raise ValueError("Holding days must be at least one.")
    dates = [item for item in provider.available_dates() if start_date <= item <= end_date]
    if len(dates) < 12:
        raise ValueError("At least 12 cached trading dates are required for the derivatives backtest.")

    daily_data = {trade_date: provider.load_date(trade_date) for trade_date in dates}
    close_panel = _cash_close_panel(daily_data, universe)
    feature_history: list[pd.DataFrame] = []
    scans: dict[date, dict[str, pd.DataFrame]] = {}
    scan_config = replace(config, result_count=max(len(universe), config.result_count))

    for index, trade_date in enumerate(dates, start=1):
        if progress_callback:
            progress_callback(index - 1, len(dates), f"Generating derivatives features for {trade_date}")
        previous = daily_data[dates[index - 2]].derivatives if index > 1 else None
        history = build_volume_history(feature_history[-20:])
        scan = scan_derivatives_momentum(
            universe,
            daily_data[trade_date],
            scan_config,
            previous_derivatives=previous,
            volume_history=history,
        )
        scans[trade_date] = scan
        feature_history.append(scan["features"])
        if progress_callback:
            progress_callback(index, len(dates), f"Generated {index:,} of {len(dates):,} daily feature sets")

    events: list[dict[str, object]] = []
    for signal_index, signal_date in enumerate(dates[:-1]):
        entry_index = signal_index + 1
        entry_date = dates[entry_index]
        scan = scans[signal_date]
        variants = _variant_candidates(scan, close_panel, signal_date, config)
        for variant, candidates in variants.items():
            if candidates.empty:
                continue
            selected = candidates.sort_values("Variant Score", ascending=False).drop_duplicates("Ticker").head(top_n)
            for candidate in selected.to_dict("records"):
                event = _build_event(
                    ticker=str(candidate["Ticker"]),
                    variant=variant,
                    score=float(candidate["Variant Score"]),
                    signal_date=signal_date,
                    entry_index=entry_index,
                    dates=dates,
                    daily_data=daily_data,
                    holding_days=holding_days,
                    cost_pct=round_trip_cost_pct,
                )
                if event:
                    events.append(event)

    events_frame = pd.DataFrame(events)
    if events_frame.empty:
        return DerivativesBacktestResult(events_frame, pd.DataFrame(), pd.DataFrame())
    events_frame["Split"] = _chronological_split(events_frame["Signal Date"])
    curve = simulate_derivatives_portfolio(
        events_frame,
        dates,
        max_positions=top_n,
        daily_data=daily_data,
    )
    summary = summarize_derivatives_events(events_frame)
    return DerivativesBacktestResult(events_frame, curve, summary)


def simulate_derivatives_portfolio(
    events: pd.DataFrame,
    trading_dates: list[date],
    max_positions: int = 10,
    initial_capital: float = 100000.0,
    daily_data: dict[date, object] | None = None,
) -> pd.DataFrame:
    """Simulate capped concurrent positions from precomputed next-open events."""
    if events.empty:
        return pd.DataFrame()
    curves: list[pd.DataFrame] = []
    for variant, variant_events in events.groupby("Variant"):
        cash = float(initial_capital)
        positions: list[dict[str, object]] = []
        values: list[dict[str, object]] = []
        entries_by_date = {
            key: value.sort_values("Signal Score", ascending=False)
            for key, value in variant_events.groupby("Entry Date")
        }
        for trade_date in trading_dates:
            remaining: list[dict[str, object]] = []
            for position in positions:
                if position["Exit Date"] == trade_date:
                    cash += float(position["Allocation"]) * (1.0 + float(position["Net Return %"]) / 100.0)
                else:
                    remaining.append(position)
            positions = remaining

            todays_entries = entries_by_date.get(trade_date, pd.DataFrame())
            held = {str(position["Ticker"]) for position in positions}
            available_slots = max(0, max_positions - len(positions))
            if available_slots and not todays_entries.empty:
                portfolio_value = cash + sum(
                    _marked_position_value(position, trade_date, daily_data)
                    for position in positions
                )
                target_allocation = portfolio_value / max_positions
                for event in todays_entries.to_dict("records"):
                    ticker = str(event["Ticker"])
                    if ticker in held or available_slots <= 0 or cash <= 0:
                        continue
                    allocation = min(target_allocation, cash)
                    positions.append(
                        {
                            "Ticker": ticker,
                            "Exit Date": event["Exit Date"],
                            "Allocation": allocation,
                            "Entry Price": float(event["Entry Price"]),
                            "Net Return %": float(event["Net Return %"]),
                        }
                    )
                    cash -= allocation
                    held.add(ticker)
                    available_slots -= 1

            estimated_value = cash + sum(
                _marked_position_value(position, trade_date, daily_data)
                for position in positions
            )
            values.append({"Date": trade_date, "Variant": variant, "Portfolio Value": round(estimated_value, 2)})
        curves.append(pd.DataFrame(values))
    return pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()


def _marked_position_value(
    position: dict[str, object],
    trade_date: date,
    daily_data: dict[date, object] | None,
) -> float:
    allocation = float(position["Allocation"])
    entry_price = float(position.get("Entry Price", 0) or 0)
    if daily_data is None or trade_date not in daily_data or entry_price <= 0:
        return allocation
    cash = daily_data[trade_date].cash
    rows = cash[cash["Ticker"].eq(str(position["Ticker"]))]
    if rows.empty:
        return allocation
    close = pd.to_numeric(rows.iloc[-1].get("Close"), errors="coerce")
    return allocation if pd.isna(close) or float(close) <= 0 else allocation * float(close) / entry_price


def summarize_derivatives_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    for (variant, split), group in events.groupby(["Variant", "Split"], dropna=False):
        returns = pd.to_numeric(group["Net Return %"], errors="coerce").dropna().to_numpy()
        if returns.size == 0:
            continue
        bootstrap_means = np.array(
            [rng.choice(returns, size=returns.size, replace=True).mean() for _ in range(1000)]
        )
        rows.append(
            {
                "Variant": variant,
                "Split": split,
                "Signals": int(returns.size),
                "Hit Rate %": round(float((returns > 0).mean() * 100), 2),
                "Mean Return %": round(float(returns.mean()), 3),
                "Median Return %": round(float(np.median(returns)), 3),
                "Mean Excess vs Nifty 50 %": _mean_column(group, "Excess Return vs Nifty 50 %"),
                "Mean 95% CI Low": round(float(np.quantile(bootstrap_means, 0.025)), 3),
                "Mean 95% CI High": round(float(np.quantile(bootstrap_means, 0.975)), 3),
            }
        )
    return pd.DataFrame(rows).sort_values(["Split", "Variant"]).reset_index(drop=True)


def _mean_column(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return round(float(values.mean()), 3) if not values.empty else np.nan


def summarize_derivatives_curve(curve: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant, group in curve.groupby("Variant"):
        group = group.sort_values("Date")
        values = pd.to_numeric(group["Portfolio Value"], errors="coerce").dropna()
        if len(values) < 2:
            continue
        daily = values.pct_change().dropna()
        start_date = pd.to_datetime(group.iloc[0]["Date"])
        end_date = pd.to_datetime(group.iloc[-1]["Date"])
        years = max((end_date - start_date).days / 365.25, 1 / 365.25)
        total_return = values.iloc[-1] / values.iloc[0] - 1.0
        cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1.0
        running_max = values.cummax()
        max_drawdown = (values / running_max - 1.0).min()
        sharpe = (daily.mean() / daily.std(ddof=0) * np.sqrt(252)) if daily.std(ddof=0) > 0 else np.nan
        variant_events = events[events["Variant"].eq(variant)] if not events.empty else pd.DataFrame()
        rows.append(
            {
                "Variant": variant,
                "Total Return %": round(total_return * 100, 2),
                "CAGR %": round(cagr * 100, 2),
                "Sharpe": round(float(sharpe), 3) if pd.notna(sharpe) else np.nan,
                "Maximum Drawdown %": round(max_drawdown * 100, 2),
                "Turnover / Signals": len(variant_events),
            }
        )
    return pd.DataFrame(rows)


def _variant_candidates(
    scan: dict[str, pd.DataFrame],
    close_panel: pd.DataFrame,
    signal_date: date,
    config: DerivativesSignalConfig,
) -> dict[str, pd.DataFrame]:
    features = scan["features"].copy()
    if features.empty:
        return {name: pd.DataFrame() for name in ("Equity Only", "Price + Call", "Full Derivatives", "Regular Momentum")}
    underlying = pd.to_numeric(features.get("Underlying Return %"), errors="coerce")
    call_return = pd.to_numeric(features.get("Call Return %"), errors="coerce")
    equity = features[underlying.ge(config.min_underlying_return_pct)].copy()
    equity["Variant Score"] = pd.to_numeric(equity["Underlying Return %"], errors="coerce")

    price_call = features[
        underlying.ge(config.min_underlying_return_pct) & call_return.ge(config.min_call_return_pct)
    ].copy()
    if not price_call.empty:
        price_call["Variant Score"] = (
            pd.to_numeric(price_call["Underlying Return %"], errors="coerce").rank(pct=True).fillna(0.5) * 40
            + pd.to_numeric(price_call["Call Return %"], errors="coerce").rank(pct=True).fillna(0.5) * 60
        )

    full = scan["signals"].copy()
    if not full.empty:
        full["Variant Score"] = full["Derivatives Momentum Score"]
    regular = _regular_momentum_candidates(close_panel, signal_date, set(features["Ticker"].astype(str)))
    return {
        "Equity Only": equity,
        "Price + Call": price_call,
        "Full Derivatives": full,
        "Regular Momentum": regular,
    }


def _regular_momentum_candidates(close_panel: pd.DataFrame, signal_date: date, eligible: set[str]) -> pd.DataFrame:
    if signal_date not in close_panel.index:
        return pd.DataFrame()
    position = close_panel.index.get_loc(signal_date)
    if not isinstance(position, (int, np.integer)) or position < 126:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for ticker in eligible:
        if ticker not in close_panel.columns:
            continue
        series = close_panel[ticker].iloc[: position + 1].dropna()
        if len(series) <= 126:
            continue
        returns = {
            "5 days ret": _series_return(series, 5),
            "15 Days Returns": _series_return(series, 15),
            "1M Return": _series_return(series, 21),
            "2 months Ret.": _series_return(series, 42),
            "3M Return": _series_return(series, 63),
            "6M Return": _series_return(series, 126),
        }
        if any(returns[key] is None or returns[key] <= 0 for key in ("5 days ret", "15 Days Returns", "1M Return")):
            continue
        score = sum(float(returns[key]) * weight for key, weight in DEFAULT_MOMENTUM_WEIGHTS.items())
        rows.append({"Ticker": ticker, "Variant Score": round(score, 3)})
    return pd.DataFrame(rows)


def _build_event(
    ticker: str,
    variant: str,
    score: float,
    signal_date: date,
    entry_index: int,
    dates: list[date],
    daily_data: dict[date, object],
    holding_days: int,
    cost_pct: float,
) -> dict[str, object] | None:
    entry_date = dates[entry_index]
    entry_cash = daily_data[entry_date].cash
    entry_rows = entry_cash[entry_cash["Ticker"].eq(ticker)]
    if entry_rows.empty:
        return None
    entry_price = float(entry_rows.iloc[-1]["Open"])
    if entry_price <= 0 or pd.isna(entry_price):
        return None

    horizon_returns: dict[str, float | None] = {}
    for horizon in (1, 3, 5, 10):
        exit_index = entry_index + horizon - 1
        if exit_index >= len(dates):
            horizon_returns[f"Forward {horizon}D Return %"] = None
            continue
        exit_rows = daily_data[dates[exit_index]].cash
        ticker_exit = exit_rows[exit_rows["Ticker"].eq(ticker)]
        if ticker_exit.empty:
            horizon_returns[f"Forward {horizon}D Return %"] = None
        else:
            exit_price = float(ticker_exit.iloc[-1]["Close"])
            horizon_returns[f"Forward {horizon}D Return %"] = round((exit_price / entry_price - 1) * 100, 3)

    exit_index = entry_index + holding_days - 1
    if exit_index >= len(dates):
        return None
    exit_date = dates[exit_index]
    exit_rows = daily_data[exit_date].cash
    ticker_exit = exit_rows[exit_rows["Ticker"].eq(ticker)]
    if ticker_exit.empty:
        return None
    exit_price = float(ticker_exit.iloc[-1]["Close"])
    gross_return = (exit_price / entry_price - 1.0) * 100.0
    return {
        "Variant": variant,
        "Ticker": ticker,
        "Signal Date": signal_date,
        "Entry Date": entry_date,
        "Exit Date": exit_date,
        "Signal Score": round(score, 3),
        "Entry Price": round(entry_price, 2),
        "Exit Price": round(exit_price, 2),
        "Gross Return %": round(gross_return, 3),
        "Net Return %": round(gross_return - cost_pct, 3),
        **horizon_returns,
    }


def _cash_close_panel(daily_data: dict[date, object], universe: pd.DataFrame) -> pd.DataFrame:
    tickers = set(universe["Ticker"].astype(str).str.upper())
    rows: list[pd.DataFrame] = []
    for trade_date, daily in daily_data.items():
        frame = daily.cash[daily.cash["Ticker"].isin(tickers)][["Ticker", "Close"]].copy()
        frame["Date"] = trade_date
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    return long.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last").sort_index()


def _series_return(series: pd.Series, days: int) -> float | None:
    if len(series) <= days:
        return None
    previous = float(series.iloc[-days - 1])
    return None if previous == 0 else (float(series.iloc[-1]) / previous - 1) * 100


def _chronological_split(signal_dates: pd.Series) -> pd.Series:
    dates = sorted(pd.Series(signal_dates).dropna().unique())
    if not dates:
        return pd.Series("Test", index=signal_dates.index)
    train_end = dates[max(0, int(len(dates) * 0.60) - 1)]
    validation_end = dates[max(0, int(len(dates) * 0.80) - 1)]
    return signal_dates.map(
        lambda value: "Train" if value <= train_end else "Validation" if value <= validation_end else "Test"
    )
