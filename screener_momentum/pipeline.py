from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .backtest import current_allocation, performance_summary, walk_forward_backtest
from .config import (
    DEFAULT_POST_EARNINGS_STOCK_RETURN_WEIGHTS,
    POST_EARNINGS_STOCK_RETURN_PERIODS,
    ScreeningConfig,
    DerivativesSignalConfig,
)
from .derivatives import NseEodDerivativesProvider, build_volume_history, scan_derivatives_momentum
from .derivatives_backtest import derivatives_backtest, summarize_derivatives_curve, summarize_derivatives_events
from .fundamentals import (
    normalize_quarter_period,
    screen_dii_holdings,
    screen_fii_holdings,
    screen_fundamentals,
    screen_quarterly_results,
)
from .momentum import calculate_returns, download_adjusted_close, score_momentum
from .sector_rotation import BENCHMARK_INDEX, NseSectorIndexProvider, calculate_sector_rotation
from .universe import load_ticker_universe

ProgressCallback = Callable[[int, int, str], None]


def output_paths(output_dir: str | Path = "output/latest") -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "root": root,
        "returns": root / "returns.csv",
        "momentum": root / "momentum.csv",
        "fundamentals_partial": root / "fundamentals_partial.csv",
        "fundamentals": root / "fundamentals.csv",
        "final": root / "final.csv",
        "backtest": root / "backtest.csv",
        "normalized_backtest": root / "normalized_backtest.csv",
        "walk_forward_backtest": root / "walk_forward_backtest.csv",
        "walk_forward_periods": root / "walk_forward_periods.csv",
        "current_allocation": root / "current_allocation.csv",
        "holdings": root / "holdings.csv",
        "performance": root / "performance.csv",
        "fii_all": root / "fii_all.csv",
        "fii_partial": root / "fii_partial.csv",
        "fii_marketcap_partial": root / "fii_marketcap_partial.csv",
        "fii_top": root / "fii_top50.csv",
        "fii_momentum": root / "fii_momentum.csv",
        "fii_final": root / "fii_final.csv",
        "dii_all": root / "dii_all.csv",
        "dii_partial": root / "dii_partial.csv",
        "dii_marketcap_partial": root / "dii_marketcap_partial.csv",
        "dii_top": root / "dii_top50.csv",
        "dii_momentum": root / "dii_momentum.csv",
        "dii_final": root / "dii_final.csv",
        "quarterly_results_partial": root / "quarterly_results_partial.csv",
        "quarterly_results_all": root / "quarterly_results_all.csv",
        "quarterly_results_matching": root / "quarterly_results_matching.csv",
        "quarterly_stock_return_returns": root / "quarterly_stock_return_returns.csv",
        "quarterly_stock_return_momentum": root / "quarterly_stock_return_momentum.csv",
        "derivatives_cache": root.parent / "derivatives_cache",
        "derivatives_contracts": root / "derivatives_contracts.csv",
        "derivatives_daily_features": root / "derivatives_daily_features.csv",
        "derivatives_signals": root / "derivatives_signals.csv",
        "derivatives_rejections": root / "derivatives_rejections.csv",
        "derivatives_data_health": root / "derivatives_data_health.csv",
        "derivatives_backtest_events": root / "derivatives_backtest_events.csv",
        "derivatives_backtest_curve": root / "derivatives_backtest_curve.csv",
        "derivatives_backtest_summary": root / "derivatives_backtest_summary.csv",
        "derivatives_event_summary": root / "derivatives_event_summary.csv",
        "sector_rotation_cache": root.parent / "sector_rotation_cache",
        "sector_rotation_prices": root / "sector_rotation_prices.csv",
        "sector_rotation_snapshot": root / "sector_rotation_snapshot.csv",
        "sector_rotation_health": root / "sector_rotation_health.csv",
    }


def save_frame(frame: pd.DataFrame, path: Path, include_index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=include_index)


def _read_saved_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run_sector_rotation_screen(
    sectors: list[str] | tuple[str, ...],
    end_date: date,
    history_months: int = 12,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, object]:
    """Refresh official NSE sector-index history and save a rotation snapshot."""
    if not sectors:
        raise ValueError("Select at least one official NSE sector index.")
    months = max(int(history_months), 7)
    start_date = end_date - timedelta(days=months * 31)
    paths = output_paths(output_dir)
    provider = NseSectorIndexProvider(paths["sector_rotation_cache"])
    prices, health = provider.fetch_many(
        [BENCHMARK_INDEX, *sectors],
        start_date=start_date,
        end_date=end_date,
        progress_callback=progress_callback,
    )
    failed = health.loc[health["Status"].eq("failed"), "Index"].astype(str).tolist() if not health.empty else []
    if BENCHMARK_INDEX in failed:
        raise RuntimeError("Official Nifty 50 history is unavailable from NSE and the local NSE cache.")
    snapshot = calculate_sector_rotation(prices, sectors)
    if snapshot.empty:
        raise RuntimeError("NSE returned no common sector and benchmark history for this selection.")
    save_frame(prices, paths["sector_rotation_prices"])
    save_frame(snapshot, paths["sector_rotation_snapshot"])
    save_frame(health, paths["sector_rotation_health"])
    return {"prices": prices, "snapshot": snapshot, "health": health, "stale": False}


def load_saved_sector_rotation(
    output_dir: str | Path = "output/latest",
) -> dict[str, object]:
    """Recover the last completed official NSE sector rotation run."""
    paths = output_paths(output_dir)
    prices = _read_saved_frame(paths["sector_rotation_prices"])
    snapshot = _read_saved_frame(paths["sector_rotation_snapshot"])
    health = _read_saved_frame(paths["sector_rotation_health"])
    if prices.empty or snapshot.empty:
        raise FileNotFoundError("No saved NSE sector rotation run is available yet.")
    prices["Date"] = pd.to_datetime(prices.get("Date"), errors="coerce")
    return {"prices": prices, "snapshot": snapshot, "health": health, "stale": True}


def _download_nifty_prices(start_date: date, end_date: date) -> pd.DataFrame:
    try:
        data = yf.download(
            "^NSEI",
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs("^NSEI", axis=1, level=1) if "^NSEI" in data.columns.get_level_values(1) else data.droplevel(1, axis=1)
        result = data.reset_index()
        result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.date
        return result.dropna(subset=["Date"]).set_index("Date")
    except Exception:
        return pd.DataFrame()


def _add_nifty_benchmark(
    curve: pd.DataFrame,
    start_date: date,
    end_date: date,
    prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if curve.empty:
        return curve
    try:
        data = prices if prices is not None else _download_nifty_prices(start_date, end_date)
        if data.empty:
            return curve
        close = data["Close"]
        close = pd.to_numeric(close, errors="coerce").dropna()
        if close.empty:
            return curve
        initial = float(pd.to_numeric(curve["Portfolio Value"], errors="coerce").dropna().iloc[0])
        benchmark = pd.DataFrame(
            {
                "Date": list(close.index),
                "Variant": "Nifty 50",
                "Portfolio Value": (close / close.iloc[0] * initial).round(2).to_numpy(),
            }
        )
        return pd.concat([curve, benchmark], ignore_index=True)
    except Exception:
        return curve


def _add_nifty_event_returns(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    if result.empty or prices.empty or not {"Open", "Close"}.issubset(prices.columns):
        return result
    opens = pd.to_numeric(prices["Open"], errors="coerce")
    closes = pd.to_numeric(prices["Close"], errors="coerce")
    benchmark_returns: list[float | None] = []
    for row in result.to_dict("records"):
        entry_date = pd.to_datetime(row["Entry Date"], errors="coerce")
        exit_date = pd.to_datetime(row["Exit Date"], errors="coerce")
        if pd.isna(entry_date) or pd.isna(exit_date):
            benchmark_returns.append(None)
            continue
        entry = opens.get(entry_date.date())
        exit_value = closes.get(exit_date.date())
        if pd.isna(entry) or pd.isna(exit_value) or float(entry) <= 0:
            benchmark_returns.append(None)
        else:
            benchmark_returns.append(round((float(exit_value) / float(entry) - 1.0) * 100.0, 3))
    result["Nifty 50 Return %"] = benchmark_returns
    result["Excess Return vs Nifty 50 %"] = (
        pd.to_numeric(result["Net Return %"], errors="coerce")
        - pd.to_numeric(result["Nifty 50 Return %"], errors="coerce")
    ).round(3)
    return result


def load_saved_returns(output_dir: str | Path = "output/latest") -> pd.DataFrame:
    path = output_paths(output_dir)["returns"]
    if not path.exists():
        raise FileNotFoundError(f"Saved returns not found: {path}")
    return pd.read_csv(path)


def run_price_returns(
    csv_path: str,
    config: ScreeningConfig,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> pd.DataFrame:
    universe = load_ticker_universe(csv_path)
    prices = download_adjusted_close(
        universe["YFinance Ticker"].tolist(),
        batch_size=config.price_batch_size,
        progress_callback=progress_callback,
    )
    returns = calculate_returns(universe, prices, progress_callback=progress_callback)
    save_frame(returns, output_paths(output_dir)["returns"])
    return returns


def score_and_save_momentum(
    returns: pd.DataFrame,
    config: ScreeningConfig,
    output_dir: str | Path = "output/latest",
) -> pd.DataFrame:
    momentum = score_momentum(
        returns.copy(),
        weights=config.momentum_weights,
        positive_filters=config.positive_return_filters,
    )
    save_frame(momentum, output_paths(output_dir)["momentum"])
    return momentum


def run_momentum(
    csv_path: str,
    config: ScreeningConfig,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
    use_saved_returns: bool = False,
) -> pd.DataFrame:
    returns = load_saved_returns(output_dir) if use_saved_returns else run_price_returns(
        csv_path,
        config,
        progress_callback=progress_callback,
        output_dir=output_dir,
    )
    return score_and_save_momentum(returns, config, output_dir=output_dir)


def download_derivatives_eod_data(
    start_date: date,
    end_date: date,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> pd.DataFrame:
    paths = output_paths(output_dir)
    provider = NseEodDerivativesProvider(paths["derivatives_cache"])
    health = provider.download_range(start_date, end_date, progress_callback=progress_callback)
    save_frame(health, paths["derivatives_data_health"])
    return health


def run_derivatives_momentum_screen(
    csv_path: str,
    config: DerivativesSignalConfig,
    scan_date: date | None = None,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    paths = output_paths(output_dir)
    provider = NseEodDerivativesProvider(paths["derivatives_cache"])
    available = [item for item in provider.available_dates() if scan_date is None or item <= scan_date]
    if not available:
        raise FileNotFoundError("No cached NSE derivatives dates are available. Download EOD data first.")
    selected_date = max(available)
    selected_index = available.index(selected_date)
    daily = provider.load_date(selected_date)
    previous = provider.load_date(available[selected_index - 1]).derivatives if selected_index > 0 else None
    universe = load_ticker_universe(csv_path)

    prior_frames: list[pd.DataFrame] = []
    history_dates = available[max(0, selected_index - 20):selected_index]
    for index, history_date in enumerate(history_dates, start=1):
        if progress_callback:
            progress_callback(index - 1, len(history_dates) + 1, f"Building option-volume history for {history_date}")
        history_daily = provider.load_date(history_date)
        history_previous = (
            provider.load_date(available[available.index(history_date) - 1]).derivatives
            if available.index(history_date) > 0
            else None
        )
        history_scan = scan_derivatives_momentum(
            universe,
            history_daily,
            config,
            previous_derivatives=history_previous,
            volume_history=build_volume_history(prior_frames[-20:]),
        )
        prior_frames.append(history_scan["features"])

    results = scan_derivatives_momentum(
        universe,
        daily,
        config,
        previous_derivatives=previous,
        volume_history=build_volume_history(prior_frames[-20:]),
        progress_callback=progress_callback,
    )
    save_frame(results["contracts"], paths["derivatives_contracts"])
    save_frame(results["features"], paths["derivatives_daily_features"])
    save_frame(results["signals"], paths["derivatives_signals"])
    save_frame(results["rejections"], paths["derivatives_rejections"])
    results["data_health"] = _read_saved_frame(paths["derivatives_data_health"])
    return results


def run_derivatives_backtest_screen(
    csv_path: str,
    config: DerivativesSignalConfig,
    start_date: date,
    end_date: date,
    holding_days: int = 5,
    top_n: int = 10,
    round_trip_cost_pct: float = 0.30,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    paths = output_paths(output_dir)
    provider = NseEodDerivativesProvider(paths["derivatives_cache"])
    universe = load_ticker_universe(csv_path)
    result = derivatives_backtest(
        provider,
        universe,
        config,
        start_date,
        end_date,
        holding_days=holding_days,
        top_n=top_n,
        round_trip_cost_pct=round_trip_cost_pct,
        progress_callback=progress_callback,
    )
    nifty_prices = _download_nifty_prices(start_date, end_date)
    events = _add_nifty_event_returns(result.events, nifty_prices)
    curve = _add_nifty_benchmark(result.curve, start_date, end_date, prices=nifty_prices)
    performance = summarize_derivatives_curve(curve, events)
    event_summary = summarize_derivatives_events(events)
    save_frame(events, paths["derivatives_backtest_events"])
    save_frame(curve, paths["derivatives_backtest_curve"])
    save_frame(performance, paths["derivatives_backtest_summary"])
    save_frame(event_summary, paths["derivatives_event_summary"])
    return {
        "events": events,
        "curve": curve,
        "performance": performance,
        "event_summary": event_summary,
    }


def load_saved_derivatives_results(output_dir: str | Path = "output/latest") -> dict[str, pd.DataFrame]:
    paths = output_paths(output_dir)
    return {
        "contracts": _read_saved_frame(paths["derivatives_contracts"]),
        "features": _read_saved_frame(paths["derivatives_daily_features"]),
        "signals": _read_saved_frame(paths["derivatives_signals"]),
        "rejections": _read_saved_frame(paths["derivatives_rejections"]),
        "data_health": _read_saved_frame(paths["derivatives_data_health"]),
        "events": _read_saved_frame(paths["derivatives_backtest_events"]),
        "curve": _read_saved_frame(paths["derivatives_backtest_curve"]),
        "performance": _read_saved_frame(paths["derivatives_backtest_summary"]),
        "event_summary": _read_saved_frame(paths["derivatives_event_summary"]),
    }


def run_fii_momentum_screen(
    csv_path: str,
    config: ScreeningConfig,
    fii_top_n: int = 50,
    final_n: int = 3,
    progress_callback: ProgressCallback | None = None,
    price_progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    paths = output_paths(output_dir)
    universe = load_ticker_universe(csv_path)
    fii_all = screen_fii_holdings(
        universe,
        progress_callback=progress_callback,
        checkpoint_path=paths["fii_partial"],
    )
    needs_market_cap = "Market Cap Cr" not in fii_all.columns or pd.to_numeric(
        fii_all.get("Market Cap Cr", pd.Series(dtype=float)),
        errors="coerce",
    ).isna().all()
    if needs_market_cap:
        fii_all = enrich_market_cap_from_yfinance(
            fii_all,
            progress_callback=price_progress_callback,
            checkpoint_path=paths["fii_marketcap_partial"],
        )
    if "Market Cap Cr" in fii_all.columns:
        fii_all["Market Cap Cr"] = pd.to_numeric(fii_all["Market Cap Cr"], errors="coerce")
        fii_all = fii_all.sort_values("Market Cap Cr", ascending=False, na_position="last").reset_index(drop=True)
    return finalize_fii_momentum_screen(
        fii_all,
        config=config,
        fii_top_n=fii_top_n,
        final_n=final_n,
        price_progress_callback=price_progress_callback,
        output_dir=output_dir,
    )


def run_dii_momentum_screen(
    csv_path: str,
    config: ScreeningConfig,
    dii_top_n: int = 50,
    final_n: int = 3,
    progress_callback: ProgressCallback | None = None,
    price_progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    paths = output_paths(output_dir)
    universe = load_ticker_universe(csv_path)
    dii_all = screen_dii_holdings(
        universe,
        progress_callback=progress_callback,
        checkpoint_path=paths["dii_partial"],
    )
    needs_market_cap = "Market Cap Cr" not in dii_all.columns or pd.to_numeric(
        dii_all.get("Market Cap Cr", pd.Series(dtype=float)),
        errors="coerce",
    ).isna().all()
    if needs_market_cap:
        dii_all = enrich_market_cap_from_yfinance(
            dii_all,
            progress_callback=price_progress_callback,
            checkpoint_path=paths["dii_marketcap_partial"],
        )
    if "Market Cap Cr" in dii_all.columns:
        dii_all["Market Cap Cr"] = pd.to_numeric(dii_all["Market Cap Cr"], errors="coerce")
        dii_all = dii_all.sort_values("Market Cap Cr", ascending=False, na_position="last").reset_index(drop=True)
    return finalize_dii_momentum_screen(
        dii_all,
        config=config,
        dii_top_n=dii_top_n,
        final_n=final_n,
        price_progress_callback=price_progress_callback,
        output_dir=output_dir,
    )


def run_quarterly_results_screen(
    csv_path: str,
    target_period: str,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    """Scan the full ticker universe for one quarterly result period."""
    normalized_target = normalize_quarter_period(target_period)
    if normalized_target is None:
        raise ValueError("Result quarter must use a format such as 'Jun 2026'.")
    paths = output_paths(output_dir)
    universe = load_ticker_universe(csv_path)
    all_results = screen_quarterly_results(
        universe,
        normalized_target,
        progress_callback=progress_callback,
        checkpoint_path=paths["quarterly_results_partial"],
    )
    return finalize_quarterly_results_screen(all_results, normalized_target, output_dir=output_dir)


def finalize_quarterly_results_screen(
    all_results: pd.DataFrame,
    target_period: str,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    """Persist a full quarterly scan plus its target-quarter subset."""
    normalized_target = normalize_quarter_period(target_period)
    if normalized_target is None:
        raise ValueError("Result quarter must use a format such as 'Jun 2026'.")
    paths = output_paths(output_dir)
    all_prepared = prepare_quarterly_results(all_results, target_period=normalized_target, matching_only=False)
    matching = prepare_quarterly_results(all_results, target_period=normalized_target, matching_only=True)
    save_frame(all_prepared, paths["quarterly_results_all"])
    save_frame(matching, paths["quarterly_results_matching"])
    return {"quarterly_all": all_prepared, "quarterly_matching": matching}


def prepare_quarterly_results(
    frame: pd.DataFrame,
    target_period: str | None = None,
    ranking_metric: str = "Sales",
    matching_only: bool = True,
) -> pd.DataFrame:
    """Filter a saved quarterly scan to one result period and sort by YoY growth."""
    result = frame.copy()
    if result.empty:
        return result
    if "Target Quarter" in result.columns and target_period:
        normalized_target = normalize_quarter_period(target_period)
        result = result[result["Target Quarter"].map(normalize_quarter_period).eq(normalized_target)].copy()
    if "Target Quarter Found" in result.columns and matching_only:
        found = result["Target Quarter Found"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        result = result[found].copy()

    ranking_column = f"{ranking_metric} YoY Growth %"
    if ranking_column in result.columns:
        result[ranking_column] = pd.to_numeric(result[ranking_column], errors="coerce")
        result = result.sort_values(ranking_column, ascending=False, na_position="last")
    return result.reset_index(drop=True)


def run_quarterly_stock_return_momentum(
    quarterly_all: pd.DataFrame,
    target_period: str,
    weights: dict[str, float] | None = None,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
    price_batch_size: int = 80,
) -> pd.DataFrame:
    """Score fresh stock-return momentum for companies reporting the selected quarter."""
    candidates = prepare_quarterly_results(
        quarterly_all,
        target_period=target_period,
        matching_only=True,
    )
    if candidates.empty:
        return pd.DataFrame()
    if "YFinance Ticker" not in candidates.columns:
        raise ValueError("Quarterly scan is missing YFinance Ticker values. Run the quarterly scan again.")

    paths = output_paths(output_dir)
    prices = download_adjusted_close(
        candidates["YFinance Ticker"].dropna().astype(str).tolist(),
        batch_size=price_batch_size,
        period="3mo",
        progress_callback=progress_callback,
    )
    returns = calculate_returns(
        candidates,
        prices,
        return_periods=POST_EARNINGS_STOCK_RETURN_PERIODS,
        progress_callback=progress_callback,
    )
    momentum = score_quarterly_stock_return_momentum(returns, weights=weights)
    save_frame(returns, paths["quarterly_stock_return_returns"])
    save_frame(momentum, paths["quarterly_stock_return_momentum"])
    return momentum


def score_quarterly_stock_return_momentum(
    returns: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank fresh post-earnings stock returns without a positive-return exclusion gate."""
    active_weights = weights or DEFAULT_POST_EARNINGS_STOCK_RETURN_WEIGHTS
    momentum = score_momentum(returns, weights=active_weights, positive_filters=())
    momentum = momentum.rename(columns={"Momentum Score": "Post-Earnings Stock Return Momentum Score"})
    momentum.insert(0, "Post-Earnings Stock Return Momentum Rank", range(1, len(momentum) + 1))
    return momentum


def prepare_fii_all(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort the full FII scan for display/export."""
    return prepare_institutional_all(frame)


def prepare_dii_all(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort the full DII scan for display/export."""
    return prepare_institutional_all(frame)


def prepare_institutional_all(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and sort the full institutional scan for display/export."""
    result = frame.copy()
    if "Market Cap Cr" in result.columns:
        result["Market Cap Cr"] = pd.to_numeric(result["Market Cap Cr"], errors="coerce")
        result = result.sort_values("Market Cap Cr", ascending=False, na_position="last")
    return result.reset_index(drop=True)


def finalize_fii_momentum_screen(
    fii_all: pd.DataFrame,
    config: ScreeningConfig,
    fii_top_n: int = 50,
    final_n: int = 3,
    price_progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    """Save FII scan outputs and momentum-score the positive FII shortlist."""
    return finalize_institutional_momentum_screen(
        all_scan=fii_all,
        holder_prefix="FII",
        config=config,
        top_n=fii_top_n,
        final_n=final_n,
        price_progress_callback=price_progress_callback,
        output_dir=output_dir,
    )


def finalize_dii_momentum_screen(
    dii_all: pd.DataFrame,
    config: ScreeningConfig,
    dii_top_n: int = 50,
    final_n: int = 3,
    price_progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    """Save DII scan outputs and momentum-score the positive DII shortlist."""
    return finalize_institutional_momentum_screen(
        all_scan=dii_all,
        holder_prefix="DII",
        config=config,
        top_n=dii_top_n,
        final_n=final_n,
        price_progress_callback=price_progress_callback,
        output_dir=output_dir,
    )


def finalize_institutional_momentum_screen(
    all_scan: pd.DataFrame,
    holder_prefix: str,
    config: ScreeningConfig,
    top_n: int = 50,
    final_n: int = 3,
    price_progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    """Save institutional scan outputs and momentum-score the positive accumulation shortlist."""
    paths = output_paths(output_dir)
    prefix = holder_prefix.lower()
    label = holder_prefix.upper()
    all_key = f"{prefix}_all"
    top_key = f"{prefix}_top"
    momentum_key = f"{prefix}_momentum"
    final_key = f"{prefix}_final"
    change_column = f"{label} Holding Change %"

    all_scan = prepare_institutional_all(all_scan)
    save_frame(all_scan, paths[all_key])

    ranked = all_scan.copy()
    if change_column not in ranked.columns:
        ranked[change_column] = pd.NA
    ranked[change_column] = pd.to_numeric(ranked[change_column], errors="coerce")
    top = (
        ranked[ranked[change_column].gt(0)]
        .sort_values(change_column, ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )
    save_frame(top, paths[top_key])

    if top.empty:
        momentum = pd.DataFrame()
        final = pd.DataFrame()
    else:
        try:
            prices = download_adjusted_close(
                top["YFinance Ticker"].astype(str).tolist(),
                batch_size=config.price_batch_size,
                progress_callback=price_progress_callback,
            )
            returns = calculate_returns(top, prices, progress_callback=price_progress_callback)
            momentum = score_momentum(
                returns,
                weights=config.momentum_weights,
                positive_filters=config.positive_return_filters,
            )
            final = momentum.head(int(final_n)).reset_index(drop=True)
        except Exception as exc:
            momentum = pd.DataFrame(
                [{"Momentum Error": f"Yahoo Finance price scoring failed after {label} scan: {exc}"}]
            )
            final = pd.DataFrame()

    save_frame(momentum, paths[momentum_key])
    save_frame(final, paths[final_key])
    return {
        all_key: all_scan,
        top_key: top,
        momentum_key: momentum,
        final_key: final,
    }


def enrich_market_cap_from_yfinance(
    frame: pd.DataFrame,
    progress_callback: ProgressCallback | None = None,
    checkpoint_path: str | Path | None = None,
    batch_size: int = 40,
) -> pd.DataFrame:
    """Add Yahoo Finance market-cap columns to a ticker frame."""
    if frame.empty or "YFinance Ticker" not in frame.columns:
        return frame

    result = frame.copy()
    if "Market Cap" not in result.columns:
        result["Market Cap"] = pd.NA
    if "Market Cap Cr" not in result.columns:
        result["Market Cap Cr"] = pd.NA

    ticker_series = result["YFinance Ticker"].dropna().astype(str).str.upper()
    existing = pd.to_numeric(result["Market Cap"], errors="coerce")
    missing_mask = existing.isna()
    tickers = result.loc[missing_mask, "YFinance Ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    market_caps: dict[str, float | None] = {}
    total = len(tickers)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, total, batch_size):
        batch = tickers[start : start + batch_size]
        if progress_callback:
            progress_callback(start, total, f"Fetching market cap for {batch[0]} to {batch[-1]}")
        tickers_obj = yf.Tickers(" ".join(batch))
        for ticker in batch:
            try:
                info = tickers_obj.tickers[ticker].fast_info
                market_cap = getattr(info, "market_cap", None)
                if market_cap is None:
                    market_cap = info.get("market_cap") if hasattr(info, "get") else None
            except Exception:
                market_cap = None
            market_caps[ticker] = float(market_cap) if market_cap else None

        completed = min(start + len(batch), total)
        result["Market Cap"] = result["YFinance Ticker"].astype(str).str.upper().map(market_caps).combine_first(
            pd.to_numeric(result["Market Cap"], errors="coerce")
        )
        result["Market Cap Cr"] = (pd.to_numeric(result["Market Cap"], errors="coerce") / 10_000_000).round(2)
        if checkpoint:
            result.sort_values("Market Cap Cr", ascending=False, na_position="last").to_csv(checkpoint, index=False)
        if progress_callback:
            progress_callback(completed, total, f"Fetched market cap for {completed:,} of {total:,} tickers")

    if total == 0 and progress_callback:
        progress_callback(0, 0, "Market cap already present")
    result["Market Cap"] = ticker_series.map(market_caps).combine_first(pd.to_numeric(result["Market Cap"], errors="coerce"))
    result["Market Cap Cr"] = (pd.to_numeric(result["Market Cap"], errors="coerce") / 10_000_000).round(2)
    return result


def run_fundamentals_screen(
    momentum: pd.DataFrame,
    config: ScreeningConfig,
    include_fundamentals: bool = True,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    paths = output_paths(output_dir)
    candidates = momentum.head(config.top_momentum_for_fundamentals).copy()

    if include_fundamentals:
        fundamentals = screen_fundamentals(
            candidates,
            config.fundamental_thresholds,
            progress_callback=progress_callback,
            checkpoint_path=paths["fundamentals_partial"],
        )
        final = fundamentals[fundamentals["Fundamental Pass"]].copy()
        save_frame(fundamentals, paths["fundamentals"])
    else:
        fundamentals = pd.DataFrame()
        final = candidates.copy()

    final = final.sort_values("Momentum Score", ascending=False).head(config.final_count).reset_index(drop=True)
    save_frame(final, paths["final"])

    if include_fundamentals and not fundamentals.empty:
        backtest_universe = fundamentals[fundamentals["Fundamental Pass"]].copy()
    else:
        backtest_universe = final.copy()

    curves, normalized, periods, allocation = walk_forward_backtest(
        backtest_universe,
        months=config.backtest_months,
        initial_capital=100000.0,
        weights=config.momentum_weights,
        positive_filters=config.positive_return_filters,
    )
    if allocation.empty:
        allocation = current_allocation(final, capital=100000.0)
    performance = performance_summary(curves)
    if not curves.empty:
        save_frame(curves, paths["backtest"], include_index=True)
        save_frame(curves, paths["walk_forward_backtest"], include_index=True)
    if not normalized.empty:
        save_frame(normalized, paths["normalized_backtest"], include_index=True)
    if not periods.empty:
        save_frame(periods, paths["walk_forward_periods"])
    if not allocation.empty:
        save_frame(allocation, paths["holdings"])
        save_frame(allocation, paths["current_allocation"])
    if not performance.empty:
        save_frame(performance, paths["performance"])
    return {
        "momentum": momentum,
        "fundamentals": fundamentals,
        "final": final,
        "backtest": curves,
        "normalized_backtest": normalized,
        "periods": periods,
        "holdings": allocation,
        "performance": performance,
    }


def run_full_screen(
    csv_path: str,
    config: ScreeningConfig,
    include_fundamentals: bool = True,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
    use_saved_returns: bool = False,
) -> dict[str, pd.DataFrame]:
    momentum = run_momentum(
        csv_path,
        config,
        progress_callback=progress_callback,
        output_dir=output_dir,
        use_saved_returns=use_saved_returns,
    )
    return run_fundamentals_screen(
        momentum,
        config,
        include_fundamentals=include_fundamentals,
        progress_callback=progress_callback,
        output_dir=output_dir,
    )
