from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from datetime import date, timedelta
import shutil

import pandas as pd
import yfinance as yf

from .backtest import current_allocation, performance_summary, walk_forward_backtest
from .correlation import (
    MacroFactorProvider,
    NseHistoricalEquityProvider,
    add_ridge_regression,
    calculate_correlations,
    select_correlation_leaders,
    select_relationship_leaders,
)
from .config import (
    DEFAULT_POST_EARNINGS_STOCK_RETURN_WEIGHTS,
    POST_EARNINGS_STOCK_RETURN_PERIODS,
    DerivativesSignalConfig,
    FundamentalThresholds,
    ScreeningConfig,
    Sma200ScanConfig,
    RETURN_PERIODS,
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
from .index_momentum import (
    DEFAULT_INDEX_MOMENTUM_WEIGHTS,
    INDEX_MOMENTUM_PERIODS,
    NseSectorIndexProvider,
    calculate_index_momentum,
)
from .sector_rotation import BENCHMARK_INDEX, calculate_sector_rotation
from .sma200 import (
    calculate_sma200_snapshot,
    combine_price_history,
    download_latest_quotes,
    insufficient_history_tickers,
    preliminary_quote_tickers,
    walk_forward_sma200_backtest,
)
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
        "index_momentum_cache": root.parent / "index_momentum_cache",
        "index_momentum_prices": root / "index_momentum_prices.csv",
        "index_momentum_ranking": root / "index_momentum_ranking.csv",
        "index_momentum_health": root / "index_momentum_health.csv",
        "index_momentum_metadata": root / "index_momentum_metadata.csv",
        "custom_stock_universe": root / "custom_stock_universe.csv",
        "custom_stock_returns": root / "custom_stock_returns.csv",
        "custom_stock_momentum": root / "custom_stock_momentum.csv",
        "custom_stock_health": root / "custom_stock_health.csv",
        "correlation_cache": root.parent / "correlation_cache",
        "correlation_stock_prices": root / "correlation_stock_prices.csv",
        "correlation_factors": root / "correlation_factors.csv",
        "correlation_all": root / "correlation_all.csv",
        "correlation_top_positive": root / "correlation_top_positive.csv",
        "correlation_top_negative": root / "correlation_top_negative.csv",
        "correlation_health": root / "correlation_health.csv",
        "correlation_stock_health": root / "correlation_stock_health.csv",
        "correlation_ridge_top_positive": root / "correlation_ridge_top_positive.csv",
        "correlation_ridge_top_negative": root / "correlation_ridge_top_negative.csv",
        "correlation_run_metadata": root / "correlation_run_metadata.csv",
        "sma200_cache": root.parent / "sma200_cache",
        "sma200_prices": root / "sma200_prices.csv",
        "sma200_universe": root / "sma200_universe.csv",
        "sma200_candidates": root / "sma200_candidates.csv",
        "sma200_rejected": root / "sma200_rejected.csv",
        "sma200_fundamentals_partial": root / "sma200_fundamentals_partial.csv",
        "sma200_fundamentals": root / "sma200_fundamentals.csv",
        "sma200_final": root / "sma200_final.csv",
        "sma200_health": root / "sma200_health.csv",
        "sma200_metadata": root / "sma200_metadata.csv",
        "sma200_backtest_curve": root / "sma200_backtest_curve.csv",
        "sma200_normalized_backtest": root / "sma200_normalized_backtest.csv",
        "sma200_backtest_periods": root / "sma200_backtest_periods.csv",
        "sma200_current_allocation": root / "sma200_current_allocation.csv",
        "sma200_backtest_summary": root / "sma200_backtest_summary.csv",
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


def run_sma200_screen(
    ticker_csv: str | Path,
    scan_config: Sma200ScanConfig,
    fundamental_thresholds: FundamentalThresholds,
    price_progress_callback: ProgressCallback | None = None,
    calculation_progress_callback: ProgressCallback | None = None,
    quote_progress_callback: ProgressCallback | None = None,
    fundamental_progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
    resume: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the complete positive-proximity 200DMA and fundamentals workflow."""
    paths = output_paths(output_dir)
    universe = load_ticker_universe(ticker_csv)
    tickers = universe["YFinance Ticker"].astype(str).str.upper().tolist()
    end_date = date.today()
    history_days = max(int(scan_config.backtest_months), 1) * 31 + 500
    start_date = end_date - timedelta(days=history_days)
    required_rows = int(scan_config.window_days) + int(scan_config.slope_lookback_days) + 1

    saved_prices = _read_sma200_prices(paths["sma200_prices"])
    saved_health = _read_saved_frame(paths["sma200_health"])
    reuse_prices = resume and _sma200_history_is_recent(saved_prices, start_date, end_date)
    if reuse_prices:
        prices = saved_prices
        source_map = _saved_sma200_source_map(saved_health)
        for ticker in prices.columns:
            source_map.setdefault(str(ticker).upper(), "Yahoo Finance")
        if price_progress_callback:
            price_progress_callback(len(tickers), len(tickers), "Reused recent saved 200DMA price history")
    else:
        yahoo_prices = download_adjusted_close(
            tickers,
            batch_size=int(scan_config.price_batch_size),
            start_date=start_date,
            end_date=end_date,
            progress_callback=price_progress_callback,
        )
        missing = insufficient_history_tickers(yahoo_prices, tickers, 1)
        if missing:
            if price_progress_callback:
                price_progress_callback(0, len(missing), f"Retrying {len(missing):,} incomplete Yahoo histories")
            retry_prices = download_adjusted_close(
                missing,
                batch_size=min(10, max(int(scan_config.price_batch_size), 1)),
                start_date=start_date,
                end_date=end_date,
                progress_callback=price_progress_callback,
            )
            yahoo_prices = _prefer_more_complete_prices(yahoo_prices, retry_prices)

        fallback_tickers = insufficient_history_tickers(yahoo_prices, tickers, 1)
        fallback_prices = pd.DataFrame()
        fallback_health = pd.DataFrame()
        if fallback_tickers:
            fallback_provider = NseHistoricalEquityProvider(paths["sma200_cache"] / "nse_equity")
            fallback_prices, fallback_health = fallback_provider.fetch_many(
                fallback_tickers,
                start_date=start_date,
                end_date=end_date,
                frequency="Daily",
                progress_callback=price_progress_callback,
            )
        prices, source_map = combine_price_history(
            yahoo_prices,
            fallback_prices,
            tickers,
            required_rows=required_rows,
        )
        _save_sma200_prices(prices, paths["sma200_prices"])
        saved_health = fallback_health

    initial_snapshot = calculate_sma200_snapshot(
        universe,
        prices,
        scan_config,
        price_sources=source_map,
        progress_callback=calculation_progress_callback,
    )

    quote_health = pd.DataFrame()
    latest_quotes = pd.DataFrame()
    if scan_config.price_mode == "Near-live shortlist":
        quote_tickers = preliminary_quote_tickers(initial_snapshot, scan_config)
        latest_quotes, quote_health = download_latest_quotes(
            quote_tickers,
            batch_size=min(50, max(int(scan_config.price_batch_size), 1)),
            progress_callback=quote_progress_callback,
        )
        snapshot = calculate_sma200_snapshot(
            universe,
            prices,
            scan_config,
            price_sources=source_map,
            latest_quotes=latest_quotes,
        )
    else:
        snapshot = initial_snapshot
        if quote_progress_callback:
            quote_progress_callback(1, 1, "Latest-close mode selected; quote refresh skipped")

    candidates = snapshot.loc[_boolean_mask(snapshot, "Proximity Pass")].copy()
    candidates = candidates.sort_values("Distance Above 200DMA %", ascending=True).reset_index(drop=True)
    rejected = snapshot.loc[~_boolean_mask(snapshot, "Proximity Pass")].copy().reset_index(drop=True)
    health = _build_sma200_health(snapshot, source_map)
    if not saved_health.empty:
        fallback_rows = saved_health.copy()
        fallback_rows["Stage"] = fallback_rows.get("Stage", "NSE fallback")
        health = pd.concat([health, fallback_rows], ignore_index=True, sort=False)
    if not quote_health.empty:
        health = pd.concat([health, quote_health], ignore_index=True, sort=False)

    save_frame(snapshot, paths["sma200_universe"])
    save_frame(candidates, paths["sma200_candidates"])
    save_frame(rejected, paths["sma200_rejected"])
    save_frame(health, paths["sma200_health"])

    fundamentals = screen_fundamentals(
        candidates,
        fundamental_thresholds,
        progress_callback=fundamental_progress_callback,
        checkpoint_path=paths["sma200_fundamentals_partial"],
        resume=resume,
    )
    pass_mask = _boolean_mask(fundamentals, "Fundamental Pass")
    final = fundamentals.loc[pass_mask].copy()
    if not final.empty:
        final = final.sort_values("Distance Above 200DMA %", ascending=True).reset_index(drop=True)
    save_frame(fundamentals, paths["sma200_fundamentals"])
    save_frame(final, paths["sma200_final"])

    curves, normalized, periods, allocation = walk_forward_sma200_backtest(
        final,
        prices,
        scan_config,
        initial_capital=100000.0,
    )
    summary = performance_summary(curves)
    if not curves.empty:
        curves.index.name = "Date"
        save_frame(curves, paths["sma200_backtest_curve"], include_index=True)
    if not normalized.empty:
        normalized.index.name = "Date"
        save_frame(normalized, paths["sma200_normalized_backtest"], include_index=True)
    save_frame(periods, paths["sma200_backtest_periods"])
    save_frame(allocation, paths["sma200_current_allocation"])
    save_frame(summary, paths["sma200_backtest_summary"])

    metadata = pd.DataFrame(
        [
            {
                "Saved At UTC": pd.Timestamp.now(tz="UTC").isoformat(),
                "Price Start": prices.index.min().date().isoformat() if not prices.empty else "",
                "Price End": prices.index.max().date().isoformat() if not prices.empty else "",
                "Price Mode": scan_config.price_mode,
                "SMA Window": int(scan_config.window_days),
                "Minimum Distance %": float(scan_config.min_distance_pct),
                "Maximum Distance %": float(scan_config.max_distance_pct),
                "Backtest Months": int(scan_config.backtest_months),
                "Universe Rows": int(len(snapshot)),
                "Proximity Candidates": int(len(candidates)),
                "Fundamental Pass": int(len(final)),
            }
        ]
    )
    save_frame(metadata, paths["sma200_metadata"])
    return {
        "prices": prices,
        "universe": snapshot,
        "candidates": candidates,
        "rejected": rejected,
        "fundamentals": fundamentals,
        "final": final,
        "health": health,
        "backtest": curves,
        "normalized_backtest": normalized,
        "periods": periods,
        "allocation": allocation,
        "performance": summary,
        "metadata": metadata,
    }


def load_saved_sma200_results(output_dir: str | Path = "output/latest") -> dict[str, pd.DataFrame]:
    paths = output_paths(output_dir)
    final = _read_saved_frame(paths["sma200_final"])
    universe = _read_saved_frame(paths["sma200_universe"])
    if universe.empty and final.empty:
        raise FileNotFoundError("No saved 200DMA scan is available yet.")
    return {
        "prices": _read_sma200_prices(paths["sma200_prices"]),
        "universe": universe,
        "candidates": _read_saved_frame(paths["sma200_candidates"]),
        "rejected": _read_saved_frame(paths["sma200_rejected"]),
        "fundamentals": _read_saved_frame(paths["sma200_fundamentals"]),
        "final": final,
        "health": _read_saved_frame(paths["sma200_health"]),
        "backtest": _read_indexed_frame(paths["sma200_backtest_curve"]),
        "normalized_backtest": _read_indexed_frame(paths["sma200_normalized_backtest"]),
        "periods": _read_saved_frame(paths["sma200_backtest_periods"]),
        "allocation": _read_saved_frame(paths["sma200_current_allocation"]),
        "performance": _read_saved_frame(paths["sma200_backtest_summary"]),
        "metadata": _read_saved_frame(paths["sma200_metadata"]),
    }


def reset_sma200_scan(output_dir: str | Path = "output/latest") -> list[Path]:
    paths = output_paths(output_dir)
    keys = (
        "sma200_prices",
        "sma200_universe",
        "sma200_candidates",
        "sma200_rejected",
        "sma200_fundamentals_partial",
        "sma200_fundamentals",
        "sma200_final",
        "sma200_health",
        "sma200_metadata",
        "sma200_backtest_curve",
        "sma200_normalized_backtest",
        "sma200_backtest_periods",
        "sma200_current_allocation",
        "sma200_backtest_summary",
    )
    removed = _remove_output_files(paths, keys)
    cache = paths["sma200_cache"]
    if cache.exists():
        shutil.rmtree(cache)
        removed.append(cache)
    return removed


def _read_sma200_prices(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()]
    return frame.apply(pd.to_numeric, errors="coerce").sort_index()


def _save_sma200_prices(prices: pd.DataFrame, path: Path) -> None:
    output = prices.copy()
    output.index.name = "Date"
    save_frame(output, path, include_index=True)


def _read_indexed_frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()


def _sma200_history_is_recent(prices: pd.DataFrame, start_date: date, end_date: date) -> bool:
    if prices.empty:
        return False
    earliest = pd.Timestamp(prices.index.min()).date()
    latest = pd.Timestamp(prices.index.max()).date()
    return earliest <= start_date + timedelta(days=10) and latest >= end_date - timedelta(days=4)


def _prefer_more_complete_prices(primary: pd.DataFrame, retry: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return retry.copy()
    result = primary.copy()
    for column in retry.columns:
        retry_rows = pd.to_numeric(retry[column], errors="coerce").notna().sum()
        primary_rows = (
            pd.to_numeric(result[column], errors="coerce").notna().sum() if column in result.columns else 0
        )
        if retry_rows > primary_rows:
            result[column] = retry[column]
    return result.sort_index()


def _saved_sma200_source_map(health: pd.DataFrame) -> dict[str, str]:
    if health.empty or "YFinance Ticker" not in health or "Price Source" not in health:
        return {}
    rows = health.copy()
    if "Stage" in rows:
        rows = rows.loc[rows["Stage"].astype(str).eq("Daily history")]
    rows["_TickerKey"] = rows["YFinance Ticker"].astype(str).str.upper()
    return (
        rows.dropna(subset=["YFinance Ticker"])
        .drop_duplicates("_TickerKey", keep="last")
        .set_index("_TickerKey")["Price Source"]
        .astype(str)
        .to_dict()
    )


def _build_sma200_health(snapshot: pd.DataFrame, source_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in snapshot.to_dict("records"):
        ticker = str(item.get("YFinance Ticker", "")).upper()
        source = source_map.get(ticker, str(item.get("Price Source", "Unavailable")))
        rows.append(
            {
                "Ticker": item.get("Ticker", ticker.removesuffix(".NS")),
                "YFinance Ticker": ticker,
                "Stage": "Daily history",
                "Price Source": source,
                "Price Basis": (
                    "Corporate-action-adjusted close"
                    if source == "Yahoo Finance"
                    else "Raw official NSE close; not corporate-action adjusted"
                    if source == "NSE India"
                    else ""
                ),
                "Status": "available" if int(item.get("Data Points", 0) or 0) > 0 else "failed",
                "Rows": int(item.get("Data Points", 0) or 0),
                "Last Date": item.get("SMA Date", ""),
                "Message": item.get("Rejection Notes", "") if source == "Unavailable" else "",
            }
        )
    return pd.DataFrame(rows)


def _boolean_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == object:
        return values.astype(str).str.lower().isin(("true", "1", "yes"))
    return values.fillna(False).astype(bool)


def run_index_momentum_screen(
    indices: list[str] | tuple[str, ...],
    end_date: date,
    weights: dict[str, float] | None = None,
    history_months: int = 12,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, object]:
    """Download official NSE index history and rank recent weighted momentum."""
    selected = list(dict.fromkeys(str(item).strip() for item in indices if str(item).strip()))
    if not selected:
        raise ValueError("Select at least one NSE index.")
    score_weights = weights or DEFAULT_INDEX_MOMENTUM_WEIGHTS
    if sum(float(score_weights.get(label, 0.0)) for label in INDEX_MOMENTUM_PERIODS) <= 0:
        raise ValueError("Index momentum weights must add to more than zero.")
    months = max(int(history_months), 4)
    start_date = end_date - timedelta(days=months * 31)
    paths = output_paths(output_dir)
    provider = NseSectorIndexProvider(paths["index_momentum_cache"])
    prices, health = provider.fetch_many(
        selected,
        start_date=start_date,
        end_date=end_date,
        progress_callback=progress_callback,
    )
    ranking = calculate_index_momentum(prices, selected, score_weights)
    if ranking.empty:
        raise RuntimeError("Official NSE history did not produce any index momentum rows.")
    metadata = _index_momentum_metadata(score_weights, end_date, selected)
    save_frame(prices, paths["index_momentum_prices"])
    save_frame(ranking, paths["index_momentum_ranking"])
    save_frame(health, paths["index_momentum_health"])
    save_frame(metadata, paths["index_momentum_metadata"])
    return {"prices": prices, "ranking": ranking, "health": health, "metadata": metadata, "stale": False}


def rescore_saved_index_momentum(
    indices: list[str] | tuple[str, ...],
    weights: dict[str, float],
    output_dir: str | Path = "output/latest",
) -> dict[str, object]:
    """Apply new weights to saved official NSE prices without another download."""
    paths = output_paths(output_dir)
    prices = _read_saved_frame(paths["index_momentum_prices"])
    if prices.empty:
        raise FileNotFoundError("No saved NSE index price history is available yet.")
    prices["Date"] = pd.to_datetime(prices.get("Date"), errors="coerce")
    available = set(prices.get("Index", pd.Series(dtype=str)).dropna().astype(str))
    selected = [str(item) for item in indices if str(item) in available]
    if not selected:
        raise ValueError("None of the selected indices exist in the saved price run.")
    ranking = calculate_index_momentum(prices, selected, weights)
    metadata = _index_momentum_metadata(
        weights,
        pd.Timestamp(prices["Date"].max()).date(),
        selected,
    )
    save_frame(ranking, paths["index_momentum_ranking"])
    save_frame(metadata, paths["index_momentum_metadata"])
    return {
        "prices": prices,
        "ranking": ranking,
        "health": _read_saved_frame(paths["index_momentum_health"]),
        "metadata": metadata,
        "stale": True,
    }


def load_saved_index_momentum(output_dir: str | Path = "output/latest") -> dict[str, object]:
    paths = output_paths(output_dir)
    prices = _read_saved_frame(paths["index_momentum_prices"])
    ranking = _read_saved_frame(paths["index_momentum_ranking"])
    if prices.empty or ranking.empty:
        raise FileNotFoundError("No saved NSE index momentum run is available yet.")
    prices["Date"] = pd.to_datetime(prices.get("Date"), errors="coerce")
    return {
        "prices": prices,
        "ranking": ranking,
        "health": _read_saved_frame(paths["index_momentum_health"]),
        "metadata": _read_saved_frame(paths["index_momentum_metadata"]),
        "stale": True,
    }


def reset_index_momentum_scan(output_dir: str | Path = "output/latest") -> list[Path]:
    paths = output_paths(output_dir)
    removed = _remove_output_files(
        paths,
        (
            "index_momentum_prices",
            "index_momentum_ranking",
            "index_momentum_health",
            "index_momentum_metadata",
        ),
    )
    cache = paths["index_momentum_cache"]
    if cache.exists():
        shutil.rmtree(cache)
        removed.append(cache)
    return removed


def run_custom_stock_momentum(
    ticker_csv: str | Path,
    config: ScreeningConfig,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, pd.DataFrame]:
    """Score a user-uploaded ticker list without changing the main screener outputs."""
    paths = output_paths(output_dir)
    universe = load_ticker_universe(ticker_csv)
    if universe.empty:
        raise ValueError("The uploaded CSV contains no usable ticker symbols.")
    periods = {
        label: RETURN_PERIODS[label]
        for label in dict.fromkeys([*config.momentum_weights, *config.positive_return_filters])
        if label in RETURN_PERIODS
    }
    max_days = max(periods.values(), default=126)
    prices = download_adjusted_close(
        universe["YFinance Ticker"].astype(str).tolist(),
        batch_size=int(config.price_batch_size),
        period=f"{max(9, int(max_days / 21) + 4)}mo",
        progress_callback=progress_callback,
    )
    returns = calculate_returns(
        universe,
        prices,
        return_periods=periods,
        progress_callback=progress_callback,
    )
    ranking = score_momentum(
        returns,
        config.momentum_weights,
        config.positive_return_filters,
        include_failed=True,
    )
    health = returns[
        [column for column in ("Ticker", "YFinance Ticker", "Data Points", "Price Error") if column in returns]
    ].copy()
    if not health.empty:
        health["Status"] = health["Price Error"].fillna("").astype(str).map(
            lambda value: "available" if not value else "failed"
        )
    save_frame(universe, paths["custom_stock_universe"])
    save_frame(returns, paths["custom_stock_returns"])
    save_frame(ranking, paths["custom_stock_momentum"])
    save_frame(health, paths["custom_stock_health"])
    return {"universe": universe, "returns": returns, "ranking": ranking, "health": health}


def load_saved_custom_stock_momentum(output_dir: str | Path = "output/latest") -> dict[str, pd.DataFrame]:
    paths = output_paths(output_dir)
    ranking = _read_saved_frame(paths["custom_stock_momentum"])
    if ranking.empty:
        raise FileNotFoundError("No saved uploaded-stock momentum run is available yet.")
    return {
        "universe": _read_saved_frame(paths["custom_stock_universe"]),
        "returns": _read_saved_frame(paths["custom_stock_returns"]),
        "ranking": ranking,
        "health": _read_saved_frame(paths["custom_stock_health"]),
    }


def _index_momentum_metadata(
    weights: dict[str, float],
    data_end: date,
    indices: list[str],
) -> pd.DataFrame:
    row: dict[str, object] = {
        "Saved At UTC": pd.Timestamp.now(tz="UTC").isoformat(),
        "Data End": data_end.isoformat(),
        "Index Count": len(indices),
    }
    row.update({f"Weight - {label}": float(weights.get(label, 0.0)) for label in INDEX_MOMENTUM_PERIODS})
    return pd.DataFrame([row])


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


def run_correlation_screen(
    ticker_csv: str | Path,
    factors: list[str] | tuple[str, ...],
    end_date: date,
    lookback_years: int = 3,
    frequency: str = "Weekly",
    method: str = "Pearson",
    relation: str = "Same period",
    min_observations: int = 40,
    top_n: int = 10,
    ridge_alpha: float = 1.0,
    progress_callback: ProgressCallback | None = None,
    output_dir: str | Path = "output/latest",
) -> dict[str, object]:
    """Download as-of stock/factor history, calculate relationships, and save recovery files."""
    if not factors:
        raise ValueError("Select at least one macro factor.")
    years = max(int(lookback_years), 1)
    start_date = end_date - timedelta(days=years * 366 + 45)
    paths = output_paths(output_dir)
    universe = load_ticker_universe(ticker_csv)
    tickers = universe["YFinance Ticker"].astype(str).tolist()

    prices = download_adjusted_close(
        tickers,
        start_date=start_date,
        end_date=end_date,
        progress_callback=progress_callback,
    )
    yahoo_available = set(prices.dropna(axis=1, how="all").columns) if not prices.empty else set()
    yahoo_missing = [ticker for ticker in tickers if ticker not in yahoo_available]
    if yahoo_missing:
        retry_prices = download_adjusted_close(
            yahoo_missing,
            batch_size=10,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback,
        )
        if not retry_prices.empty:
            prices = pd.concat([prices, retry_prices], axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
            yahoo_available.update(retry_prices.dropna(axis=1, how="all").columns)

    nse_missing = [ticker for ticker in tickers if ticker not in yahoo_available]
    nse_health = pd.DataFrame()
    nse_available: set[str] = set()
    if nse_missing:
        nse_provider = NseHistoricalEquityProvider(paths["correlation_cache"] / "nse_equity")
        nse_prices, nse_health = nse_provider.fetch_many(
            nse_missing,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            progress_callback=progress_callback,
        )
        if not nse_prices.empty:
            prices = pd.concat([prices, nse_prices], axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated(keep="last")]
            nse_available.update(nse_prices.dropna(axis=1, how="all").columns)

    if prices.empty:
        raise RuntimeError("Neither Yahoo Finance nor the official NSE fallback returned stock price history.")
    prices = prices.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)].copy()
    save_frame(prices.rename_axis("Date").reset_index(), paths["correlation_stock_prices"])

    nse_health_lookup = (
        nse_health.set_index("YFinance Ticker").to_dict("index") if not nse_health.empty else {}
    )
    stock_health_rows: list[dict[str, object]] = []
    for ticker in tickers:
        if ticker in yahoo_available:
            series = prices[ticker].dropna() if ticker in prices else pd.Series(dtype=float)
            stock_health_rows.append(
                {
                    "Ticker": ticker.removesuffix(".NS"),
                    "YFinance Ticker": ticker,
                    "Price Source": "Yahoo Finance",
                    "Price Basis": "Auto-adjusted close",
                    "Status": "downloaded",
                    "Rows": int(series.shape[0]),
                    "First Date": series.index.min().date().isoformat() if not series.empty else "",
                    "Last Date": series.index.max().date().isoformat() if not series.empty else "",
                    "Message": "",
                }
            )
        elif ticker in nse_available:
            stock_health_rows.append(
                {
                    **nse_health_lookup.get(ticker, {}),
                    "YFinance Ticker": ticker,
                }
            )
        else:
            fallback_health = nse_health_lookup.get(
                ticker,
                {
                    "Ticker": ticker.removesuffix(".NS"),
                    "Price Source": "Unavailable",
                    "Price Basis": "",
                    "Status": "failed",
                    "Rows": 0,
                    "First Date": "",
                    "Last Date": "",
                    "Message": "No usable Yahoo Finance or official NSE equity-series history.",
                },
            )
            stock_health_rows.append({**fallback_health, "YFinance Ticker": ticker})
    stock_health = pd.DataFrame(stock_health_rows)
    save_frame(stock_health, paths["correlation_stock_health"])

    provider = MacroFactorProvider(paths["correlation_cache"])
    factor_levels, health = provider.fetch_many(
        list(factors),
        start_date=start_date,
        end_date=end_date,
        progress_callback=progress_callback,
    )
    if not factor_levels:
        raise RuntimeError("No selected macro factor could be downloaded or recovered from cache.")

    results, factor_history = calculate_correlations(
        universe,
        prices,
        factor_levels,
        requested_frequency=frequency,
        method=method,
        relation=relation,
        min_observations=min_observations,
        progress_callback=progress_callback,
    )
    results = add_ridge_regression(
        results,
        prices,
        factor_history,
        relation=relation,
        alpha=ridge_alpha,
        min_observations=min_observations,
    )
    if results.empty:
        raise RuntimeError(
            "No stock-factor pair met the minimum observation requirement. Reduce the minimum observations or increase the lookback."
        )
    positive, inverse = select_correlation_leaders(results, top_n=top_n)
    ridge_positive, ridge_inverse = select_relationship_leaders(
        results,
        top_n=top_n,
        ranking_metric="Ridge Coefficient",
    )
    health = health.copy()
    health["Requested Frequency"] = frequency
    health["Relation"] = relation
    health["Method"] = method
    health["Analysis End Date"] = end_date.isoformat()
    health["Ticker Universe"] = len(universe)
    health["Stocks With Prices"] = int(prices.notna().any(axis=0).sum())
    metadata = pd.DataFrame(
        [
            {
                "Saved At UTC": pd.Timestamp.now(tz="UTC").isoformat(),
                "Analysis End Date": end_date.isoformat(),
                "Lookback Years": years,
                "Requested Frequency": frequency,
                "Correlation Method": method,
                "Relationship": relation,
                "Minimum Observations": int(min_observations),
                "Top Results Per Direction": int(top_n),
                "Ridge Alpha": float(ridge_alpha),
                "Factors": " | ".join(factors),
                "Ticker Universe": len(universe),
                "Stocks With Prices": int(prices.notna().any(axis=0).sum()),
                "Stock-Factor Pairs": len(results),
            }
        ]
    )

    save_frame(factor_history, paths["correlation_factors"])
    save_frame(results, paths["correlation_all"])
    save_frame(positive, paths["correlation_top_positive"])
    save_frame(inverse, paths["correlation_top_negative"])
    save_frame(ridge_positive, paths["correlation_ridge_top_positive"])
    save_frame(ridge_inverse, paths["correlation_ridge_top_negative"])
    save_frame(health, paths["correlation_health"])
    save_frame(metadata, paths["correlation_run_metadata"])
    return {
        "results": results,
        "positive": positive,
        "inverse": inverse,
        "ridge_positive": ridge_positive,
        "ridge_inverse": ridge_inverse,
        "prices": prices.rename_axis("Date").reset_index(),
        "factors": factor_history,
        "health": health,
        "stock_health": stock_health,
        "metadata": metadata,
        "stale": False,
    }


def load_saved_correlation(
    output_dir: str | Path = "output/latest",
) -> dict[str, object]:
    """Recover the most recent completed correlation scan without another market-data request."""
    paths = output_paths(output_dir)
    results = _read_saved_frame(paths["correlation_all"])
    factor_history = _read_saved_frame(paths["correlation_factors"])
    health = _read_saved_frame(paths["correlation_health"])
    stock_health = _read_saved_frame(paths["correlation_stock_health"])
    prices = _read_saved_frame(paths["correlation_stock_prices"])
    metadata = _read_saved_frame(paths["correlation_run_metadata"])
    if results.empty:
        raise FileNotFoundError("No saved correlation scan is available yet.")
    positive = _read_saved_frame(paths["correlation_top_positive"])
    inverse = _read_saved_frame(paths["correlation_top_negative"])
    if positive.empty and inverse.empty:
        positive, inverse = select_correlation_leaders(results, top_n=10)
    if not factor_history.empty:
        factor_history["Date"] = pd.to_datetime(factor_history["Date"], errors="coerce")
    if not prices.empty:
        prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    relation = (
        str(metadata.iloc[0].get("Relationship", "Same period"))
        if not metadata.empty
        else str(results.get("Relation", pd.Series(["Same period"])).iloc[0])
    )
    ridge_alpha = float(metadata.iloc[0].get("Ridge Alpha", 1.0)) if not metadata.empty else 1.0
    min_observations = (
        int(metadata.iloc[0].get("Minimum Observations", 24))
        if not metadata.empty
        else int(pd.to_numeric(results.get("Observations"), errors="coerce").dropna().min())
    )
    if "Ridge Coefficient" not in results.columns and not prices.empty and not factor_history.empty:
        price_frame = prices.set_index("Date")
        results = add_ridge_regression(
            results,
            price_frame,
            factor_history,
            relation=relation,
            alpha=ridge_alpha,
            min_observations=min_observations,
        )
        save_frame(results, paths["correlation_all"])
    ridge_positive, ridge_inverse = select_relationship_leaders(
        results,
        top_n=10,
        ranking_metric="Ridge Coefficient",
    )
    save_frame(ridge_positive, paths["correlation_ridge_top_positive"])
    save_frame(ridge_inverse, paths["correlation_ridge_top_negative"])
    if metadata.empty:
        metadata = pd.DataFrame(
            [
                {
                    "Saved At UTC": pd.Timestamp.now(tz="UTC").isoformat(),
                    "Analysis End Date": results.get("Data End", pd.Series([""])).max(),
                    "Lookback Years": "",
                    "Requested Frequency": results.get("Effective Frequency", pd.Series([""])).mode().iloc[0],
                    "Correlation Method": results.get("Method", pd.Series([""])).iloc[0],
                    "Relationship": relation,
                    "Minimum Observations": min_observations,
                    "Top Results Per Direction": 10,
                    "Ridge Alpha": ridge_alpha,
                    "Factors": " | ".join(results["Factor"].dropna().astype(str).unique()),
                    "Ticker Universe": len(stock_health),
                    "Stocks With Prices": int(prices.drop(columns=["Date"], errors="ignore").notna().any().sum()),
                    "Stock-Factor Pairs": len(results),
                }
            ]
        )
        save_frame(metadata, paths["correlation_run_metadata"])
    return {
        "results": results,
        "positive": positive,
        "inverse": inverse,
        "ridge_positive": ridge_positive,
        "ridge_inverse": ridge_inverse,
        "prices": prices,
        "factors": factor_history,
        "health": health,
        "stock_health": stock_health,
        "metadata": metadata,
        "stale": True,
    }


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


def reset_derivatives_eod_data(
    start_date: date,
    end_date: date,
    output_dir: str | Path = "output/latest",
) -> dict[str, object]:
    """Clear a selected NSE EOD cache range and all derived derivatives outputs."""
    if start_date > end_date:
        raise ValueError("Derivatives data start date must be on or before the end date.")
    paths = output_paths(output_dir)
    provider = NseEodDerivativesProvider(paths["derivatives_cache"])
    removed_cache_directories = provider.reset_range(start_date, end_date)
    output_keys = (
        "derivatives_contracts",
        "derivatives_daily_features",
        "derivatives_signals",
        "derivatives_rejections",
        "derivatives_data_health",
        "derivatives_backtest_events",
        "derivatives_backtest_curve",
        "derivatives_backtest_summary",
        "derivatives_event_summary",
    )
    removed_outputs = _remove_output_files(paths, output_keys)
    return {
        "removed_cache_directories": removed_cache_directories,
        "removed_outputs": removed_outputs,
    }


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


def reset_fii_momentum_scan(output_dir: str | Path = "output/latest") -> list[Path]:
    """Clear only FII scan checkpoints and derived FII rankings."""
    paths = output_paths(output_dir)
    return _remove_output_files(
        paths,
        (
            "fii_partial",
            "fii_marketcap_partial",
            "fii_all",
            "fii_top",
            "fii_momentum",
            "fii_final",
        ),
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


def reset_dii_momentum_scan(output_dir: str | Path = "output/latest") -> list[Path]:
    """Clear only DII scan checkpoints and derived DII rankings."""
    paths = output_paths(output_dir)
    return _remove_output_files(
        paths,
        (
            "dii_partial",
            "dii_marketcap_partial",
            "dii_all",
            "dii_top",
            "dii_momentum",
            "dii_final",
        ),
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


def reset_quarterly_results_scan(
    target_period: str,
    output_dir: str | Path = "output/latest",
) -> list[Path]:
    """Clear only quarterly-result checkpoints so the next scan starts at ticker one."""
    normalized_target = normalize_quarter_period(target_period)
    if normalized_target is None:
        raise ValueError("Result quarter must use a format such as 'Jun 2026'.")
    paths = output_paths(output_dir)
    reset_keys = (
        "quarterly_results_partial",
        "quarterly_results_all",
        "quarterly_results_matching",
        "quarterly_stock_return_returns",
        "quarterly_stock_return_momentum",
    )
    return _remove_output_files(paths, reset_keys)


def _remove_output_files(paths: dict[str, Path], keys: tuple[str, ...]) -> list[Path]:
    removed: list[Path] = []
    for key in keys:
        path = paths[key]
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


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
