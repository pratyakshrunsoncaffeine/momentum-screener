from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import yfinance as yf

from .backtest import current_allocation, performance_summary, walk_forward_backtest
from .config import ScreeningConfig
from .fundamentals import screen_fii_holdings, screen_fundamentals
from .momentum import calculate_returns, download_adjusted_close, score_momentum
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
    }


def save_frame(frame: pd.DataFrame, path: Path, include_index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=include_index)


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
    fii_all = enrich_market_cap_from_yfinance(
        fii_all,
        progress_callback=price_progress_callback,
        checkpoint_path=paths["fii_marketcap_partial"],
    )
    if "Market Cap Cr" in fii_all.columns:
        fii_all = fii_all.sort_values("Market Cap Cr", ascending=False, na_position="last").reset_index(drop=True)
    save_frame(fii_all, paths["fii_all"])

    fii_ranked = fii_all.copy()
    fii_ranked["FII Holding Change %"] = pd.to_numeric(fii_ranked["FII Holding Change %"], errors="coerce")
    fii_top = (
        fii_ranked[fii_ranked["FII Holding Change %"].gt(0)]
        .sort_values("FII Holding Change %", ascending=False)
        .head(int(fii_top_n))
        .reset_index(drop=True)
    )
    save_frame(fii_top, paths["fii_top"])

    if fii_top.empty:
        fii_momentum = pd.DataFrame()
        fii_final = pd.DataFrame()
    else:
        prices = download_adjusted_close(
            fii_top["YFinance Ticker"].astype(str).tolist(),
            batch_size=config.price_batch_size,
            progress_callback=price_progress_callback,
        )
        returns = calculate_returns(fii_top, prices, progress_callback=price_progress_callback)
        fii_momentum = score_momentum(
            returns,
            weights=config.momentum_weights,
            positive_filters=config.positive_return_filters,
        )
        fii_final = fii_momentum.head(int(final_n)).reset_index(drop=True)

    save_frame(fii_momentum, paths["fii_momentum"])
    save_frame(fii_final, paths["fii_final"])
    return {
        "fii_all": fii_all,
        "fii_top": fii_top,
        "fii_momentum": fii_momentum,
        "fii_final": fii_final,
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
