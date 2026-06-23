from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .backtest import backtest_top10, performance_summary
from .config import ScreeningConfig
from .fundamentals import screen_fundamentals
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
        "holdings": root / "holdings.csv",
        "performance": root / "performance.csv",
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

    curves, holdings = backtest_top10(final, months=config.backtest_months)
    performance = performance_summary(curves)
    if not curves.empty:
        save_frame(curves, paths["backtest"], include_index=True)
    if not holdings.empty:
        save_frame(holdings, paths["holdings"])
    if not performance.empty:
        save_frame(performance, paths["performance"])
    return {
        "momentum": momentum,
        "fundamentals": fundamentals,
        "final": final,
        "backtest": curves,
        "holdings": holdings,
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
