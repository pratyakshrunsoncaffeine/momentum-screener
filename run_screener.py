from __future__ import annotations

import argparse
from pathlib import Path

from screener_momentum.config import FundamentalThresholds, ScreeningConfig
from screener_momentum.pipeline import run_full_screen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NSE momentum and fundamentals screener.")
    parser.add_argument("--csv", default="ticker.csv", help="Path to ticker CSV with a Ticker column.")
    parser.add_argument("--top", type=int, default=100, help="Momentum names to pass into Screener.in filters.")
    parser.add_argument("--final", type=int, default=100, help="Maximum final rows to export.")
    parser.add_argument("--no-fundamentals", action="store_true", help="Skip Screener.in fundamentals scraping.")
    parser.add_argument("--backtest-months", type=int, default=6, help="Backtest lookback in months.")
    parser.add_argument("--out", default="output", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ScreeningConfig(
        fundamental_thresholds=FundamentalThresholds(),
        top_momentum_for_fundamentals=args.top,
        final_count=args.final,
        backtest_months=args.backtest_months,
    )
    results = run_full_screen(
        args.csv,
        config,
        include_fundamentals=not args.no_fundamentals,
        output_dir=output_dir,
    )

    for name, frame in results.items():
        if frame is not None and not frame.empty:
            frame.to_csv(output_dir / f"{name}.csv", index=name in {"backtest", "normalized_backtest"})

    print(f"Momentum rows: {len(results['momentum'])}")
    print(f"Final rows: {len(results['final'])}")
    print(f"Saved CSV files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
