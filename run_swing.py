from __future__ import annotations

import argparse
import json
from pathlib import Path

from screener_momentum.swing import SwingConfig, run_swing_scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evening long-only swing scan on the full ticker list.")
    parser.add_argument("--csv", default="ticker.csv")
    parser.add_argument("--out", default="output/swing/latest")
    parser.add_argument("--config", help="JSON overrides for SwingConfig; see docs/swing_rules.md.")
    parser.add_argument("--events", help="Optional known-events CSV: Ticker,Date,Event.")
    parser.add_argument("--refresh", action="store_true", help="Fetch daily data again instead of reusing same-session history.")
    args = parser.parse_args()
    overrides = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    config = SwingConfig(**overrides)
    last_stage = [""]

    def progress(completed: int, total: int, message: str) -> None:
        stage = message.split(" ")[0]
        if stage != last_stage[0] or completed == total or "Daily scan" in message:
            print(message, flush=True)
            last_stage[0] = stage

    result = run_swing_scan(args.csv, config, args.out, resume=not args.refresh, events_csv=args.events, callback=progress)
    m = result["metadata"]
    print(f"Session {m['session']} | {m['universe_count']:,} tickers attempted | {m['usable_count']:,} usable | {m['technical_pass_count']} technical setups | {m['final_count']} final picks")
    print(f"NIFTY regime: {m['market_regime']}. Scores measure setup strength, not probability.")
    columns = ["Swing Rank", "Ticker", "Setup", "Setup Score", "Entry Trigger", "Max Entry", "Initial Stop", "Target", "Session VWAP Status"]
    print(result["picks"][columns].round(2).to_string(index=False) if not result["picks"].empty else "No qualifying picks. Filters were not relaxed.")
    print(f"Saved results: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
