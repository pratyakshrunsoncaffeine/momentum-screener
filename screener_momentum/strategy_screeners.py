from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .momentum import chunked


ProgressCallback = Callable[[int, int, str], None]
OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")


def _yf_ticker(value: object) -> str:
    ticker = str(value).strip().upper()
    return ticker if ticker.endswith(".NS") else f"{ticker}.NS"


def _ticker_cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = re.sub(r"[^A-Z0-9._-]", "_", ticker.upper())
    return cache_dir / f"{safe}.csv"


def _extract_field(data: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if field in data.columns.get_level_values(-1):
            result = data.xs(field, axis=1, level=-1)
        elif field in data.columns.get_level_values(0):
            result = data.xs(field, axis=1, level=0)
        else:
            return pd.DataFrame()
    elif len(tickers) == 1 and field in data.columns:
        result = data[[field]].rename(columns={field: tickers[0]})
    else:
        return pd.DataFrame()
    if isinstance(result, pd.Series):
        result = result.to_frame(name=tickers[0])
    result.columns = [str(column).strip().upper() for column in result.columns]
    result.index = pd.to_datetime(result.index, errors="coerce")
    return result.loc[result.index.notna()].sort_index()


def _download_batch(tickers: list[str], period: str, threads: bool) -> pd.DataFrame:
    try:
        return yf.download(
            tickers=tickers,
            period=period,
            auto_adjust=True,
            group_by="ticker",
            threads=threads,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()


def download_strategy_history(
    yahoo_tickers: list[str],
    cache_dir: str | Path,
    batch_size: int = 50,
    period: str = "2y",
    progress_callback: ProgressCallback | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted daily OHLCV and checkpoint each ticker for resumable scans."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tickers = list(dict.fromkeys(_yf_ticker(ticker) for ticker in yahoo_tickers))
    available: set[str] = set()
    if resume:
        for ticker in tickers:
            path = _ticker_cache_path(cache, ticker)
            if path.exists():
                try:
                    cached = pd.read_csv(path, parse_dates=["Date"])
                    if not cached.empty and {"Close", "High", "Low", "Volume"}.issubset(cached.columns):
                        available.add(ticker)
                except (OSError, ValueError, pd.errors.ParserError):
                    continue

    missing = [ticker for ticker in tickers if ticker not in available]
    completed = len(available)
    for batch in chunked(missing, max(1, int(batch_size))):
        if progress_callback:
            progress_callback(completed, len(tickers), f"Downloading OHLCV: {batch[0]} to {batch[-1]}")
        data = _download_batch(batch, period=period, threads=True)
        fields = {field: _extract_field(data, batch, field) for field in OHLCV_FIELDS}
        batch_saved: set[str] = set()
        for ticker in batch:
            parts = []
            for field, frame in fields.items():
                if ticker in frame.columns:
                    parts.append(frame[ticker].rename(field))
            if len(parts) < 4:
                continue
            history = pd.concat(parts, axis=1).dropna(subset=["Close", "High", "Low", "Volume"])
            if history.empty:
                continue
            history.index.name = "Date"
            history.reset_index().to_csv(_ticker_cache_path(cache, ticker), index=False)
            batch_saved.add(ticker)

        failed = [ticker for ticker in batch if ticker not in batch_saved]
        for ticker in failed:
            retry = _download_batch([ticker], period=period, threads=False)
            retry_fields = {field: _extract_field(retry, [ticker], field) for field in OHLCV_FIELDS}
            parts = [frame[ticker].rename(field) for field, frame in retry_fields.items() if ticker in frame]
            if len(parts) < 4:
                continue
            history = pd.concat(parts, axis=1).dropna(subset=["Close", "High", "Low", "Volume"])
            if not history.empty:
                history.index.name = "Date"
                history.reset_index().to_csv(_ticker_cache_path(cache, ticker), index=False)
                batch_saved.add(ticker)

        available.update(batch_saved)
        completed += len(batch)
        if progress_callback:
            progress_callback(completed, len(tickers), f"Checkpointed prices for {completed:,} of {len(tickers):,} tickers")
    return pd.DataFrame(
        [{"YFinance Ticker": ticker, "Price History Available": ticker in available} for ticker in tickers]
    )


def _read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, parse_dates=["Date"])
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()
    required = {"Date", "High", "Low", "Close", "Volume"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame.dropna(subset=list(required)).sort_values("Date").drop_duplicates("Date", keep="last")
    for column in OHLCV_FIELDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["High", "Low", "Close", "Volume"])


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "Price Date": "",
        "Price Observations": 0,
        "CMP Rs.": np.nan,
        "Average Volume 30D": np.nan,
        "Return 1M %": np.nan,
        "Return 3M %": np.nan,
        "Return 5D %": np.nan,
        "ADR 20D %": np.nan,
        "EMA 20": np.nan,
        "EMA 50": np.nan,
        "EMA 200": np.nan,
        "Distance From 60D High %": np.nan,
        "Down Days 5D": np.nan,
        "Dip 5D %": np.nan,
        "Strategy Score": np.nan,
        "Status": "Rejected",
        "Rejection Reasons": reason,
    }


def evaluate_strategy(
    universe: pd.DataFrame,
    cache_dir: str | Path,
    strategy: str,
    min_history: int = 201,
    min_average_volume: int = 200_000,
    insta_min_return_3m: float = 30.0,
    insta_min_adr_20d: float = 4.0,
    draw_min_return_3m: float = 10.0,
    draw_min_dip_5d: float = 2.0,
    draw_max_dip_5d: float = 10.0,
    draw_max_below_60d_high: float = 12.0,
) -> pd.DataFrame:
    """Evaluate one of the two daily-candle strategies and retain rejection reasons."""
    strategy_key = strategy.strip().casefold()
    if strategy_key not in {"test insta strategy", "m mom d draw"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    cache = Path(cache_dir)
    rows: list[dict[str, Any]] = []
    for item in universe.to_dict("records"):
        ticker = str(item.get("YFinance Ticker") or _yf_ticker(item.get("Ticker", ""))).upper()
        history = _read_history(_ticker_cache_path(cache, ticker))
        base = {**item, "YFinance Ticker": ticker}
        if len(history) < int(min_history):
            rows.append({**base, **_empty_result(f"Insufficient OHLCV history: {len(history)} of {min_history} sessions")})
            continue

        close = history["Close"].astype(float).reset_index(drop=True)
        high = history["High"].astype(float).reset_index(drop=True)
        low = history["Low"].astype(float).reset_index(drop=True)
        volume = history["Volume"].astype(float).reset_index(drop=True)
        current = float(close.iloc[-1])
        ret_1m = (current / float(close.iloc[-22]) - 1.0) * 100.0
        ret_3m = (current / float(close.iloc[-64]) - 1.0) * 100.0
        ret_5d = (current / float(close.iloc[-6]) - 1.0) * 100.0
        avg_volume = float(volume.tail(30).mean())
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        high60 = float(high.tail(60).max())
        distance_high = (current / high60 - 1.0) * 100.0 if high60 > 0 else np.nan
        down_days = int(close.diff().tail(5).lt(0).sum())
        dip_5d = -ret_5d
        adr20 = float((((high - low) / close.replace(0, np.nan)) * 100.0).tail(20).mean())
        reasons: list[str] = []
        volume_failed = (
            avg_volume <= min_average_volume
            if strategy_key == "test insta strategy"
            else avg_volume < min_average_volume
        )
        if not np.isfinite(avg_volume) or volume_failed:
            reasons.append(f"30-session average volume {avg_volume:,.0f} below {min_average_volume:,}" if np.isfinite(avg_volume) else "30-session average volume unavailable")

        if strategy_key == "test insta strategy":
            if current <= ema50:
                reasons.append("latest close is not above EMA50")
            if current <= ema200:
                reasons.append("latest close is not above EMA200")
            if not ret_3m > insta_min_return_3m:
                reasons.append(f"3-month return {ret_3m:.2f}% is not greater than {insta_min_return_3m:.2f}%")
            if adr20 < insta_min_adr_20d:
                reasons.append(f"20-session ADR {adr20:.2f}% is below {insta_min_adr_20d:.2f}%")
            score = ret_3m
        else:
            if ret_3m < draw_min_return_3m:
                reasons.append(f"3-month return {ret_3m:.2f}% is below {draw_min_return_3m:.2f}%")
            if ret_1m <= 0:
                reasons.append(f"1-month return {ret_1m:.2f}% is not positive")
            for period, ema in ((20, ema20), (50, ema50), (200, ema200)):
                if current <= ema:
                    reasons.append(f"latest close is not above EMA{period}")
            if dip_5d < draw_min_dip_5d or dip_5d > draw_max_dip_5d:
                reasons.append(f"5-session dip {dip_5d:.2f}% is outside {draw_min_dip_5d:.2f}% to {draw_max_dip_5d:.2f}%")
            if down_days < 2:
                reasons.append(f"only {down_days} down sessions in the last five; at least 2 required")
            if not np.isfinite(distance_high) or distance_high < -abs(draw_max_below_60d_high):
                reasons.append(f"price is more than {draw_max_below_60d_high:.2f}% below its 60-session high")
            score = 0.7 * ret_3m + 0.3 * ret_1m

        rows.append(
            {
                **base,
                "Price Date": pd.Timestamp(history["Date"].iloc[-1]).date().isoformat(),
                "Price Observations": len(history),
                "CMP Rs.": round(current, 2),
                "Average Volume 30D": round(avg_volume, 0),
                "Return 1M %": round(ret_1m, 2),
                "Return 3M %": round(ret_3m, 2),
                "Return 5D %": round(ret_5d, 2),
                "ADR 20D %": round(adr20, 2),
                "EMA 20": round(ema20, 2),
                "EMA 50": round(ema50, 2),
                "EMA 200": round(ema200, 2),
                "Distance From 60D High %": round(distance_high, 2),
                "Down Days 5D": down_days,
                "Dip 5D %": round(dip_5d, 2),
                "Strategy Score": round(score, 3),
                "Status": "Qualified" if not reasons else "Rejected",
                "Rejection Reasons": "; ".join(reasons) if reasons else "passed",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["Status", "Strategy Score", "Ticker"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "Rank", pd.Series(pd.NA, index=result.index, dtype="Int64"))
    passed = result["Status"].eq("Qualified")
    result.loc[passed, "Rank"] = range(1, int(passed.sum()) + 1)
    return result
