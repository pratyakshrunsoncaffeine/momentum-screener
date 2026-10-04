"""Long-only end-of-day swing research screen; scores are not probabilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from collections.abc import Callable

import numpy as np
import pandas as pd
import yfinance as yf

from .momentum import chunked  # Also configures the project's writable Yahoo cache.
from .universe import load_ticker_universe

IST = timezone(timedelta(hours=5, minutes=30))
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
Progress = Callable[[int, int, str], None]
RULE_VERSION = "swing-1.0"


def load_swing_universe(ticker_csv: str | Path) -> pd.DataFrame:
    universe = load_ticker_universe(ticker_csv)
    # Numeric six-digit codes are BSE identifiers; preserve explicit BSE suffixes.
    universe["YFinance Ticker"] = universe["Ticker"].map(
        lambda s: s if s.endswith(".BO") else f"{s}.BO" if re.fullmatch(r"\d{6}", s) else f"{s}.NS")
    if "Industry" not in universe:
        universe["Industry"] = pd.NA
    universe["Industry Source"] = np.where(universe["Industry"].notna(), "Ticker CSV", "Unclassified")
    for filename in ("quality_momentum_universe.csv", "correlation_universe.csv"):
        metadata_path = Path(ticker_csv).resolve().parent / filename
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            if {"Ticker", "Industry"}.issubset(metadata):
                lookup = metadata.dropna(subset=["Industry"]).drop_duplicates("Ticker").set_index("Ticker")["Industry"]
                mapped = universe["Ticker"].map(lookup)
                missing = universe["Industry"].isna() & mapped.notna()
                universe.loc[missing, "Industry"] = mapped[missing]
                universe.loc[missing, "Industry Source"] = filename
    return universe


@dataclass(frozen=True)
class SwingConfig:
    min_price: float = 20.0
    min_turnover_cr: float = 5.0
    min_history: int = 200
    min_rsi: float = 52.0
    max_rsi: float = 78.0
    breakout_rvol: float = 1.5
    min_atr_pct: float = 0.8
    max_atr_pct: float = 6.0
    max_extension_atr: float = 3.0
    max_stop_pct: float = 8.0
    max_stop_atr: float = 2.5
    max_5d_return: float = 20.0
    max_5d_range_atr: float = 2.5
    top_n: int = 10
    max_per_industry: int = 3
    min_score: float = 60.0
    batch_size: int = 50
    session_vwap_limit: int = 200
    require_session_vwap: bool = True
    require_weekly_trend: bool = True
    require_bullish_macd: bool = False
    estimated_roundtrip_cost_pct: float = 0.30
    reward_risk: float = 2.0
    event_horizon_days: int = 14

    def __post_init__(self):
        if self.top_n < 1 or self.batch_size < 1 or self.max_per_industry < 1:
            raise ValueError("Counts and batch size must be positive.")
        if self.min_history < 100 or not 0 <= self.min_rsi < self.max_rsi <= 100:
            raise ValueError("Use at least 100 history bars and valid RSI bounds.")
        if self.session_vwap_limit < self.top_n or not 0 <= self.min_score <= 100:
            raise ValueError("VWAP limit must cover top N, and minimum score must be 0–100.")
        if self.min_turnover_cr < 0 or self.reward_risk <= 0 or self.estimated_roundtrip_cost_pct < 0:
            raise ValueError("Liquidity/cost cannot be negative; reward/risk must be positive.")
        if not 0 < self.max_stop_pct < 100 or self.max_stop_atr <= 0 or not 0 < self.min_atr_pct <= self.max_atr_pct:
            raise ValueError("Stop and volatility limits must be positive, with stop percentage below 100.")


def wilder(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder smoothing seeded with the first period's arithmetic mean."""
    valid = series.dropna()
    if len(valid) < period:
        return pd.Series(np.nan, index=series.index)
    seeded = series.astype(float).copy()
    position = series.index.get_loc(valid.index[period - 1])
    seeded.iloc[:position] = np.nan
    seeded.iloc[position] = valid.iloc[:period].mean()
    return seeded.ewm(alpha=1 / period, adjust=False, ignore_na=True).mean()


def indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Causal indicators: trailing windows only, no future pivot confirmation."""
    out = frame.copy()
    close, high, low, volume = (out[x].astype(float) for x in ["Close", "High", "Low", "Volume"])
    for n in (10, 20, 50):
        out[f"EMA{n}"] = close.ewm(span=n, adjust=False, min_periods=n).mean()
    out["SMA200"] = close.rolling(200).mean()
    delta = close.diff()
    gain, loss = wilder(delta.clip(lower=0)), wilder(-delta.clip(upper=0))
    out["RSI14"] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).where(loss.ne(0), 100)
    out.loc[gain.eq(0) & loss.eq(0), "RSI14"] = 50
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    out["ATR14"] = wilder(tr)
    macd = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["MACD"] = macd
    out["MACD Signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["MACD Histogram"] = out["MACD"] - out["MACD Signal"]
    up, down = high.diff(), -low.diff()
    plus = wilder(up.where((up > down) & (up > 0), 0).where(up.notna()))
    minus = wilder(down.where((down > up) & (down > 0), 0).where(down.notna()))
    out["+DI14"], out["-DI14"] = 100 * plus / out["ATR14"], 100 * minus / out["ATR14"]
    di_sum = out["+DI14"] + out["-DI14"]
    dx = (100 * (out["+DI14"] - out["-DI14"]).abs() / di_sum.replace(0, np.nan)).where(di_sum.ne(0), 0)
    out["ADX14"] = wilder(dx)
    out["RVOL20"] = volume / volume.shift(1).rolling(20).mean().replace(0, np.nan)
    out["Average Turnover Cr"] = (close * volume).shift(1).rolling(20).mean() / 10_000_000
    typical = (high + low + close) / 3
    out["Daily VWAP Proxy20"] = (typical * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    out["OBV"] = (np.sign(delta).fillna(0) * volume).cumsum()
    out["OBV Flow5"] = out["OBV"].diff(5) / volume.shift(1).rolling(20).mean().replace(0, np.nan) / 5
    middle, std = close.rolling(20).mean(), close.rolling(20).std(ddof=0)
    out["BB Width %"] = 400 * std / middle
    out["BB Width Percentile120"] = out["BB Width %"].rolling(120, min_periods=60).apply(lambda x: np.mean(x <= x[-1]) * 100, raw=True)
    out["Resistance20"] = high.shift(1).rolling(20).max()
    out["Overhead60"] = high.shift(1).rolling(60).max()
    for n in (5, 10, 20, 63):
        out[f"Return{n} %"] = close.pct_change(n, fill_method=None) * 100
    spread = (high - low).replace(0, np.nan)
    out["Close Location"] = (close - low) / spread
    out["Upper Wick"] = (high - pd.concat([out["Open"], close], axis=1).max(axis=1)) / spread
    return out


def confirmed_structure(frame: pd.DataFrame, legs: int = 2) -> dict:
    """Pivots become usable only after their right-hand confirmation bars exist."""
    lows, highs = [], []
    low, high = frame.Low.to_numpy(), frame.High.to_numpy()
    for i in range(legs, len(frame) - legs):
        low_window, high_window = low[i-legs:i+legs+1], high[i-legs:i+legs+1]
        if low[i] == low_window.min() and np.count_nonzero(low_window == low[i]) == 1:
            lows.append((frame.index[i], float(low[i])))
        if high[i] == high_window.max() and np.count_nonzero(high_window == high[i]) == 1:
            highs.append((frame.index[i], float(high[i])))
    return {"Confirmed Higher Low": bool(lows[-1][1] > lows[-2][1]) if len(lows) >= 2 else None,
            "Confirmed Higher High": bool(highs[-1][1] > highs[-2][1]) if len(highs) >= 2 else None,
            "Last Confirmed Swing Low": lows[-1][1] if lows else np.nan,
            "Last Confirmed Swing High": highs[-1][1] if highs else np.nan,
            "Pivot Confirmation Lag Sessions": legs}


def clean_daily(frame: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    if frame.empty or not set(FIELDS).issubset(frame.columns):
        return pd.DataFrame(columns=FIELDS)
    out = frame[FIELDS].copy()
    index = pd.to_datetime(out.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    out.index = index.normalize()
    out = out.loc[out.index.date <= cutoff].sort_index()
    out = out[~out.index.duplicated(keep="last")].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(out).all(axis=1) & out[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
    valid &= out["High"].ge(out[["Open", "Low", "Close"]].max(axis=1)) & out["Low"].le(out[["Open", "High", "Close"]].min(axis=1)) & out["Volume"].ge(0)
    return out.loc[valid]


def split_download(data: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    result = {}
    if data.empty:
        return result
    for symbol in symbols:
        if isinstance(data.columns, pd.MultiIndex):
            for level in range(data.columns.nlevels):
                if symbol in data.columns.get_level_values(level):
                    result[symbol] = data.xs(symbol, axis=1, level=level).dropna(subset=["Close"])
                    break
        elif len(symbols) == 1 and "Close" in data:
            result[symbol] = data.dropna(subset=["Close"])
    return result


def fetch_bars(symbols: list[str], intraday: bool = False) -> dict[str, pd.DataFrame]:
    try:
        raw = yf.download(symbols, period="5d" if intraday else "2y", interval="5m" if intraday else "1d",
                          auto_adjust=True, group_by="ticker", progress=False, threads=8, timeout=20,
                          ignore_tz=not intraday)
        return split_download(raw, symbols)
    except Exception:
        return {}


def _cache_path(directory: Path, symbol: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", symbol)
    return directory / f"{safe}_{hashlib.sha256(symbol.encode()).hexdigest()[:8]}.csv"


def download_universe(universe: pd.DataFrame, config: SwingConfig, cutoff: date, session: date,
                      cache: Path, resume: bool, callback: Progress | None) -> tuple[dict, pd.DataFrame]:
    cache.mkdir(parents=True, exist_ok=True)
    symbols = universe["YFinance Ticker"].tolist()
    histories, sources, pending = {}, {}, []
    failed_same_session = set()
    previous_meta, previous_health = cache.parent / "metadata.json", cache.parent / "health.csv"
    if resume and previous_meta.exists() and previous_health.exists():
        try:
            if json.loads(previous_meta.read_text(encoding="utf-8")).get("session") == str(session):
                previous = pd.read_csv(previous_health)
                failed_same_session = set(previous.loc[previous["Data Status"].eq("Missing data"), "YFinance Ticker"]) & set(symbols)
        except (OSError, ValueError, KeyError):
            pass
    for symbol in symbols:
        path = _cache_path(cache, symbol)
        saved = pd.DataFrame()
        if resume and path.exists():
            try:
                saved = clean_daily(pd.read_csv(path, index_col=0, parse_dates=True), cutoff)
            except (ValueError, pd.errors.ParserError, OSError):
                pass
        if not saved.empty and saved.index[-1].date() == session:
            histories[symbol], sources[symbol] = saved, "Same-session cache"
        elif symbol in failed_same_session:
            sources[symbol] = "Unavailable in prior same-session attempt; use Refresh to retry"
        else:
            pending.append(symbol)
    completed = len(histories) + len(failed_same_session - set(histories))
    if callback:
        callback(completed, len(symbols), f"{completed:,} current histories recovered; {len(pending):,} to fetch")
    for batch in chunked(pending, config.batch_size):
        fetched = fetch_bars(batch)
        missing = [s for s in batch if s not in fetched or fetched[s].empty]
        for retry in chunked(missing, 10):
            fetched.update(fetch_bars(retry))
        for symbol in batch:
            frame = clean_daily(fetched.get(symbol, pd.DataFrame()), cutoff)
            if not frame.empty:
                histories[symbol] = frame
                sources[symbol] = "Yahoo adjusted daily OHLCV"
                frame.to_csv(_cache_path(cache, symbol), index_label="Date")
        completed += len(batch)
        if callback:
            callback(completed, len(symbols), f"Daily scan {completed:,}/{len(symbols):,}; {len(histories):,} histories available")
    health = []
    for record in universe.to_dict("records"):
        symbol = record["YFinance Ticker"]
        frame = histories.get(symbol, pd.DataFrame())
        last = frame.index[-1].date() if not frame.empty else None
        status = "Ready" if last == session and len(frame) >= config.min_history and frame.iloc[-1]["Volume"] > 0 else "Missing data" if frame.empty else "Stale / inactive" if last != session or frame.iloc[-1]["Volume"] <= 0 else "Insufficient history"
        health.append({**record, "Data Status": status, "Last Session": str(last or ""), "Bars": len(frame), "Source": sources.get(symbol, "Unavailable")})
    return histories, pd.DataFrame(health)


def snapshot(frame: pd.DataFrame, benchmark: pd.DataFrame, config: SwingConfig) -> dict:
    f = indicators(frame)
    x, previous = f.iloc[-1], f.iloc[-2]
    close, atr = float(x["Close"]), float(x["ATR14"])
    row = {key: float(x[key]) for key in f.columns if key != "Volume"}
    row["Volume"] = float(x["Volume"])
    row.update(confirmed_structure(frame.tail(80)))
    row["ATR %"] = atr / close * 100
    row["Extension ATR"] = (close - x["EMA20"]) / atr
    row["EMA20 Rising"] = bool(x["EMA20"] > f["EMA20"].iloc[-6])
    row["Long Trend"] = bool(close > x["EMA20"] > x["EMA50"] and row["EMA20 Rising"])
    weekly = frame["Close"].resample("W-FRI").last().dropna()
    weekly = weekly[weekly.index <= frame.index[-1]]  # Last fully completed Friday-labelled week only.
    weekly_ema = weekly.ewm(span=10, adjust=False, min_periods=10).mean()
    row["Weekly Trend"] = bool(len(weekly) >= 11 and weekly.iloc[-1] > weekly_ema.iloc[-1] > weekly_ema.iloc[-2])
    # Match sessions before measuring relative strength; no returns across different endpoints.
    paired = pd.concat([frame["Close"].rename("Stock"), benchmark["Close"].rename("Market")], axis=1).dropna()
    row["Excess Return20 %"] = float((paired["Stock"].pct_change(20, fill_method=None).iloc[-1] - paired["Market"].pct_change(20, fill_method=None).iloc[-1]) * 100) if len(paired) >= 21 else np.nan
    market_close = benchmark["Close"]
    market_ok = market_close.iloc[-1] > market_close.ewm(span=20, adjust=False).mean().iloc[-1] > market_close.ewm(span=50, adjust=False).mean().iloc[-1]
    row["Market Regime"] = "Supportive" if market_ok else "Cautious"
    consolidation = (frame["High"].iloc[-6:-1].max() - frame["Low"].iloc[-6:-1].min()) / atr
    current_range = (frame["High"].tail(5).max() - frame["Low"].tail(5).min()) / atr
    row["Range5 ATR"] = float(current_range)
    row["Previous Range5 ATR"] = float(consolidation)
    setup, setup_points = "None", 0.0
    strong_close = close > x["Open"] and x["Close Location"] >= 0.65 and x["Upper Wick"] <= 0.30
    row["Bullish Engulfing"] = bool(previous["Close"] < previous["Open"] and x["Open"] <= previous["Close"] and close >= previous["Open"])
    body = abs(close - x["Open"])
    row["Bullish Rejection Wick"] = bool(min(close, x["Open"]) - x["Low"] >= max(2 * body, 0.3 * atr) and x["Close Location"] >= 0.65)
    row["Strong Close"] = bool(strong_close)
    retest_level = np.nan
    for j in range(max(len(f) - 7, 20), len(f) - 1):
        bar = f.iloc[j]
        if bar["Close"] > bar["Resistance20"] and bar["RVOL20"] >= config.breakout_rvol:
            retest_level = float(bar["Resistance20"])
    retest = (np.isfinite(retest_level) and retest_level - 0.5 * atr <= x["Low"] <= retest_level + 0.5 * atr
              and close > retest_level and strong_close and x["RVOL20"] >= 0.8)
    row["Retest Level"] = retest_level
    if close > x["Resistance20"] and x["RVOL20"] >= config.breakout_rvol and strong_close and consolidation <= config.max_5d_range_atr:
        setup, setup_points = "Fresh breakout", 25.0
        support = x["Low"]
    elif retest:
        setup, setup_points = "Breakout retest", 25.0
        support = frame["Low"].tail(3).min()
    elif x["Low"] <= x["EMA20"] + 0.5 * atr and close > x["EMA10"] and strong_close and x["RVOL20"] >= 1.0 and previous["Low"] <= previous["EMA20"] + 0.5 * atr:
        setup, setup_points = "Pullback recovery", 22.0
        support = frame["Low"].tail(3).min()
    elif close <= x["Resistance20"] and x["Resistance20"] - close <= 0.75 * atr and current_range <= config.max_5d_range_atr and x["BB Width Percentile120"] <= 35:
        setup, setup_points = "Tight base near breakout", 18.0
        support = frame["Low"].tail(5).min()
    else:
        support = frame["Low"].tail(5).min()
    row["Setup"] = setup
    row["Entry Trigger"] = float(max(x["High"], x["Resistance20"] if setup == "Tight base near breakout" else x["High"]) + 0.05 * atr)
    row["Initial Stop"] = float(support - 0.10 * atr)
    row["Max Entry"] = float(min(row["Entry Trigger"] + 0.25 * atr,
                                 row["Initial Stop"] / (1 - config.max_stop_pct / 100),
                                 row["Initial Stop"] + config.max_stop_atr * atr))
    risk = row["Entry Trigger"] - row["Initial Stop"]
    row["Stop Distance %"] = risk / row["Entry Trigger"] * 100
    row["Stop Distance ATR"] = risk / atr
    row["Worst Fill Stop %"] = (row["Max Entry"] - row["Initial Stop"]) / row["Max Entry"] * 100
    # Plan target from the WORST allowed fill, including an explicit cost assumption.
    worst_risk = row["Max Entry"] - row["Initial Stop"]
    cost = row["Max Entry"] * config.estimated_roundtrip_cost_pct / 100
    row["Target"] = row["Max Entry"] + config.reward_risk * (worst_risk + cost) + cost
    row["Net Planned R:R"] = config.reward_risk
    overhead = x["Overhead60"]
    row["Overhead Room OK"] = bool(overhead <= row["Entry Trigger"] or overhead >= row["Target"])
    row["Trailing EMA10"] = float(x["EMA10"])
    row["Chandelier3ATR"] = float(frame["High"].tail(10).max() - 3 * atr)
    row["Time Exit Sessions"] = 10
    row["Early Review Sessions"] = 3
    row["Exit Warning"] = "; ".join(s for condition, s in [
        (x["MACD Histogram"] <= 0, "MACD below signal"),
        (close < x["EMA10"], "Below EMA10"),
        (x["RVOL20"] >= 3 and x["Upper Wick"] >= 0.4, "High volume / upper wick"),
        (x["OBV Flow5"] < 0, "Negative 5-session OBV flow")
    ] if condition) or "None"
    reasons = []
    gates = [
        (close >= config.min_price, "Price below minimum"),
        (x["Average Turnover Cr"] >= config.min_turnover_cr, "Low liquidity"),
        (row["Long Trend"], "Trend not rising"),
        (not config.require_weekly_trend or row["Weekly Trend"], "Completed-week trend not rising"),
        (config.min_rsi <= x["RSI14"] <= config.max_rsi, "RSI outside momentum band"),
        (not config.require_bullish_macd or (x["MACD"] > 0 and x["MACD Histogram"] > 0), "MACD not bullish"),
        (config.min_atr_pct <= row["ATR %"] <= config.max_atr_pct, "Volatility outside range"),
        (0 <= row["Extension ATR"] <= config.max_extension_atr, "Too extended from EMA20"),
        (x["Return20 %"] > 0 and row["Excess Return20 %"] > 0, "No positive 20-session market outperformance"),
        (x["Return5 %"] <= config.max_5d_return, "Recent vertical surge"),
        (close >= x["Daily VWAP Proxy20"], "Below daily volume-weighted price proxy"),
        (setup != "None", "No qualifying price structure"),
        (0 < row["Stop Distance %"] <= config.max_stop_pct and 0 < row["Stop Distance ATR"] <= config.max_stop_atr, "Stop too distant"),
        (row["Max Entry"] >= row["Entry Trigger"], "No fill fits stop limit"),
        (row["Overhead Room OK"], "Prior 60-session high before target"),
        (not (x["RVOL20"] >= 3 and x["Upper Wick"] >= 0.4), "High-volume rejection candle"),
    ]
    reasons.extend(label for passed, label in gates if not passed)
    bounded = lambda value: float(np.clip(value, 0, 1)) if np.isfinite(value) else 0.0
    row["Trend Points"] = 10 * row["Long Trend"] + 5 * row["Weekly Trend"] + 5 * bool(x["ADX14"] >= 20 and x["+DI14"] > x["-DI14"])
    row["Momentum Points"] = 10 * bounded(1 - abs(x["RSI14"] - 65) / 25) + 5 * bool(x["MACD Histogram"] > 0) + 5 * bool(x["MACD Histogram"] > previous["MACD Histogram"])
    row["Structure Points"] = setup_points
    row["Participation Points"] = 10 * bounded((x["RVOL20"] - 0.5) / 1.5) + 5 * bounded(x["OBV Flow5"])
    row["Relative Strength Points"] = 10 * bounded(row["Excess Return20 %"] / 10)
    row["VWAP Points"] = 5 * bool(close >= x["Daily VWAP Proxy20"])
    row["Market Points"] = 5 * bool(market_ok)
    row["Setup Score"] = sum(row[key] for key in row if key.endswith(" Points"))
    if row["Setup Score"] < config.min_score:
        reasons.append("Score below minimum")
    row["Technical Pass"] = not reasons
    row["Rejection Reasons"] = "; ".join(reasons)
    row["Why Selected"] = f"{setup}; RSI {x['RSI14']:.1f}; volume {x['RVOL20']:.2f}x; 20-session NIFTY outperformance {row['Excess Return20 %']:.1f}pp"
    return row


def session_vwap(frame: pd.DataFrame, session: date) -> dict:
    unavailable = {"Session VWAP Status": "Unavailable / incomplete", "Session VWAP5m": np.nan, "Above Session VWAP": False, "Intraday Bars": 0}
    if frame.empty or not set(FIELDS).issubset(frame):
        return unavailable
    f = frame[FIELDS].copy()
    index = pd.to_datetime(f.index)
    # Yahoo exchange-local naive timestamps are interpreted as IST; aware timestamps are converted.
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    f.index = index
    f = f[(f.index.date == session) & (f.index.time >= time(9, 15)) & (f.index.time <= time(15, 25))]
    f = f[~f.index.duplicated()].sort_index().dropna()
    if f.empty or len(f) < 70 or f.index[0].time() > time(9, 20) or f.index[-1].time() < time(15, 25) or f["Volume"].sum() <= 0:
        return {**unavailable, "Intraday Bars": len(f)}
    if not np.isfinite(f.to_numpy()).all() or f["Volume"].lt(0).any():
        return {**unavailable, "Intraday Bars": len(f)}
    vwap = float((((f.High + f.Low + f.Close) / 3) * f.Volume).sum() / f.Volume.sum())
    above = bool(f.Close.iloc[-1] >= vwap)
    return {"Session VWAP Status": "Pass" if above else "Below session VWAP", "Session VWAP5m": vwap, "Above Session VWAP": above, "Intraday Bars": len(f)}


def apply_events(metrics: pd.DataFrame, events: pd.DataFrame, as_of: date, config: SwingConfig) -> pd.DataFrame:
    result = metrics.copy()
    result["Event Status"] = "Unverified — check exchange announcements"
    result["Known Event"] = ""
    if events.empty:
        return result
    if not {"Ticker", "Date", "Event"}.issubset(events.columns):
        raise ValueError("Events CSV must contain Ticker, Date (YYYY-MM-DD), and Event.")
    e = events.copy()
    e["Date"] = pd.to_datetime(e["Date"], errors="raise").dt.date
    e["Ticker"] = e["Ticker"].astype(str).str.upper().str.strip().str.removesuffix(".NS")
    e = e[e.Date.ge(as_of) & e.Date.le(as_of + timedelta(days=config.event_horizon_days))]
    if e.empty:
        return result
    descriptions = e.groupby("Ticker").apply(lambda g: "; ".join(g.Date.astype(str) + " " + g.Event.astype(str)), include_groups=False)
    result["Known Event"] = result.Ticker.map(descriptions).fillna("")
    blocked = result["Known Event"].ne("")
    result.loc[blocked, "Event Status"] = "Known event in holding window — excluded"
    result.loc[blocked, "Technical Pass"] = False
    result.loc[blocked, "Rejection Reasons"] = result.loc[blocked, "Rejection Reasons"].fillna("").map(lambda s: (s + "; " if s else "") + "Known event risk")
    return result


def select_picks(metrics: pd.DataFrame, config: SwingConfig) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    mask = metrics["Technical Pass"].eq(True)
    if config.require_session_vwap:
        mask &= metrics["Session VWAP Status"].eq("Pass")
    ranked = metrics.loc[mask].sort_values(["Setup Score", "Excess Return20 %", "Ticker"], ascending=[False, False, True]).copy()
    ranked["Industry"] = ranked.get("Industry", pd.Series("Unknown", index=ranked.index)).fillna("Unknown")
    ranked = ranked[ranked.groupby("Industry").cumcount() < config.max_per_industry].head(config.top_n).reset_index(drop=True)
    ranked.insert(0, "Swing Rank", range(1, len(ranked) + 1))
    return ranked


def industry_pockets(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    f = metrics.copy()
    f["Industry"] = f.get("Industry", pd.Series("Unknown", index=f.index)).fillna("Unknown")
    pockets = f.groupby("Industry").agg(Stocks=("Ticker", "size"), **{
        "Rising Trend %": ("Long Trend", lambda x: x.mean() * 100),
        "Technical Setups": ("Technical Pass", "sum"),
        "Median Excess Return20 %": ("Excess Return20 %", "median"),
        "Median Setup Score": ("Setup Score", "median")}).reset_index()
    pockets["Breadth Score"] = pockets["Rising Trend %"] * 0.5 + pockets["Median Excess Return20 %"].clip(-20, 20) * 2.5
    pockets["Rankable"] = pockets.Industry.ne("Unknown") & pockets.Stocks.ge(5)
    pockets["Coverage Status"] = np.where(pockets.Industry.eq("Unknown"), "Unclassified stocks — not a sector pocket", np.where(pockets.Stocks.lt(5), "Small group: fewer than 5 usable stocks", "Measured CSV industry group"))
    return pockets.sort_values(["Rankable", "Technical Setups", "Breadth Score"], ascending=False).reset_index(drop=True)


def build_watchlist(metrics: pd.DataFrame, previous: pd.DataFrame, session: date) -> pd.DataFrame:
    """Persist sightings across scans; repeated runs in one session count once."""
    columns = ["Ticker", "Name", "Industry", "First Seen", "Last Seen", "Sessions Seen", "Last Setup", "Last Score", "Current Status"]
    records = {}
    if not previous.empty:
        for r in previous.to_dict("records"):
            last = date.fromisoformat(str(r["Last Seen"]))
            if session - timedelta(days=30) <= last <= session:
                records[r["Ticker"]] = r
    current = metrics.loc[metrics["Technical Pass"]]
    for row in current.to_dict("records"):
        old = records.get(row["Ticker"], {})
        records[row["Ticker"]] = {"Ticker":row["Ticker"], "Name":row.get("Name", ""), "Industry":row.get("Industry", "Unknown"),
                                  "First Seen":old.get("First Seen", str(session)), "Last Seen":str(session),
                                  "Sessions Seen":int(old.get("Sessions Seen", 0)) + int(old.get("Last Seen") != str(session)),
                                  "Last Setup":row["Setup"], "Last Score":row["Setup Score"]}
    tickers = set(current.Ticker)
    for ticker, row in records.items():
        row["Current Status"] = "Qualifies today — check VWAP and entry" if ticker in tickers else "Monitor — does not qualify today"
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records.values())[columns].sort_values(["Last Seen", "Last Score"], ascending=False).reset_index(drop=True)


def load_swing_results(output_dir: str | Path) -> dict:
    root = Path(output_dir)
    if not (root / "metadata.json").exists():
        raise FileNotFoundError("No saved swing scan. Run the evening scan first.")
    result = {"metadata": json.loads((root / "metadata.json").read_text(encoding="utf-8"))}
    for name in ("picks", "metrics", "rejected", "health", "pockets", "watchlist"):
        try:
            result[name] = pd.read_csv(root / f"{name}.csv")
        except (pd.errors.EmptyDataError, FileNotFoundError):
            result[name] = pd.DataFrame()
    return result


def run_swing_scan(ticker_csv: str | Path, config: SwingConfig | None = None,
                   output_dir: str | Path = "output/swing/latest", resume: bool = True,
                   events_csv: str | Path | None = None, callback: Progress | None = None,
                   now: datetime | None = None) -> dict:
    config = config or SwingConfig()
    now = (now or datetime.now(IST)).astimezone(IST)
    # Ignore today's bar until after the cash-session close plus a data-delivery buffer.
    cutoff = now.date() if now.time() >= time(16, 0) else now.date() - timedelta(days=1)
    universe = load_swing_universe(ticker_csv)
    if universe.empty:
        raise ValueError("Ticker universe is empty.")
    root = Path(output_dir)
    benchmark = clean_daily(fetch_bars(["^NSEI"]).get("^NSEI", pd.DataFrame()), cutoff)
    if len(benchmark) < config.min_history:
        raise RuntimeError("NIFTY benchmark has no sufficient current history. Existing saved scan was preserved.")
    session = benchmark.index[-1].date()
    if (cutoff - session).days > 7:
        raise RuntimeError("NIFTY history is over a week old; cannot establish a current scan session.")
    histories, health = download_universe(universe, config, cutoff, session, root / "daily_cache", resume, callback)
    ready = set(health.loc[health["Data Status"].eq("Ready"), "YFinance Ticker"])
    rows = []
    # Compute benchmark indicators once; snapshot operates on causal, matched dates.
    for i, record in enumerate(universe.to_dict("records")):
        symbol = record["YFinance Ticker"]
        if symbol in ready:
            try:
                rows.append({**record, **snapshot(histories[symbol].loc[:pd.Timestamp(session)], benchmark, config), "As Of": str(session)})
            except (ValueError, IndexError, ZeroDivisionError) as exc:
                health.loc[health["YFinance Ticker"].eq(symbol), "Data Status"] = f"Calculation failed: {type(exc).__name__}"
        if callback and (i % 100 == 0 or i == len(universe) - 1):
            callback(i + 1, len(universe), f"Calculating daily indicators {i + 1:,}/{len(universe):,}")
    if not rows:
        raise RuntimeError("No ticker has sufficient fresh OHLCV data. Existing saved scan was preserved.")
    metrics = pd.DataFrame(rows)
    events = pd.read_csv(events_csv) if events_csv else pd.DataFrame()
    metrics = apply_events(metrics, events, now.date(), config)
    metrics["Session VWAP Status"] = "Not checked"
    metrics["Session VWAP5m"] = np.nan
    metrics["Above Session VWAP"] = False
    metrics["Intraday Bars"] = 0
    shortlist = metrics[metrics["Technical Pass"]].sort_values(["Setup Score", "Excess Return20 %", "Ticker"], ascending=[False, False, True]).head(config.session_vwap_limit)
    symbols = shortlist["YFinance Ticker"].tolist()
    for start in range(0, len(symbols), 20):
        batch = symbols[start:start + 20]
        intraday = fetch_bars(batch, intraday=True)
        for symbol in batch:
            verification = session_vwap(intraday.get(symbol, pd.DataFrame()), session)
            for key, value in verification.items():
                metrics.loc[metrics["YFinance Ticker"].eq(symbol), key] = value
        if callback:
            callback(min(start + len(batch), len(symbols)), len(symbols), "Checking shortlist VWAP from completed 5-minute session bars")
    metrics = metrics.sort_values(["Setup Score", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    picks = select_picks(metrics, config)
    rejected = metrics.loc[~metrics["Technical Pass"] | (config.require_session_vwap & ~metrics["Session VWAP Status"].eq("Pass"))].copy()
    rejected["Final Exclusion"] = rejected["Rejection Reasons"].fillna("")
    needs_vwap = rejected["Technical Pass"] & ~rejected["Session VWAP Status"].eq("Pass")
    rejected.loc[needs_vwap, "Final Exclusion"] = rejected.loc[needs_vwap, "Session VWAP Status"]
    pockets = industry_pockets(metrics)
    try:
        previous_watchlist = pd.read_csv(root / "watchlist.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        previous_watchlist = pd.DataFrame()
    watchlist = build_watchlist(metrics, previous_watchlist, session)
    metadata = {"rule_version": RULE_VERSION, "run_time_ist": now.isoformat(), "session": str(session),
                "universe_count": len(universe), "usable_count": len(metrics), "technical_pass_count": int(metrics["Technical Pass"].sum()),
                "vwap_checked_count": len(symbols), "final_count": len(picks),
                "market_regime": metrics["Market Regime"].iloc[0], "config": asdict(config),
                "classified_usable_count": int(metrics.Industry.notna().sum()), "watchlist_count": len(watchlist),
                "ticker_file_sha256": hashlib.sha256(Path(ticker_csv).read_bytes()).hexdigest(),
                "events_file": str(events_csv or "None — events unverified"),
                "coverage_note": "All listed tickers attempted. Missing, inactive, stale, and short-history stocks excluded. NIFTY's latest completed bar defines the scan session; confirm exchange data has updated.",
                "score_note": "Heuristic setup strength, not a calibrated probability or research-validated return forecast."}
    # Keep archived snapshots and publish latest metadata only after all CSVs are saved.
    root.mkdir(parents=True, exist_ok=True)
    archive = root.parent / "runs" / now.strftime("%Y%m%d_%H%M%S")
    archive.mkdir(parents=True, exist_ok=True)
    result = {"picks": picks, "metrics": metrics, "rejected": rejected, "health": health, "pockets": pockets, "watchlist":watchlist, "metadata": metadata}
    for name in ("picks", "metrics", "rejected", "health", "pockets", "watchlist"):
        result[name].to_csv(root / f"{name}.csv", index=False)
        result[name].to_csv(archive / f"{name}.csv", index=False)
    benchmark.to_csv(root / "benchmark.csv", index_label="Date")
    payload = json.dumps(metadata, indent=2)
    (root / "metadata.json").write_text(payload, encoding="utf-8")
    (archive / "metadata.json").write_text(payload, encoding="utf-8")
    return result
