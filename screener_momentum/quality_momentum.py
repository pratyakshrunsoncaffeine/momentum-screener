from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from io import BytesIO
import gzip
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import QualityMomentumConfig
from .fundamentals import HEADERS, fetch_company_quality_metrics, normalize_quarter_period
from .momentum import chunked


ProgressCallback = Callable[[int, int, str], None]

SURVEILLANCE_DESCRIPTIONS = {
    99: "GSM watch list",
    1: "GSM Stage I",
    2: "GSM Stage II",
    3: "GSM Stage III",
    4: "GSM Stage IV",
    5: "GSM Stage V",
    6: "GSM Stage VI",
    11: "Short-term ASM Stage I",
    12: "Short-term ASM Stage II",
    13: "Long-term ASM Stage I",
    14: "Long-term ASM Stage II",
    15: "Long-term ASM Stage III",
    16: "Long-term ASM Stage IV",
    34: "ESM Stage I",
    35: "ESM Stage II",
    36: "ESM Stage I and GSM watch list",
    37: "ESM Stage II and GSM watch list",
}
FLAG_ONLY_CODES = {1, 11, 13, 34, 36, 99}
REJECT_SURVEILLANCE_CODES = {2, 3, 4, 5, 6, 12, 14, 15, 16, 35, 37, *range(50, 63)}


def download_quality_history(
    yahoo_tickers: list[str],
    batch_size: int = 50,
    period: str = "2y",
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted closes and volumes with serialized retries for missing symbols."""
    tickers = list(dict.fromkeys(str(ticker).strip().upper() for ticker in yahoo_tickers))
    close_parts: list[pd.DataFrame] = []
    volume_parts: list[pd.DataFrame] = []
    total = len(tickers)
    completed = 0
    for batch in chunked(tickers, max(1, int(batch_size))):
        if progress_callback:
            progress_callback(completed, total, f"Downloading quality price history for {batch[0]} to {batch[-1]}")
        data = _download_market_batch(batch, period=period, threads=True)
        close = _extract_market_field(data, batch, "Close")
        volume = _extract_market_field(data, batch, "Volume")
        available = {
            column for column in close.columns if close[column].notna().any()
        } if not close.empty else set()
        if not close.empty:
            close_parts.append(close)
        if not volume.empty:
            volume_parts.append(volume)

        missing = [ticker for ticker in batch if ticker not in available]
        for retry_batch in chunked(missing, 5):
            retry_data = _download_market_batch(retry_batch, period=period, threads=False)
            retry_close = _extract_market_field(retry_data, retry_batch, "Close")
            retry_volume = _extract_market_field(retry_data, retry_batch, "Volume")
            if not retry_close.empty:
                close_parts.append(retry_close)
            if not retry_volume.empty:
                volume_parts.append(retry_volume)
        completed += len(batch)
        if progress_callback:
            progress_callback(completed, total, f"Downloaded {completed:,} of {total:,} quality histories")

    return _merge_market_parts(close_parts), _merge_market_parts(volume_parts)


def _download_market_batch(tickers: list[str], period: str, threads: bool) -> pd.DataFrame:
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


def _extract_market_field(data: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
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
    result.columns = [str(column).upper() for column in result.columns]
    result.index = pd.to_datetime(result.index, errors="coerce")
    return result.loc[result.index.notna()].sort_index()


def _merge_market_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, axis=1)
    if merged.columns.duplicated().any():
        merged = merged.T.groupby(level=0).first().T
    return merged.sort_index()


def trendline_momentum(series: pd.Series, lookback_days: int = 126) -> tuple[float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().tail(lookback_days)
    if len(values) < lookback_days or (values <= 0).any():
        return np.nan, np.nan, np.nan
    x = np.arange(len(values), dtype=float)
    y = np.log(values.to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if total == 0 else max(0.0, 1.0 - residual / total)
    annualized = (math.exp(float(slope) * 252.0) - 1.0) * 100.0
    return round(annualized * r_squared, 3), round(annualized, 3), round(r_squared, 4)


def calculate_rsi(series: pd.Series, period: int = 14) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= period:
        return np.nan
    changes = values.diff()
    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    gain = float(average_gain.iloc[-1])
    loss = float(average_loss.iloc[-1])
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return round(100.0 - (100.0 / (1.0 + gain / loss)), 2)


def calculate_quality_price_metrics(
    universe: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    config: QualityMomentumConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in universe.to_dict("records"):
        yahoo_ticker = str(item["YFinance Ticker"]).upper()
        close = closes[yahoo_ticker].dropna() if yahoo_ticker in closes else pd.Series(dtype=float)
        volume = volumes[yahoo_ticker].dropna() if yahoo_ticker in volumes else pd.Series(dtype=float)
        aligned = pd.concat([close.rename("Close"), volume.rename("Volume")], axis=1).dropna()
        row: dict[str, Any] = {
            **item,
            "Price Date": close.index[-1].date().isoformat() if not close.empty else "",
            "Price Observations": int(len(close)),
            "CMP Rs.": round(float(close.iloc[-1]), 2) if not close.empty else np.nan,
        }
        required = max(
            config.sma_slow_days,
            config.high_lookback_days,
            config.trend_lookback_days + config.score_comparison_days,
        )
        if len(close) < required:
            row.update(_empty_price_metrics(f"Only {len(close)} valid closes; {required} required"))
            rows.append(row)
            continue

        current = float(close.iloc[-1])
        sma50 = float(close.tail(config.sma_fast_days).mean())
        sma200 = float(close.tail(config.sma_slow_days).mean())
        score, annualized, r_squared = trendline_momentum(close, config.trend_lookback_days)
        prior_score, _, _ = trendline_momentum(
            close.iloc[: -config.score_comparison_days], config.trend_lookback_days
        )
        high_52w = float(close.tail(config.high_lookback_days).max())
        distance_high = (current / high_52w - 1.0) * 100.0
        return_21d = (
            (current / float(close.iloc[-config.score_comparison_days - 1]) - 1.0) * 100.0
        )
        extension = (current / sma50 - 1.0) * 100.0
        adtv_cr = (
            float((aligned["Close"] * aligned["Volume"]).tail(config.liquidity_days).mean()) / 10_000_000
            if len(aligned) >= config.liquidity_days
            else np.nan
        )
        rsi = calculate_rsi(close, config.rsi_days)
        vertical_surge = distance_high >= -config.high_proximity_pct and return_21d >= config.vertical_surge_return_pct

        reasons: list[str] = []
        if pd.isna(adtv_cr) or adtv_cr < config.min_average_traded_value_cr:
            reasons.append(f"average traded value {adtv_cr:.2f} Cr below {config.min_average_traded_value_cr:.2f} Cr" if pd.notna(adtv_cr) else "missing traded value")
        if pd.isna(score) or pd.isna(prior_score) or score < prior_score:
            reasons.append("trendline score is below its one-month-earlier value")
        if not (current > sma50 > sma200):
            reasons.append("price > SMA50 > SMA200 is not satisfied")
        if pd.isna(rsi) or rsi > config.max_rsi:
            reasons.append(f"RSI {rsi:.2f} exceeds {config.max_rsi:.2f}" if pd.notna(rsi) else "RSI unavailable")
        if extension > config.max_sma50_extension_pct:
            reasons.append(f"price is {extension:.2f}% above SMA50")
        if vertical_surge:
            reasons.append("vertical surge: near 52-week high with 21-day return above limit")

        row.update(
            {
                "Average Daily Traded Value Cr": round(adtv_cr, 3),
                "Trendline Momentum Score": score,
                "Annualized Trend %": annualized,
                "Trend R Squared": r_squared,
                "Score One Month Ago": prior_score,
                "SMA50": round(sma50, 2),
                "SMA200": round(sma200, 2),
                "RSI 14": rsi,
                "SMA50 Extension %": round(extension, 2),
                "52 Week High": round(high_52w, 2),
                "Distance From 52W High %": round(distance_high, 2),
                "21D Return %": round(return_21d, 2),
                "Vertical Surge": vertical_surge,
                "Price Filter Pass": not reasons,
                "Price Rejection Reasons": "; ".join(reasons) if reasons else "passed",
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["Price Momentum Rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    valid = result["Trendline Momentum Score"].notna()
    ordered = result.loc[valid].sort_values("Trendline Momentum Score", ascending=False).index
    result.loc[ordered, "Price Momentum Rank"] = range(1, len(ordered) + 1)
    eligible = result.loc[result["Price Filter Pass"].fillna(False)].sort_values(
        "Trendline Momentum Score", ascending=False
    )
    keep_count = max(1, math.ceil(len(eligible) * config.top_percent / 100.0)) if len(eligible) else 0
    keep_indices = set(eligible.head(keep_count).index)
    result["Top Momentum Decile"] = result.index.to_series().isin(keep_indices)
    result["Price Stage Pass"] = result["Price Filter Pass"].fillna(False) & result["Top Momentum Decile"]
    outside = result["Price Filter Pass"].fillna(False) & ~result["Top Momentum Decile"]
    result.loc[outside, "Price Rejection Reasons"] = (
        f"outside top {config.top_percent:g}% after price-quality filters"
    )
    return result.sort_values(
        ["Price Stage Pass", "Trendline Momentum Score"], ascending=[False, False]
    ).reset_index(drop=True)


def _empty_price_metrics(reason: str) -> dict[str, Any]:
    return {
        "Average Daily Traded Value Cr": np.nan,
        "Trendline Momentum Score": np.nan,
        "Annualized Trend %": np.nan,
        "Trend R Squared": np.nan,
        "Score One Month Ago": np.nan,
        "SMA50": np.nan,
        "SMA200": np.nan,
        "RSI 14": np.nan,
        "SMA50 Extension %": np.nan,
        "52 Week High": np.nan,
        "Distance From 52W High %": np.nan,
        "21D Return %": np.nan,
        "Vertical Surge": False,
        "Price Filter Pass": False,
        "Price Rejection Reasons": reason,
    }


class NseQualityReferenceProvider:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch(self, as_of: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        master, master_health = self.fetch_security_master(as_of)
        surveillance, surveillance_health = self.fetch_surveillance(as_of)
        return master, surveillance, pd.DataFrame([master_health, surveillance_health])

    def fetch_security_master(self, as_of: date) -> tuple[pd.DataFrame, dict[str, Any]]:
        for offset in range(10):
            report_date = as_of - timedelta(days=offset)
            url = (
                "https://nsearchives.nseindia.com/content/cm/"
                f"NSE_CM_security_{report_date.strftime('%d%m%Y')}.csv.gz"
            )
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                frame = pd.read_csv(BytesIO(gzip.decompress(response.content)), low_memory=False)
                parsed = parse_security_master(frame)
                if not parsed.empty:
                    path = self.cache_dir / f"security_master_{report_date:%Y%m%d}.csv"
                    parsed.to_csv(path, index=False)
                    return parsed, _health("NSE security master", "downloaded", report_date, url, len(parsed))
            except Exception:
                continue
        cached = sorted(self.cache_dir.glob("security_master_*.csv"), reverse=True)
        if cached:
            parsed = pd.read_csv(cached[0])
            report_date = _date_from_cache_name(cached[0])
            return parsed, _health("NSE security master", "saved fallback", report_date, str(cached[0]), len(parsed))

        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            parsed = parse_security_master(pd.read_csv(BytesIO(response.content)))
            return parsed, _health("NSE security master", "static fallback", as_of, url, len(parsed))
        except Exception as exc:
            return pd.DataFrame(), _health("NSE security master", "failed", as_of, url, 0, str(exc))

    def fetch_surveillance(self, as_of: date) -> tuple[pd.DataFrame, dict[str, Any]]:
        for offset in range(10):
            report_date = as_of - timedelta(days=offset)
            for prefix in ("REG1_IND", "REG_IND"):
                filename = f"{prefix}{report_date.strftime('%d%m%y')}.csv"
                for base_url in (
                    "https://nsearchives.nseindia.com/content/cm/",
                    "https://nsearchives.nseindia.com/archives/equities/mkt/",
                ):
                    url = f"{base_url}{filename}"
                    try:
                        response = self.session.get(url, timeout=30)
                        response.raise_for_status()
                        if response.content.lstrip().startswith(b"<"):
                            continue
                        parsed = parse_surveillance_report(
                            pd.read_csv(BytesIO(response.content), low_memory=False)
                        )
                        if not parsed.empty:
                            path = self.cache_dir / f"surveillance_{report_date:%Y%m%d}.csv"
                            parsed.to_csv(path, index=False)
                            return parsed, _health(
                                "NSE surveillance", "downloaded", report_date, url, len(parsed)
                            )
                    except Exception:
                        continue
        cached = sorted(self.cache_dir.glob("surveillance_*.csv"), reverse=True)
        if cached:
            parsed = pd.read_csv(cached[0])
            report_date = _date_from_cache_name(cached[0])
            return parsed, _health("NSE surveillance", "saved fallback", report_date, str(cached[0]), len(parsed))
        return pd.DataFrame(), _health(
            "NSE surveillance", "unavailable", as_of, "NSE Surveillance Indicator report", 0,
            "No current or saved NSE surveillance report was available; candidates remain pending verification.",
        )


def parse_security_master(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {str(column).strip().lower(): column for column in frame.columns}
    symbol = next((aliases[name] for name in ("tckrsymb", "symbol") if name in aliases), None)
    isin = next((aliases[name] for name in ("isin", "isin number") if name in aliases), None)
    name = next((aliases[name] for name in ("fininstrmnm", "name of company") if name in aliases), None)
    series = next((aliases[name] for name in ("sctysrs", "series") if name in aliases), None)
    if symbol is None or isin is None:
        return pd.DataFrame()
    result = pd.DataFrame(
        {
            "Ticker": frame[symbol].astype(str).str.strip().str.upper(),
            "NSE Name": frame[name].astype(str).str.strip() if name else "",
            "NSE Series": frame[series].astype(str).str.strip().str.upper() if series else "",
            "ISIN": frame[isin].astype(str).str.strip().str.upper(),
        }
    )
    if series:
        preferred = result["NSE Series"].eq("EQ")
        result = pd.concat([result[preferred], result[~preferred]])
    result = result[result["Ticker"].ne("") & result["ISIN"].str.match(r"^INE[A-Z0-9]{9}$", na=False)]
    return result.drop_duplicates("Ticker", keep="first").reset_index(drop=True)


def parse_surveillance_report(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    aliases = {str(column).strip().lower(): column for column in frame.columns}
    symbol = next(
        (aliases[name] for name in ("symbol", "tckrsymb", "security symbol", "scrip symbol") if name in aliases),
        None,
    )
    if symbol is None:
        symbol = frame.columns[0]
    indicator_columns = [
        column for column in frame.columns
        if re.search(r"surv|gsm|asm|esm|indicator|stage|short.?code", str(column), re.I)
        and column != symbol
    ]
    if not indicator_columns and len(frame.columns) <= 5:
        indicator_columns = [column for column in frame.columns if column != symbol]
    if not indicator_columns:
        return pd.DataFrame()

    by_symbol: dict[str, set[int]] = {}
    for _, row in frame.iterrows():
        ticker = str(row[symbol]).strip().upper()
        if not ticker or ticker == "NAN":
            continue
        codes = by_symbol.setdefault(ticker, set())
        for column in indicator_columns:
            codes.update(
                int(match) for match in re.findall(r"(?<!\d)(?:99|[1-9]|[1-8]\d)(?!\d)", str(row[column]))
                if int(match) != 0
            )
    rows = []
    for ticker, codes in by_symbol.items():
        ordered = sorted(codes)
        descriptions = [SURVEILLANCE_DESCRIPTIONS.get(code, f"Surveillance code {code}") for code in ordered]
        rows.append(
            {
                "Ticker": ticker,
                "Surveillance Codes": ", ".join(map(str, ordered)),
                "Surveillance Status": "; ".join(descriptions) if descriptions else "Clear",
                "Surveillance Reject": bool(set(ordered) & REJECT_SURVEILLANCE_CODES),
                "Surveillance Flag": bool(set(ordered) & FLAG_ONLY_CODES),
            }
        )
    return pd.DataFrame(rows)


def screen_quality_fundamentals(
    shortlist: pd.DataFrame,
    security_master: pd.DataFrame,
    surveillance: pd.DataFrame,
    config: QualityMomentumConfig,
    progress_callback: ProgressCallback | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    sleep_seconds: float = 0.6,
    as_of: date | None = None,
) -> pd.DataFrame:
    if shortlist.empty:
        return shortlist.copy()
    scan_date = as_of or date.today()
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    records = shortlist.to_dict("records")
    candidate_by_ticker = {str(item["Ticker"]).upper(): item for item in records}
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if resume and checkpoint and checkpoint.exists() and checkpoint.stat().st_size:
        try:
            saved = pd.read_csv(checkpoint)
            saved["Ticker"] = saved["Ticker"].astype(str).str.upper()
            saved = saved[saved["Ticker"].isin(candidate_by_ticker)].drop_duplicates("Ticker", keep="last")
            rows = [
                {**saved_row, **candidate_by_ticker[str(saved_row["Ticker"]).upper()]}
                for saved_row in saved.to_dict("records")
            ]
            completed = set(saved["Ticker"].astype(str).str.upper())
        except (pd.errors.EmptyDataError, KeyError):
            rows = []
    master_lookup = security_master.set_index("Ticker").to_dict("index") if not security_master.empty else {}
    surveillance_lookup = surveillance.set_index("Ticker").to_dict("index") if not surveillance.empty else {}
    rows = [
        evaluate_quality_fundamentals(
            saved_row,
            master_lookup.get(str(saved_row["Ticker"]).upper()),
            surveillance_lookup.get(str(saved_row["Ticker"]).upper()),
            config,
            scan_date,
            surveillance_available=not surveillance.empty,
            fetch_error=(
                ""
                if pd.isna(saved_row.get("Screener Fetch Error"))
                else str(saved_row.get("Screener Fetch Error", ""))
            ),
        )
        for saved_row in rows
    ]
    session = requests.Session()
    session.headers.update(HEADERS)
    total = len(records)
    for item in records:
        ticker = str(item["Ticker"]).upper()
        if ticker in completed:
            continue
        if progress_callback:
            progress_callback(len(completed), total, f"Verifying fundamentals and ownership for {ticker}")
        try:
            metrics = fetch_company_quality_metrics(ticker, session=session)
            fetch_error = ""
        except Exception as exc:
            metrics = {}
            fetch_error = str(exc)
        row = evaluate_quality_fundamentals(
            {**item, **metrics},
            master_lookup.get(ticker),
            surveillance_lookup.get(ticker),
            config,
            scan_date,
            surveillance_available=not surveillance.empty,
            fetch_error=fetch_error,
        )
        rows.append(row)
        completed.add(ticker)
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
        if progress_callback:
            progress_callback(len(completed), total, f"Verified {len(completed):,} of {total:,} shortlisted stocks")
        time.sleep(sleep_seconds)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    order = {str(item["Ticker"]).upper(): index for index, item in enumerate(records)}
    result["_Order"] = result["Ticker"].astype(str).str.upper().map(order)
    return result.sort_values("_Order").drop(columns="_Order").reset_index(drop=True)


def evaluate_quality_fundamentals(
    row: dict[str, Any],
    master: dict[str, Any] | None,
    surveillance: dict[str, Any] | None,
    config: QualityMomentumConfig,
    scan_date: date,
    surveillance_available: bool,
    fetch_error: str = "",
) -> dict[str, Any]:
    result = dict(row)
    master = master or {}
    surveillance = surveillance or {}
    isin = str(master.get("ISIN", ""))
    identity_pass = bool(master) and isin.startswith("INE") and str(result.get("Ticker", "")).upper() == str(master.get("Ticker", result.get("Ticker", ""))).upper()
    result.update(master)
    result.update(surveillance)
    result["Identity Pass"] = identity_pass
    result["Governance Fetched Date"] = scan_date.isoformat()
    result["Screener Fetch Error"] = fetch_error

    pledge = _number(result.get("Promoter Pledge %"))
    revenue_growth = _number(result.get("Quarterly Revenue YoY Growth %"))
    profit_growth = _number(result.get("Quarterly Profit YoY Growth %"))
    deterioration = (
        revenue_growth < 0 and profit_growth < -50
        if revenue_growth is not None and profit_growth is not None
        else None
    )
    fii_change = _number(result.get("FII Holding Change %"))
    dii_change = _number(result.get("DII Holding Change %"))
    institutional_complete = all(
        _number(result.get(column)) is not None
        for column in (
            "FII Previous Holding %", "FII Latest Holding %",
            "DII Previous Holding %", "DII Latest Holding %",
        )
    )
    combined_change = fii_change + dii_change if fii_change is not None and dii_change is not None else None
    result["Combined FII+DII Change %"] = round(combined_change, 2) if combined_change is not None else np.nan
    result["Institutional Data Complete"] = institutional_complete

    latest_periods = [
        normalize_quarter_period(result.get("FII Latest Period")),
        normalize_quarter_period(result.get("DII Latest Period")),
    ]
    latest_periods = [period for period in latest_periods if period]
    shareholding_date = max((_quarter_end(period) for period in latest_periods), default=None)
    shareholding_age = (scan_date - shareholding_date).days if shareholding_date else None
    result["Shareholding Data Date"] = shareholding_date.isoformat() if shareholding_date else ""
    result["Shareholding Age Days"] = shareholding_age

    missing: list[str] = []
    rejected: list[str] = []
    flags: list[str] = []
    if fetch_error:
        missing.append(f"Screener fetch failed: {fetch_error}")
    if not identity_pass:
        missing.append("symbol/ISIN identity could not be verified from NSE")
    if pledge is None:
        missing.append("promoter pledge unavailable")
    elif pledge > config.max_promoter_pledge_pct:
        rejected.append(f"promoter pledge {pledge:.2f}% exceeds {config.max_promoter_pledge_pct:.2f}%")
    if deterioration is None:
        missing.append("quarterly revenue/profit YoY comparison unavailable")
    elif deterioration:
        rejected.append("business deterioration: revenue negative and profit below -50% YoY")
    if not institutional_complete or combined_change is None:
        missing.append("two complete quarters of FII and DII data unavailable")
    elif combined_change <= -config.max_combined_institutional_drop_pct:
        rejected.append(
            f"combined FII+DII holding fell {abs(combined_change):.2f} percentage points"
        )
    if shareholding_age is None:
        missing.append("shareholding quarter date unavailable")
    elif shareholding_age > config.max_shareholding_age_days:
        rejected.append(f"shareholding data is {shareholding_age} days old")
    if not surveillance_available:
        missing.append("NSE surveillance report unavailable")
    elif bool(surveillance.get("Surveillance Reject", False)):
        rejected.append(str(surveillance.get("Surveillance Status", "severe NSE surveillance flag")))
    elif bool(surveillance.get("Surveillance Flag", False)):
        flags.append(str(surveillance.get("Surveillance Status", "NSE Stage I surveillance flag")))

    result["Business Deterioration"] = deterioration
    result["Quality Data Complete"] = not missing
    result["Quality Pass"] = not missing and not rejected
    result["Quality Status"] = "Qualified" if result["Quality Pass"] else ("Pending verification" if missing and not rejected else "Rejected")
    result["Quality Flags"] = "; ".join(flags)
    result["Quality Rejection Reasons"] = "; ".join(rejected + missing) if rejected or missing else "passed"
    return result


def verify_latest_trend(
    quality_rows: pd.DataFrame,
    latest_closes: pd.DataFrame,
    config: QualityMomentumConfig,
    scan_date: date | None = None,
) -> pd.DataFrame:
    if quality_rows.empty:
        return quality_rows.copy()
    today = scan_date or date.today()
    rows = []
    for item in quality_rows.to_dict("records"):
        yahoo_ticker = str(item["YFinance Ticker"]).upper()
        series = latest_closes[yahoo_ticker].dropna() if yahoo_ticker in latest_closes else pd.Series(dtype=float)
        if series.empty:
            price = _number(item.get("CMP Rs."))
            price_date = pd.to_datetime(item.get("Price Date"), errors="coerce")
            source = "Initial Yahoo close fallback"
            sma50 = _number(item.get("SMA50"))
            sma200 = _number(item.get("SMA200"))
        else:
            price = float(series.iloc[-1])
            price_date = pd.Timestamp(series.index[-1])
            source = "Yahoo Finance refresh"
            sma50 = float(series.tail(config.sma_fast_days).mean()) if len(series) >= config.sma_fast_days else _number(item.get("SMA50"))
            sma200 = float(series.tail(config.sma_slow_days).mean()) if len(series) >= config.sma_slow_days else _number(item.get("SMA200"))
        age = (today - price_date.date()).days if pd.notna(price_date) else None
        trend_pass = (
            price is not None and sma50 is not None and sma200 is not None
            and price > sma50 > sma200
            and age is not None and age <= config.max_quote_age_days
        )
        result = dict(item)
        result.update(
            {
                "Verification Price": round(price, 2) if price is not None else np.nan,
                "Verification Price Date": price_date.date().isoformat() if pd.notna(price_date) else "",
                "Verification Price Age Days": age,
                "Verification Price Source": source,
                "Verification SMA50": round(sma50, 2) if sma50 is not None else np.nan,
                "Verification SMA200": round(sma200, 2) if sma200 is not None else np.nan,
                "Trend Confirmation Pass": trend_pass,
                "Final Pass": bool(item.get("Quality Pass", False)) and trend_pass,
            }
        )
        if not trend_pass:
            existing = str(result.get("Quality Rejection Reasons", "")).strip()
            trend_reason = "latest price failed price > SMA50 > SMA200 or quote freshness"
            result["Quality Rejection Reasons"] = f"{existing}; {trend_reason}".strip("; ")
        rows.append(result)
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["Final Pass", "Trendline Momentum Score"], ascending=[False, False]
    ).reset_index(drop=True)


def build_quality_rejections(price_metrics: pd.DataFrame, verified: pd.DataFrame) -> pd.DataFrame:
    price_rejected = price_metrics.loc[~price_metrics["Price Stage Pass"].fillna(False)].copy()
    price_rejected["Rejection Stage"] = "Price and momentum"
    price_rejected["Rejection Reasons"] = price_rejected["Price Rejection Reasons"]
    quality_rejected = verified.loc[~verified["Final Pass"].fillna(False)].copy()
    quality_rejected["Rejection Stage"] = "Live verification"
    quality_rejected["Rejection Reasons"] = quality_rejected["Quality Rejection Reasons"]
    columns = list(dict.fromkeys([*price_metrics.columns, *verified.columns, "Rejection Stage", "Rejection Reasons"]))
    return pd.concat(
        [price_rejected.reindex(columns=columns), quality_rejected.reindex(columns=columns)],
        ignore_index=True,
    )


def _number(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(parsed) else float(parsed)


def _quarter_end(period: str) -> date:
    timestamp = pd.to_datetime(period.title(), format="%b %Y") + pd.offsets.MonthEnd(0)
    return timestamp.date()


def _health(
    source: str,
    status: str,
    data_date: date,
    location: str,
    rows: int,
    message: str = "",
) -> dict[str, Any]:
    return {
        "Source": source,
        "Status": status,
        "Data Date": data_date.isoformat(),
        "Rows": rows,
        "Location": location,
        "Message": message,
    }


def _date_from_cache_name(path: Path) -> date:
    match = re.search(r"(\d{8})", path.stem)
    return pd.to_datetime(match.group(1), format="%Y%m%d").date() if match else date.today()
