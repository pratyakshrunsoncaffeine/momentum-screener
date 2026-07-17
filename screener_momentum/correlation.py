from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    source: str
    identifier: str
    kind: str
    native_frequency: str = "Daily"


FACTOR_CATALOG: dict[str, FactorDefinition] = {
    "Brent Crude": FactorDefinition("Brent Crude", "Yahoo Finance", "BZ=F", "price"),
    "Gold": FactorDefinition("Gold", "Yahoo Finance", "GC=F", "price"),
    "US 10Y Yield": FactorDefinition("US 10Y Yield", "Yahoo Finance", "^TNX", "yield"),
    "India 10Y Yield": FactorDefinition(
        "India 10Y Yield",
        "FRED/OECD",
        "INDIRLTLT01STM",
        "yield",
        native_frequency="Monthly",
    ),
    "USD/INR": FactorDefinition("USD/INR", "Yahoo Finance", "INR=X", "price"),
}

FREQUENCY_RULES = {
    "Daily": None,
    "Weekly": "W-FRI",
    "Monthly": "ME",
}

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
NSE_EQUITY_SERIES_PRIORITY = {
    "EQ": 0,
    "BE": 1,
    "BZ": 2,
    "SM": 3,
    "ST": 4,
    "MT": 5,
    "RR": 6,
}


class MacroFactorProvider:
    """Download macro factor levels and retain local source-level checkpoints."""

    def __init__(self, cache_dir: str | Path, timeout: float = 30.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def fetch_many(
        self,
        factors: list[str] | tuple[str, ...],
        start_date: date,
        end_date: date,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
        levels: dict[str, pd.Series] = {}
        health_rows: list[dict[str, object]] = []
        total = len(factors)
        for position, factor_name in enumerate(factors, start=1):
            definition = FACTOR_CATALOG[factor_name]
            if progress_callback:
                progress_callback(position - 1, total, f"Downloading {factor_name}")
            try:
                series = self._fetch(definition, start_date, end_date)
                status = "downloaded"
                message = ""
            except Exception as exc:
                series = self._load_cache(definition, start_date, end_date)
                status = "cached" if not series.empty else "failed"
                message = str(exc)
            if not series.empty:
                levels[factor_name] = series
            health_rows.append(
                {
                    "Factor": factor_name,
                    "Source": definition.source,
                    "Identifier": definition.identifier,
                    "Native Frequency": definition.native_frequency,
                    "Status": status,
                    "Rows": int(series.shape[0]),
                    "First Date": series.index.min().date().isoformat() if not series.empty else "",
                    "Last Date": series.index.max().date().isoformat() if not series.empty else "",
                    "Message": message,
                }
            )
            if progress_callback:
                progress_callback(position, total, f"Loaded {position} of {total} factors")
        return levels, pd.DataFrame(health_rows)

    def _fetch(self, definition: FactorDefinition, start_date: date, end_date: date) -> pd.Series:
        if definition.source == "Yahoo Finance":
            series = self._fetch_yahoo(definition.identifier, start_date, end_date)
        else:
            series = self._fetch_fred(definition.identifier, start_date, end_date)
        if series.empty:
            raise RuntimeError(f"{definition.name} returned no observations.")
        self._save_cache(definition, series)
        return series

    @staticmethod
    def _fetch_yahoo(identifier: str, start_date: date, end_date: date) -> pd.Series:
        data = yf.download(
            identifier,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data.empty:
            return pd.Series(dtype=float)
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                close = data.xs("Close", axis=1, level=0).iloc[:, 0]
            elif "Close" in data.columns.get_level_values(-1):
                close = data.xs("Close", axis=1, level=-1).iloc[:, 0]
            else:
                return pd.Series(dtype=float)
        else:
            close = data["Close"]
        return _clean_series(close, start_date, end_date)

    def _fetch_fred(self, identifier: str, start_date: date, end_date: date) -> pd.Series:
        response = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={
                "id": identifier,
                "cosd": start_date.isoformat(),
                "coed": end_date.isoformat(),
            },
            headers={"User-Agent": "momentum-screener/1.0"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text))
        if frame.shape[1] < 2:
            return pd.Series(dtype=float)
        series = pd.Series(
            pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame.iloc[:, 0], errors="coerce"),
            name=identifier,
        )
        return _clean_series(series, start_date, end_date)

    def _cache_path(self, definition: FactorDefinition) -> Path:
        slug = "".join(character.lower() if character.isalnum() else "_" for character in definition.name)
        return self.cache_dir / f"{slug}.csv"

    def _save_cache(self, definition: FactorDefinition, series: pd.Series) -> None:
        frame = series.rename("Value").rename_axis("Date").reset_index()
        frame.to_csv(self._cache_path(definition), index=False)

    def _load_cache(self, definition: FactorDefinition, start_date: date, end_date: date) -> pd.Series:
        path = self._cache_path(definition)
        if not path.exists() or path.stat().st_size == 0:
            return pd.Series(dtype=float)
        frame = pd.read_csv(path)
        if not {"Date", "Value"}.issubset(frame.columns):
            return pd.Series(dtype=float)
        series = pd.Series(
            pd.to_numeric(frame["Value"], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame["Date"], errors="coerce"),
            name=definition.identifier,
        )
        return _clean_series(series, start_date, end_date)


class NseHistoricalEquityProvider:
    """Best-effort official NSE fallback for symbols absent from Yahoo Finance."""

    def __init__(self, cache_dir: str | Path, timeout: float = 25.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def fetch_many(
        self,
        tickers: list[str] | tuple[str, ...],
        start_date: date,
        end_date: date,
        frequency: str = "Weekly",
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        requested = [str(ticker).upper().removesuffix(".NS") for ticker in tickers]
        series_by_ticker: dict[str, pd.Series] = {}
        pending: set[str] = set()
        for ticker in requested:
            cached = self._load_cache(ticker, start_date, end_date)
            if not cached.empty and cached.index.min() <= pd.Timestamp(start_date + timedelta(days=7)):
                series_by_ticker[ticker] = cached
            else:
                pending.add(ticker)

        target_groups = _nse_archive_target_groups(start_date, end_date, frequency)
        total = len(target_groups)
        report_errors: list[str] = []
        if pending:
            for position, candidates in enumerate(target_groups, start=1):
                if progress_callback:
                    progress_callback(
                        position - 1,
                        total,
                        f"Checking official NSE cash archive near {candidates[0].isoformat()}",
                    )
                report = pd.DataFrame()
                report_date: date | None = None
                for candidate in candidates:
                    try:
                        report = self._load_archive_date(candidate)
                    except Exception as exc:
                        report_errors.append(f"{candidate.isoformat()}: {exc}")
                        continue
                    if not report.empty:
                        report_date = candidate
                        break
                if report_date is not None:
                    available = report.loc[report["Ticker"].isin(pending)]
                    for row in available.to_dict("records"):
                        ticker = str(row["Ticker"])
                        existing = series_by_ticker.get(ticker, pd.Series(dtype=float))
                        addition = pd.Series(
                            [row["Close"]],
                            index=pd.DatetimeIndex([pd.Timestamp(report_date)]),
                            name=f"{ticker}.NS",
                        )
                        series_by_ticker[ticker] = pd.concat([existing, addition])
                if progress_callback:
                    progress_callback(position, total, f"Processed {position:,} of {total:,} NSE archive periods")

        series_list: list[pd.Series] = []
        health_rows: list[dict[str, object]] = []
        for ticker in requested:
            series = series_by_ticker.get(ticker, pd.Series(dtype=float))
            if not series.empty:
                series = series.loc[~series.index.duplicated(keep="last")].sort_index()
                series.name = f"{ticker}.NS"
                self._save_cache(ticker, series)
                series_list.append(series)
                status = "cached" if ticker not in pending else "archive fallback"
                message = ""
            else:
                status = "failed"
                message = (
                    "No EQ-series rows in the official NSE archive files checked."
                    if not report_errors
                    else f"NSE archive errors encountered: {report_errors[-1]}"
                )
            health_rows.append(
                {
                    "Ticker": ticker,
                    "YFinance Ticker": f"{ticker}.NS",
                    "Price Source": "NSE India" if not series.empty else "Unavailable",
                    "Price Basis": "Raw NSE close; not corporate-action adjusted" if not series.empty else "",
                    "Status": status,
                    "Rows": int(series.shape[0]),
                    "First Date": series.index.min().date().isoformat() if not series.empty else "",
                    "Last Date": series.index.max().date().isoformat() if not series.empty else "",
                    "Message": message,
                }
            )
        prices = pd.concat(series_list, axis=1).sort_index() if series_list else pd.DataFrame()
        return prices, pd.DataFrame(health_rows)

    def _load_archive_date(self, trade_date: date) -> pd.DataFrame:
        report_dir = self.cache_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"{trade_date:%Y%m%d}.csv"
        if path.exists() and path.stat().st_size > 0:
            frame = pd.read_csv(path)
        else:
            response = requests.get(
                "https://nsearchives.nseindia.com/products/content/"
                f"sec_bhavdata_full_{trade_date:%d%m%Y}.csv",
                headers=NSE_HEADERS,
                timeout=self.timeout,
            )
            if response.status_code in {403, 404}:
                return pd.DataFrame()
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
            frame.to_csv(path, index=False)
            time.sleep(0.05)
        frame.columns = [str(column).strip().upper() for column in frame.columns]
        required = {"SYMBOL", "SERIES", "CLOSE_PRICE"}
        if not required.issubset(frame.columns):
            return pd.DataFrame()
        result = frame.copy()
        result["Series"] = result["SERIES"].astype(str).str.strip().str.upper()
        result = result.loc[result["Series"].isin(NSE_EQUITY_SERIES_PRIORITY)].copy()
        result["Ticker"] = result["SYMBOL"].astype(str).str.strip().str.upper()
        result["Close"] = pd.to_numeric(result["CLOSE_PRICE"], errors="coerce")
        result["Series Priority"] = result["Series"].map(NSE_EQUITY_SERIES_PRIORITY)
        result = result.sort_values(["Ticker", "Series Priority"])
        return result[["Ticker", "Close"]].dropna(subset=["Ticker", "Close"]).drop_duplicates("Ticker", keep="first")

    def fetch_symbol(self, ticker: str, start_date: date, end_date: date) -> tuple[pd.Series, str]:
        cached = self._load_cache(ticker, start_date, end_date)
        if not cached.empty and cached.index.min() <= pd.Timestamp(start_date + timedelta(days=7)):
            return cached, "cached"

        session = requests.Session()
        session.headers.update(NSE_HEADERS)
        session.get("https://www.nseindia.com/", timeout=self.timeout)
        frames: list[pd.DataFrame] = []
        segment_start = start_date
        while segment_start <= end_date:
            segment_end = min(segment_start + timedelta(days=364), end_date)
            request_options = (
                (
                    "https://www.nseindia.com/api/historical/securityArchives",
                    {
                        "symbol": ticker,
                        "series": "EQ",
                        "dataType": "priceVolumeDeliverable",
                        "from": segment_start.strftime("%d-%m-%Y"),
                        "to": segment_end.strftime("%d-%m-%Y"),
                    },
                ),
                (
                    "https://www.nseindia.com/api/historical/cm/equity",
                    {
                        "symbol": ticker,
                        "series": '["EQ"]',
                        "from": segment_start.strftime("%d-%m-%Y"),
                        "to": segment_end.strftime("%d-%m-%Y"),
                    },
                ),
            )
            records: list[dict[str, object]] = []
            last_error: Exception | None = None
            for url, params in request_options:
                response = session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )
                if response.status_code in {401, 403, 429}:
                    session.get("https://www.nseindia.com/", timeout=self.timeout)
                    response = session.get(url, params=params, timeout=self.timeout)
                try:
                    response.raise_for_status()
                    payload = response.json()
                    records = payload.get("data", payload if isinstance(payload, list) else [])
                    if records:
                        break
                except Exception as exc:
                    last_error = exc
            if not records and last_error is not None:
                raise last_error
            if records:
                frames.append(pd.DataFrame(records))
            segment_start = segment_end + timedelta(days=1)

        series = parse_nse_equity_history(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
        if series.empty:
            raise RuntimeError("No EQ-series close history returned by NSE.")
        self._save_cache(ticker, series)
        return _clean_series(series, start_date, end_date), "downloaded"

    def _cache_path(self, ticker: str) -> Path:
        safe = "".join(character if character.isalnum() or character in "-_&" else "_" for character in ticker)
        return self.cache_dir / f"{safe}.csv"

    def _save_cache(self, ticker: str, series: pd.Series) -> None:
        series.rename("Close").rename_axis("Date").reset_index().to_csv(self._cache_path(ticker), index=False)

    def _load_cache(self, ticker: str, start_date: date, end_date: date) -> pd.Series:
        path = self._cache_path(ticker)
        if not path.exists() or path.stat().st_size == 0:
            return pd.Series(dtype=float)
        frame = pd.read_csv(path)
        if not {"Date", "Close"}.issubset(frame.columns):
            return pd.Series(dtype=float)
        series = pd.Series(
            pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
            index=pd.to_datetime(frame["Date"], errors="coerce"),
            name=f"{ticker}.NS",
        )
        return _clean_series(series, start_date, end_date)


def parse_nse_equity_history(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    date_column = next(
        (column for column in ("CH_TIMESTAMP", "TIMESTAMP", "mTIMESTAMP", "Date", "DATE") if column in frame),
        None,
    )
    close_column = next(
        (
            column
            for column in ("CH_CLOSING_PRICE", "CLOSE", "Close", "close", "CH_LAST_TRADED_PRICE")
            if column in frame
        ),
        None,
    )
    if date_column is None or close_column is None:
        return pd.Series(dtype=float)
    dates = pd.to_datetime(frame[date_column], errors="coerce", dayfirst=False)
    if dates.isna().all():
        dates = pd.to_datetime(frame[date_column], errors="coerce", dayfirst=True)
    result = pd.Series(pd.to_numeric(frame[close_column], errors="coerce").to_numpy(), index=dates)
    result = result.loc[~result.index.isna()].dropna()
    result.index = result.index.normalize()
    return result.loc[~result.index.duplicated(keep="last")].sort_index()


def _nse_archive_target_groups(start_date: date, end_date: date, frequency: str) -> list[list[date]]:
    if frequency not in FREQUENCY_RULES:
        raise ValueError(f"Unsupported frequency: {frequency}")
    if frequency == "Daily":
        targets = [item.date() for item in pd.bdate_range(start_date, end_date)]
    else:
        rule = "W-FRI" if frequency == "Weekly" else "ME"
        targets = [item.date() for item in pd.date_range(start_date, end_date, freq=rule)]
        if not targets or targets[-1] < end_date:
            targets.append(end_date)
    groups: list[list[date]] = []
    for target in targets:
        candidates = [
            item.date()
            for item in pd.bdate_range(end=target, periods=5 if frequency != "Daily" else 1)[::-1]
            if start_date <= item.date() <= end_date
        ]
        if candidates:
            groups.append(candidates)
    return groups

def _clean_series(series: pd.Series, start_date: date, end_date: date) -> pd.Series:
    cleaned = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(),
        index=pd.to_datetime(series.index, errors="coerce"),
        name=series.name,
    )
    cleaned = cleaned.loc[~cleaned.index.isna()].dropna()
    if cleaned.index.tz is not None:
        cleaned.index = cleaned.index.tz_localize(None)
    cleaned.index = cleaned.index.normalize()
    cleaned = cleaned.loc[~cleaned.index.duplicated(keep="last")].sort_index()
    return cleaned.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]


def effective_frequency(factor_name: str, requested_frequency: str) -> str:
    definition = FACTOR_CATALOG[factor_name]
    if definition.native_frequency == "Monthly" and requested_frequency != "Monthly":
        return "Monthly"
    return requested_frequency


def resample_levels(values: pd.Series | pd.DataFrame, frequency: str) -> pd.Series | pd.DataFrame:
    if frequency not in FREQUENCY_RULES:
        raise ValueError(f"Unsupported frequency: {frequency}")
    cleaned = values.sort_index()
    rule = FREQUENCY_RULES[frequency]
    if rule is None:
        return cleaned
    return cleaned.resample(rule).last()


def stock_returns(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    levels = resample_levels(prices.apply(pd.to_numeric, errors="coerce"), frequency)
    return levels.pct_change(fill_method=None) * 100.0


def factor_changes(levels: pd.Series, definition: FactorDefinition, frequency: str) -> pd.Series:
    sampled = resample_levels(levels, frequency)
    if definition.kind == "yield":
        changed = sampled.diff() * 100.0
    else:
        changed = sampled.pct_change(fill_method=None) * 100.0
    return changed.rename(definition.name)


def calculate_correlations(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    factor_levels: dict[str, pd.Series],
    requested_frequency: str = "Weekly",
    method: str = "pearson",
    relation: str = "Same period",
    min_observations: int = 40,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate stock-factor relationships using only the supplied as-of history."""
    method = method.lower()
    if method not in {"pearson", "spearman"}:
        raise ValueError("Correlation method must be Pearson or Spearman.")
    if relation not in {"Same period", "Next stock period"}:
        raise ValueError("Relation must be Same period or Next stock period.")
    if prices.empty:
        raise ValueError("No stock price history is available.")

    metadata = universe.set_index("YFinance Ticker").to_dict("index")
    returns_by_frequency: dict[str, pd.DataFrame] = {}
    result_rows: list[dict[str, object]] = []
    factor_rows: list[pd.DataFrame] = []
    total = len(factor_levels) * max(len(prices.columns), 1)
    completed = 0

    for factor_name, levels in factor_levels.items():
        definition = FACTOR_CATALOG[factor_name]
        frequency = effective_frequency(factor_name, requested_frequency)
        if frequency not in returns_by_frequency:
            returns_by_frequency[frequency] = stock_returns(prices, frequency)
        returns = returns_by_frequency[frequency]
        changes = factor_changes(levels, definition, frequency).dropna()

        factor_output = pd.DataFrame(
            {
                "Date": changes.index,
                "Factor": factor_name,
                "Level": resample_levels(levels, frequency).reindex(changes.index).to_numpy(),
                "Change": changes.to_numpy(),
                "Change Unit": "Basis points" if definition.kind == "yield" else "Percent",
                "Effective Frequency": frequency,
            }
        )
        factor_rows.append(factor_output)

        for yahoo_ticker in prices.columns:
            stock = returns[yahoo_ticker].rename("Stock Return %")
            if relation == "Next stock period":
                stock = stock.shift(-1)
            aligned = pd.concat([stock, changes.rename("Factor Change")], axis=1).dropna()
            observations = int(aligned.shape[0])
            if observations < int(min_observations):
                completed += 1
                continue
            if aligned["Stock Return %"].nunique() < 2 or aligned["Factor Change"].nunique() < 2:
                completed += 1
                continue
            correlation = aligned["Stock Return %"].corr(aligned["Factor Change"], method=method)
            if pd.isna(correlation):
                completed += 1
                continue
            variance = float(aligned["Factor Change"].var(ddof=1))
            covariance = float(aligned["Stock Return %"].cov(aligned["Factor Change"]))
            raw_sensitivity = covariance / variance if variance > 0 else np.nan
            if definition.kind == "yield":
                sensitivity = raw_sensitivity * 100.0
                sensitivity_unit = "Stock return % per 100 bp yield move"
            else:
                sensitivity = raw_sensitivity
                sensitivity_unit = "Stock return % per 1% factor move"

            rises = aligned.loc[aligned["Factor Change"] > 0, "Stock Return %"]
            falls = aligned.loc[aligned["Factor Change"] < 0, "Stock Return %"]
            upper_cutoff = aligned["Factor Change"].quantile(0.80)
            lower_cutoff = aligned["Factor Change"].quantile(0.20)
            strong_rises = aligned.loc[aligned["Factor Change"] >= upper_cutoff, "Stock Return %"]
            strong_falls = aligned.loc[aligned["Factor Change"] <= lower_cutoff, "Stock Return %"]
            item = metadata.get(str(yahoo_ticker).upper(), {})
            result_rows.append(
                {
                    "Factor": factor_name,
                    "Ticker": item.get("Ticker", str(yahoo_ticker).removesuffix(".NS")),
                    "Name": item.get("Name", ""),
                    "Industry": item.get("Industry", ""),
                    "YFinance Ticker": str(yahoo_ticker).upper(),
                    "Correlation": round(float(correlation), 4),
                    "R Squared": round(float(correlation) ** 2, 4),
                    "Sensitivity": round(float(sensitivity), 4) if pd.notna(sensitivity) else np.nan,
                    "Sensitivity Unit": sensitivity_unit,
                    "Avg Return When Factor Rises %": _mean_or_nan(rises),
                    "Avg Return When Factor Falls %": _mean_or_nan(falls),
                    "Positive Hit Rate When Factor Rises %": _hit_rate(rises),
                    "Positive Hit Rate When Factor Falls %": _hit_rate(falls),
                    "Strong Rise Avg Return %": _mean_or_nan(strong_rises),
                    "Strong Fall Avg Return %": _mean_or_nan(strong_falls),
                    "Observations": observations,
                    "Effective Frequency": frequency,
                    "Relation": relation,
                    "Method": method.title(),
                    "Factor Change Unit": "Basis points" if definition.kind == "yield" else "Percent",
                    "Data Start": aligned.index.min().date().isoformat(),
                    "Data End": aligned.index.max().date().isoformat(),
                }
            )
            completed += 1
            if progress_callback and (completed == total or completed % 250 == 0):
                progress_callback(completed, total, f"Calculated {completed:,} of {total:,} stock-factor pairs")

    results = pd.DataFrame(result_rows)
    if not results.empty:
        results = results.sort_values(["Factor", "Correlation"], ascending=[True, False]).reset_index(drop=True)
        results["Positive Rank"] = results.groupby("Factor")["Correlation"].rank(
            method="first", ascending=False
        ).astype(int)
        results["Inverse Rank"] = results.groupby("Factor")["Correlation"].rank(
            method="first", ascending=True
        ).astype(int)
    factor_history = pd.concat(factor_rows, ignore_index=True) if factor_rows else pd.DataFrame()
    return results, factor_history


def add_ridge_regression(
    results: pd.DataFrame,
    prices: pd.DataFrame,
    factor_history: pd.DataFrame,
    relation: str = "Same period",
    alpha: float = 1.0,
    min_observations: int = 24,
) -> pd.DataFrame:
    """Attach standardized multivariate ridge coefficients to stock-factor results."""
    if results.empty or prices.empty or factor_history.empty:
        return results.copy()
    if relation not in {"Same period", "Next stock period"}:
        raise ValueError("Relation must be Same period or Next stock period.")
    ridge_alpha = max(float(alpha), 0.0)
    history = factor_history.copy()
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history["Change"] = pd.to_numeric(history["Change"], errors="coerce")
    history = history.dropna(subset=["Date", "Factor", "Change", "Effective Frequency"])

    regression_rows: list[dict[str, object]] = []
    for frequency, frequency_history in history.groupby("Effective Frequency"):
        factor_frame = (
            frequency_history.pivot_table(index="Date", columns="Factor", values="Change", aggfunc="last")
            .sort_index()
        )
        if factor_frame.empty:
            continue
        available_factors = [factor for factor in factor_frame.columns if factor_frame[factor].nunique() > 1]
        factor_frame = factor_frame[available_factors]
        if factor_frame.empty:
            continue
        returns = stock_returns(prices, str(frequency))
        for yahoo_ticker in returns.columns:
            stock = returns[yahoo_ticker].rename("Stock Return %")
            if relation == "Next stock period":
                stock = stock.shift(-1)
            aligned = pd.concat([stock, factor_frame], axis=1).dropna()
            if aligned.shape[0] < int(min_observations) or aligned["Stock Return %"].nunique() < 2:
                continue
            x = aligned[available_factors].astype(float)
            y = aligned["Stock Return %"].astype(float)
            x_std = x.std(ddof=0).replace(0, np.nan)
            valid_factors = x_std.dropna().index.tolist()
            if not valid_factors:
                continue
            x_scaled = ((x[valid_factors] - x[valid_factors].mean()) / x_std[valid_factors]).to_numpy()
            y_std = float(y.std(ddof=0))
            if not np.isfinite(y_std) or y_std == 0:
                continue
            y_scaled = ((y - y.mean()) / y_std).to_numpy()
            penalty = np.eye(len(valid_factors), dtype=float) * ridge_alpha
            try:
                coefficients = np.linalg.solve(x_scaled.T @ x_scaled + penalty, x_scaled.T @ y_scaled)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.pinv(x_scaled.T @ x_scaled + penalty) @ x_scaled.T @ y_scaled
            predicted = x_scaled @ coefficients
            residual_sum = float(np.square(y_scaled - predicted).sum())
            total_sum = float(np.square(y_scaled - y_scaled.mean()).sum())
            model_r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
            for factor_name, coefficient in zip(valid_factors, coefficients):
                regression_rows.append(
                    {
                        "YFinance Ticker": str(yahoo_ticker).upper(),
                        "Factor": factor_name,
                        "Ridge Coefficient": round(float(coefficient), 6),
                        "Ridge Model R Squared": round(float(model_r_squared), 4),
                        "Ridge Observations": int(aligned.shape[0]),
                        "Ridge Alpha": ridge_alpha,
                        "Ridge Factors": len(valid_factors),
                    }
                )

    regression = pd.DataFrame(regression_rows)
    base = results.drop(
        columns=[
            "Ridge Coefficient",
            "Ridge Model R Squared",
            "Ridge Observations",
            "Ridge Alpha",
            "Ridge Factors",
            "Ridge Positive Rank",
            "Ridge Inverse Rank",
        ],
        errors="ignore",
    )
    if regression.empty:
        return base
    merged = base.merge(regression, on=["YFinance Ticker", "Factor"], how="left")
    merged["Ridge Positive Rank"] = (
        merged.groupby("Factor")["Ridge Coefficient"].rank(method="first", ascending=False, na_option="bottom")
    )
    merged["Ridge Inverse Rank"] = (
        merged.groupby("Factor")["Ridge Coefficient"].rank(method="first", ascending=True, na_option="bottom")
    )
    return merged


def select_correlation_leaders(
    results: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()
    positive = (
        results.loc[results["Correlation"] > 0]
        .sort_values(["Factor", "Correlation"], ascending=[True, False])
        .groupby("Factor", as_index=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    inverse = (
        results.loc[results["Correlation"] < 0]
        .sort_values(["Factor", "Correlation"], ascending=[True, True])
        .groupby("Factor", as_index=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return positive, inverse


def select_relationship_leaders(
    results: pd.DataFrame,
    top_n: int = 5,
    ranking_metric: str = "Ridge Coefficient",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ranking_metric not in results.columns or results[ranking_metric].notna().sum() == 0:
        ranking_metric = "Correlation"
    eligible = results.dropna(subset=[ranking_metric]).copy()
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame()
    positive = (
        eligible.loc[eligible[ranking_metric] > 0]
        .sort_values(["Factor", ranking_metric], ascending=[True, False])
        .groupby("Factor", as_index=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    inverse = (
        eligible.loc[eligible[ranking_metric] < 0]
        .sort_values(["Factor", ranking_metric], ascending=[True, True])
        .groupby("Factor", as_index=False, group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return positive, inverse


def normalized_factor_stock_performance(
    prices: pd.DataFrame,
    factor_history: pd.DataFrame,
    factor_name: str,
    tickers: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Normalize a factor level and selected stock closes to a common starting value of 100."""
    if prices.empty or factor_history.empty or not tickers:
        return pd.DataFrame()
    history = factor_history.loc[factor_history["Factor"].astype(str).eq(factor_name)].copy()
    if history.empty:
        return pd.DataFrame()
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history["Level"] = pd.to_numeric(history["Level"], errors="coerce")
    history = history.dropna(subset=["Date", "Level"]).sort_values("Date")
    if history.empty:
        return pd.DataFrame()
    frequency = str(history["Effective Frequency"].dropna().iloc[-1])

    stock_prices = prices.copy()
    if "Date" in stock_prices.columns:
        stock_prices["Date"] = pd.to_datetime(stock_prices["Date"], errors="coerce")
        stock_prices = stock_prices.dropna(subset=["Date"]).set_index("Date")
    else:
        stock_prices.index = pd.to_datetime(stock_prices.index, errors="coerce")
        stock_prices = stock_prices.loc[~stock_prices.index.isna()]
    yahoo_tickers = [ticker if str(ticker).upper().endswith(".NS") else f"{str(ticker).upper()}.NS" for ticker in tickers]
    available = [ticker for ticker in yahoo_tickers if ticker in stock_prices.columns]
    if not available:
        return pd.DataFrame()
    sampled = resample_levels(stock_prices[available].apply(pd.to_numeric, errors="coerce"), frequency)
    factor_series = history.set_index("Date")["Level"].rename(factor_name)
    combined = pd.concat([factor_series, sampled], axis=1).sort_index()
    first_dates = [combined[column].first_valid_index() for column in combined if combined[column].notna().any()]
    if not first_dates:
        return pd.DataFrame()
    common_start = max(first_dates)
    combined = combined.loc[common_start:]
    normalized = pd.DataFrame(index=combined.index)
    for column in combined:
        series = combined[column].dropna()
        if series.empty or float(series.iloc[0]) == 0:
            continue
        normalized[column] = combined[column] / float(series.iloc[0]) * 100.0
    if normalized.empty:
        return pd.DataFrame()
    normalized = normalized.rename(columns={ticker: ticker.removesuffix(".NS") for ticker in available})
    output = normalized.rename_axis("Date").reset_index().melt(
        id_vars="Date",
        var_name="Series",
        value_name="Normalized Value",
    )
    output["Type"] = np.where(output["Series"].eq(factor_name), "Factor", "Stock")
    return output.dropna(subset=["Normalized Value"])


def correlation_matrix(results: pd.DataFrame, stocks_per_factor: int = 5) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    positive, inverse = select_correlation_leaders(results, stocks_per_factor)
    tickers = pd.concat([positive["Ticker"], inverse["Ticker"]], ignore_index=True).drop_duplicates()
    selected = results.loc[results["Ticker"].isin(tickers)]
    return selected.pivot_table(index="Ticker", columns="Factor", values="Correlation", aggfunc="first")


def _mean_or_nan(values: pd.Series) -> float:
    return round(float(values.mean()), 4) if not values.empty else np.nan


def _hit_rate(values: pd.Series) -> float:
    return round(float((values > 0).mean() * 100.0), 2) if not values.empty else np.nan
