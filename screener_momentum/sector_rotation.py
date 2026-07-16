from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ProgressCallback = Callable[[int, int, str], None]
NSE_HISTORICAL_URL = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"
NSE_HISTORICAL_PAGE = "https://www.niftyindices.com/reports/historical-data"
BENCHMARK_INDEX = "Nifty 50"
RETURN_PERIODS = {"1W Return %": 5, "1M Return %": 21, "3M Return %": 63, "6M Return %": 126}


class NseSectorIndexProvider:
    """Fetch and cache official daily NSE Indices price-index history."""

    def __init__(self, cache_dir: str | Path, timeout: int = 25, retries: int = 3) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
                ),
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Referer": NSE_HISTORICAL_PAGE,
                "Origin": "https://www.niftyindices.com",
            }
        )

    def fetch_many(
        self,
        indices: Iterable[str],
        start_date: date,
        end_date: date,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if start_date > end_date:
            raise ValueError("Sector history start date must be on or before the end date.")
        unique_indices = list(dict.fromkeys(str(item).strip() for item in indices if str(item).strip()))
        frames: list[pd.DataFrame] = []
        health_rows: list[dict[str, object]] = []
        total = len(unique_indices)
        for position, index_name in enumerate(unique_indices, start=1):
            if progress_callback:
                progress_callback(position - 1, total, f"Fetching official NSE history for {index_name}")
            try:
                frame = self.fetch_index(index_name, start_date, end_date)
                status = "downloaded"
                note = "official NSE Indices endpoint"
            except Exception as exc:
                frame = self.load_cached_index(index_name, start_date, end_date)
                if frame.empty:
                    status = "failed"
                    note = f"{type(exc).__name__}: {exc}"
                else:
                    status = "cached fallback"
                    note = f"NSE refresh failed; cached NSE data used: {type(exc).__name__}"
            if not frame.empty:
                frames.append(frame)
            health_rows.append(
                {
                    "Index": index_name,
                    "Status": status,
                    "Rows": len(frame),
                    "First Date": frame["Date"].min() if not frame.empty else pd.NaT,
                    "Last Date": frame["Date"].max() if not frame.empty else pd.NaT,
                    "Notes": note,
                }
            )
            if progress_callback:
                progress_callback(position, total, f"Processed {position:,} of {total:,} NSE indices")
        prices = pd.concat(frames, ignore_index=True) if frames else empty_sector_prices()
        return prices, pd.DataFrame(health_rows)

    def fetch_index(self, index_name: str, start_date: date, end_date: date) -> pd.DataFrame:
        payload = {
            "cinfo": json.dumps(
                {
                    "name": index_name.upper(),
                    "startDate": start_date.strftime("%d-%b-%Y"),
                    "endDate": end_date.strftime("%d-%b-%Y"),
                    "indexName": index_name,
                }
            )
        }
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.post(NSE_HISTORICAL_URL, json=payload, timeout=self.timeout)
                response.raise_for_status()
                frame = parse_nse_historical_response(response.json(), requested_index=index_name)
                frame = frame[frame["Date"].dt.date.between(start_date, end_date)].reset_index(drop=True)
                if frame.empty:
                    raise ValueError(f"NSE returned no historical rows for {index_name}.")
                self._save_cache(index_name, frame)
                return frame
            except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(float(attempt))
        raise RuntimeError(f"Could not fetch {index_name} from NSE Indices: {last_error}")

    def load_cached_index(self, index_name: str, start_date: date, end_date: date) -> pd.DataFrame:
        path = self._cache_path(index_name)
        if not path.exists() or path.stat().st_size == 0:
            return empty_sector_prices()
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return empty_sector_prices()
        frame["Date"] = pd.to_datetime(frame.get("Date"), errors="coerce")
        frame["Close"] = pd.to_numeric(frame.get("Close"), errors="coerce")
        frame = frame[frame["Date"].dt.date.between(start_date, end_date)].copy()
        return normalize_sector_prices(frame, requested_index=index_name)

    def _save_cache(self, index_name: str, frame: pd.DataFrame) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(index_name)
        if path.exists() and path.stat().st_size > 0:
            try:
                existing = pd.read_csv(path)
                existing["Date"] = pd.to_datetime(existing.get("Date"), errors="coerce")
                frame = pd.concat([existing, frame], ignore_index=True)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pass
        normalize_sector_prices(frame, requested_index=index_name).to_csv(path, index=False)

    def _cache_path(self, index_name: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "_", index_name.lower()).strip("_")
        return self.cache_dir / f"{slug}.csv"


def parse_nse_historical_response(payload: object, requested_index: str) -> pd.DataFrame:
    data = payload
    if isinstance(data, dict) and "d" in data:
        data = data["d"]
    if isinstance(data, str):
        data = json.loads(data)
        if isinstance(data, dict) and "d" in data:
            data = data["d"]
            if isinstance(data, str):
                data = json.loads(data)
    if not isinstance(data, list):
        raise ValueError("Unexpected NSE historical response shape.")
    rows = pd.DataFrame(data)
    if rows.empty:
        return empty_sector_prices()
    date_values = _first_column(rows, "HistoricalDate", "Date", "date")
    close_values = _first_column(rows, "CLOSE", "Close", "close")
    index_values = _first_column(rows, "INDEX_NAME", "Index Name", "Index", default=requested_index)
    frame = pd.DataFrame({"Index": index_values, "Date": date_values, "Close": close_values})
    return normalize_sector_prices(frame, requested_index=requested_index)


def normalize_sector_prices(frame: pd.DataFrame, requested_index: str | None = None) -> pd.DataFrame:
    result = frame.copy()
    if "Index" not in result.columns:
        result["Index"] = requested_index
    if requested_index:
        # NSE occasionally changes only the response label's casing mid-series.
        # The request name is the stable identity for every returned row.
        result["Index"] = requested_index
    result["Index"] = result["Index"].astype(str).str.strip()
    result["Date"] = pd.to_datetime(result.get("Date"), errors="coerce")
    result["Close"] = pd.to_numeric(
        result.get("Close").astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    result = result.dropna(subset=["Index", "Date", "Close"])
    result = result[result["Close"].gt(0)].drop_duplicates(["Index", "Date"], keep="last")
    return result.sort_values(["Index", "Date"]).reset_index(drop=True)


def calculate_sector_rotation(
    prices: pd.DataFrame,
    sectors: Iterable[str],
    benchmark: str = BENCHMARK_INDEX,
) -> pd.DataFrame:
    selected = list(dict.fromkeys(str(item).strip() for item in sectors if str(item).strip()))
    normalized = normalize_sector_prices(prices)
    if normalized.empty:
        return pd.DataFrame()
    panel = normalized.pivot_table(index="Date", columns="Index", values="Close", aggfunc="last").sort_index()
    available_sectors = [item for item in selected if item in panel.columns]
    if benchmark not in panel.columns:
        raise ValueError("Official Nifty 50 benchmark history is unavailable.")
    if not available_sectors:
        return pd.DataFrame()
    panel = panel[[benchmark, *available_sectors]].dropna(how="any")
    if panel.empty:
        raise ValueError("No common NSE trading date exists across the selected sectors and Nifty 50.")

    current = _rotation_metrics_at(panel, available_sectors, benchmark, len(panel) - 1)
    if current.empty:
        return current
    current = _score_rotation(current)
    previous_position = len(panel) - 22
    if previous_position >= 0:
        previous = _score_rotation(_rotation_metrics_at(panel, available_sectors, benchmark, previous_position))
        previous_ranks = previous.set_index("Sector")["Rotation Rank"] if not previous.empty else pd.Series(dtype=float)
        current["Previous Rank"] = current["Sector"].map(previous_ranks)
        current["Rank Change"] = current["Previous Rank"] - current["Rotation Rank"]
    else:
        current["Previous Rank"] = np.nan
        current["Rank Change"] = np.nan
    current["Data Date"] = panel.index[-1].date()
    columns = [
        "Rotation Rank",
        "Sector",
        "Rotation Status",
        "Rotation Score",
        "Rank Change",
        "Previous Rank",
        "Data Date",
        *RETURN_PERIODS,
        "1W Excess vs Nifty 50 %",
        "1M Excess vs Nifty 50 %",
        "3M Excess vs Nifty 50 %",
        "6M Excess vs Nifty 50 %",
        "Relative Strength 3M %",
        "Relative Momentum %",
        "Latest Close",
    ]
    return current.reindex(columns=columns).reset_index(drop=True)


def normalized_performance(
    prices: pd.DataFrame,
    sectors: Iterable[str],
    benchmark: str = BENCHMARK_INDEX,
    trading_days: int = 126,
) -> pd.DataFrame:
    selected = [benchmark, *list(dict.fromkeys(sectors))]
    panel = normalize_sector_prices(prices).pivot_table(index="Date", columns="Index", values="Close", aggfunc="last")
    columns = [item for item in selected if item in panel.columns]
    if not columns:
        return pd.DataFrame()
    panel = panel[columns].sort_index().dropna(how="any").tail(int(trading_days) + 1)
    if panel.empty:
        return pd.DataFrame()
    normalized = panel.divide(panel.iloc[0]).multiply(100.0)
    return normalized.reset_index().melt(id_vars="Date", var_name="Index", value_name="Normalized Value")


def empty_sector_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=["Index", "Date", "Close"])


def _rotation_metrics_at(
    panel: pd.DataFrame,
    sectors: list[str],
    benchmark: str,
    position: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sector in sectors:
        row: dict[str, object] = {"Sector": sector, "Latest Close": float(panel[sector].iloc[position])}
        for label, days in RETURN_PERIODS.items():
            sector_return = _panel_return(panel[sector], position, days)
            benchmark_return = _panel_return(panel[benchmark], position, days)
            row[label] = sector_return
            row[label.replace("Return %", "Excess vs Nifty 50 %")] = _difference(sector_return, benchmark_return)
        current_excess = row.get("1M Excess vs Nifty 50 %")
        previous_sector_return = _window_return(panel[sector], position - 21, position - 42)
        previous_benchmark_return = _window_return(panel[benchmark], position - 21, position - 42)
        previous_excess = _difference(previous_sector_return, previous_benchmark_return)
        row["Relative Strength 3M %"] = row.get("3M Excess vs Nifty 50 %")
        row["Relative Momentum %"] = _difference(current_excess, previous_excess)
        strength = row["Relative Strength 3M %"]
        momentum = row["Relative Momentum %"]
        row["Rotation Status"] = _rotation_status(strength, momentum)
        rows.append(row)
    return pd.DataFrame(rows)


def _score_rotation(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    score = pd.Series(0.0, index=result.index)
    for column, weight in (
        ("1M Excess vs Nifty 50 %", 0.40),
        ("3M Excess vs Nifty 50 %", 0.35),
        ("Relative Momentum %", 0.25),
    ):
        values = pd.to_numeric(result[column], errors="coerce")
        score += values.rank(pct=True, method="average").fillna(0.0) * weight
    result["Rotation Score"] = (score * 100.0).round(2)
    result = result.sort_values(["Rotation Score", "Sector"], ascending=[False, True]).reset_index(drop=True)
    result.insert(0, "Rotation Rank", range(1, len(result) + 1))
    return result


def _rotation_status(strength: object, momentum: object) -> str:
    if pd.isna(strength) or pd.isna(momentum):
        return "Insufficient History"
    if float(strength) > 0 and float(momentum) > 0:
        return "Leading"
    if float(strength) <= 0 and float(momentum) > 0:
        return "Improving"
    if float(strength) > 0 and float(momentum) <= 0:
        return "Weakening"
    return "Lagging"


def _panel_return(series: pd.Series, position: int, days: int) -> float | None:
    return _window_return(series, position, position - days)


def _window_return(series: pd.Series, end_position: int, start_position: int) -> float | None:
    if start_position < 0 or end_position < 0 or end_position >= len(series):
        return None
    start = float(series.iloc[start_position])
    end = float(series.iloc[end_position])
    return None if start <= 0 else round((end / start - 1.0) * 100.0, 3)


def _difference(first: object, second: object) -> float | None:
    if first is None or second is None or pd.isna(first) or pd.isna(second):
        return None
    return round(float(first) - float(second), 3)


def _first_column(frame: pd.DataFrame, *names: str, default: object = None) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index)
