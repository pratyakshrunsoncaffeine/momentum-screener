from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import FundamentalThresholds


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def screen_fundamentals(
    momentum_frame: pd.DataFrame,
    thresholds: FundamentalThresholds,
    sleep_seconds: float = 0.6,
    progress_callback: Callable[[int, int, str], None] | None = None,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Scrape Screener.in fundamentals and apply the configured filters."""
    if momentum_frame.empty:
        return momentum_frame.copy()

    rows: list[dict[str, Any]] = []
    session = requests.Session()
    session.headers.update(HEADERS)
    records = momentum_frame.to_dict("records")
    total = len(records)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(records, start=1):
        symbol = str(item["Ticker"]).strip().upper()
        if progress_callback:
            progress_callback(index - 1, total, f"Scraping {symbol} ({index:,}/{total:,})")
        try:
            metrics = fetch_company_fundamentals(symbol, session=session)
            passed, reasons = passes_fundamental_filters(metrics, thresholds)
            row = {
                **item,
                **metrics,
                "Fundamental Pass": passed,
                "Fundamental Notes": "; ".join(reasons),
            }
        except Exception as exc:  # Network pages can fail per ticker; keep the run alive.
            row = {
                **item,
                "Market Cap Cr": None,
                "Quarterly Revenue Growth %": None,
                "Annual Revenue Growth %": None,
                "Promoter Holding Change %": None,
                "Fundamental Pass": False,
                "Fundamental Notes": f"Screener fetch failed: {exc}",
            }
        rows.append(row)
        if checkpoint:
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
        if progress_callback:
            progress_callback(index, total, f"Completed {index:,} of {total:,} fundamentals")
        time.sleep(sleep_seconds)

    return pd.DataFrame(rows)


def screen_fii_holdings(
    universe: pd.DataFrame,
    sleep_seconds: float = 0.6,
    progress_callback: Callable[[int, int, str], None] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """Scrape Screener.in FII holding changes for a full ticker universe."""
    return screen_institutional_holdings(
        universe,
        holder_prefix="FII",
        row_labels=("FIIs", "FII", "Foreign Institutional", "Foreign Portfolio"),
        sleep_seconds=sleep_seconds,
        progress_callback=progress_callback,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )


def screen_dii_holdings(
    universe: pd.DataFrame,
    sleep_seconds: float = 0.6,
    progress_callback: Callable[[int, int, str], None] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """Scrape Screener.in DII holding changes for a full ticker universe."""
    return screen_institutional_holdings(
        universe,
        holder_prefix="DII",
        row_labels=("DIIs", "DII", "Domestic Institutional", "Domestic Institutions"),
        sleep_seconds=sleep_seconds,
        progress_callback=progress_callback,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )


def screen_institutional_holdings(
    universe: pd.DataFrame,
    holder_prefix: str,
    row_labels: tuple[str, ...],
    sleep_seconds: float = 0.6,
    progress_callback: Callable[[int, int, str], None] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """Scrape Screener.in institutional holding changes for a full ticker universe."""
    if universe.empty:
        return universe.copy()

    prefix = holder_prefix.upper()
    rows: list[dict[str, Any]] = []
    completed_tickers: set[str] = set()
    session = requests.Session()
    session.headers.update(HEADERS)
    records = universe.to_dict("records")
    total = len(records)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        rows, completed_tickers = load_institutional_checkpoint(checkpoint, prefix)

    for index, item in enumerate(records, start=1):
        symbol = str(item["Ticker"]).strip().upper()
        if symbol in completed_tickers:
            continue
        if progress_callback:
            progress_callback(len(completed_tickers), total, f"Scraping {prefix} holding for {symbol} ({index:,}/{total:,})")
        try:
            metrics = fetch_company_institutional_holding(
                symbol,
                holder_prefix=prefix,
                row_labels=row_labels,
                session=session,
            )
            row = {
                **item,
                **metrics,
                f"{prefix} Scan Notes": "ok" if metrics.get(f"{prefix} Holding Change %") is not None else f"missing {prefix} row",
            }
        except Exception as exc:
            row = {
                **item,
                "Screener URL": screener_company_url(symbol),
                "Market Cap": None,
                "Market Cap Cr": None,
                f"{prefix} Previous Period": None,
                f"{prefix} Latest Period": None,
                f"{prefix} Previous Holding %": None,
                f"{prefix} Latest Holding %": None,
                f"{prefix} Holding Change %": None,
                f"{prefix} Scan Notes": f"Screener fetch failed: {exc}",
            }
        rows.append(row)
        completed_tickers.add(symbol)
        if checkpoint and (len(completed_tickers) == total or len(completed_tickers) % checkpoint_every == 0):
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
        if progress_callback:
            progress_callback(len(completed_tickers), total, f"Completed {len(completed_tickers):,} of {total:,} {prefix} scans")
        time.sleep(sleep_seconds)

    if checkpoint:
        pd.DataFrame(rows).to_csv(checkpoint, index=False)
    return pd.DataFrame(rows)


def load_fii_checkpoint(checkpoint: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Resume a modern FII checkpoint and retry rows that previously failed."""
    return load_institutional_checkpoint(checkpoint, "FII")


def load_institutional_checkpoint(checkpoint: Path, holder_prefix: str) -> tuple[list[dict[str, Any]], set[str]]:
    """Resume a modern institutional checkpoint and retry rows that previously failed."""
    if not checkpoint.exists() or checkpoint.stat().st_size == 0:
        return [], set()

    try:
        frame = pd.read_csv(checkpoint)
    except Exception:
        return [], set()

    prefix = holder_prefix.upper()
    required_columns = {"Ticker", "Market Cap Cr", f"{prefix} Holding Change %", f"{prefix} Scan Notes"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return [], set()

    frame = frame.drop_duplicates(subset=["Ticker"], keep="last")
    retry_mask = frame[f"{prefix} Scan Notes"].astype(str).str.startswith("Screener fetch failed", na=False)
    frame = frame[~retry_mask].copy()
    tickers = set(frame["Ticker"].astype(str).str.strip().str.upper())
    return frame.to_dict("records"), tickers


def fetch_company_fundamentals(
    symbol: str,
    session: requests.Session | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    session = session or requests.Session()
    session.headers.update(HEADERS)

    url, html = fetch_screener_html(symbol, session=session, timeout=timeout)
    soup = BeautifulSoup(html, "lxml")
    return {
        "Screener URL": url,
        "Market Cap Cr": extract_market_cap(soup),
        "Quarterly Revenue Growth %": extract_quarterly_sales_growth(soup),
        "Annual Revenue Growth %": extract_annual_sales_growth(soup),
        "Promoter Holding Change %": extract_promoter_holding_change(soup),
    }


def fetch_company_fii_holding(
    symbol: str,
    session: requests.Session | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    return fetch_company_institutional_holding(
        symbol,
        holder_prefix="FII",
        row_labels=("FIIs", "FII", "Foreign Institutional", "Foreign Portfolio"),
        session=session,
        timeout=timeout,
    )


def fetch_company_dii_holding(
    symbol: str,
    session: requests.Session | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    return fetch_company_institutional_holding(
        symbol,
        holder_prefix="DII",
        row_labels=("DIIs", "DII", "Domestic Institutional", "Domestic Institutions"),
        session=session,
        timeout=timeout,
    )


def fetch_company_institutional_holding(
    symbol: str,
    holder_prefix: str,
    row_labels: tuple[str, ...],
    session: requests.Session | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    session = session or requests.Session()
    session.headers.update(HEADERS)

    url, html = fetch_screener_html(symbol, session=session, timeout=timeout)
    soup = BeautifulSoup(html, "lxml")
    prefix = holder_prefix.upper()
    holding = extract_institutional_holding_change(soup, prefix, row_labels)
    market_cap_cr = extract_market_cap(soup)
    return {
        "Screener URL": url,
        "Market Cap": market_cap_cr * 10_000_000 if market_cap_cr is not None else None,
        "Market Cap Cr": market_cap_cr,
        **holding,
    }


def screener_company_url(symbol: str) -> str:
    return f"https://www.screener.in/company/{quote(symbol, safe='')}/consolidated/"


def fetch_screener_html(
    symbol: str,
    session: requests.Session,
    timeout: int = 20,
    max_attempts: int = 3,
) -> tuple[str, str]:
    """Fetch a Screener company page with small retries for rate limits/transient failures."""
    url = screener_company_url(symbol)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404 and url.endswith("/consolidated/"):
                fallback_url = url.replace("/consolidated/", "/")
                response = session.get(fallback_url, timeout=timeout)
                effective_url = fallback_url
            else:
                effective_url = url

            if response.status_code in RETRY_STATUS_CODES and attempt < max_attempts:
                time.sleep(2.0 * attempt)
                continue

            response.raise_for_status()
            return effective_url, response.text
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            time.sleep(2.0 * attempt)

    raise RuntimeError(f"Screener fetch failed for {symbol}: {last_exc}")


def passes_fundamental_filters(
    metrics: dict[str, Any],
    thresholds: FundamentalThresholds,
) -> tuple[bool, list[str]]:
    values = asdict(thresholds)
    checks = [
        (
            "Market Cap Cr",
            metrics.get("Market Cap Cr"),
            values["min_market_cap_cr"],
            ">=",
            "market cap",
        ),
        (
            "Quarterly Revenue Growth %",
            metrics.get("Quarterly Revenue Growth %"),
            values["min_quarterly_revenue_growth_pct"],
            ">=",
            "quarterly revenue growth",
        ),
        (
            "Annual Revenue Growth %",
            metrics.get("Annual Revenue Growth %"),
            values["min_annual_revenue_growth_pct"],
            ">=",
            "annual revenue growth",
        ),
        (
            "Promoter Holding Change %",
            abs(metrics.get("Promoter Holding Change %") or 9999),
            values["max_promoter_holding_change_pct"],
            "<=",
            "promoter holding change",
        ),
    ]

    reasons: list[str] = []
    passed = True
    for _field, actual, threshold, operator, label in checks:
        if actual is None or pd.isna(actual):
            passed = False
            reasons.append(f"missing {label}")
            continue
        check_passed = actual >= threshold if operator == ">=" else actual <= threshold
        if not check_passed:
            passed = False
            reasons.append(f"{label} {actual:.2f} fails {operator} {threshold:.2f}")
    if passed:
        reasons.append("passed")
    return passed, reasons


def extract_market_cap(soup: BeautifulSoup) -> float | None:
    for li in soup.select("#top-ratios li, .company-ratios li"):
        text = " ".join(li.get_text(" ", strip=True).split())
        if "Market Cap" in text:
            return parse_number(text)
    return None


def extract_quarterly_sales_growth(soup: BeautifulSoup) -> float | None:
    table = section_table(soup, "quarters")
    if table is None:
        return None
    row = metric_row(table, ("Sales", "Revenue", "Operating Revenue"))
    return trailing_growth(row, include_ttm=False)


def extract_annual_sales_growth(soup: BeautifulSoup) -> float | None:
    table = section_table(soup, "profit-loss")
    if table is None:
        table = section_table(soup, "profit-loss-consolidated")
    if table is None:
        return None
    row = metric_row(table, ("Sales", "Revenue", "Operating Revenue"))
    return trailing_growth(row, include_ttm=False)


def extract_promoter_holding_change(soup: BeautifulSoup) -> float | None:
    table = section_table(soup, "shareholding")
    if table is None:
        return None
    row = metric_row(table, ("Promoters", "Promoter"))
    values = numeric_values(row, include_ttm=False)
    if len(values) < 2:
        return None
    return round(values[-1] - values[-2], 2)


def extract_fii_holding_change(soup: BeautifulSoup) -> dict[str, Any]:
    return extract_institutional_holding_change(
        soup,
        "FII",
        ("FIIs", "FII", "Foreign Institutional", "Foreign Portfolio"),
    )


def extract_dii_holding_change(soup: BeautifulSoup) -> dict[str, Any]:
    return extract_institutional_holding_change(
        soup,
        "DII",
        ("DIIs", "DII", "Domestic Institutional", "Domestic Institutions"),
    )


def extract_institutional_holding_change(
    soup: BeautifulSoup,
    holder_prefix: str,
    row_labels: tuple[str, ...],
) -> dict[str, Any]:
    prefix = holder_prefix.upper()
    table = section_table(soup, "shareholding")
    empty = {
        f"{prefix} Previous Period": None,
        f"{prefix} Latest Period": None,
        f"{prefix} Previous Holding %": None,
        f"{prefix} Latest Holding %": None,
        f"{prefix} Holding Change %": None,
    }
    if table is None:
        return empty
    row = metric_row(table, row_labels)
    values = numeric_values_with_periods(row, include_ttm=False)
    if len(values) < 2:
        return empty

    previous_period, previous_value = values[-2]
    latest_period, latest_value = values[-1]
    return {
        f"{prefix} Previous Period": previous_period,
        f"{prefix} Latest Period": latest_period,
        f"{prefix} Previous Holding %": round(previous_value, 2),
        f"{prefix} Latest Holding %": round(latest_value, 2),
        f"{prefix} Holding Change %": round(latest_value - previous_value, 2),
    }


def section_table(soup: BeautifulSoup, section_id: str) -> pd.DataFrame | None:
    section = soup.find(id=section_id)
    if section is None:
        return None
    table = section.find("table")
    if table is None:
        return None
    frames = pd.read_html(StringIO(str(table)))
    if not frames:
        return None
    frame = frames[0]
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def metric_row(table: pd.DataFrame, labels: tuple[str, ...]) -> pd.Series | None:
    if table.empty:
        return None
    first_column = table.columns[0]
    for _, row in table.iterrows():
        label = clean_label(row[first_column])
        if any(label.lower().startswith(target.lower()) for target in labels):
            return row
    return None


def trailing_growth(row: pd.Series | None, include_ttm: bool) -> float | None:
    values = numeric_values(row, include_ttm=include_ttm)
    if len(values) < 2:
        return None
    previous, latest = values[-2], values[-1]
    if previous == 0:
        return None
    return round(((latest / previous) - 1.0) * 100.0, 2)


def numeric_values(row: pd.Series | None, include_ttm: bool) -> list[float]:
    if row is None:
        return []
    values: list[float] = []
    for column, value in row.iloc[1:].items():
        if not include_ttm and str(column).strip().upper() == "TTM":
            continue
        parsed = parse_number(value)
        if parsed is not None:
            values.append(parsed)
    return values


def numeric_values_with_periods(row: pd.Series | None, include_ttm: bool) -> list[tuple[str, float]]:
    if row is None:
        return []
    values: list[tuple[str, float]] = []
    for column, value in row.iloc[1:].items():
        period = str(column).strip()
        if not include_ttm and period.upper() == "TTM":
            continue
        parsed = parse_number(value)
        if parsed is not None:
            values.append((period, parsed))
    return values


def clean_label(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).replace("+", "")).strip()


def parse_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace(",", "").replace("%", "").replace("₹", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))
