"""Source-dated Screener market-cap snapshots for a probability shortlist.

Market capitalisation is for display and filtering only. It is never passed to
the earnings model. A ticker is accepted only when the profile links back to
the same NSE symbol.
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


DEFAULT_CAPS = Path(__file__).resolve().parent / "data" / "market_caps.csv"
DEFAULT_RANKINGS = Path(__file__).resolve().parent / "data" / "default_rankings.csv"
FIELDS = ["ticker", "market_cap_cr", "retrieved_at_ist", "source_url", "status"]
CAP_PATTERN = re.compile(
    r'<span\s+class="name"[^>]*>\s*Market Cap\s*</span>.*?'
    r'<span\s+class="number"[^>]*>\s*([\d,]+(?:\.\d+)?)\s*</span>',
    re.IGNORECASE | re.DOTALL,
)
SYMBOL_PATTERN = re.compile(r"nseindia\.com/get-quotes/equity\?symbol=([^\"&<>]+)", re.IGNORECASE)


def load_market_caps(path: Path = DEFAULT_CAPS) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame(columns=FIELDS)
    frame = pd.read_csv(path, dtype={"ticker": "string", "status": "string"})
    missing = sorted(set(FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"Market-cap CSV is missing: {', '.join(missing)}")
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["market_cap_cr"] = pd.to_numeric(frame["market_cap_cr"], errors="coerce")
    return frame.drop_duplicates("ticker", keep="last")


def fetch_market_cap(ticker: str) -> dict:
    symbol = ticker.removesuffix(".NS").upper()
    url = f"https://www.screener.in/company/{urllib.parse.quote(symbol)}/"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; research-dashboard/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        page = response.read().decode("utf-8", errors="replace")
        source_url = response.geturl()
    verified_symbols = {
        urllib.parse.unquote(value).strip().upper()
        for value in SYMBOL_PATTERN.findall(page)
    }
    if symbol not in verified_symbols:
        raise ValueError("Screener profile did not verify the NSE symbol")
    match = CAP_PATTERN.search(page)
    if not match:
        raise ValueError("No market-cap value on Screener profile")
    return {
        "ticker": f"{symbol}.NS",
        "market_cap_cr": float(match.group(1).replace(",", "")),
        "retrieved_at_ist": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
        "source_url": source_url,
        "status": "ok",
    }


def refresh_market_caps(
    tickers: list[str], path: Path = DEFAULT_CAPS, delay_seconds: float = 0.5
) -> tuple[pd.DataFrame, list[str]]:
    """Refresh requested tickers; retain other saved snapshots for future seasons."""
    existing = load_market_caps(path)
    updates: list[dict] = []
    errors: list[str] = []
    requested = list(dict.fromkeys(str(ticker).upper() for ticker in tickers))
    for position, ticker in enumerate(requested):
        try:
            updates.append(fetch_market_cap(ticker))
        except (ValueError, TimeoutError, urllib.error.URLError) as exc:
            errors.append(f"{ticker}: {exc}")
            if ticker not in set(existing["ticker"]):
                updates.append({
                    "ticker": ticker, "market_cap_cr": float("nan"),
                    "retrieved_at_ist": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
                    "source_url": f"https://www.screener.in/company/{ticker.removesuffix('.NS')}/",
                    "status": str(exc),
                })
        if position < len(requested) - 1:
            time.sleep(delay_seconds)
    updated = pd.concat([existing, pd.DataFrame(updates, columns=FIELDS)], ignore_index=True)
    updated = updated.drop_duplicates("ticker", keep="last").sort_values("ticker")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    updated[FIELDS].to_csv(path, index=False, encoding="utf-8-sig")
    return updated.reset_index(drop=True), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh Screener market caps for the top earnings forecasts")
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_CAPS)
    args = parser.parse_args()
    rankings = pd.read_csv(args.rankings)
    tickers = rankings.loc[rankings["status"].eq("Scored")].sort_values("rank")["ticker"].head(args.top).tolist()
    updated, errors = refresh_market_caps(tickers, args.output)
    covered = updated.loc[updated["ticker"].isin(tickers) & updated["market_cap_cr"].notna()]
    print(f"Market caps available for {len(covered)} of {len(tickers)} top-probability tickers")
    for error in errors:
        print(error)


if __name__ == "__main__":
    main()
