from __future__ import annotations

import gzip
import io
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import requests

from .config import DerivativesSignalConfig


ProgressCallback = Callable[[int, int, str], None]
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class DerivativesDataProvider(Protocol):
    def download_range(
        self,
        start_date: date,
        end_date: date,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.DataFrame: ...

    def available_dates(self) -> list[date]: ...

    def load_date(self, trade_date: date) -> "DailyDerivativesData": ...


class NseReportUnavailable(RuntimeError):
    pass


@dataclass
class DailyDerivativesData:
    trade_date: date
    cash: pd.DataFrame
    derivatives: pd.DataFrame
    contracts: pd.DataFrame


class NseEodDerivativesProvider:
    """Download and cache official NSE EOD cash, F&O, and contract reports."""

    def __init__(self, cache_dir: str | Path, timeout: int = 25, sleep_seconds: float = 0.15) -> None:
        self.cache_dir = Path(cache_dir)
        self.raw_dir = self.cache_dir / "raw"
        self.normalized_dir = self.cache_dir / "normalized"
        self.manifest_path = self.cache_dir / "manifest.csv"
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.session = requests.Session()
        self.session.headers.update(NSE_HEADERS)

    def download_range(
        self,
        start_date: date,
        end_date: date,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        if start_date > end_date:
            raise ValueError("Derivatives data start date must be on or before the end date.")

        trading_days = [item.date() for item in pd.date_range(start_date, end_date, freq="B")]
        total = len(trading_days)
        manifest = self._load_manifest()
        rows = manifest.to_dict("records") if not manifest.empty else []
        by_date = {str(row.get("Trade Date")): row for row in rows}

        for index, trade_date in enumerate(trading_days, start=1):
            date_text = trade_date.isoformat()
            if progress_callback:
                progress_callback(index - 1, total, f"Checking NSE EOD reports for {date_text}")
            if self._is_cached(trade_date):
                row = {
                    "Trade Date": date_text,
                    "Status": "cached",
                    "Cash Rows": len(pd.read_csv(self._normalized_path(trade_date, "cash"))),
                    "F&O Rows": len(pd.read_csv(self._normalized_path(trade_date, "fo"))),
                    "Contract Rows": _csv_row_count(self._normalized_path(trade_date, "contracts")),
                    "Notes": "normalized files already available",
                }
            else:
                try:
                    daily = self.download_date(trade_date)
                    row = {
                        "Trade Date": date_text,
                        "Status": "downloaded",
                        "Cash Rows": len(daily.cash),
                        "F&O Rows": len(daily.derivatives),
                        "Contract Rows": len(daily.contracts),
                        "Notes": "complete",
                    }
                except NseReportUnavailable as exc:
                    row = {
                        "Trade Date": date_text,
                        "Status": "unavailable",
                        "Cash Rows": 0,
                        "F&O Rows": 0,
                        "Contract Rows": 0,
                        "Notes": str(exc),
                    }
                except Exception as exc:
                    row = {
                        "Trade Date": date_text,
                        "Status": "failed",
                        "Cash Rows": 0,
                        "F&O Rows": 0,
                        "Contract Rows": 0,
                        "Notes": f"{type(exc).__name__}: {exc}",
                    }

            by_date[date_text] = row
            self._save_manifest(pd.DataFrame(by_date.values()))
            if progress_callback:
                progress_callback(index, total, f"Processed {index:,} of {total:,} EOD report dates")
            time.sleep(self.sleep_seconds)

        result = self._load_manifest()
        if result.empty:
            return result
        report_dates = pd.to_datetime(result["Trade Date"], errors="coerce").dt.date
        return result[report_dates.between(start_date, end_date)].reset_index(drop=True)

    def download_date(self, trade_date: date) -> DailyDerivativesData:
        cash_url, cash_bytes = self._download_first(self._cash_urls(trade_date), required=True)
        fo_url, fo_bytes = self._download_first(self._fo_urls(trade_date), required=True)
        contract_url, contract_bytes = self._download_first(self._contract_urls(trade_date), required=False)

        raw_day = self.raw_dir / trade_date.strftime("%Y%m%d")
        normalized_day = self.normalized_dir / trade_date.strftime("%Y%m%d")
        raw_day.mkdir(parents=True, exist_ok=True)
        normalized_day.mkdir(parents=True, exist_ok=True)
        self._write_raw(raw_day, "cash", cash_url, cash_bytes)
        self._write_raw(raw_day, "fo", fo_url, fo_bytes)
        if contract_url and contract_bytes:
            self._write_raw(raw_day, "contracts", contract_url, contract_bytes)

        cash = normalize_cash_bhavcopy(_read_report(cash_bytes, cash_url), trade_date)
        derivatives = normalize_fo_bhavcopy(_read_report(fo_bytes, fo_url), trade_date)
        if contract_url and contract_bytes:
            contracts = normalize_contract_master(_read_report(contract_bytes, contract_url))
        else:
            contracts = contracts_from_derivatives(derivatives)

        cash.to_csv(normalized_day / "cash.csv", index=False)
        derivatives.to_csv(normalized_day / "fo.csv", index=False)
        contracts.to_csv(normalized_day / "contracts.csv", index=False)
        return DailyDerivativesData(trade_date, cash, derivatives, contracts)

    def available_dates(self) -> list[date]:
        if not self.normalized_dir.exists():
            return []
        dates: list[date] = []
        for child in self.normalized_dir.iterdir():
            if child.is_dir() and (child / "cash.csv").exists() and (child / "fo.csv").exists():
                try:
                    dates.append(datetime.strptime(child.name, "%Y%m%d").date())
                except ValueError:
                    continue
        return sorted(dates)

    def load_date(self, trade_date: date) -> DailyDerivativesData:
        if not self._is_cached(trade_date):
            raise FileNotFoundError(f"No cached NSE derivatives reports for {trade_date.isoformat()}.")
        cash = pd.read_csv(self._normalized_path(trade_date, "cash"))
        derivatives = pd.read_csv(self._normalized_path(trade_date, "fo"))
        contract_path = self._normalized_path(trade_date, "contracts")
        contracts = pd.read_csv(contract_path) if contract_path.exists() and contract_path.stat().st_size else pd.DataFrame()
        return DailyDerivativesData(trade_date, cash, derivatives, contracts)

    def _download_first(self, urls: Iterable[str], required: bool) -> tuple[str, bytes]:
        last_error = ""
        for url in urls:
            for attempt in range(1, 4):
                try:
                    response = self.session.get(url, timeout=self.timeout)
                    if response.status_code == 404:
                        break
                    if response.status_code in RETRY_STATUS_CODES and attempt < 3:
                        time.sleep(attempt * 1.5)
                        continue
                    response.raise_for_status()
                    return url, response.content
                except requests.RequestException as exc:
                    last_error = str(exc)
                    if attempt < 3:
                        time.sleep(attempt * 1.5)
        if required:
            raise NseReportUnavailable(last_error or "official NSE report not published for this date")
        return "", b""

    @staticmethod
    def _cash_urls(trade_date: date) -> list[str]:
        return [
            "https://nsearchives.nseindia.com/products/content/"
            f"sec_bhavdata_full_{trade_date:%d%m%Y}.csv"
        ]

    @staticmethod
    def _fo_urls(trade_date: date) -> list[str]:
        return [
            "https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{trade_date:%Y%m%d}_F_0000.csv.zip",
            "https://nsearchives.nseindia.com/content/fo/"
            f"fo{trade_date.strftime('%d%b%Y').upper()}bhav.csv.zip",
        ]

    @staticmethod
    def _contract_urls(trade_date: date) -> list[str]:
        return [
            "https://nsearchives.nseindia.com/content/fo/"
            f"NSE_FO_contract_{trade_date:%d%m%Y}.csv.gz"
        ]

    def _is_cached(self, trade_date: date) -> bool:
        return self._normalized_path(trade_date, "cash").exists() and self._normalized_path(trade_date, "fo").exists()

    def _normalized_path(self, trade_date: date, name: str) -> Path:
        return self.normalized_dir / trade_date.strftime("%Y%m%d") / f"{name}.csv"

    @staticmethod
    def _write_raw(directory: Path, name: str, url: str, content: bytes) -> None:
        suffix = ".zip" if url.endswith(".zip") else ".gz" if url.endswith(".gz") else ".csv"
        (directory / f"{name}{suffix}").write_bytes(content)

    def _load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists() or self.manifest_path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(self.manifest_path)

    def _save_manifest(self, frame: pd.DataFrame) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        frame = frame.sort_values("Trade Date").drop_duplicates("Trade Date", keep="last")
        frame.to_csv(self.manifest_path, index=False)


def normalize_cash_bhavcopy(frame: pd.DataFrame, trade_date: date | None = None) -> pd.DataFrame:
    source = _clean_columns(frame)
    result = pd.DataFrame(
        {
            "Ticker": _column(source, "SYMBOL", "TckrSymb"),
            "Series": _column(source, "SERIES", "SctySrs"),
            "Trade Date": _column(source, "DATE1", "TradDt", default=trade_date),
            "Previous Close": _numeric_column(source, "PREV_CLOSE", "PrvsClsgPric"),
            "Open": _numeric_column(source, "OPEN_PRICE", "OpnPric"),
            "High": _numeric_column(source, "HIGH_PRICE", "HghPric"),
            "Low": _numeric_column(source, "LOW_PRICE", "LwPric"),
            "Close": _numeric_column(source, "CLOSE_PRICE", "ClsPric"),
            "Volume": _numeric_column(source, "TTL_TRD_QNTY", "TtlTradgVol"),
            "Turnover": _numeric_column(source, "TURNOVER_LACS", "TtlTrfVal"),
        }
    )
    result["Ticker"] = result["Ticker"].astype(str).str.strip().str.upper()
    result["Series"] = result["Series"].astype(str).str.strip().str.upper()
    result = result[result["Series"].eq("EQ")].copy()
    result["Trade Date"] = _parse_report_dates(result["Trade Date"])
    result = result.dropna(subset=["Ticker", "Close"]).drop_duplicates("Ticker", keep="last")
    return result.reset_index(drop=True)


def normalize_fo_bhavcopy(frame: pd.DataFrame, trade_date: date | None = None) -> pd.DataFrame:
    source = _clean_columns(frame)
    instrument = _column(source, "FinInstrmTp", "INSTRUMENT")
    instrument = instrument.astype(str).str.strip().str.upper().replace(
        {"STO": "OPTSTK", "STF": "FUTSTK", "IDO": "OPTIDX", "IDF": "FUTIDX"}
    )
    result = pd.DataFrame(
        {
            "Trade Date": _column(source, "TradDt", "TIMESTAMP", default=trade_date),
            "Instrument": instrument,
            "Instrument ID": _numeric_column(source, "FinInstrmId", "INSTRUMENT_ID"),
            "Ticker": _column(source, "TckrSymb", "SYMBOL"),
            "Expiry": _column(source, "XpryDt", "EXPIRY_DT"),
            "Strike": _numeric_column(source, "StrkPric", "STRIKE_PR"),
            "Option Type": _column(source, "OptnTp", "OPTION_TYP"),
            "Contract": _column(source, "FinInstrmNm", "CONTRACT", default=""),
            "Open": _numeric_column(source, "OpnPric", "OPEN"),
            "High": _numeric_column(source, "HghPric", "HIGH"),
            "Low": _numeric_column(source, "LwPric", "LOW"),
            "Close": _numeric_column(source, "ClsPric", "CLOSE"),
            "Previous Close": _numeric_column(source, "PrvsClsgPric", "PREV_CLOSE"),
            "Underlying Price": _numeric_column(source, "UndrlygPric", "UNDERLYING_VALUE"),
            "OI Units": _numeric_column(source, "OpnIntrst", "OPEN_INT"),
            "OI Change Units": _numeric_column(source, "ChngInOpnIntrst", "CHG_IN_OI"),
            "Volume Contracts": _numeric_column(source, "TtlTradgVol", "CONTRACTS"),
            "Turnover": _numeric_column(source, "TtlTrfVal", "VAL_INLAKH"),
            "Lot Size": _numeric_column(source, "NewBrdLotQty", "LOT_SIZE", default=1),
        }
    )
    result["Ticker"] = result["Ticker"].astype(str).str.strip().str.upper()
    result["Option Type"] = result["Option Type"].astype(str).str.strip().str.upper().replace("NAN", "")
    result["Trade Date"] = _parse_report_dates(result["Trade Date"])
    result["Expiry"] = _parse_report_dates(result["Expiry"])
    missing_contract = result["Contract"].astype(str).str.strip().isin({"", "nan", "None"})
    result.loc[missing_contract, "Contract"] = result.loc[missing_contract].apply(_contract_label, axis=1)
    result = result[result["Instrument"].isin({"OPTSTK", "FUTSTK", "OPTIDX", "FUTIDX"})].copy()
    result = result.dropna(subset=["Ticker", "Expiry", "Close"])
    return result.reset_index(drop=True)


def normalize_contract_master(frame: pd.DataFrame) -> pd.DataFrame:
    source = _clean_columns(frame)
    result = pd.DataFrame(
        {
            "Instrument ID": _numeric_column(source, "FinInstrmId", "INSTRUMENT_ID"),
            "Ticker": _column(source, "TckrSymb", "SYMBOL"),
            "Contract": _column(source, "StockNm", "FinInstrmNm", "CONTRACT", default=""),
            "Corporate Adjustment": _column(source, "CorpAdjstmnt", "CORP_ADJUSTMENT", default="N"),
        }
    )
    result["Ticker"] = result["Ticker"].astype(str).str.strip().str.upper()
    result["Corporate Adjustment"] = result["Corporate Adjustment"].astype(str).str.strip().str.upper()
    return result.dropna(subset=["Instrument ID"]).drop_duplicates("Instrument ID", keep="last").reset_index(drop=True)


def contracts_from_derivatives(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Instrument ID", "Ticker", "Contract", "Corporate Adjustment"])
    result = frame[["Instrument ID", "Ticker", "Contract"]].drop_duplicates().copy()
    result["Corporate Adjustment"] = "N"
    return result.reset_index(drop=True)


def scan_derivatives_momentum(
    universe: pd.DataFrame,
    daily: DailyDerivativesData,
    config: DerivativesSignalConfig,
    previous_derivatives: pd.DataFrame | None = None,
    volume_history: dict[str, list[float]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, pd.DataFrame]:
    """Build daily features, hard-gate signals, rejections, and F&O eligibility."""
    weights = config.score_weights()
    if sum(weights.values()) <= 0:
        raise ValueError("Derivatives momentum score weights must add to more than zero.")
    if config.min_call_return_pct < 8:
        raise ValueError("The call-option return threshold cannot be below the required 8% minimum.")

    fo = daily.derivatives.copy()
    fo["Expiry"] = pd.to_datetime(fo["Expiry"], errors="coerce").dt.date
    cash = daily.cash.copy()
    cash["Ticker"] = cash["Ticker"].astype(str).str.upper()
    universe_frame = universe.copy()
    universe_frame["Ticker"] = universe_frame["Ticker"].astype(str).str.upper()
    stock_fo = fo[fo["Instrument"].isin({"OPTSTK", "FUTSTK"})].copy()
    eligible = set(stock_fo["Ticker"].dropna().astype(str).str.upper())
    contracts = universe_frame.copy()
    contracts["Listed Derivatives"] = contracts["Ticker"].isin(eligible)
    contracts["Derivatives Status"] = np.where(contracts["Listed Derivatives"], "Eligible", "No listed derivatives")

    previous_lookup = _previous_close_lookup(previous_derivatives)
    corporate_lookup = _corporate_adjustment_lookup(daily.contracts)
    volume_history = volume_history or {}
    cash_lookup = cash.set_index("Ticker", drop=False)
    feature_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    contract_records = contracts.to_dict("records")
    total_contracts = len(contract_records)
    for index, item in enumerate(contract_records, start=1):
        ticker = str(item["Ticker"]).upper()
        if ticker not in eligible:
            rejection_rows.append({**item, "Trade Date": daily.trade_date, "Rejection Reasons": "No listed derivatives"})
            if progress_callback:
                progress_callback(index, total_contracts, f"Checked {ticker}: no listed derivatives")
            continue
        if ticker not in cash_lookup.index:
            rejection_rows.append({**item, "Trade Date": daily.trade_date, "Rejection Reasons": "Missing cash-market close"})
            if progress_callback:
                progress_callback(index, total_contracts, f"Checked {ticker}: missing cash close")
            continue
        cash_row = cash_lookup.loc[ticker]
        if isinstance(cash_row, pd.DataFrame):
            cash_row = cash_row.iloc[-1]
        feature = _build_ticker_feature(
            ticker,
            cash_row,
            stock_fo[stock_fo["Ticker"].eq(ticker)],
            daily.trade_date,
            config,
            previous_lookup,
            corporate_lookup,
            volume_history.get(ticker, []),
        )
        feature_rows.append({**item, **feature})
        if feature["Rejection Reasons"]:
            rejection_rows.append({**item, **feature})
        if progress_callback:
            outcome = "signal" if not feature["Rejection Reasons"] else "not qualified"
            progress_callback(index, total_contracts, f"Checked {ticker}: {outcome}")

    features = pd.DataFrame(feature_rows)
    if features.empty:
        signals = pd.DataFrame()
    else:
        signals = features[features["Rejection Reasons"].eq("")].copy()
        signals = rank_derivatives_signals(signals, weights).head(int(config.result_count)).reset_index(drop=True)
    rejections = pd.DataFrame(rejection_rows)
    return {
        "contracts": contracts.reset_index(drop=True),
        "features": features.reset_index(drop=True),
        "signals": signals,
        "rejections": rejections.reset_index(drop=True),
    }


def rank_derivatives_signals(signals: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    frame = signals.copy()
    if frame.empty:
        return frame
    total_weight = sum(max(float(weight), 0.0) for weight in weights.values())
    if total_weight <= 0:
        raise ValueError("Derivatives momentum score weights must add to more than zero.")
    score = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        values = pd.to_numeric(frame.get(column, pd.Series(index=frame.index, dtype=float)), errors="coerce")
        percentile = values.rank(pct=True, method="average").fillna(0.5)
        score += percentile * (max(float(weight), 0.0) / total_weight)
    frame["Derivatives Momentum Score"] = (score * 100).round(2)
    frame = frame.sort_values("Derivatives Momentum Score", ascending=False, na_position="last").reset_index(drop=True)
    frame.insert(0, "Derivatives Momentum Rank", range(1, len(frame) + 1))
    return frame


def futures_oi_classification(futures_return: float | None, oi_change_units: float | None) -> str:
    if futures_return is None or pd.isna(futures_return) or oi_change_units is None or pd.isna(oi_change_units):
        return "Unconfirmed"
    if futures_return > 0 and oi_change_units > 0:
        return "Long build-up"
    if futures_return > 0 and oi_change_units < 0:
        return "Short covering"
    if futures_return < 0 and oi_change_units > 0:
        return "Short build-up"
    if futures_return < 0 and oi_change_units < 0:
        return "Long unwinding"
    return "Unconfirmed"


def build_volume_history(features_by_date: Iterable[pd.DataFrame], window: int = 20) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for frame in features_by_date:
        if frame.empty or not {"Ticker", "Call Volume Contracts"}.issubset(frame.columns):
            continue
        for row in frame[["Ticker", "Call Volume Contracts"]].dropna().to_dict("records"):
            ticker = str(row["Ticker"]).upper()
            values.setdefault(ticker, []).append(float(row["Call Volume Contracts"]))
            values[ticker] = values[ticker][-window:]
    return values


def _build_ticker_feature(
    ticker: str,
    cash_row: pd.Series,
    ticker_fo: pd.DataFrame,
    trade_date: date,
    config: DerivativesSignalConfig,
    previous_lookup: dict[str, float],
    corporate_lookup: dict[int, str],
    volume_history: list[float],
) -> dict[str, Any]:
    cash_close = _number(cash_row.get("Close"))
    cash_previous = _number(cash_row.get("Previous Close"))
    underlying_return = _return_pct(cash_close, cash_previous)
    base: dict[str, Any] = {
        "Trade Date": trade_date,
        "Ticker": ticker,
        "Cash Previous Close": cash_previous,
        "Cash Close": cash_close,
        "Underlying Return %": underlying_return,
        "Rejection Reasons": "",
    }

    options = ticker_fo[
        ticker_fo["Instrument"].eq("OPTSTK") & ticker_fo["Option Type"].eq("CE")
    ].copy()
    options["Days to Expiry"] = options["Expiry"].map(lambda expiry: (expiry - trade_date).days if pd.notna(expiry) else np.nan)
    options = options[
        options["Days to Expiry"].between(config.min_days_to_expiry, config.max_days_to_expiry, inclusive="both")
    ].copy()
    if options.empty or cash_close is None:
        return {**base, "Rejection Reasons": "No eligible call option in expiry window"}

    nearest_expiry = options["Expiry"].min()
    options = options[options["Expiry"].eq(nearest_expiry)].copy()
    options["Strike Distance"] = (pd.to_numeric(options["Strike"], errors="coerce") - cash_close).abs()
    options = options.sort_values(["Strike Distance", "Volume Contracts"], ascending=[True, False], na_position="last")
    call = options.iloc[0]
    call_previous = _number(call.get("Previous Close"))
    if call_previous is None or call_previous <= 0:
        call_previous = previous_lookup.get(str(call.get("Contract")))
    call_close = _number(call.get("Close"))
    call_return = _return_pct(call_close, call_previous)
    lot_size = max(_number(call.get("Lot Size")) or 1.0, 1.0)
    call_oi_units = _number(call.get("OI Units"))
    call_oi_change_units = _number(call.get("OI Change Units"))
    call_oi_contracts = call_oi_units / lot_size if call_oi_units is not None else None
    call_oi_change_contracts = call_oi_change_units / lot_size if call_oi_change_units is not None else None
    previous_oi = (
        call_oi_units - call_oi_change_units
        if call_oi_units is not None and call_oi_change_units is not None
        else None
    )
    call_oi_change_pct = _return_pct(call_oi_units, previous_oi)
    call_volume = _number(call.get("Volume Contracts"))
    volume_median = float(np.median(volume_history[-20:])) if volume_history else None
    volume_ratio = call_volume / volume_median if call_volume is not None and volume_median and volume_median > 0 else None

    futures = ticker_fo[ticker_fo["Instrument"].eq("FUTSTK")].copy()
    exact_future = futures[futures["Expiry"].eq(nearest_expiry)]
    if not exact_future.empty:
        future = exact_future.sort_values("Volume Contracts", ascending=False).iloc[0]
    elif not futures.empty:
        futures["Days to Expiry"] = futures["Expiry"].map(lambda expiry: (expiry - trade_date).days if pd.notna(expiry) else np.nan)
        eligible_futures = futures[futures["Days to Expiry"].ge(0)].sort_values(
            ["Expiry", "Volume Contracts"],
            ascending=[True, False],
        )
        future = eligible_futures.iloc[0] if not eligible_futures.empty else pd.Series(dtype=object)
    else:
        future = pd.Series(dtype=object)
    futures_close = _number(future.get("Close"))
    futures_previous = _number(future.get("Previous Close"))
    if futures_previous is None or futures_previous <= 0:
        futures_previous = previous_lookup.get(str(future.get("Contract")))
    futures_return = _return_pct(futures_close, futures_previous)
    futures_oi_change = _number(future.get("OI Change Units"))

    instrument_id = _number(call.get("Instrument ID"))
    corp_adjustment = corporate_lookup.get(int(instrument_id), "N") if instrument_id is not None else "N"
    corporate_action = corp_adjustment not in {"", "N", "NAN"} or (
        underlying_return is not None and abs(underlying_return) >= config.corporate_action_return_pct
    )
    reasons: list[str] = []
    if underlying_return is None or underlying_return < config.min_underlying_return_pct:
        reasons.append(f"Underlying return below {config.min_underlying_return_pct:.2f}%")
    if call_return is None or call_return < config.min_call_return_pct:
        reasons.append(f"Call return below {config.min_call_return_pct:.2f}%")
    if call_close is None or call_close < config.min_call_price:
        reasons.append(f"Call close below Rs. {config.min_call_price:.2f}")
    if call_volume is None or call_volume < config.min_volume_contracts:
        reasons.append(f"Call volume below {config.min_volume_contracts} contracts")
    if call_oi_contracts is None or call_oi_contracts < config.min_open_interest_contracts:
        reasons.append(f"Call OI below {config.min_open_interest_contracts} contracts")
    if call_previous is None or call_previous <= 0:
        reasons.append("Missing valid previous call close")
    if corporate_action:
        reasons.append("Potential corporate-action distortion")

    return {
        **base,
        "Call Contract": call.get("Contract"),
        "Call Instrument ID": instrument_id,
        "Call Strike": _number(call.get("Strike")),
        "Call Expiry": nearest_expiry,
        "Days to Expiry": int(call.get("Days to Expiry")),
        "Call Previous Close": call_previous,
        "Call Close": call_close,
        "Call Return %": call_return,
        "Call Volume Contracts": call_volume,
        "Call Volume 20D Median": volume_median,
        "Call Volume Ratio": volume_ratio,
        "Call OI Contracts": call_oi_contracts,
        "Call OI Change Contracts": call_oi_change_contracts,
        "Call OI Change %": call_oi_change_pct,
        "Futures Contract": future.get("Contract"),
        "Futures Previous Close": futures_previous,
        "Futures Close": futures_close,
        "Futures Return %": futures_return,
        "Futures OI Change Units": futures_oi_change,
        "Futures Activity": futures_oi_classification(futures_return, futures_oi_change),
        "Corporate Adjustment": corp_adjustment,
        "Corporate Action Flag": corporate_action,
        "Rejection Reasons": "; ".join(reasons),
    }


def _previous_close_lookup(previous: pd.DataFrame | None) -> dict[str, float]:
    if previous is None or previous.empty or not {"Contract", "Close"}.issubset(previous.columns):
        return {}
    frame = previous[["Contract", "Close"]].dropna().drop_duplicates("Contract", keep="last")
    return {str(row["Contract"]): float(row["Close"]) for row in frame.to_dict("records")}


def _corporate_adjustment_lookup(contracts: pd.DataFrame) -> dict[int, str]:
    if contracts.empty or not {"Instrument ID", "Corporate Adjustment"}.issubset(contracts.columns):
        return {}
    frame = contracts.dropna(subset=["Instrument ID"]).drop_duplicates("Instrument ID", keep="last")
    return {
        int(float(row["Instrument ID"])): str(row["Corporate Adjustment"]).strip().upper()
        for row in frame.to_dict("records")
    }


def _read_report(content: bytes, url: str) -> pd.DataFrame:
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith((".csv", ".dat"))]
            if not names:
                raise ValueError(f"No CSV/DAT report found inside {url}")
            raw = archive.read(names[0])
    elif url.endswith(".gz"):
        raw = gzip.decompress(content)
    else:
        raw = content
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def _parse_report_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    iso_mask = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    parsed.loc[iso_mask] = pd.to_datetime(text.loc[iso_mask], format="%Y-%m-%d", errors="coerce")
    parsed.loc[~iso_mask] = pd.to_datetime(text.loc[~iso_mask], errors="coerce", dayfirst=True)
    return parsed.dt.date


def _column(frame: pd.DataFrame, *names: str, default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _numeric_column(frame: pd.DataFrame, *names: str, default: Any = np.nan) -> pd.Series:
    return pd.to_numeric(_column(frame, *names, default=default), errors="coerce")


def _return_pct(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0 or pd.isna(current) or pd.isna(previous):
        return None
    return round(((current / previous) - 1.0) * 100.0, 2)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _contract_label(row: pd.Series) -> str:
    expiry = row.get("Expiry")
    expiry_text = expiry.strftime("%Y%m%d") if isinstance(expiry, date) else str(expiry)
    strike = _number(row.get("Strike")) or 0
    return f"{row.get('Instrument')}|{row.get('Ticker')}|{expiry_text}|{strike:g}|{row.get('Option Type')}"


def _csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return len(pd.read_csv(path))
