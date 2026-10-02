"""Point-in-time batch scoring of the user-supplied NSE ticker universe.

The eligibility, feature construction, preprocessing, and MLP arithmetic match
the existing single-company scorer in ``work/score_caelion_ticker.py``. The
archive and model bundle are loaded once per run, even for thousands of names.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TICKERS = Path(__file__).resolve().parent / "data" / "tickers.csv"
DEFAULT_ARCHIVE = Path(__file__).resolve().parent / "data" / "caelion_model_panel.zip"
DEFAULT_BUNDLE = Path(__file__).resolve().parent / "data" / "model_bundle.json"
IST = "Asia/Kolkata"

PANEL_COLUMNS = {
    "isin", "current_symbol", "board", "period_months", "quality",
    "revision_type", "is_first", "basis", "basis_switch", "period_end",
    "release_ts", "xbrl_broadcast_ts", "net_profit_cr", "revenue_cr",
    "operating_profit_cr",
}
FINANCIAL_FIELDS = ("net_profit_cr", "revenue_cr", "operating_profit_cr")


@dataclass
class BatchResult:
    rows: pd.DataFrame
    summary: dict


def normalize_symbol(value: object) -> str:
    symbol = str(value).strip().upper()
    return symbol[:-3] if symbol.endswith(".NS") else symbol


def as_ist(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("The as-of timestamp is missing.")
    return timestamp.tz_localize(IST) if timestamp.tzinfo is None else timestamp.tz_convert(IST)


def previous_completed_quarter(value: object) -> pd.Timestamp:
    day = pd.Timestamp(value).normalize()
    if day.is_quarter_end:
        return day
    return day.to_period("Q").start_time - pd.Timedelta(days=1)


def load_tickers(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"Ticker": "string"}, keep_default_na=False)
    if "Ticker" not in table.columns:
        raise ValueError("Ticker CSV must have a 'Ticker' column.")
    table = table.copy()
    table["input_row"] = np.arange(1, len(table) + 1)
    table["symbol"] = table["Ticker"].map(normalize_symbol)
    table["ticker"] = table["symbol"] + ".NS"
    table["company"] = table["Name"].astype(str) if "Name" in table else table["symbol"]
    table["industry"] = table["Industry"].astype(str) if "Industry" in table else "Unspecified"
    table["industry"] = table["industry"].replace({"": "Unspecified", "N/A": "Unspecified"})
    return table[["input_row", "ticker", "symbol", "company", "industry"]]


def load_bundle(path: Path) -> dict:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    required = {"feature_columns", "preprocessing", "platt_a_b", "training_n", "calibration_n"}
    missing = sorted(required - bundle.keys())
    if missing:
        raise ValueError(f"Model bundle is missing: {', '.join(missing)}")
    return bundle


def load_eligible_panel(path: Path) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("/model_panel.csv")]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one model_panel.csv inside the filing archive.")
        panel = pd.read_csv(
            archive.open(candidates[0]),
            usecols=lambda column: column in PANEL_COLUMNS,
            low_memory=False,
        )
    missing = sorted(PANEL_COLUMNS - set(panel.columns))
    if missing:
        raise ValueError(f"Filing panel is missing: {', '.join(missing)}")
    raw_rows = len(panel)
    for column in (*FINANCIAL_FIELDS, "period_months", "basis_switch", "is_first"):
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["period_end"] = pd.to_datetime(panel["period_end"], errors="coerce")
    panel["release_ts"] = pd.to_datetime(panel["release_ts"], errors="coerce", utc=True)
    panel["xbrl_broadcast_ts"] = pd.to_datetime(panel["xbrl_broadcast_ts"], errors="coerce", utc=True)
    panel["knowledge_ts"] = panel[["release_ts", "xbrl_broadcast_ts"]].max(axis=1)
    eligible = panel.loc[
        panel["board"].eq("MAIN")
        & panel["period_months"].eq(3)
        & panel["quality"].eq("ok")
        & panel["revision_type"].isin(["UNMARKED", "ORIGINAL"])
        & panel["is_first"].fillna(0).eq(1)
        & panel["net_profit_cr"].notna()
        & panel["period_end"].notna()
        & panel["release_ts"].notna()
        & panel["xbrl_broadcast_ts"].notna()
    ].copy()
    eligible["basis"] = eligible["basis"].fillna("UNKNOWN")
    eligible["basis_switch"] = eligible["basis_switch"].fillna(0)
    eligible = eligible.sort_values(
        ["isin", "basis", "period_end", "knowledge_ts"]
    ).drop_duplicates(["isin", "basis", "period_end"], keep="first")
    eligible["symbol"] = eligible["current_symbol"].astype(str).str.strip().str.upper()
    return eligible, {
        "panel_rows": raw_rows,
        "eligible_rows": len(eligible),
        "latest_archived_knowledge_utc": str(eligible["knowledge_ts"].max()),
    }


def _signed_log(value: float) -> float:
    return float(np.sign(value) * np.log1p(abs(value))) if pd.notna(value) else float("nan")


def _relative_change(current: float, prior: float) -> float:
    if pd.isna(current) or pd.isna(prior):
        return float("nan")
    return float(np.clip((current - prior) / (abs(prior) + 1.0), -10, 10))


def make_features(history: pd.DataFrame, target_end: pd.Timestamp) -> dict[str, float]:
    """Replicate the deployed 17-feature transformation exactly."""
    history = history.sort_values("period_end").drop_duplicates("period_end", keep="last")
    by_end = {row["period_end"]: row for _, row in history.iterrows()}
    rows = [by_end.get(target_end - pd.offsets.QuarterEnd(k)) for k in range(1, 6)]
    quarter = int((target_end.month - 1) // 3 + 1)
    features = {f"target_q{q}": float(quarter == q) for q in range(1, 5)}
    for field, prefix in [
        ("net_profit_cr", "np"),
        ("revenue_cr", "sales"),
        ("operating_profit_cr", "op"),
    ]:
        values = [
            float(row[field]) if row is not None and pd.notna(row[field]) else float("nan")
            for row in rows
        ]
        features[f"{prefix}_last_log"] = _signed_log(values[0])
        features[f"{prefix}_qoq_change"] = _relative_change(values[0], values[1])
        features[f"{prefix}_yoy_change"] = _relative_change(values[0], values[4])
    latest = rows[0]
    if latest is not None:
        sales, operating_profit, net_profit = (
            latest["revenue_cr"], latest["operating_profit_cr"], latest["net_profit_cr"]
        )
        features["net_margin_last"] = (
            float(net_profit / sales)
            if pd.notna(net_profit) and pd.notna(sales) and abs(sales) > 1e-9
            else float("nan")
        )
        features["op_margin_last"] = (
            float(operating_profit / sales)
            if pd.notna(operating_profit) and pd.notna(sales) and abs(sales) > 1e-9
            else float("nan")
        )
    else:
        features["net_margin_last"] = float("nan")
        features["op_margin_last"] = float("nan")
    profits = [
        float(row["net_profit_cr"]) if row is not None and pd.notna(row["net_profit_cr"]) else float("nan")
        for row in rows
    ]
    transitions = [
        profits[index] > profits[index + 1]
        for index in range(4)
        if np.isfinite(profits[index]) and np.isfinite(profits[index + 1])
    ]
    features["np_up_share_4"] = float(np.mean(transitions)) if transitions else float("nan")
    recent = np.asarray([value for value in profits[:4] if np.isfinite(value)], dtype=float)
    features["np_volatility_4"] = (
        float(np.std(recent) / (np.mean(np.abs(recent)) + 1.0))
        if len(recent) >= 3 else float("nan")
    )
    return features


def _sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=float)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def probabilities(feature_rows: pd.DataFrame, bundle: dict) -> np.ndarray:
    names = bundle["feature_columns"]
    prep = bundle["preprocessing"]
    x = feature_rows.reindex(columns=names).to_numpy(dtype=float)
    median = np.asarray([prep["median"][name] for name in names], dtype=float)
    low = np.asarray([prep["low"][name] for name in names], dtype=float)
    high = np.asarray([prep["high"][name] for name in names], dtype=float)
    mean = np.asarray([prep["mean"][name] for name in names], dtype=float)
    std = np.asarray([prep["std"][name] for name in names], dtype=float)
    if not np.all(np.isfinite(std)) or np.any(std == 0):
        raise ValueError("Model preprocessing has an invalid standard deviation.")
    x = np.where(np.isfinite(x), x, median)
    z = (np.clip(x, low, high) - mean) / std
    model_kind = bundle.get("model", "ridge_logistic_regression_numpy")
    if model_kind == "two_hidden_layer_relu_mlp_numpy":
        weights = {key: np.asarray(value, dtype=float) for key, value in bundle["parameters"].items()}
        hidden_1 = np.maximum(z @ weights["W1"] + weights["b1"], 0)
        hidden_2 = np.maximum(hidden_1 @ weights["W2"] + weights["b2"], 0)
        raw = (hidden_2 @ weights["W3"] + weights["b3"]).reshape(-1)
    elif model_kind == "ridge_logistic_regression_numpy":
        coefficients = np.asarray(bundle["coefficients"], dtype=float)
        raw = np.c_[np.ones(len(z)), z] @ coefficients
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    platt_a, platt_b = bundle["platt_a_b"]
    return _sigmoid(float(platt_a) + float(platt_b) * raw)


def score_universe(
    tickers_path: Path,
    archive_path: Path,
    bundle_path: Path,
    target_period_end: object,
    as_of: object,
) -> BatchResult:
    tickers_path, archive_path, bundle_path = map(Path, (tickers_path, archive_path, bundle_path))
    target_end = pd.Timestamp(target_period_end).normalize()
    if not target_end.is_quarter_end:
        raise ValueError("Target period must end on 31 March, 30 June, 30 September, or 31 December.")
    cutoff = as_ist(as_of)
    bundle = load_bundle(bundle_path)
    model_cutoff = as_ist(bundle["calibration_release_cutoff_ist"])
    if cutoff < model_cutoff:
        raise ValueError(f"This model uses calibration results through {model_cutoff}; choose a later as-of time.")

    universe = load_tickers(tickers_path)
    eligible, archive_info = load_eligible_panel(archive_path)
    eligible = eligible.loc[eligible["knowledge_ts"].le(cutoff.tz_convert("UTC"))]
    groups = {symbol: group for symbol, group in eligible.groupby("symbol", sort=False)}
    previous_end = target_end - pd.offsets.QuarterEnd(1)
    rows: list[dict] = []
    feature_rows: list[dict] = []
    feature_positions: list[int] = []

    for input_row in universe.itertuples(index=False):
        record = {
            "input_row": input_row.input_row,
            "company": input_row.company,
            "ticker": input_row.ticker,
            "industry": input_row.industry,
            "status": "Unscored",
            "reason": "",
            "probability_up": float("nan"),
            "probability_down": float("nan"),
            "rank": pd.NA,
            "issuer_isin": "",
            "basis": "",
            "last_report_period": "",
            "last_net_profit_cr": float("nan"),
            "last_xbrl_available_ist": "",
            "history_quarters": 0,
            "missing_financial_features": pd.NA,
            "recent_profit_up_share": float("nan"),
        }
        if not input_row.symbol or input_row.symbol == "NAN":
            record["reason"] = "Empty ticker"
            rows.append(record)
            continue
        matching = groups.get(input_row.symbol)
        if matching is None or matching.empty:
            record["reason"] = "No eligible filing under this ticker by the as-of time"
            rows.append(record)
            continue
        issuer_isin = matching.sort_values("knowledge_ts").iloc[-1]["isin"]
        issuer = matching.loc[matching["isin"].eq(issuer_isin)].sort_values(
            ["period_end", "knowledge_ts"]
        )
        basis = issuer.iloc[-1]["basis"]
        history = issuer.loc[issuer["basis"].eq(basis)].drop_duplicates(
            "period_end", keep="last"
        ).sort_values("period_end")
        latest = history.iloc[-1]
        record.update({
            "issuer_isin": str(issuer_isin),
            "basis": str(basis),
            "last_report_period": str(latest["period_end"].date()),
            "last_net_profit_cr": float(latest["net_profit_cr"]),
            "last_xbrl_available_ist": latest["knowledge_ts"].tz_convert(IST).isoformat(),
            "history_quarters": len(history),
        })
        if latest["period_end"] >= target_end:
            record["reason"] = "Target quarter already reported" if latest["period_end"] == target_end else "Target quarter has passed"
        elif latest["period_end"] != previous_end:
            record["reason"] = f"Latest eligible quarter is not {previous_end.date()}"
        elif latest["basis_switch"] != 0:
            record["reason"] = "Latest filing has an accounting-basis switch"
        else:
            features = make_features(history, target_end)
            record["status"] = "Scored"
            record["missing_financial_features"] = sum(
                not np.isfinite(value) for name, value in features.items()
                if not name.startswith("target_q")
            )
            record["recent_profit_up_share"] = features["np_up_share_4"]
            feature_positions.append(len(rows))
            feature_rows.append(features)
        rows.append(record)

    if feature_rows:
        scores = probabilities(pd.DataFrame(feature_rows), bundle)
        for position, probability in zip(feature_positions, scores, strict=True):
            rows[position]["probability_up"] = float(probability)
            rows[position]["probability_down"] = float(1.0 - probability)
    result = pd.DataFrame(rows)
    scored = result.loc[result["status"].eq("Scored")].sort_values(
        ["probability_up", "ticker"], ascending=[False, True]
    )
    for rank, index in enumerate(scored.index, start=1):
        result.at[index, "rank"] = rank
    result["_scored_first"] = result["status"].eq("Scored")
    result = result.sort_values(
        ["_scored_first", "rank", "input_row"], ascending=[False, True, True]
    ).drop(columns="_scored_first")
    summary = {
        "input_count": len(universe),
        "unique_input_tickers": universe["symbol"].nunique(),
        "scored_count": len(scored),
        "unscored_count": len(universe) - len(scored),
        "target_period_end": str(target_end.date()),
        "as_of_ist": cutoff.isoformat(),
        "previous_period_end": str(previous_end.date()),
        "model_kind": bundle.get("model"),
        "model_training_n": bundle["training_n"],
        "model_calibration_n": bundle["calibration_n"],
        "model_calibration_cutoff_ist": str(model_cutoff),
        "archive_file": str(archive_path),
        "archive_info": archive_info,
        "status_counts": result["reason"].where(result["status"].ne("Scored"), "Scored").value_counts().to_dict(),
    }
    return BatchResult(result.reset_index(drop=True), summary)
