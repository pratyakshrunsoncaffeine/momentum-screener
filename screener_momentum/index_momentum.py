from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .sector_rotation import NseSectorIndexProvider, normalize_sector_prices


INDEX_MOMENTUM_PERIODS: dict[str, int] = {
    "2D Return %": 2,
    "5D Return %": 5,
    "10D Return %": 10,
    "1M Return %": 21,
    "2M Return %": 42,
    "3M Return %": 63,
}

DEFAULT_INDEX_MOMENTUM_WEIGHTS: dict[str, float] = {
    "2D Return %": 0.10,
    "5D Return %": 0.20,
    "10D Return %": 0.25,
    "1M Return %": 0.25,
    "2M Return %": 0.15,
    "3M Return %": 0.05,
}

SHORT_TERM_INDEX_FILTERS = ("2D Return %", "5D Return %", "10D Return %")


NSE_INDEX_CATALOGUE: tuple[tuple[str, str], ...] = (
    ("Derivatives Eligible", "Nifty 50"),
    ("Derivatives Eligible", "Nifty Next 50"),
    ("Derivatives Eligible", "Nifty Bank"),
    ("Derivatives Eligible", "Nifty Financial Services"),
    ("Derivatives Eligible", "Nifty Midcap Select"),
    ("Derivatives Eligible", "Nifty India FPI 150"),
    ("Broad Market", "Nifty 100"),
    ("Broad Market", "Nifty 200"),
    ("Broad Market", "Nifty 500"),
    ("Broad Market", "Nifty Midcap 50"),
    ("Broad Market", "Nifty Midcap 100"),
    ("Broad Market", "Nifty Smallcap 100"),
    ("Broad Market", "India VIX"),
    ("Broad Market", "Nifty Midcap 150"),
    ("Broad Market", "Nifty Smallcap 50"),
    ("Broad Market", "Nifty Smallcap 250"),
    ("Broad Market", "Nifty Midsmallcap 400"),
    ("Broad Market", "Nifty500 Multicap 50:25:25"),
    ("Broad Market", "Nifty Largemidcap 250"),
    ("Broad Market", "Nifty Total Market"),
    ("Broad Market", "Nifty Microcap 250"),
    ("Broad Market", "Nifty500 Largemidsmall Equal-Cap Weighted"),
    ("Broad Market", "Nifty Smallcap 500"),
    ("Broad Market", "Nifty Midsmallcap400 50:50"),
    ("Sectoral", "Nifty Auto"),
    ("Sectoral", "Nifty Financial Services 25/50"),
    ("Sectoral", "Nifty FMCG"),
    ("Sectoral", "Nifty IT"),
    ("Sectoral", "Nifty Media"),
    ("Sectoral", "Nifty Metal"),
    ("Sectoral", "Nifty Pharma"),
    ("Sectoral", "Nifty PSU Bank"),
    ("Sectoral", "Nifty Private Bank"),
    ("Sectoral", "Nifty Realty"),
    ("Sectoral", "Nifty Healthcare Index"),
    ("Sectoral", "Nifty Consumer Durables"),
    ("Sectoral", "Nifty Oil & Gas"),
    ("Sectoral", "Nifty Midsmall Healthcare"),
    ("Sectoral", "Nifty Financial Services Ex-Bank"),
    ("Sectoral", "Nifty Midsmall Financial Services"),
    ("Sectoral", "Nifty Midsmall IT & Telecom"),
    ("Sectoral", "Nifty Chemicals"),
    ("Sectoral", "Nifty500 Healthcare"),
    ("Sectoral", "Nifty REITs & Realty"),
    ("Sectoral", "Nifty Cement"),
    ("Strategy", "Nifty Dividend Opportunities 50"),
    ("Strategy", "Nifty Growth Sectors 15"),
    ("Strategy", "Nifty100 Quality 30"),
    ("Strategy", "Nifty50 Value 20"),
    ("Strategy", "Nifty50 TR 2x Leverage"),
    ("Strategy", "Nifty50 PR 2x Leverage"),
    ("Strategy", "Nifty50 TR 1x Inverse"),
    ("Strategy", "Nifty50 PR 1x Inverse"),
    ("Strategy", "Nifty50 Dividend Points"),
    ("Strategy", "Nifty Alpha 50"),
    ("Strategy", "Nifty50 Equal Weight"),
    ("Strategy", "Nifty100 Equal Weight"),
    ("Strategy", "Nifty100 Low Volatility 30"),
    ("Strategy", "Nifty200 Quality 30"),
    ("Strategy", "Nifty Alpha Low-Volatility 30"),
    ("Strategy", "Nifty200 Momentum 30"),
    ("Strategy", "Nifty Midcap150 Quality 50"),
    ("Strategy", "Nifty200 Alpha 30"),
    ("Strategy", "Nifty Midcap150 Momentum 50"),
    ("Strategy", "Nifty500 Momentum 50"),
    ("Strategy", "Nifty Midsmallcap400 Momentum Quality 100"),
    ("Strategy", "Nifty Smallcap250 Momentum Quality 100"),
    ("Strategy", "Nifty Top 10 Equal Weight"),
    ("Strategy", "Nifty Alpha Quality Low-Volatility 30"),
    ("Strategy", "Nifty Alpha Quality Value Low-Volatility 30"),
    ("Strategy", "Nifty High Beta 50"),
    ("Strategy", "Nifty Low Volatility 50"),
    ("Strategy", "Nifty Quality Low-Volatility 30"),
    ("Strategy", "Nifty Smallcap250 Quality 50"),
    ("Strategy", "Nifty Top 15 Equal Weight"),
    ("Strategy", "Nifty100 Alpha 30"),
    ("Strategy", "Nifty200 Value 30"),
    ("Strategy", "Nifty500 Equal Weight"),
    ("Strategy", "Nifty500 Multicap Momentum Quality 50"),
    ("Strategy", "Nifty500 Value 50"),
    ("Strategy", "Nifty Top 20 Equal Weight"),
    ("Strategy", "Nifty500 Quality 50"),
    ("Strategy", "Nifty500 Low Volatility 50"),
    ("Strategy", "Nifty500 Multifactor MQVLV 50"),
    ("Strategy", "Nifty50 USD"),
    ("Strategy", "Nifty500 Flexicap Quality 30"),
    ("Strategy", "Nifty Total Market Momentum Quality 50"),
    ("Thematic", "Nifty Commodities"),
    ("Thematic", "Nifty India Consumption"),
    ("Thematic", "Nifty CPSE"),
    ("Thematic", "Nifty Energy"),
    ("Thematic", "Nifty Infrastructure"),
    ("Thematic", "Nifty100 Liquid 15"),
    ("Thematic", "Nifty Midcap Liquid 15"),
    ("Thematic", "Nifty MNC"),
    ("Thematic", "Nifty PSE"),
    ("Thematic", "Nifty Services Sector"),
    ("Thematic", "Nifty100 ESG Sector Leaders"),
    ("Thematic", "Nifty India Digital"),
    ("Thematic", "Nifty100 ESG"),
    ("Thematic", "Nifty India Manufacturing"),
    ("Thematic", "Nifty India Corporate Group Index - Tata Group 25% Cap"),
    ("Thematic", "Nifty500 Multicap India Manufacturing 50:30:20"),
    ("Thematic", "Nifty500 Multicap Infrastructure 50:30:20"),
    ("Thematic", "Nifty India Defence"),
    ("Thematic", "Nifty India Tourism"),
    ("Thematic", "Nifty Capital Markets"),
    ("Thematic", "Nifty EV & New Age Automotive"),
    ("Thematic", "Nifty India New Age Consumption"),
    ("Thematic", "Nifty India Select 5 Corporate Groups (MAATR)"),
    ("Thematic", "Nifty Mobility"),
    ("Thematic", "Nifty100 Enhanced ESG"),
    ("Thematic", "Nifty Core Housing"),
    ("Thematic", "Nifty Housing"),
    ("Thematic", "Nifty IPO"),
    ("Thematic", "Nifty Midsmall India Consumption"),
    ("Thematic", "Nifty Non-Cyclical Consumer"),
    ("Thematic", "Nifty Rural"),
    ("Thematic", "Nifty Shariah 25"),
    ("Thematic", "Nifty Transportation & Logistics"),
    ("Thematic", "Nifty50 Shariah"),
    ("Thematic", "Nifty500 Shariah"),
    ("Thematic", "Nifty SME Emerge"),
    ("Thematic", "Nifty India Internet"),
    ("Thematic", "Nifty Waves"),
    ("Thematic", "Nifty India Infrastructure & Logistics"),
    ("Thematic", "Nifty India Railways PSU"),
    ("Thematic", "Nifty Conglomerate 50"),
)


def index_catalogue_frame() -> pd.DataFrame:
    return pd.DataFrame(NSE_INDEX_CATALOGUE, columns=["Category", "Index"])


def calculate_index_momentum(
    prices: pd.DataFrame,
    selected_indices: Iterable[str],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Calculate a recent-return-weighted momentum score for each NSE index independently."""
    normalized = normalize_sector_prices(prices)
    if normalized.empty:
        return pd.DataFrame()
    score_weights = weights or DEFAULT_INDEX_MOMENTUM_WEIGHTS
    total_weight = sum(float(score_weights.get(label, 0.0)) for label in INDEX_MOMENTUM_PERIODS)
    if total_weight <= 0:
        raise ValueError("Index momentum weights must add to more than zero.")

    categories = index_catalogue_frame().drop_duplicates("Index").set_index("Index")["Category"].to_dict()
    requested = list(dict.fromkeys(str(item).strip() for item in selected_indices if str(item).strip()))
    rows: list[dict[str, object]] = []
    for index_name in requested:
        series = (
            normalized.loc[normalized["Index"].eq(index_name), ["Date", "Close"]]
            .drop_duplicates("Date", keep="last")
            .sort_values("Date")
            .set_index("Date")["Close"]
        )
        row: dict[str, object] = {
            "Category": categories.get(index_name, "Other"),
            "Index": index_name,
            "Latest Close": float(series.iloc[-1]) if not series.empty else np.nan,
            "Data Date": series.index[-1].date() if not series.empty else pd.NaT,
            "Data Points": int(series.shape[0]),
        }
        for label, days in INDEX_MOMENTUM_PERIODS.items():
            row[label] = _period_return(series, days)

        required_values = [row[label] for label in INDEX_MOMENTUM_PERIODS]
        if any(pd.isna(value) for value in required_values):
            row["Momentum Score"] = np.nan
            row["Short-Term Positive"] = False
            row["Momentum Status"] = "Insufficient History"
        else:
            row["Momentum Score"] = round(
                sum(float(row[label]) * float(score_weights.get(label, 0.0)) for label in INDEX_MOMENTUM_PERIODS)
                / total_weight,
                3,
            )
            positive = all(float(row[label]) > 0 for label in SHORT_TERM_INDEX_FILTERS)
            row["Short-Term Positive"] = positive
            if positive:
                row["Momentum Status"] = "Positive Momentum"
            elif all(float(row[label]) <= 0 for label in SHORT_TERM_INDEX_FILTERS):
                row["Momentum Status"] = "Weak Momentum"
            else:
                row["Momentum Status"] = "Mixed Momentum"
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["Momentum Score", "10D Return %", "Index"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    valid = result["Momentum Score"].notna()
    result.insert(0, "Momentum Rank", pd.Series(pd.NA, index=result.index, dtype="Int64"))
    result.loc[valid, "Momentum Rank"] = range(1, int(valid.sum()) + 1)
    return result


def normalized_index_performance(
    prices: pd.DataFrame,
    indices: Iterable[str],
    trading_days: int = 126,
) -> pd.DataFrame:
    normalized = normalize_sector_prices(prices)
    selected = list(dict.fromkeys(str(item) for item in indices))
    frames: list[pd.DataFrame] = []
    for index_name in selected:
        series = (
            normalized.loc[normalized["Index"].eq(index_name), ["Date", "Close"]]
            .drop_duplicates("Date", keep="last")
            .sort_values("Date")
            .tail(int(trading_days) + 1)
        )
        if series.empty or float(series["Close"].iloc[0]) <= 0:
            continue
        series = series.copy()
        series["Normalized Value"] = series["Close"] / float(series["Close"].iloc[0]) * 100.0
        series["Index"] = index_name
        frames.append(series[["Date", "Index", "Normalized Value"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _period_return(series: pd.Series, trading_days: int) -> float:
    if series.shape[0] <= int(trading_days):
        return np.nan
    current = float(series.iloc[-1])
    previous = float(series.iloc[-int(trading_days) - 1])
    if previous <= 0:
        return np.nan
    return round((current / previous - 1.0) * 100.0, 3)
