from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")
NEWS_CUTOFF = time(16, 30)

NEWS_HORIZONS: dict[str, int] = {
    "5D": 5,
    "1M": 21,
    "3M": 63,
}

SPECIAL_INDEX_RULES: tuple[tuple[str, str], ...] = (
    ("INDIA VIX", "Volatility index"),
    ("LEVERAGE", "Leveraged index"),
    ("INVERSE", "Inverse index"),
    ("DIVIDEND POINT", "Dividend-point index"),
    ("TR 2X", "Leveraged index"),
    ("PR 2X", "Leveraged index"),
    ("TR 1X INVERSE", "Inverse index"),
    ("PR 1X INVERSE", "Inverse index"),
)


@dataclass(frozen=True)
class NoiseShrinkageConfig:
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    sentiment_prior_strength: float = 5.0
    embedding_components: int = 24
    minimum_articles: int = 2
    minimum_sources: int = 2
    headline_dropout: float = 0.10
    embedding_noise_std: float = 0.01
    feature_jitter_pct: float = 0.015
    source_mask_probability: float = 0.08
    relevance_jitter_pct: float = 0.05


@dataclass(frozen=True)
class NewsCatalystConfig:
    benchmark_index: str = "Nifty 50"
    history_years: int = 5
    cutoff: time = NEWS_CUTOFF
    horizons: dict[str, int] = field(default_factory=lambda: NEWS_HORIZONS.copy())
    gdelt_query: str = (
        '(india OR indian OR nse OR nifty OR "reserve bank of india" OR rbi OR sebi) '
        '(market OR economy OR company OR bank OR oil OR gold OR policy OR inflation OR earnings)'
    )
    late_arrival_hours: int = 48
    maximum_articles_per_fetch: int = 250
    bigquery_sandbox: bool = True
    bigquery_partitions_per_run: int = 90
    bigquery_monthly_budget_gib: float = 900.0
    bigquery_max_query_gib: float = 5.0
    bigquery_min_sample_pct: float = 1.0
    # Keeps five years of sampled GDELT metadata inside Supabase's free database tier.
    bigquery_result_row_limit: int = 40
    minimum_history_days: int = 252
    transaction_cost_pct: float = 0.20
    random_seed: int = 42
    shrinkage: NoiseShrinkageConfig = field(default_factory=NoiseShrinkageConfig)


def as_ist(value: datetime | None = None) -> datetime:
    current = value or datetime.now(tz=IST)
    if current.tzinfo is None:
        return current.replace(tzinfo=IST)
    return current.astimezone(IST)


def signal_cutoff_for(day: date, cutoff: time = NEWS_CUTOFF) -> datetime:
    return datetime.combine(day, cutoff, tzinfo=IST)


def after_news_cutoff(value: datetime | None = None, cutoff: time = NEWS_CUTOFF) -> bool:
    current = as_ist(value)
    return current.time().replace(tzinfo=None) >= cutoff


def index_model_eligibility(index_name: str) -> tuple[bool, str]:
    normalized = " ".join(str(index_name).upper().split())
    for token, reason in SPECIAL_INDEX_RULES:
        if token in normalized:
            return False, reason
    return True, "Eligible standard equity index"


def eligibility_rows(indices: list[str] | tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index_name in indices:
        eligible, reason = index_model_eligibility(index_name)
        rows.append({"Index": index_name, "Model Eligible": eligible, "Eligibility Reason": reason})
    return rows
