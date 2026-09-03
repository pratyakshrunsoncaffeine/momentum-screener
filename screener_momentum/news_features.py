from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import re

import numpy as np
import pandas as pd

from .news_config import IST, NEWS_CUTOFF, NewsCatalystConfig, signal_cutoff_for
from .news_sources import normalize_news_frame, normalize_text
from .sector_rotation import normalize_sector_prices


INDEX_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AUTO": ("auto", "automobile", "vehicle", "car", "two wheeler", "ev", "electric vehicle"),
    "BANK": ("bank", "lending", "deposit", "credit growth", "rbi", "interest rate", "repo rate"),
    "FINANCIAL": ("finance", "nbfc", "insurance", "asset management", "capital market", "lending"),
    "FMCG": ("fmcg", "consumer staples", "rural demand", "food prices", "household products"),
    "HEALTHCARE": ("healthcare", "hospital", "diagnostic", "medical", "drug", "pharma"),
    "PHARMA": ("pharma", "drug", "medicine", "usfda", "clinical trial", "generic"),
    "IT": ("information technology", "software", "technology", "ai", "digital", "outsourcing"),
    "MEDIA": ("media", "broadcast", "advertising", "entertainment", "streaming"),
    "METAL": ("metal", "steel", "aluminium", "copper", "zinc", "iron ore"),
    "OIL & GAS": ("oil", "gas", "crude", "brent", "refinery", "petroleum", "lng"),
    "REALTY": ("realty", "real estate", "housing", "property", "home sales"),
    "INFRASTRUCTURE": ("infrastructure", "roads", "highway", "construction", "capex"),
    "DEFENCE": ("defence", "defense", "military", "arms", "missile", "geopolitical"),
    "CONSUMPTION": ("consumption", "consumer", "retail", "demand", "discretionary"),
    "COMMODITIES": ("commodity", "oil", "gold", "silver", "metal", "agriculture"),
    "ENERGY": ("energy", "power", "electricity", "renewable", "oil", "gas", "coal"),
    "PSU": ("public sector", "psu", "government company", "divestment", "privatisation"),
    "CEMENT": ("cement", "construction material", "clinker"),
    "CHEMICAL": ("chemical", "specialty chemical", "fertilizer", "petrochemical"),
    "RAIL": ("railway", "rail", "irctc", "freight corridor"),
    "TOURISM": ("tourism", "hotel", "travel", "airline", "hospitality"),
    "TELECOM": ("telecom", "5g", "spectrum", "mobile subscriber"),
}

MACRO_KEYWORDS = (
    "india",
    "indian economy",
    "nifty",
    "nse",
    "rbi",
    "reserve bank",
    "sebi",
    "inflation",
    "gdp",
    "fiscal",
    "budget",
    "interest rate",
    "rupee",
)

NEWS_BASE_FEATURES = (
    "news_article_count",
    "news_source_count",
    "news_relevance_sum",
    "news_sentiment_raw",
    "news_sentiment_shrunk",
    "news_positive_probability",
    "news_negative_probability",
    "news_neutral_probability",
    "news_gdelt_tone",
    "news_text_coverage",
    "news_source_diversity",
    "news_novelty",
)


@dataclass
class FinBertFeatureExtractor:
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 32
    _classifier: object | None = field(default=None, init=False, repr=False)

    def transform(self, articles: pd.DataFrame, allow_tone_fallback: bool = True) -> pd.DataFrame:
        result = normalize_news_frame(articles).copy()
        result["Positive Probability"] = np.nan
        result["Negative Probability"] = np.nan
        result["Neutral Probability"] = np.nan
        text_mask = result["Text Available"].fillna(False).astype(bool)
        texts = (
            result.loc[text_mask, "Title"].fillna("") + ". " + result.loc[text_mask, "Snippet"].fillna("")
        ).map(normalize_text)
        if not texts.empty:
            try:
                from transformers import pipeline

                if self._classifier is None:
                    self._classifier = pipeline(
                        "text-classification",
                        model=self.model_name,
                        tokenizer=self.model_name,
                        return_all_scores=True,
                        truncation=True,
                    )
                outputs = self._classifier(texts.tolist(), batch_size=self.batch_size)
                for row_index, scores in zip(texts.index, outputs):
                    lookup = {str(item["label"]).lower(): float(item["score"]) for item in scores}
                    result.loc[row_index, "Positive Probability"] = _label_score(lookup, "positive")
                    result.loc[row_index, "Negative Probability"] = _label_score(lookup, "negative")
                    result.loc[row_index, "Neutral Probability"] = _label_score(lookup, "neutral")
            except (ImportError, OSError, RuntimeError):
                if not allow_tone_fallback:
                    raise RuntimeError(
                        "FinBERT is unavailable. Install requirements-ml.txt or enable GDELT-tone fallback."
                    )
        if allow_tone_fallback:
            missing = result["Positive Probability"].isna()
            tone = pd.to_numeric(result["GDELT Tone"], errors="coerce").fillna(0.0).clip(-10, 10) / 10.0
            positive = ((tone + 1.0) / 2.0).clip(0.05, 0.90)
            negative = ((1.0 - tone) / 2.0).clip(0.05, 0.90)
            neutral = (1.0 - positive - negative).clip(0.05, 0.90)
            total = positive + negative + neutral
            result.loc[missing, "Positive Probability"] = positive[missing] / total[missing]
            result.loc[missing, "Negative Probability"] = negative[missing] / total[missing]
            result.loc[missing, "Neutral Probability"] = neutral[missing] / total[missing]
            result["Sentiment Source"] = np.where(text_mask & ~missing, "FinBERT", "GDELT tone fallback")
        result["Sentiment Score"] = result["Positive Probability"] - result["Negative Probability"]
        return result


@dataclass
class SentenceEmbeddingExtractor:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    _encoder: object | None = field(default=None, init=False, repr=False)

    def transform(self, articles: pd.DataFrame) -> pd.DataFrame:
        result = articles.copy()
        text = (result.get("Title", "").fillna("") + ". " + result.get("Snippet", "").fillna("")).map(
            normalize_text
        )
        result["Embedding"] = pd.Series([None] * len(result), dtype=object)
        usable = text.str.len() > 0
        if not usable.any():
            return result
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Sentence embeddings require sentence-transformers from requirements-ml.txt.") from exc
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model_name)
        vectors = self._encoder.encode(
            text[usable].tolist(), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        result.loc[usable, "Embedding"] = pd.Series([vector.tolist() for vector in vectors], index=text[usable].index)
        return result


def assign_signal_dates(
    articles: pd.DataFrame,
    trading_dates: Iterable[object],
    cutoff=NEWS_CUTOFF,
) -> pd.DataFrame:
    result = articles.copy()
    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates), errors="coerce")).dropna().normalize().unique().sort_values()
    if dates.empty:
        result["Signal Date"] = pd.NaT
        return result
    cutoffs = pd.DatetimeIndex([pd.Timestamp(signal_cutoff_for(item.date(), cutoff)).tz_convert("UTC") for item in dates])
    published = pd.to_datetime(result["Published At UTC"], utc=True, errors="coerce")
    positions = np.searchsorted(cutoffs.asi8, published.astype("int64"), side="left")
    result["Signal Date"] = [
        dates[position] if pd.notna(timestamp) and 0 <= position < len(dates) else pd.NaT
        for timestamp, position in zip(published, positions)
    ]
    return result


def map_articles_to_indices(
    articles: pd.DataFrame,
    index_catalogue: pd.DataFrame,
    constituents: pd.DataFrame | None = None,
    minimum_relevance: float = 0.20,
) -> pd.DataFrame:
    if articles.empty or index_catalogue.empty:
        return pd.DataFrame(
            columns=["Article ID", "Index", "Relevance", "Attribution Method", "Attribution Reason"]
        )
    company_rows = _constituent_aliases(constituents)
    rows: list[dict[str, object]] = []
    for item in articles.to_dict("records"):
        article_id = str(item.get("Article ID", ""))
        published = pd.Timestamp(item.get("Published At UTC"))
        text = normalize_text(f"{item.get('Title', '')} {item.get('Snippet', '')} {item.get('Organizations', '')}").lower()
        if not text:
            text = normalize_text(f"{item.get('Themes', '')} {item.get('Organizations', '')}").lower()
        for index_row in index_catalogue.itertuples(index=False):
            index_name = str(getattr(index_row, "Index"))
            category = str(getattr(index_row, "Category", ""))
            reasons: list[str] = []
            score = 0.0
            index_tokens = _index_keywords(index_name, category)
            matched = [keyword for keyword in index_tokens if _contains_term(text, keyword)]
            if matched:
                score += min(0.70, 0.18 + len(matched) * 0.12)
                reasons.append("topics: " + ", ".join(matched[:5]))
            aliases = company_rows.get(index_name, [])
            valid_aliases = [
                alias
                for alias, valid_from, valid_to in aliases
                if _membership_valid(published, valid_from, valid_to) and _contains_term(text, alias)
            ]
            if valid_aliases:
                score = max(score, min(1.0, 0.75 + 0.05 * len(valid_aliases)))
                reasons.append("constituents: " + ", ".join(valid_aliases[:4]))
            semantic_similarity = _semantic_similarity(item.get("Embedding"), getattr(index_row, "Embedding", None))
            if semantic_similarity >= 0.25:
                score = max(score, min(0.70, semantic_similarity))
                reasons.append(f"semantic similarity: {semantic_similarity:.2f}")
            if category == "Broad Market" and any(_contains_term(text, word) for word in MACRO_KEYWORDS):
                score = max(score, 0.35)
                reasons.append("India macro/market news")
            if score >= minimum_relevance:
                rows.append(
                    {
                        "Article ID": article_id,
                        "Index": index_name,
                        "Relevance": round(min(score, 1.0), 4),
                        "Attribution Method": "point-in-time constituent + topic taxonomy + semantic similarity",
                        "Attribution Reason": "; ".join(reasons),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(["Article ID", "Index"], keep="last")


def build_daily_news_features(
    articles: pd.DataFrame,
    links: pd.DataFrame,
    index_prices: pd.DataFrame,
    config: NewsCatalystConfig | None = None,
    constituent_activity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    settings = config or NewsCatalystConfig()
    prices = normalize_sector_prices(index_prices)
    if prices.empty:
        raise ValueError("Index price history is required to build point-in-time news features.")
    enriched = articles.copy()
    if "Sentiment Score" not in enriched:
        enriched = FinBertFeatureExtractor().transform(enriched)
    enriched = assign_signal_dates(enriched, prices["Date"].drop_duplicates(), settings.cutoff)
    merged = links.merge(enriched, on="Article ID", how="inner")
    merged = merged.dropna(subset=["Signal Date"])
    if merged.empty:
        news_daily = pd.DataFrame(columns=["Date", "Index", *NEWS_BASE_FEATURES])
    else:
        merged["Weighted Sentiment"] = merged["Sentiment Score"] * merged["Relevance"]
        grouped = merged.groupby(["Signal Date", "Index"], observed=True)
        news_daily = grouped.agg(
            news_article_count=("Article ID", "nunique"),
            news_source_count=("Publisher", "nunique"),
            news_relevance_sum=("Relevance", "sum"),
            news_sentiment_raw=("Weighted Sentiment", "sum"),
            news_positive_probability=("Positive Probability", "mean"),
            news_negative_probability=("Negative Probability", "mean"),
            news_neutral_probability=("Neutral Probability", "mean"),
            news_gdelt_tone=("GDELT Tone", "mean"),
            news_text_coverage=("Text Available", "mean"),
            news_novelty=("Title", lambda values: values.nunique() / max(len(values), 1)),
        ).reset_index().rename(columns={"Signal Date": "Date"})
        denominator = news_daily["news_relevance_sum"].replace(0, np.nan)
        news_daily["news_sentiment_raw"] = news_daily["news_sentiment_raw"] / denominator
        news_daily["news_source_diversity"] = (
            news_daily["news_source_count"] / news_daily["news_article_count"].clip(lower=1)
        )
        global_mean = news_daily.groupby("Date")["news_sentiment_raw"].transform("mean").fillna(0.0)
        count = news_daily["news_article_count"].astype(float)
        prior = settings.shrinkage.sentiment_prior_strength
        news_daily["news_sentiment_shrunk"] = (
            count * news_daily["news_sentiment_raw"].fillna(global_mean) + prior * global_mean
        ) / (count + prior)
        embedding_daily = _aggregate_daily_embeddings(merged)
        if not embedding_daily.empty:
            news_daily = news_daily.merge(embedding_daily, on=["Date", "Index"], how="left")
    base = prices[["Date", "Index", "Close"]].drop_duplicates(["Date", "Index"], keep="last")
    feature_frame = base.merge(news_daily, on=["Date", "Index"], how="left")
    for column in NEWS_BASE_FEATURES:
        if column not in feature_frame:
            feature_frame[column] = 0.0
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(0.0)
    feature_frame["news_reliable"] = (
        feature_frame["news_article_count"].ge(settings.shrinkage.minimum_articles)
        & feature_frame["news_source_count"].ge(settings.shrinkage.minimum_sources)
    ).astype(float)
    feature_frame["news_sentiment_reliable"] = (
        feature_frame["news_sentiment_shrunk"] * feature_frame["news_reliable"]
    )
    feature_frame = _add_exponential_news_windows(feature_frame)
    feature_frame = _add_price_features(feature_frame, settings.benchmark_index)
    if constituent_activity is not None and not constituent_activity.empty:
        activity = constituent_activity.copy()
        activity["Date"] = pd.to_datetime(activity["Date"], errors="coerce").dt.normalize()
        feature_frame = feature_frame.merge(activity, on=["Date", "Index"], how="left")
    for column in ("volume_abnormal", "turnover_abnormal", "breadth_positive", "participation_rate"):
        if column not in feature_frame:
            feature_frame[column] = np.nan
    feature_frame["Feature Cutoff UTC"] = feature_frame["Date"].map(
        lambda value: pd.Timestamp(signal_cutoff_for(pd.Timestamp(value).date(), settings.cutoff)).tz_convert("UTC")
    )
    return feature_frame.sort_values(["Date", "Index"]).reset_index(drop=True)


def build_forward_labels(
    features: pd.DataFrame,
    benchmark_index: str = "Nifty 50",
    horizons: dict[str, int] | None = None,
) -> pd.DataFrame:
    periods = horizons or {"5D": 5, "1M": 21, "3M": 63}
    result = features.copy().sort_values(["Index", "Date"])
    close = pd.to_numeric(result["Close"], errors="coerce")
    grouped_close = close.groupby(result["Index"])
    benchmark = (
        result.loc[result["Index"].eq(benchmark_index), ["Date", "Close"]]
        .drop_duplicates("Date", keep="last")
        .set_index("Date")["Close"]
        .sort_index()
    )
    for label, days in periods.items():
        future = grouped_close.shift(-int(days))
        absolute = (future / close - 1.0) * 100.0
        benchmark_forward = (benchmark.shift(-int(days)) / benchmark - 1.0) * 100.0
        benchmark_return = result["Date"].map(benchmark_forward)
        result[f"Absolute Return {label} %"] = absolute
        result[f"Benchmark Return {label} %"] = benchmark_return
        result[f"Excess Return {label} %"] = absolute - benchmark_return
        excess = result[f"Excess Return {label} %"]
        positive = (excess > 0).astype("Int64")
        result[f"Positive Excess {label}"] = positive.mask(excess.isna(), pd.NA)
    return result


def winsorize_features(
    frame: pd.DataFrame,
    columns: Iterable[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    result = frame.copy()
    limits: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        low = float(values.quantile(lower)) if values.notna().any() else 0.0
        high = float(values.quantile(upper)) if values.notna().any() else 0.0
        limits[column] = (low, high)
        result[column] = values.clip(low, high)
    return result, limits


def apply_winsor_limits(
    frame: pd.DataFrame, limits: dict[str, tuple[float, float]]
) -> pd.DataFrame:
    result = frame.copy()
    for column, (low, high) in limits.items():
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").clip(low, high)
    return result


def inject_training_noise(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: NewsCatalystConfig | None = None,
) -> pd.DataFrame:
    settings = config or NewsCatalystConfig()
    noise = settings.shrinkage
    rng = np.random.default_rng(settings.random_seed)
    result = frame.copy()
    numeric = result[feature_columns].apply(pd.to_numeric, errors="coerce")
    jitter_columns = [column for column in feature_columns if "sentiment" in column or "price_" in column or "volume_" in column]
    for column in jitter_columns:
        scale = numeric[column].std(skipna=True)
        if pd.notna(scale) and scale > 0:
            result[column] = numeric[column] + rng.normal(0.0, scale * noise.feature_jitter_pct, len(result))
    embedding_columns = [column for column in feature_columns if column.startswith("embedding_")]
    for column in embedding_columns:
        result[column] = numeric[column] + rng.normal(0.0, noise.embedding_noise_std, len(result))
    headline_mask = rng.random(len(result)) < noise.headline_dropout
    for column in [item for item in feature_columns if item.startswith("news_")]:
        result.loc[headline_mask, column] = pd.to_numeric(result.loc[headline_mask, column], errors="coerce") * 0.9
    source_mask = rng.random(len(result)) < noise.source_mask_probability
    for column in ("news_source_count", "news_source_diversity"):
        if column in result:
            result.loc[source_mask, column] = 0.0
    if "news_relevance_sum" in result:
        relevance = pd.to_numeric(result["news_relevance_sum"], errors="coerce")
        result["news_relevance_sum"] = relevance * rng.normal(
            1.0, noise.relevance_jitter_pct, len(result)
        ).clip(0.75, 1.25)
    return result


def _aggregate_daily_embeddings(merged: pd.DataFrame) -> pd.DataFrame:
    if "Embedding" not in merged or not merged["Embedding"].notna().any():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (signal_date, index_name), group in merged.groupby(["Signal Date", "Index"], observed=True):
        vectors: list[np.ndarray] = []
        weights: list[float] = []
        for vector, relevance in zip(group["Embedding"], group["Relevance"]):
            try:
                array = np.asarray(vector, dtype=float)
            except (TypeError, ValueError):
                continue
            if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
                continue
            vectors.append(array)
            weights.append(max(float(relevance), 0.0))
        if not vectors or len({vector.size for vector in vectors}) != 1:
            continue
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.sum() <= 0:
            weight_array = np.ones(len(vectors), dtype=float)
        mean = np.average(np.vstack(vectors), axis=0, weights=weight_array)
        row: dict[str, object] = {"Date": signal_date, "Index": index_name}
        row.update({f"embedding_{position:03d}": value for position, value in enumerate(mean)})
        rows.append(row)
    return pd.DataFrame(rows)


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded_prefixes = (
        "Absolute Return ",
        "Benchmark Return ",
        "Excess Return ",
        "Positive Excess ",
    )
    excluded = {"Date", "Index", "Close", "Feature Cutoff UTC"}
    return [
        column
        for column in frame.columns
        if column not in excluded
        and not column.startswith(excluded_prefixes)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _add_exponential_news_windows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["Index", "Date"]).copy()
    for span in (3, 7, 21, 63):
        for source in ("news_sentiment_shrunk", "news_article_count", "news_relevance_sum"):
            result[f"{source}_ewm_{span}"] = result.groupby("Index", observed=True)[source].transform(
                lambda values: values.ewm(span=span, adjust=False, min_periods=1).mean()
            )
    result["news_acceleration_7_21"] = (
        result["news_sentiment_shrunk_ewm_7"] - result["news_sentiment_shrunk_ewm_21"]
    )
    result["news_acceleration_21_63"] = (
        result["news_sentiment_shrunk_ewm_21"] - result["news_sentiment_shrunk_ewm_63"]
    )
    return result


def _add_price_features(frame: pd.DataFrame, benchmark_index: str) -> pd.DataFrame:
    result = frame.sort_values(["Index", "Date"]).copy()
    close = pd.to_numeric(result["Close"], errors="coerce")
    grouped = close.groupby(result["Index"])
    for days in (1, 5, 21, 63):
        result[f"price_return_{days}d"] = grouped.pct_change(days) * 100.0
    result["price_volatility_21d"] = grouped.pct_change().groupby(result["Index"]).transform(
        lambda values: values.rolling(21, min_periods=10).std() * math.sqrt(252) * 100.0
    )
    rolling_high = grouped.transform(lambda values: values.rolling(63, min_periods=20).max())
    result["price_drawdown_63d"] = (close / rolling_high - 1.0) * 100.0
    benchmark = result[result["Index"].eq(benchmark_index)].set_index("Date")
    for days in (1, 5, 21, 63):
        result[f"market_return_{days}d"] = result["Date"].map(benchmark[f"price_return_{days}d"])
    return result


def _constituent_aliases(constituents: pd.DataFrame | None) -> dict[str, list[tuple[str, object, object]]]:
    if constituents is None or constituents.empty:
        return {}
    frame = constituents.copy()
    lookup: dict[str, list[tuple[str, object, object]]] = {}
    for row in frame.to_dict("records"):
        index_name = str(row.get("Index", "")).strip()
        aliases = {
            normalize_text(row.get("Ticker", "")).lower(),
            normalize_text(row.get("Company", row.get("Name", ""))).lower(),
        }
        for alias in aliases:
            if len(alias) >= 3:
                lookup.setdefault(index_name, []).append((alias, row.get("Valid From"), row.get("Valid To")))
    return lookup


def _membership_valid(published: pd.Timestamp, valid_from: object, valid_to: object) -> bool:
    day = published.tz_convert(IST).tz_localize(None).normalize()
    start = pd.to_datetime(valid_from, errors="coerce")
    end = pd.to_datetime(valid_to, errors="coerce")
    return (pd.isna(start) or day >= start) and (pd.isna(end) or day <= end)


def _index_keywords(index_name: str, category: str) -> tuple[str, ...]:
    normalized = index_name.upper()
    keywords: set[str] = set()
    for marker, values in INDEX_TOPIC_KEYWORDS.items():
        if marker in normalized:
            keywords.update(values)
    cleaned = re.sub(r"\b(NIFTY|INDEX|INDIA|500|200|150|100|50|30|25|20|15|10)\b", " ", index_name, flags=re.I)
    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z&-]+", cleaned) if len(token) >= 3]
    keywords.update(tokens)
    if category == "Broad Market":
        keywords.update(MACRO_KEYWORDS)
    return tuple(sorted(keywords, key=len, reverse=True))


def _contains_term(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text))


def _label_score(scores: dict[str, float], wanted: str) -> float:
    for label, score in scores.items():
        if wanted in label:
            return float(score)
    return np.nan


def _semantic_similarity(first: object, second: object) -> float:
    if first is None or second is None:
        return 0.0
    try:
        left = np.asarray(first, dtype=float)
        right = np.asarray(second, dtype=float)
    except (TypeError, ValueError):
        return 0.0
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or left.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)
