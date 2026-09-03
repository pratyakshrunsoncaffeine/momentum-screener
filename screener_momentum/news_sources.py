from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import html
import json
import os
import re
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests

from .news_config import UTC


NEWS_COLUMNS = (
    "Article ID",
    "Published At UTC",
    "Publisher",
    "Title",
    "Snippet",
    "URL",
    "Language",
    "Source Country",
    "GDELT Tone",
    "Themes",
    "Organizations",
    "Source Type",
    "Text Available",
    "Ingested At UTC",
)

TRACKING_QUERY_PREFIXES = ("utm_", "fbclid", "gclid", "mc_", "ref")


class NewsProvider(Protocol):
    def fetch(self, start: datetime, end: datetime, query: str) -> pd.DataFrame: ...


def canonicalize_url(value: object) -> str:
    raw = "" if _is_missing(value) else str(value).strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    clean_query = [
        (key, item)
        for key, item in parse_qsl(parts.query, keep_blank_values=True)
        if not any(key.lower().startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    path = re.sub(r"/{2,}", "/", parts.path or "/").rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(clean_query), ""))


def normalize_text(value: object) -> str:
    text = html.unescape("" if _is_missing(value) else str(value))
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def article_identifier(url: object, title: object, publisher: object, published_at: object) -> str:
    canonical = canonicalize_url(url)
    fallback = "|".join(
        [normalize_text(title).lower(), normalize_text(publisher).lower(), str(published_at)]
    )
    return sha256((canonical or fallback).encode("utf-8")).hexdigest()


def normalize_news_frame(frame: pd.DataFrame, source_type: str = "unknown") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    aliases = {
        "published_at": "Published At UTC",
        "published_at_utc": "Published At UTC",
        "publisher": "Publisher",
        "source": "Publisher",
        "title": "Title",
        "headline": "Title",
        "snippet": "Snippet",
        "description": "Snippet",
        "url": "URL",
        "language": "Language",
        "source_country": "Source Country",
        "tone": "GDELT Tone",
        "gdelt_tone": "GDELT Tone",
        "themes": "Themes",
        "organizations": "Organizations",
    }
    result = frame.rename(
        columns={column: aliases.get(str(column).strip().lower(), column) for column in frame.columns}
    ).copy()
    for column in NEWS_COLUMNS:
        if column not in result:
            result[column] = pd.NA
    result["Published At UTC"] = pd.to_datetime(result["Published At UTC"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    result["Ingested At UTC"] = pd.to_datetime(result["Ingested At UTC"], utc=True, errors="coerce").fillna(now)
    for column in ("Publisher", "Title", "Snippet", "Language", "Source Country", "Themes", "Organizations"):
        result[column] = result[column].map(normalize_text)
    result["URL"] = result["URL"].map(canonicalize_url)
    result["GDELT Tone"] = pd.to_numeric(result["GDELT Tone"], errors="coerce")
    result["Source Type"] = result["Source Type"].fillna(source_type).astype(str)
    result["Text Available"] = (result["Title"].str.len() > 0) | (result["Snippet"].str.len() > 0)
    result["Article ID"] = [
        article_identifier(url, title, publisher, published)
        for url, title, publisher, published in zip(
            result["URL"], result["Title"], result["Publisher"], result["Published At UTC"]
        )
    ]
    result = result.dropna(subset=["Published At UTC"]).drop_duplicates("Article ID", keep="last")
    return result.loc[:, NEWS_COLUMNS].sort_values("Published At UTC").reset_index(drop=True)


def deduplicate_articles(*frames: pd.DataFrame) -> pd.DataFrame:
    available = [normalize_news_frame(frame) for frame in frames if frame is not None and not frame.empty]
    if not available:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return pd.concat(available, ignore_index=True).drop_duplicates("Article ID", keep="last").sort_values(
        "Published At UTC"
    ).reset_index(drop=True)


@dataclass
class GdeltDocProvider:
    endpoint: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    timeout: int = 45
    maximum_records: int = 250

    def fetch(self, start: datetime, end: datetime, query: str) -> pd.DataFrame:
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": min(max(int(self.maximum_records), 1), 250),
            "startdatetime": _gdelt_datetime(start),
            "enddatetime": _gdelt_datetime(end),
        }
        response = requests.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        rows: list[dict[str, object]] = []
        for item in payload.get("articles", []):
            rows.append(
                {
                    "Published At UTC": item.get("seendate"),
                    "Publisher": item.get("domain"),
                    "Title": item.get("title"),
                    "Snippet": item.get("description", ""),
                    "URL": item.get("url"),
                    "Language": item.get("language"),
                    "Source Country": item.get("sourcecountry"),
                    "GDELT Tone": item.get("tone"),
                    "Source Type": "gdelt_doc",
                }
            )
        return normalize_news_frame(pd.DataFrame(rows), "gdelt_doc")


@dataclass
class RssNewsProvider:
    feeds: dict[str, str]

    @classmethod
    def from_environment(cls) -> "RssNewsProvider":
        raw = os.getenv("NEWS_RSS_FEEDS_JSON", "{}").strip()
        try:
            feeds = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError("NEWS_RSS_FEEDS_JSON must be a JSON object of publisher to permitted feed URL.") from exc
        if not isinstance(feeds, dict):
            raise ValueError("NEWS_RSS_FEEDS_JSON must be a JSON object of publisher to permitted feed URL.")
        return cls({str(key): str(value) for key, value in feeds.items()})

    def fetch(self, start: datetime, end: datetime, query: str = "") -> pd.DataFrame:
        del query
        try:
            import feedparser
        except ImportError as exc:
            raise RuntimeError("RSS ingestion requires the optional feedparser package.") from exc
        rows: list[dict[str, object]] = []
        start_utc = _utc_timestamp(start)
        end_utc = _utc_timestamp(end)
        for publisher, url in self.feeds.items():
            parsed = feedparser.parse(url)
            for entry in parsed.entries:
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                timestamp = pd.Timestamp(*published[:6], tz="UTC") if published else pd.NaT
                if pd.isna(timestamp) or timestamp < start_utc or timestamp > end_utc:
                    continue
                rows.append(
                    {
                        "Published At UTC": timestamp,
                        "Publisher": publisher,
                        "Title": entry.get("title", ""),
                        "Snippet": entry.get("summary", ""),
                        "URL": entry.get("link", ""),
                        "Language": parsed.feed.get("language", "English"),
                        "Source Type": "rss",
                    }
                )
        return normalize_news_frame(pd.DataFrame(rows), "rss")


@dataclass
class GdeltBigQueryProvider:
    table: str = "gdelt-bq.gdeltv2.gkg_partitioned"
    project: str | None = None
    maximum_query_bytes: int = 5 * 1024**3
    minimum_sample_percent: float = 1.0
    maximum_rows: int = 25_000

    @classmethod
    def from_environment(
        cls,
        maximum_query_gib: float = 5.0,
        minimum_sample_percent: float = 1.0,
        maximum_rows: int = 25_000,
    ) -> "GdeltBigQueryProvider":
        return cls(
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or None,
            maximum_query_bytes=max(1, int(maximum_query_gib * 1024**3)),
            minimum_sample_percent=min(max(float(minimum_sample_percent), 0.01), 100.0),
            maximum_rows=max(1, int(maximum_rows)),
        )

    def fetch(self, start: datetime, end: datetime, query: str) -> pd.DataFrame:
        frame, _ = self.fetch_with_stats(start, end, query)
        return frame

    def estimate_bytes(
        self,
        start: datetime,
        end: datetime,
        query: str,
        sample_percent: float = 100.0,
    ) -> int:
        bigquery, client = self._client()
        sql, parameters = self._query(bigquery, start, end, query, sample_percent)
        config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            query_parameters=parameters,
        )
        job = client.query(sql, job_config=config)
        return int(job.total_bytes_processed or 0)

    def plan_sample_percent(
        self,
        start: datetime,
        end: datetime,
        query: str,
        remaining_budget_bytes: int | None = None,
    ) -> tuple[float, int]:
        full_estimate = self.estimate_bytes(start, end, query)
        allowed = self.maximum_query_bytes
        if remaining_budget_bytes is not None:
            allowed = min(allowed, max(int(remaining_budget_bytes), 0))
        if allowed <= 0:
            raise BigQuerySandboxBudgetExceeded("The configured monthly BigQuery Sandbox budget is exhausted.")
        if full_estimate <= allowed:
            return 100.0, full_estimate

        sample = max(
            self.minimum_sample_percent,
            min(99.0, (allowed / max(full_estimate, 1)) * 90.0),
        )
        estimate = self.estimate_bytes(start, end, query, sample)
        while estimate > allowed and sample > self.minimum_sample_percent:
            sample = max(self.minimum_sample_percent, sample * 0.70)
            estimate = self.estimate_bytes(start, end, query, sample)
        if estimate > allowed:
            raise BigQuerySandboxBudgetExceeded(
                f"Even a {sample:.2f}% GDELT sample needs {_format_bytes(estimate)}, "
                f"above the remaining {_format_bytes(allowed)} sandbox allowance."
            )
        return round(sample, 4), estimate

    def fetch_with_stats(
        self,
        start: datetime,
        end: datetime,
        query: str,
        remaining_budget_bytes: int | None = None,
    ) -> tuple[pd.DataFrame, "GdeltQueryStats"]:
        sample_percent, estimated_bytes = self.plan_sample_percent(
            start, end, query, remaining_budget_bytes
        )
        bigquery, client = self._client()
        sql, parameters = self._query(bigquery, start, end, query, sample_percent)
        allowed = self.maximum_query_bytes
        if remaining_budget_bytes is not None:
            allowed = min(allowed, max(int(remaining_budget_bytes), 1))
        config = bigquery.QueryJobConfig(
            query_parameters=parameters,
            maximum_bytes_billed=allowed,
            use_query_cache=True,
        )
        job = client.query(sql, job_config=config)
        raw = job.to_dataframe(create_bqstorage_client=False)
        processed_bytes = int(job.total_bytes_processed or estimated_bytes)
        frame = self._normalize_result(raw, sample_percent)
        return frame, GdeltQueryStats(
            sample_percent=sample_percent,
            estimated_bytes=estimated_bytes,
            processed_bytes=processed_bytes,
            row_count=len(frame),
        )

    def _client(self):
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            raise RuntimeError("Historical GDELT backfill requires google-cloud-bigquery.") from exc
        return bigquery, bigquery.Client(project=self.project)

    def _query(
        self,
        bigquery,
        start: datetime,
        end: datetime,
        query: str,
        sample_percent: float,
    ):
        term_groups = _query_term_groups(query)
        sample = min(max(float(sample_percent), 0.01), 100.0)
        sample_clause = "" if sample >= 100.0 else f" TABLESAMPLE SYSTEM ({sample:.4f} PERCENT)"
        row_limit = max(1, int(self.maximum_rows))
        search_text = (
            "LOWER(CONCAT(IFNULL(Themes, ''), ' ', IFNULL(Organizations, ''), "
            "' ', IFNULL(Locations, '')))"
        )
        group_filters = "\n              ".join(
            f"AND EXISTS (SELECT 1 FROM UNNEST(@patterns_{position}) AS pattern "
            f"WHERE {search_text} LIKE pattern)"
            for position in range(len(term_groups))
        )
        sql = f"""
            SELECT
              SAFE.PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS published_at,
              SourceCommonName AS publisher,
              DocumentIdentifier AS url,
              V2Tone AS tone_raw,
              Themes AS themes,
              Organizations AS organizations
            FROM `{self.table}`{sample_clause}
            WHERE _PARTITIONTIME >= @start_time
              AND _PARTITIONTIME < @end_time
              {group_filters}
            ORDER BY FARM_FINGERPRINT(IFNULL(DocumentIdentifier, ''))
            LIMIT {row_limit}
        """
        parameters = [
            bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", _utc_timestamp(start).to_pydatetime()),
            bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", _utc_timestamp(end).to_pydatetime()),
            *[
                bigquery.ArrayQueryParameter(
                    f"patterns_{position}", "STRING", [f"%{token}%" for token in tokens]
                )
                for position, tokens in enumerate(term_groups)
            ],
        ]
        return sql, parameters

    @staticmethod
    def _normalize_result(raw: pd.DataFrame, sample_percent: float) -> pd.DataFrame:
        tone = raw.get("tone_raw", pd.Series(dtype=str)).astype(str).str.split(",").str[0]
        frame = pd.DataFrame(
            {
                "Published At UTC": raw.get("published_at"),
                "Publisher": raw.get("publisher"),
                "Title": "",
                "Snippet": "",
                "URL": raw.get("url"),
                "GDELT Tone": pd.to_numeric(tone, errors="coerce"),
                "Themes": _bounded_text_series(raw.get("themes"), len(raw), 4_000),
                "Organizations": _bounded_text_series(raw.get("organizations"), len(raw), 4_000),
                "Source Type": (
                    "gdelt_bigquery" if sample_percent >= 100.0 else "gdelt_bigquery_sampled"
                ),
            }
        )
        return normalize_news_frame(frame, "gdelt_bigquery")


@dataclass(frozen=True)
class GdeltQueryStats:
    sample_percent: float
    estimated_bytes: int
    processed_bytes: int
    row_count: int


class BigQuerySandboxBudgetExceeded(RuntimeError):
    pass


def _query_term_groups(query: str) -> list[list[str]]:
    grouped = re.findall(r"\(([^()]*)\)", str(query))
    sources = grouped if grouped else [str(query)]
    result: list[list[str]] = []
    for source in sources[:4]:
        phrases = re.findall(r'"([^"]+)"', source)
        unquoted = re.sub(r'"[^"]+"', " ", source)
        words = re.findall(r"\b[A-Za-z][A-Za-z&-]{2,}\b", unquoted)
        tokens = list(
            dict.fromkeys(
                token.lower().strip()
                for token in [*phrases, *words]
                if token.lower().strip() not in {"and", "or", "not"}
            )
        )[:40]
        if tokens:
            result.append(tokens)
    return result or [["india"]]


def _bounded_text_series(value: object, length: int, maximum_length: int) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.reset_index(drop=True)
    else:
        series = pd.Series([value] * length)
    return series.fillna("").astype(str).str.slice(0, maximum_length)


def _format_bytes(value: int) -> str:
    return f"{value / 1024**3:,.2f} GiB"


class AuthorizedPulseProvider:
    """Reserved integration point; unauthorized Pulse scraping is deliberately unsupported."""

    def fetch(self, start: datetime, end: datetime, query: str) -> pd.DataFrame:
        del start, end, query
        raise PermissionError(
            "Pulse by Zerodha is not scraped. Configure an official authorized interface before enabling this provider."
        )


def fetch_incremental_news(
    providers: Iterable[NewsProvider], start: datetime, end: datetime, query: str
) -> pd.DataFrame:
    frames = [provider.fetch(start, end, query) for provider in providers]
    return deduplicate_articles(*frames)


def _gdelt_datetime(value: datetime) -> str:
    return _utc_timestamp(value).strftime("%Y%m%d%H%M%S")


def _utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)
