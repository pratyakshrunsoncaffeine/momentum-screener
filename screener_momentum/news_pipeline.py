from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import os
import subprocess

import pandas as pd

from .index_momentum import NSE_INDEX_CATALOGUE, index_catalogue_frame
from .news_config import IST, UTC, NewsCatalystConfig, eligibility_rows
from .news_features import (
    FinBertFeatureExtractor,
    SentenceEmbeddingExtractor,
    build_daily_news_features,
    build_forward_labels,
    map_articles_to_indices,
)
from .news_model import (
    catalyst_headlines,
    explain_model_features,
    load_news_model,
    predict_news_catalysts,
    train_news_models,
)
from .news_market import calculate_constituent_daily_activity, download_constituent_activity_prices
from .news_sources import (
    BigQuerySandboxBudgetExceeded,
    GdeltBigQueryProvider,
    GdeltDocProvider,
    RssNewsProvider,
    deduplicate_articles,
)
from .news_store import (
    GitHubActionsDispatcher,
    LocalNewsResultStore,
    SupabaseNewsResultStore,
    SupabaseObjectStore,
    news_store_from_environment,
)
from .sector_rotation import NseSectorIndexProvider


WORKFLOW_FILES = {
    "daily": "news_daily.yml",
    "backfill": "news_backfill.yml",
    "retrain": "news_retrain.yml",
}


def news_environment_status() -> pd.DataFrame:
    checks = [
        ("Supabase API", "SUPABASE_URL", "Streamlit and GitHub Actions"),
        ("Supabase service role", "SUPABASE_SERVICE_ROLE_KEY", "Streamlit and GitHub Actions"),
        ("Supabase database host", "SUPABASE_DB_HOST", "GitHub Actions"),
        ("GitHub Actions dispatch", "NEWS_GITHUB_ACTIONS_TOKEN", "Streamlit only"),
        ("Google Cloud project", "GOOGLE_CLOUD_PROJECT", "GitHub Actions only"),
    ]
    return pd.DataFrame(
        [
            {
                "Component": label,
                "Environment Variable": variable,
                "Configured In This Process": bool(os.getenv(variable, "").strip()),
                "Required In": target,
            }
            for label, variable, target in checks
        ]
        + [
            {
                "Component": "BigQuery mode",
                "Environment Variable": "NEWS_BIGQUERY_SANDBOX",
                "Configured In This Process": True,
                "Required In": "Free Sandbox; billing is not required",
            }
        ]
    )


def load_saved_news_results(output_dir: str | Path = "output/latest") -> dict[str, pd.DataFrame]:
    local = LocalNewsResultStore(output_dir)
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        try:
            remote = SupabaseNewsResultStore.from_environment().load_dashboard()
            if any(not frame.empty for frame in remote.values()):
                local.save_dashboard(remote)
                remote["Storage"] = pd.DataFrame([{"Source": "Supabase PostgreSQL", "Stale": False}])
                return remote
        except requests_exception_types():
            pass
    results = local.load_dashboard()
    results["Storage"] = pd.DataFrame([{"Source": "Local saved recovery", "Stale": True}])
    return results


def queue_news_workflow(
    job_type: str,
    as_of_date: date,
    output_dir: str | Path = "output/latest",
) -> dict[str, object]:
    if job_type not in WORKFLOW_FILES:
        raise ValueError(f"Unsupported news job type: {job_type}")
    if not (os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY")):
        raise RuntimeError("Supabase secrets are required before a background news job can be queued.")
    dispatcher = GitHubActionsDispatcher.from_environment()
    store = SupabaseNewsResultStore.from_environment()
    requested = pd.Timestamp.now(tz="UTC").isoformat()
    payload = {"as_of_date": as_of_date.isoformat(), "job_type": job_type}
    job = store.create_job(job_type, requested, payload)
    job_id = str(job.get("job_id", job.get("Job ID")))
    try:
        dispatcher.dispatch(
            WORKFLOW_FILES[job_type],
            {"job_id": job_id, "as_of_date": as_of_date.isoformat()},
        )
    except Exception:
        store.update_job(job_id, status="dispatch_failed", message="GitHub Actions dispatch failed")
        raise
    return job


def save_news_eligibility(output_dir: str | Path = "output/latest") -> pd.DataFrame:
    frame = pd.DataFrame(eligibility_rows([item[1] for item in NSE_INDEX_CATALOGUE]))
    LocalNewsResultStore(output_dir).save_dashboard({"eligibility": frame})
    return frame


class NewsWorker:
    def __init__(
        self,
        job_id: str,
        output_dir: str | Path = "output/latest",
        config: NewsCatalystConfig | None = None,
    ) -> None:
        self.job_id = job_id
        self.output_dir = Path(output_dir)
        self.config = config or NewsCatalystConfig()
        self.store = SupabaseNewsResultStore.from_environment()
        self.objects = SupabaseObjectStore.from_environment()
        self.local = LocalNewsResultStore(self.output_dir)
        self.sentiment_extractor = FinBertFeatureExtractor()
        self.embedding_extractor = SentenceEmbeddingExtractor()
        self._embedded_catalogue: pd.DataFrame | None = None

    def run(self, job_type: str, as_of_date: date) -> None:
        self._update(status="running", started_at_utc=pd.Timestamp.now(tz="UTC").isoformat(), message="Worker started")
        try:
            finished = True
            if job_type == "backfill":
                finished = self.backfill(as_of_date)
            elif job_type == "daily":
                self.daily(as_of_date)
            elif job_type == "retrain":
                self.retrain(as_of_date)
            else:
                raise ValueError(f"Unknown news worker job type: {job_type}")
            if not finished:
                return
            self._update(
                status="completed",
                completed_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                message="News pipeline completed",
            )
        except Exception as exc:
            self._update(
                status="failed",
                completed_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                message=str(exc)[:1000],
            )
            raise

    def backfill(self, as_of_date: date) -> bool:
        start = as_of_date - timedelta(days=self.config.history_years * 366)
        partitions = [item.date() for item in pd.date_range(start, as_of_date, freq="D")]
        checkpoints = self._backfill_checkpoints()
        completed_dates = set(
            checkpoints.loc[checkpoints["status"].eq("completed"), "partition_date"].dropna()
        )
        pending = [item for item in partitions if item not in completed_dates]
        partitions_per_run = _environment_int(
            "NEWS_BIGQUERY_PARTITIONS_PER_RUN", self.config.bigquery_partitions_per_run
        )
        monthly_budget_bytes = int(
            _environment_float(
                "NEWS_BIGQUERY_MONTHLY_BUDGET_GIB", self.config.bigquery_monthly_budget_gib
            )
            * 1024**3
        )
        provider = GdeltBigQueryProvider.from_environment(
            maximum_query_gib=_environment_float(
                "NEWS_BIGQUERY_MAX_QUERY_GIB", self.config.bigquery_max_query_gib
            ),
            minimum_sample_percent=_environment_float(
                "NEWS_BIGQUERY_MIN_SAMPLE_PCT", self.config.bigquery_min_sample_pct
            ),
            maximum_rows=_environment_int(
                "NEWS_BIGQUERY_RESULT_ROW_LIMIT", self.config.bigquery_result_row_limit
            ),
        )
        bytes_used = self._sandbox_bytes_this_month(checkpoints)
        total = len(partitions) + 4
        self._update(
            total=total,
            completed=len(completed_dates),
            message=(
                f"Resuming free BigQuery Sandbox backfill: {len(completed_dates):,} of "
                f"{len(partitions):,} dates complete; {_format_gib(bytes_used)} used this month"
            ),
        )
        catalogue = self._catalogue()
        constituents = self._constituents()
        batch = pending[:partitions_per_run]
        budget_blocked = False
        failures = 0
        for partition_date in batch:
            partition_start = datetime.combine(partition_date, datetime.min.time(), tzinfo=UTC)
            partition_end = partition_start + timedelta(days=1)
            remaining_budget = monthly_budget_bytes - bytes_used
            try:
                articles, stats = provider.fetch_with_stats(
                    partition_start,
                    partition_end,
                    self.config.gdelt_query,
                    remaining_budget_bytes=remaining_budget,
                )
                articles = self._enrich_articles(articles)
                links = map_articles_to_indices(articles, catalogue, constituents)
                object_uri = self._archive_frame(
                    articles,
                    f"backfills/news_articles/{partition_date.isoformat()}.parquet",
                )
                self._upsert_articles(articles)
                self.store.upsert("news_index_links", links, "article_id,index_name")
                self._save_backfill_checkpoint(
                    partition_date,
                    status="completed",
                    sample_percent=stats.sample_percent,
                    estimated_bytes=stats.estimated_bytes,
                    processed_bytes=stats.processed_bytes,
                    article_count=len(articles),
                    object_uri=object_uri,
                )
                completed_dates.add(partition_date)
                bytes_used += stats.processed_bytes
                self._update(
                    completed=len(completed_dates),
                    message=(
                        f"Saved {partition_date}: {len(articles):,} articles at "
                        f"{stats.sample_percent:.2f}% sampling; {_format_gib(bytes_used)} "
                        "of the monthly sandbox guard used"
                    ),
                )
            except BigQuerySandboxBudgetExceeded as exc:
                self._save_backfill_checkpoint(
                    partition_date,
                    status="skipped_budget",
                    error_message=str(exc),
                )
                budget_blocked = True
                break
            except Exception as exc:
                failures += 1
                self._save_backfill_checkpoint(
                    partition_date,
                    status="failed",
                    error_message=str(exc)[:1000],
                )

        remaining = len(partitions) - len(completed_dates)
        if remaining:
            reason = (
                "Monthly sandbox guard reached; resume after the BigQuery monthly quota resets"
                if budget_blocked
                else f"Batch checkpoint saved; run again to continue the remaining {remaining:,} dates"
            )
            if failures:
                reason += f" ({failures} date(s) will be retried)"
            self._update(
                status="waiting_for_resume",
                completed=len(completed_dates),
                total=total,
                message=reason,
            )
            return False

        self._sync_index_prices(start, as_of_date)
        self._sync_constituent_activity(start, as_of_date, constituents)
        self._update(completed=len(partitions) + 1, message="Official NSE index prices saved")
        self._run_sqlmesh("plan")
        self._update(completed=len(partitions) + 2, message="SQLMesh historical features and labels built")
        self._train_and_publish()
        self._update(completed=len(partitions) + 4, message="Initial models trained and published")
        return True

    def daily(self, as_of_date: date) -> None:
        watermark = self._watermark("news_incremental")
        end = datetime.combine(as_of_date, self.config.cutoff, tzinfo=IST)
        start = watermark - timedelta(hours=self.config.late_arrival_hours) if watermark else end - timedelta(days=3)
        providers = [GdeltDocProvider(maximum_records=self.config.maximum_articles_per_fetch)]
        rss = RssNewsProvider.from_environment()
        if rss.feeds:
            providers.append(rss)
        frames = [provider.fetch(start, end, self.config.gdelt_query) for provider in providers]
        articles = deduplicate_articles(*frames)
        articles = self._enrich_articles(articles)
        links = map_articles_to_indices(articles, self._catalogue(), self._constituents())
        self._archive_frame(articles, f"daily/news_articles/{as_of_date.isoformat()}.parquet")
        self._update(total=6, completed=1, message=f"Fetched {len(articles):,} incremental articles")
        self._upsert_articles(articles)
        self.store.upsert("news_index_links", links, "article_id,index_name")
        self._sync_index_prices(as_of_date - timedelta(days=120), as_of_date)
        self._sync_constituent_activity(as_of_date - timedelta(days=120), as_of_date, self._constituents())
        self._update(completed=3, message="News and latest NSE index prices saved")
        self._run_sqlmesh("run")
        self._update(completed=4, message="SQLMesh daily features refreshed")
        self._infer_and_publish(articles, links)
        self._advance_watermark("news_incremental", end)
        self._update(completed=6, message="Predictions and catalyst explanations published")

    def retrain(self, as_of_date: date) -> None:
        del as_of_date
        self._update(total=3, completed=1, message="Loading matured point-in-time labels")
        self._run_sqlmesh("run")
        self._train_and_publish()
        self._update(completed=3, message="Champion/challenger evaluation completed")

    def _train_and_publish(self) -> None:
        training = self._query_frame("SELECT * FROM news_ml.training_frame ORDER BY date, index")
        result = train_news_models(training, self.output_dir / "news_models", self.config)
        bundle = result["bundle"]
        artifact_path = Path(result["artifact"])
        object_path = f"models/{bundle.model_version}/{artifact_path.name}"
        artifact_uri = self.objects.upload(artifact_path, object_path)
        metrics = result["metrics"].copy()
        metrics["Model Version"] = bundle.model_version
        current_metrics = self._current_champion_metrics()
        deployment_status = (
            "Champion" if current_metrics.empty or _challenger_can_replace(metrics, current_metrics, bundle.status)
            else "Challenger"
        )
        if deployment_status == "Champion" and not current_metrics.empty:
            self.store.patch(
                "news_model_runs",
                {"deployment_status": "eq.Champion"},
                deployment_status="Archived",
            )
        model_row = pd.DataFrame(
            [
                {
                    "Model Version": bundle.model_version,
                    "Status": bundle.status,
                    "Deployment Status": deployment_status,
                    "Trained At UTC": bundle.trained_at_utc,
                    "Artifact URI": artifact_uri,
                    "Feature Schema": bundle.feature_columns,
                    "Config": json.loads(json.dumps(asdict(self.config), default=str)),
                }
            ]
        )
        self.store.upsert("news_model_runs", model_row, "model_version")
        self.store.upsert("news_model_metrics", metrics, "model_version,horizon")
        evaluation = result["test_predictions"].copy()
        if not evaluation.empty:
            self.store.upsert(
                "news_model_evaluation", evaluation, "date,index_name,horizon,model_version"
            )
        self.local.save_dashboard({"metrics": metrics, "evaluation": evaluation})

    def _infer_and_publish(self, articles: pd.DataFrame, links: pd.DataFrame) -> None:
        model = self._query_frame(
            "SELECT model_version, artifact_uri FROM news_model_runs "
            "WHERE deployment_status = 'Champion' ORDER BY trained_at_utc DESC LIMIT 1"
        )
        if model.empty:
            raise RuntimeError("No trained news model is available. Run the historical backfill first.")
        uri = str(model.iloc[0]["artifact_uri"])
        object_path = uri.split("/", 1)[1] if "/" in uri else uri
        local_model = self.objects.download(object_path, self.output_dir / "news_models" / "champion.joblib")
        bundle = load_news_model(local_model)
        latest = self._query_frame('SELECT * FROM public.news_latest_features ORDER BY "Index"')
        predictions = predict_news_catalysts(bundle, latest)
        catalyst_frames: list[pd.DataFrame] = []
        driver_frames: list[pd.DataFrame] = []
        for horizon in bundle.horizons:
            horizon_catalysts = catalyst_headlines(predictions, articles, links, horizon)
            if not horizon_catalysts.empty:
                horizon_catalysts["Horizon"] = horizon
                catalyst_frames.append(horizon_catalysts)
            horizon_drivers = explain_model_features(bundle, latest, horizon)
            horizon_drivers["Model Version"] = bundle.model_version
            horizon_drivers["As Of Date"] = pd.to_datetime(latest["Date"], errors="coerce").max().date()
            driver_frames.append(horizon_drivers)
        catalysts = pd.concat(catalyst_frames, ignore_index=True) if catalyst_frames else pd.DataFrame()
        drivers = pd.concat(driver_frames, ignore_index=True) if driver_frames else pd.DataFrame()
        self.store.upsert("news_predictions", predictions, "as_of_date,index_name,horizon,model_version")
        if not catalysts.empty:
            catalyst_columns = [
                "Article ID", "Index", "Model Version", "Horizon", "Published At UTC", "Publisher",
                "Title", "URL", "Themes", "Relevance", "Attribution Reason", "Sentiment Score",
                "Sentiment Source", "Signal", "Catalyst Contribution", "Catalyst Magnitude",
            ]
            self.store.upsert(
                "news_prediction_catalysts",
                catalysts.reindex(columns=catalyst_columns),
                "article_id,index_name,model_version,horizon",
            )
        if not drivers.empty:
            self.store.upsert(
                "news_prediction_drivers",
                drivers,
                "as_of_date,index_name,horizon,model_version,driver_rank",
            )
        self.local.save_dashboard(
            {
                "predictions": predictions,
                "catalysts": catalysts,
                "drivers": drivers,
                "features": latest,
            }
        )

    def _sync_index_prices(self, start: date, end: date) -> None:
        eligibility = pd.DataFrame(eligibility_rows([item[1] for item in NSE_INDEX_CATALOGUE]))
        catalogue = index_catalogue_frame().merge(eligibility, on="Index", how="left")
        catalogue["Description"] = catalogue["Category"].astype(str) + " NSE India equity index"
        eligible = eligibility.loc[eligibility["Model Eligible"], "Index"].tolist()
        provider = NseSectorIndexProvider(self.output_dir.parent / "news_index_cache")
        prices, _ = provider.fetch_many(eligible, start_date=start, end_date=end)
        self.store.upsert("index_prices", prices, "date,index_name")
        self.store.upsert("indices", catalogue, "index_name")
        self.store.upsert("news_index_eligibility", eligibility, "index_name")
        self.local.save_dashboard({"eligibility": eligibility})

    def _upsert_articles(self, articles: pd.DataFrame) -> None:
        frame = articles.drop(columns=["Embedding"], errors="ignore")
        self.store.upsert("news_articles", frame, "article_id")
        if "Embedding" in articles:
            vectors = articles.loc[articles["Embedding"].notna(), ["Article ID", "Embedding"]].copy()
            if not vectors.empty:
                vectors["Embedding"] = vectors["Embedding"].map(json.dumps)
                self.store.upsert("news_embeddings", vectors, "article_id")

    def _sync_constituent_activity(
        self, start: date, end: date, constituents: pd.DataFrame
    ) -> None:
        if constituents.empty:
            return
        tickers = constituents["Ticker"].dropna().astype(str).unique().tolist()
        prices = download_constituent_activity_prices(tickers, start, end)
        activity = calculate_constituent_daily_activity(prices, constituents)
        if not activity.empty:
            self.store.upsert("constituent_daily_activity", activity, "date,index_name")

    def _enrich_articles(self, articles: pd.DataFrame) -> pd.DataFrame:
        enriched = self.sentiment_extractor.transform(articles, allow_tone_fallback=True)
        if not enriched.empty and enriched["Text Available"].fillna(False).any():
            try:
                enriched = self.embedding_extractor.transform(enriched)
            except RuntimeError:
                enriched["Embedding"] = None
        elif "Embedding" not in enriched:
            enriched["Embedding"] = None
        return enriched

    def _catalogue(self) -> pd.DataFrame:
        if self._embedded_catalogue is not None:
            return self._embedded_catalogue
        catalogue = index_catalogue_frame().copy()
        descriptions = pd.DataFrame(
            {
                "Title": catalogue["Index"],
                "Snippet": catalogue["Category"].astype(str) + " NSE India equity index",
            }
        )
        try:
            embedded = self.embedding_extractor.transform(descriptions)
            catalogue["Embedding"] = embedded["Embedding"]
        except RuntimeError:
            catalogue["Embedding"] = None
        self._embedded_catalogue = catalogue
        return catalogue

    def _archive_frame(self, frame: pd.DataFrame, object_path: str) -> str | None:
        if frame.empty:
            return None
        archive = frame.drop(columns=["Embedding"], errors="ignore")
        local_path = self.output_dir / "news_archives" / Path(object_path).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        archive.to_parquet(local_path, index=False)
        return self.objects.upload(
            local_path, object_path, content_type="application/vnd.apache.parquet"
        )

    def _backfill_checkpoints(self) -> pd.DataFrame:
        try:
            frame = pd.DataFrame(
                self.store.select("news_backfill_checkpoints", limit=10_000)
            )
        except Exception:
            return pd.DataFrame(columns=["partition_date", "status", "processed_bytes", "processed_at_utc"])
        if frame.empty:
            return pd.DataFrame(columns=["partition_date", "status", "processed_bytes", "processed_at_utc"])
        frame["partition_date"] = pd.to_datetime(
            frame.get("partition_date"), errors="coerce"
        ).dt.date
        frame["processed_bytes"] = pd.to_numeric(
            frame.get("processed_bytes"), errors="coerce"
        ).fillna(0)
        frame["processed_at_utc"] = pd.to_datetime(
            frame.get("processed_at_utc"), utc=True, errors="coerce"
        )
        return frame

    def _sandbox_bytes_this_month(self, checkpoints: pd.DataFrame) -> int:
        if checkpoints.empty:
            return 0
        now = pd.Timestamp.now(tz="UTC")
        month_start = pd.Timestamp(year=now.year, month=now.month, day=1, tz="UTC")
        mask = checkpoints["processed_at_utc"].ge(month_start)
        return int(checkpoints.loc[mask, "processed_bytes"].sum())

    def _save_backfill_checkpoint(
        self,
        partition_date: date,
        status: str,
        sample_percent: float = 100.0,
        estimated_bytes: int = 0,
        processed_bytes: int = 0,
        article_count: int = 0,
        object_uri: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.store.upsert(
            "news_backfill_checkpoints",
            pd.DataFrame(
                [
                    {
                        "Source": "gdelt_bigquery_sandbox",
                        "Partition Date": partition_date,
                        "Status": status,
                        "Sample Percent": sample_percent,
                        "Estimated Bytes": int(estimated_bytes),
                        "Processed Bytes": int(processed_bytes),
                        "Article Count": int(article_count),
                        "Object URI": object_uri,
                        "Error Message": error_message,
                        "Processed At UTC": pd.Timestamp.now(tz="UTC"),
                    }
                ]
            ),
            "source,partition_date",
        )

    def _constituents(self) -> pd.DataFrame:
        try:
            return pd.DataFrame(self.store.select("index_constituents", limit=100000)).rename(
                columns={
                    "index_name": "Index",
                    "ticker": "Ticker",
                    "company": "Company",
                    "valid_from": "Valid From",
                    "valid_to": "Valid To",
                }
            )
        except Exception:
            return pd.DataFrame()

    def _watermark(self, pipeline: str) -> datetime | None:
        rows = self.store.select("news_ingestion_watermarks", filters={"pipeline": f"eq.{pipeline}"}, limit=1)
        if not rows:
            return None
        value = pd.to_datetime(rows[0].get("watermark_utc"), utc=True, errors="coerce")
        return None if pd.isna(value) else value.to_pydatetime()

    def _advance_watermark(self, pipeline: str, value: datetime) -> None:
        self.store.upsert(
            "news_ingestion_watermarks",
            pd.DataFrame([{"Pipeline": pipeline, "Watermark UTC": pd.Timestamp(value).tz_convert("UTC")}]),
            "pipeline",
        )

    def _run_sqlmesh(self, command: str) -> None:
        environment = {**os.environ, "NEWS_SQLMESH_COMMAND": command}
        arguments = ["sqlmesh", command]
        if command == "plan":
            arguments.extend(["prod", "--auto-apply", "--no-prompts"])
        subprocess.run(arguments, cwd=Path(__file__).resolve().parents[1] / "sqlmesh", env=environment, check=True)

    def _query_frame(self, query: str) -> pd.DataFrame:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Direct training queries require psycopg from requirements-ml.txt.") from exc
        connection_string = os.getenv("SUPABASE_DB_URL", "").strip()
        if connection_string:
            connection = psycopg.connect(connection_string)
        else:
            values = {
                "host": os.getenv("SUPABASE_DB_HOST", "").strip(),
                "user": os.getenv("SUPABASE_DB_USER", "").strip(),
                "password": os.getenv("SUPABASE_DB_PASSWORD", ""),
                "dbname": os.getenv("SUPABASE_DB_NAME", "postgres").strip() or "postgres",
            }
            missing = [key for key, value in values.items() if not value]
            if missing:
                raise RuntimeError(
                    "Supabase database settings are incomplete: " + ", ".join(missing)
                )
            connection = psycopg.connect(**values, port=5432, sslmode="require")
        with connection:
            frame = pd.read_sql_query(query, connection)
        return _expand_embedding_column(frame)

    def _current_champion_metrics(self) -> pd.DataFrame:
        try:
            return self._query_frame(
                "SELECT metrics.* FROM news_model_metrics AS metrics "
                "JOIN news_model_runs AS runs USING (model_version) "
                "WHERE runs.deployment_status = 'Champion'"
            )
        except Exception:
            return pd.DataFrame()

    def _update(self, **fields: object) -> None:
        self.store.update_job(self.job_id, **fields)


def _month_windows(start: date, end: date):
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        next_month = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
        yield max(cursor, start), min(next_month - timedelta(days=1), end)
        cursor = next_month


def requests_exception_types():
    import requests

    return (requests.RequestException, RuntimeError, ValueError)


def _environment_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(1, int(default))


def _environment_float(name: str, default: float) -> float:
    try:
        return max(0.01, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(0.01, float(default))


def _format_gib(value: int) -> str:
    return f"{int(value) / 1024**3:,.2f} GiB"


def _expand_embedding_column(frame: pd.DataFrame) -> pd.DataFrame:
    column = next((item for item in ("news_embedding", "News Embedding") if item in frame), None)
    if column is None:
        return frame
    parsed: list[list[float] | None] = []
    for value in frame[column]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            parsed.append(None)
            continue
        try:
            vector = json.loads(value) if isinstance(value, str) else list(value)
            parsed.append([float(item) for item in vector])
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed.append(None)
    width = max((len(item) for item in parsed if item), default=0)
    result = frame.drop(columns=[column]).copy()
    for position in range(width):
        result[f"embedding_{position:03d}"] = [
            vector[position] if vector and position < len(vector) else pd.NA for vector in parsed
        ]
    return result


def _challenger_can_replace(
    challenger: pd.DataFrame,
    champion: pd.DataFrame,
    model_status: str,
    maximum_drawdown_tolerance_pct: float = 5.0,
) -> bool:
    if model_status != "Validated" or challenger.empty:
        return False
    challenger_frame = challenger.rename(columns={column: _comparison_name(column) for column in challenger.columns})
    champion_frame = champion.rename(columns={column: _comparison_name(column) for column in champion.columns})
    required = {"horizon", "rank_ic", "price_only_rank_ic", "maximum_drawdown_pct"}
    if not required.issubset(challenger_frame.columns):
        return False
    beats_price = (
        pd.to_numeric(challenger_frame["rank_ic"], errors="coerce") > 0
    ) & (
        pd.to_numeric(challenger_frame["rank_ic"], errors="coerce")
        > pd.to_numeric(challenger_frame["price_only_rank_ic"], errors="coerce")
    )
    if int(beats_price.sum()) < 2:
        return False
    if required.issubset(champion_frame.columns):
        comparison = challenger_frame.merge(
            champion_frame[["horizon", "maximum_drawdown_pct"]],
            on="horizon",
            how="inner",
            suffixes=("_new", "_old"),
        )
        if not comparison.empty:
            new_drawdown = pd.to_numeric(comparison["maximum_drawdown_pct_new"], errors="coerce")
            old_drawdown = pd.to_numeric(comparison["maximum_drawdown_pct_old"], errors="coerce")
            if (new_drawdown < old_drawdown - maximum_drawdown_tolerance_pct).any():
                return False
    return True


def _comparison_name(value: object) -> str:
    normalized = str(value).strip().lower().replace("%", "pct").replace("-", " ")
    return "_".join(normalized.split())
