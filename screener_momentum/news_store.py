from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import json
import os
import time
from typing import Protocol
from uuid import uuid4

import pandas as pd
import requests


LOCAL_RESULT_FILES: dict[str, str] = {
    "predictions": "news_predictions.csv",
    "catalysts": "news_prediction_catalysts.csv",
    "metrics": "news_model_metrics.csv",
    "jobs": "news_pipeline_jobs.csv",
    "eligibility": "news_index_eligibility.csv",
    "features": "news_latest_features.csv",
    "evaluation": "news_model_evaluation.csv",
    "drivers": "news_prediction_drivers.csv",
    "articles": "news_articles_latest.csv",
    "links": "news_index_links_latest.csv",
}


class NewsResultStore(Protocol):
    def load_dashboard(self) -> dict[str, pd.DataFrame]: ...

    def create_job(self, job_type: str, requested_at_utc: str, payload: dict[str, object]) -> dict[str, object]: ...

    def update_job(self, job_id: str, **fields: object) -> None: ...


@dataclass
class LocalNewsResultStore:
    output_dir: Path

    def __init__(self, output_dir: str | Path = "output/latest") -> None:
        self.output_dir = Path(output_dir)

    def load_dashboard(self) -> dict[str, pd.DataFrame]:
        return {
            key: _read_csv(self.output_dir / file_name)
            for key, file_name in LOCAL_RESULT_FILES.items()
        }

    def save_dashboard(self, results: dict[str, pd.DataFrame]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for key, frame in results.items():
            if key in LOCAL_RESULT_FILES and frame is not None:
                _atomic_csv(frame, self.output_dir / LOCAL_RESULT_FILES[key])

    def create_job(self, job_type: str, requested_at_utc: str, payload: dict[str, object]) -> dict[str, object]:
        jobs = _read_csv(self.output_dir / LOCAL_RESULT_FILES["jobs"])
        row = {
            "Job ID": str(uuid4()),
            "Job Type": job_type,
            "Status": "queued",
            "Requested At UTC": requested_at_utc,
            "Started At UTC": pd.NA,
            "Completed At UTC": pd.NA,
            "Completed": 0,
            "Total": 0,
            "Message": "Waiting for worker",
            "Payload": json.dumps(payload, sort_keys=True),
        }
        jobs = pd.concat([jobs, pd.DataFrame([row])], ignore_index=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_csv(jobs, self.output_dir / LOCAL_RESULT_FILES["jobs"])
        return row

    def update_job(self, job_id: str, **fields: object) -> None:
        path = self.output_dir / LOCAL_RESULT_FILES["jobs"]
        jobs = _read_csv(path)
        if jobs.empty or "Job ID" not in jobs:
            return
        mask = jobs["Job ID"].astype(str).eq(str(job_id))
        for key, value in fields.items():
            jobs.loc[mask, key] = value
        _atomic_csv(jobs, path)


@dataclass
class SupabaseNewsResultStore:
    url: str
    service_key: str
    timeout: int = 45
    upsert_batch_rows: int = 250
    upsert_batch_bytes: int = 750_000
    upsert_retry_attempts: int = 3

    @classmethod
    def from_environment(cls) -> "SupabaseNewsResultStore":
        url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")
        return cls(url, key)

    def load_dashboard(self) -> dict[str, pd.DataFrame]:
        tables = {
            "predictions": "news_predictions",
            "catalysts": "news_prediction_catalysts",
            "metrics": "news_model_metrics",
            "jobs": "news_pipeline_jobs",
            "eligibility": "news_index_eligibility",
            "features": "news_latest_features",
            "evaluation": "news_model_evaluation",
            "drivers": "news_prediction_drivers",
        }
        results: dict[str, pd.DataFrame] = {}
        for key, table in tables.items():
            try:
                results[key] = _display_columns(pd.DataFrame(self.select(table, limit=10000)))
            except requests.RequestException:
                results[key] = pd.DataFrame()
        return results

    def select(
        self,
        table: str,
        columns: str = "*",
        filters: dict[str, str] | None = None,
        order: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, object]]:
        params: dict[str, object] = {"select": columns, "limit": limit}
        if filters:
            params.update(filters)
        if order:
            params["order"] = order
        response = requests.get(
            f"{self.url}/rest/v1/{table}",
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def patch(self, table: str, filters: dict[str, str], **fields: object) -> None:
        payload = {_snake_case(key): value for key, value in fields.items()}
        response = requests.patch(
            f"{self.url}/rest/v1/{table}",
            headers={**self._headers(), "Prefer": "return=minimal", "Content-Type": "application/json"},
            params=filters,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def upsert(self, table: str, frame: pd.DataFrame, on_conflict: str) -> None:
        if frame.empty:
            return
        records = _database_records(frame)
        for batch in _record_batches(
            records,
            maximum_rows=self.upsert_batch_rows,
            maximum_bytes=self.upsert_batch_bytes,
        ):
            self._upsert_batch(table, batch, on_conflict)

    def _upsert_batch(
        self,
        table: str,
        records: list[dict[str, object]],
        on_conflict: str,
        split_depth: int = 0,
    ) -> None:
        last_error: requests.RequestException | None = None
        for attempt in range(max(1, self.upsert_retry_attempts)):
            try:
                self._post_upsert(table, records, on_conflict)
                return
            except requests.RequestException as exc:
                last_error = exc
                if not _is_transient_request_error(exc) or attempt + 1 >= self.upsert_retry_attempts:
                    break
                time.sleep(0.5 * (2**attempt))

        if len(records) > 10 and split_depth < 3:
            midpoint = len(records) // 2
            self._upsert_batch(table, records[:midpoint], on_conflict, split_depth + 1)
            self._upsert_batch(table, records[midpoint:], on_conflict, split_depth + 1)
            return
        raise RuntimeError(_supabase_error_message(table, len(records), last_error)) from last_error

    def _post_upsert(
        self,
        table: str,
        records: list[dict[str, object]],
        on_conflict: str,
    ) -> None:
        response = requests.post(
            f"{self.url}/rest/v1/{table}",
            headers={
                **self._headers(),
                "Prefer": "resolution=merge-duplicates,return=minimal",
                "Content-Type": "application/json",
            },
            params={"on_conflict": on_conflict},
            json=records,
            timeout=max(self.timeout, 120),
        )
        response.raise_for_status()

    def create_job(self, job_type: str, requested_at_utc: str, payload: dict[str, object]) -> dict[str, object]:
        row = {
            "job_id": str(uuid4()),
            "job_type": job_type,
            "status": "queued",
            "requested_at_utc": requested_at_utc,
            "completed": 0,
            "total": 0,
            "message": "Waiting for worker",
            "payload": payload,
        }
        response = requests.post(
            f"{self.url}/rest/v1/news_pipeline_jobs",
            headers={**self._headers(), "Prefer": "return=representation", "Content-Type": "application/json"},
            json=row,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()[0]

    def update_job(self, job_id: str, **fields: object) -> None:
        payload = {_snake_case(key): value for key, value in fields.items()}
        response = requests.patch(
            f"{self.url}/rest/v1/news_pipeline_jobs",
            headers={**self._headers(), "Prefer": "return=minimal", "Content-Type": "application/json"},
            params={"job_id": f"eq.{job_id}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

    def _headers(self) -> dict[str, str]:
        return {"apikey": self.service_key, "Authorization": f"Bearer {self.service_key}"}


@dataclass
class SupabaseObjectStore:
    url: str
    service_key: str
    bucket: str = "news-models"
    timeout: int = 120

    @classmethod
    def from_environment(cls) -> "SupabaseObjectStore":
        return cls(
            os.environ["SUPABASE_URL"].rstrip("/"),
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
            os.getenv("SUPABASE_NEWS_BUCKET", "news-models"),
        )

    def upload(self, local_path: str | Path, object_path: str, content_type: str = "application/octet-stream") -> str:
        with Path(local_path).open("rb") as source:
            response = requests.post(
                f"{self.url}/storage/v1/object/{self.bucket}/{object_path}",
                headers={
                    "apikey": self.service_key,
                    "Authorization": f"Bearer {self.service_key}",
                    "Content-Type": content_type,
                    "x-upsert": "true",
                },
                data=source,
                timeout=self.timeout,
            )
        response.raise_for_status()
        return f"{self.bucket}/{object_path}"

    def download(self, object_path: str, local_path: str | Path) -> Path:
        response = requests.get(
            f"{self.url}/storage/v1/object/authenticated/{self.bucket}/{object_path}",
            headers={"apikey": self.service_key, "Authorization": f"Bearer {self.service_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return target


@dataclass
class GitHubActionsDispatcher:
    repository: str
    token: str
    branch: str = "main"
    timeout: int = 30

    @classmethod
    def from_environment(cls) -> "GitHubActionsDispatcher":
        repository = os.getenv("NEWS_GITHUB_REPOSITORY", "pratyakshrunsoncaffeine/momentum-screener")
        token = os.getenv("NEWS_GITHUB_ACTIONS_TOKEN", "").strip()
        if not token:
            raise RuntimeError("NEWS_GITHUB_ACTIONS_TOKEN is required to dispatch background news jobs.")
        return cls(repository, token, os.getenv("NEWS_GITHUB_BRANCH", "main"))

    def dispatch(self, workflow: str, inputs: dict[str, object]) -> None:
        response = requests.post(
            f"https://api.github.com/repos/{self.repository}/actions/workflows/{workflow}/dispatches",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"ref": self.branch, "inputs": {key: str(value) for key, value in inputs.items()}},
            timeout=self.timeout,
        )
        response.raise_for_status()


def news_store_from_environment(output_dir: str | Path = "output/latest") -> NewsResultStore:
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        return SupabaseNewsResultStore.from_environment()
    return LocalNewsResultStore(output_dir)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def _database_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    normalized = frame.rename(columns={column: _snake_case(column) for column in frame.columns}).copy()
    normalized = normalized.astype(object).where(pd.notna(normalized), None)
    return [
        {key: _json_value(value) for key, value in record.items()}
        for record in normalized.to_dict("records")
    ]


def _record_batches(
    records: list[dict[str, object]],
    maximum_rows: int = 250,
    maximum_bytes: int = 750_000,
) -> list[list[dict[str, object]]]:
    row_limit = max(1, int(maximum_rows))
    byte_limit = max(1, int(maximum_bytes))
    batches: list[list[dict[str, object]]] = []
    batch: list[dict[str, object]] = []
    batch_bytes = 2
    for record in records:
        record_bytes = len(
            json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ) + (1 if batch else 0)
        if batch and (len(batch) >= row_limit or batch_bytes + record_bytes > byte_limit):
            batches.append(batch)
            batch = []
            batch_bytes = 2
            record_bytes -= 1
        batch.append(record)
        batch_bytes += record_bytes
    if batch:
        batches.append(batch)
    return batches


def _is_transient_request_error(exc: requests.RequestException) -> bool:
    response = getattr(exc, "response", None)
    if response is None:
        return True
    return response.status_code == 429 or response.status_code >= 500


def _supabase_error_message(
    table: str,
    record_count: int,
    exc: requests.RequestException | None,
) -> str:
    if exc is None:
        return f"Supabase upsert failed for {table} ({record_count} records)."
    response = getattr(exc, "response", None)
    if response is None:
        detail = str(exc)
    else:
        detail = " ".join(str(response.text or "").split())[:1000]
        detail = f"HTTP {response.status_code}" + (f": {detail}" if detail else "")
    return f"Supabase upsert failed for {table} ({record_count} records): {detail}"


def _json_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    names = {}
    for column in frame.columns:
        if column == "index_name":
            names[column] = "Index"
            continue
        parts = ["%" if part == "pct" else part.upper() if part in {"utc", "mae", "auc", "ic"} else part.capitalize()
                 for part in column.split("_")]
        names[column] = " ".join(parts)
    return frame.rename(columns=names)


def _snake_case(value: str) -> str:
    normalized = "_".join(str(value).strip().lower().replace("%", "pct").replace("/", " ").split())
    return "index_name" if normalized == "index" else normalized
