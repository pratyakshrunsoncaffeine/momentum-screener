from __future__ import annotations

import argparse
from datetime import date
import os

import pandas as pd

from .news_pipeline import NewsWorker
from .news_store import SupabaseNewsResultStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a News Catalysts background job.")
    parser.add_argument("--job-type", choices=("daily", "backfill", "retrain"), required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    return parser


def ensure_job(job_type: str, job_id: str, as_of_date: date) -> str:
    if job_id.strip():
        return job_id.strip()
    store = SupabaseNewsResultStore.from_environment()
    row = store.create_job(
        job_type,
        pd.Timestamp.now(tz="UTC").isoformat(),
        {"job_type": job_type, "as_of_date": as_of_date.isoformat(), "source": "scheduled workflow"},
    )
    return str(row["job_id"])


def main() -> None:
    args = build_parser().parse_args()
    as_of_date = date.fromisoformat(args.as_of_date)
    job_id = ensure_job(args.job_type, args.job_id, as_of_date)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    NewsWorker(job_id=job_id).run(args.job_type, as_of_date)


if __name__ == "__main__":
    main()
