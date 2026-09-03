create extension if not exists vector with schema extensions;

create table if not exists public.news_articles (
    article_id text primary key,
    published_at_utc timestamptz not null,
    publisher text,
    title text,
    snippet text,
    url text,
    language text,
    source_country text,
    gdelt_tone double precision,
    themes text,
    organizations text,
    source_type text not null,
    text_available boolean not null default false,
    ingested_at_utc timestamptz not null default now(),
    positive_probability double precision,
    negative_probability double precision,
    neutral_probability double precision,
    sentiment_source text,
    sentiment_score double precision,
    unique (url)
);

create index if not exists news_articles_published_idx on public.news_articles (published_at_utc);
create index if not exists news_articles_publisher_idx on public.news_articles (publisher);

create table if not exists public.news_embeddings (
    article_id text primary key references public.news_articles(article_id) on delete cascade,
    embedding extensions.vector(384) not null,
    model_name text not null default 'sentence-transformers/all-MiniLM-L6-v2',
    created_at_utc timestamptz not null default now()
);

create index if not exists news_embeddings_hnsw_idx
    on public.news_embeddings using hnsw (embedding extensions.vector_cosine_ops);

create table if not exists public.indices (
    index_name text primary key,
    category text,
    description text,
    model_eligible boolean not null default true,
    eligibility_reason text,
    created_at_utc timestamptz not null default now()
);

create table if not exists public.index_constituents (
    index_name text not null,
    ticker text not null,
    company text,
    valid_from date not null,
    valid_to date,
    source text,
    primary key (index_name, ticker, valid_from),
    check (valid_to is null or valid_to >= valid_from)
);

create index if not exists index_constituents_ticker_validity_idx
    on public.index_constituents (ticker, valid_from, valid_to);

create table if not exists public.index_prices (
    date date not null,
    index_name text not null,
    close double precision not null check (close > 0),
    primary key (date, index_name)
);

create table if not exists public.constituent_daily_activity (
    date date not null,
    index_name text not null,
    volume_abnormal double precision,
    turnover_abnormal double precision,
    breadth_positive double precision,
    participation_rate double precision,
    constituent_count integer,
    source text,
    primary key (date, index_name)
);

create table if not exists public.news_index_links (
    article_id text not null references public.news_articles(article_id) on delete cascade,
    index_name text not null,
    relevance double precision not null check (relevance >= 0 and relevance <= 1),
    attribution_method text,
    attribution_reason text,
    created_at_utc timestamptz not null default now(),
    primary key (article_id, index_name)
);

create index if not exists news_index_links_index_idx on public.news_index_links (index_name);

create table if not exists public.news_ingestion_watermarks (
    pipeline text primary key,
    watermark_utc timestamptz not null,
    updated_at_utc timestamptz not null default now()
);

create table if not exists public.news_backfill_checkpoints (
    source text not null,
    partition_date date not null,
    status text not null check (status in ('completed', 'failed', 'skipped_budget')),
    sample_percent double precision not null default 100.0,
    estimated_bytes bigint not null default 0,
    processed_bytes bigint not null default 0,
    article_count integer not null default 0,
    object_uri text,
    error_message text,
    processed_at_utc timestamptz not null default now(),
    primary key (source, partition_date)
);

create index if not exists news_backfill_checkpoints_processed_idx
    on public.news_backfill_checkpoints (processed_at_utc);

create table if not exists public.news_pipeline_jobs (
    job_id uuid primary key,
    job_type text not null check (job_type in ('daily', 'backfill', 'retrain')),
    status text not null,
    requested_at_utc timestamptz not null,
    started_at_utc timestamptz,
    completed_at_utc timestamptz,
    completed integer not null default 0,
    total integer not null default 0,
    message text,
    payload jsonb not null default '{}'::jsonb
);

create table if not exists public.news_model_runs (
    model_version text primary key,
    status text not null,
    deployment_status text not null default 'Challenger',
    trained_at_utc timestamptz not null,
    artifact_uri text not null,
    feature_schema jsonb not null,
    config jsonb not null
);

alter table public.news_model_runs
    add column if not exists deployment_status text not null default 'Challenger';
create unique index if not exists one_news_model_champion
    on public.news_model_runs (deployment_status)
    where deployment_status = 'Champion';

create table if not exists public.news_model_metrics (
    model_version text not null references public.news_model_runs(model_version) on delete cascade,
    horizon text not null,
    days integer,
    test_mae double precision,
    test_auc double precision,
    directional_accuracy double precision,
    rank_ic double precision,
    price_only_rank_ic double precision,
    rank_ic_improvement double precision,
    top_five_mean_excess_pct double precision,
    top_five_hit_rate_pct double precision,
    theoretical_excess_after_cost_pct double precision,
    sharpe_ratio double precision,
    maximum_drawdown_pct double precision,
    observations integer,
    noise_enabled boolean,
    validation_score double precision,
    lightgbm_ensemble_weight double precision,
    primary key (model_version, horizon)
);

create table if not exists public.news_model_evaluation (
    date date not null,
    index_name text not null,
    horizon text not null,
    model_version text not null references public.news_model_runs(model_version) on delete cascade,
    actual_absolute_return_pct double precision,
    actual_excess_return_pct double precision,
    actual_positive_excess integer,
    predicted_excess_return_pct double precision,
    predicted_positive_probability double precision,
    price_only_predicted_excess_return_pct double precision,
    primary key (date, index_name, horizon, model_version)
);

create table if not exists public.news_predictions (
    as_of_date date not null,
    index_name text not null,
    horizon text not null,
    expected_excess_return_pct double precision,
    expected_absolute_return_pct double precision,
    probability_positive_excess_pct double precision,
    confidence_pct double precision,
    model_version text not null references public.news_model_runs(model_version),
    model_status text not null,
    noise_enabled boolean not null default false,
    signal text not null,
    rank integer not null,
    created_at_utc timestamptz not null default now(),
    primary key (as_of_date, index_name, horizon, model_version)
);

create table if not exists public.news_prediction_catalysts (
    article_id text not null references public.news_articles(article_id),
    index_name text not null,
    model_version text not null references public.news_model_runs(model_version),
    horizon text not null default '5D',
    published_at_utc timestamptz,
    publisher text,
    title text,
    url text,
    themes text,
    relevance double precision,
    attribution_reason text,
    sentiment_score double precision,
    sentiment_source text,
    signal text,
    catalyst_contribution double precision,
    catalyst_magnitude double precision,
    primary key (article_id, index_name, model_version, horizon)
);

create table if not exists public.news_prediction_drivers (
    as_of_date date not null,
    index_name text not null,
    horizon text not null,
    model_version text not null references public.news_model_runs(model_version) on delete cascade,
    driver_rank integer not null,
    feature text not null,
    contribution double precision not null,
    primary key (as_of_date, index_name, horizon, model_version, driver_rank)
);

create table if not exists public.news_index_eligibility (
    index_name text primary key,
    model_eligible boolean not null,
    eligibility_reason text not null
);

insert into storage.buckets (id, name, public)
values ('news-models', 'news-models', false)
on conflict (id) do nothing;

alter table public.news_articles enable row level security;
alter table public.news_backfill_checkpoints enable row level security;
alter table public.news_embeddings enable row level security;
alter table public.news_index_links enable row level security;
alter table public.index_prices enable row level security;
alter table public.index_constituents enable row level security;
alter table public.constituent_daily_activity enable row level security;
alter table public.news_pipeline_jobs enable row level security;
alter table public.news_model_runs enable row level security;
alter table public.news_model_metrics enable row level security;
alter table public.news_model_evaluation enable row level security;
alter table public.news_predictions enable row level security;
alter table public.news_prediction_catalysts enable row level security;
alter table public.news_prediction_drivers enable row level security;
alter table public.news_index_eligibility enable row level security;

drop policy if exists "Public read news predictions" on public.news_predictions;
drop policy if exists "Public read news catalysts" on public.news_prediction_catalysts;
drop policy if exists "Public read news metrics" on public.news_model_metrics;
drop policy if exists "Public read news jobs" on public.news_pipeline_jobs;
drop policy if exists "Public read news evaluation" on public.news_model_evaluation;
drop policy if exists "Public read news eligibility" on public.news_index_eligibility;
drop policy if exists "Public read news drivers" on public.news_prediction_drivers;
create policy "Public read news predictions" on public.news_predictions for select using (true);
create policy "Public read news catalysts" on public.news_prediction_catalysts for select using (true);
create policy "Public read news metrics" on public.news_model_metrics for select using (true);
create policy "Public read news jobs" on public.news_pipeline_jobs for select using (true);
create policy "Public read news evaluation" on public.news_model_evaluation for select using (true);
create policy "Public read news eligibility" on public.news_index_eligibility for select using (true);
create policy "Public read news drivers" on public.news_prediction_drivers for select using (true);
