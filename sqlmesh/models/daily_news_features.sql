MODEL (
  name news_ml.daily_news_features,
  kind INCREMENTAL_BY_TIME_RANGE (
    time_column feature_date
  ),
  grain (feature_date, index_name),
  audits (
    not_null(columns := (feature_date, index_name)),
    unique_combination_of_columns(columns := (feature_date, index_name))
  )
);

WITH market_dates AS (
  SELECT DISTINCT date
  FROM public.index_prices
  WHERE index_name = 'Nifty 50'
),
point_in_time_news AS (
  SELECT
    links.index_name,
    article.article_id,
    article.publisher,
    article.title,
    article.text_available,
    article.gdelt_tone,
    article.positive_probability,
    article.negative_probability,
    article.neutral_probability,
    article.sentiment_score,
    embedding.embedding,
    links.relevance,
    signal_date.date AS feature_date
  FROM public.news_index_links AS links
  JOIN public.news_articles AS article USING (article_id)
  LEFT JOIN public.news_embeddings AS embedding USING (article_id)
  JOIN LATERAL (
    SELECT date
    FROM market_dates
    WHERE date >= CASE
      WHEN (article.published_at_utc AT TIME ZONE 'Asia/Kolkata')::time <= TIME '16:30'
        THEN (article.published_at_utc AT TIME ZONE 'Asia/Kolkata')::date
      ELSE (article.published_at_utc AT TIME ZONE 'Asia/Kolkata')::date + 1
    END
    ORDER BY date
    LIMIT 1
  ) AS signal_date ON TRUE
  WHERE signal_date.date BETWEEN @start_date AND @end_date
),
daily AS (
  SELECT
    feature_date,
    index_name,
    COUNT(DISTINCT article_id)::double precision AS news_article_count,
    COUNT(DISTINCT publisher)::double precision AS news_source_count,
    SUM(relevance) AS news_relevance_sum,
    SUM(COALESCE(sentiment_score, 0) * relevance) / NULLIF(SUM(relevance), 0) AS news_sentiment_raw,
    AVG(positive_probability) AS news_positive_probability,
    AVG(negative_probability) AS news_negative_probability,
    AVG(neutral_probability) AS news_neutral_probability,
    AVG(gdelt_tone) AS news_gdelt_tone,
    AVG(text_available::integer) AS news_text_coverage,
    COUNT(DISTINCT publisher)::double precision / NULLIF(COUNT(DISTINCT article_id), 0) AS news_source_diversity,
    COUNT(DISTINCT title)::double precision / NULLIF(COUNT(*), 0) AS news_novelty,
    AVG(embedding) AS news_embedding
  FROM point_in_time_news
  GROUP BY feature_date, index_name
),
global_prior AS (
  SELECT feature_date, AVG(news_sentiment_raw) AS global_sentiment
  FROM daily
  GROUP BY feature_date
)
SELECT
  daily.*,
  (
    daily.news_article_count * COALESCE(daily.news_sentiment_raw, prior.global_sentiment, 0)
    + 5.0 * COALESCE(prior.global_sentiment, 0)
  ) / (daily.news_article_count + 5.0) AS news_sentiment_shrunk
FROM daily
LEFT JOIN global_prior AS prior USING (feature_date)
;
