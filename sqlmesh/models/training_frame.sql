MODEL (
  name news_ml.training_frame,
  kind FULL,
  grain ("Date", "Index"),
  audits (
    not_null(columns := ("Date", "Index", "Close")),
    unique_combination_of_columns(columns := ("Date", "Index"))
  )
);

WITH price_lags AS (
  SELECT
    date,
    index_name,
    close,
    LAG(close, 1) OVER index_window AS lag_1,
    LAG(close, 5) OVER index_window AS lag_5,
    LAG(close, 21) OVER index_window AS lag_21,
    LAG(close, 63) OVER index_window AS lag_63,
    LEAD(close, 5) OVER index_window AS lead_5,
    LEAD(close, 21) OVER index_window AS lead_21,
    LEAD(close, 63) OVER index_window AS lead_63
  FROM public.index_prices
  WINDOW index_window AS (PARTITION BY index_name ORDER BY date)
),
prices AS (
  SELECT
    price_lags.*,
    STDDEV_SAMP(LN(close / NULLIF(lag_1, 0)))
      OVER (PARTITION BY index_name ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS price_volatility_21d
  FROM price_lags
),
benchmark AS (
  SELECT
    date,
    100.0 * (lead_5 / close - 1.0) AS benchmark_5d,
    100.0 * (lead_21 / close - 1.0) AS benchmark_1m,
    100.0 * (lead_63 / close - 1.0) AS benchmark_3m
  FROM prices
  WHERE index_name = 'Nifty 50'
),
combined AS (
  SELECT
    prices.*,
    COALESCE(news.news_article_count, 0) AS news_article_count,
    COALESCE(news.news_source_count, 0) AS news_source_count,
    COALESCE(news.news_relevance_sum, 0) AS news_relevance_sum,
    COALESCE(news.news_sentiment_raw, 0) AS news_sentiment_raw,
    COALESCE(news.news_sentiment_shrunk, 0) AS news_sentiment_shrunk,
    COALESCE(news.news_positive_probability, 0) AS news_positive_probability,
    COALESCE(news.news_negative_probability, 0) AS news_negative_probability,
    COALESCE(news.news_neutral_probability, 0) AS news_neutral_probability,
    COALESCE(news.news_gdelt_tone, 0) AS news_gdelt_tone,
    COALESCE(news.news_text_coverage, 0) AS news_text_coverage,
    COALESCE(news.news_source_diversity, 0) AS news_source_diversity,
    COALESCE(news.news_novelty, 0) AS news_novelty,
    news.news_embedding,
    activity.volume_abnormal,
    activity.turnover_abnormal,
    activity.breadth_positive,
    activity.participation_rate,
    benchmark.benchmark_5d,
    benchmark.benchmark_1m,
    benchmark.benchmark_3m
  FROM prices
  LEFT JOIN news_ml.daily_news_features AS news
    ON news.feature_date = prices.date AND news.index_name = prices.index_name
  LEFT JOIN public.constituent_daily_activity AS activity
    ON activity.date = prices.date AND activity.index_name = prices.index_name
  LEFT JOIN benchmark USING (date)
)
SELECT
  date AS "Date",
  index_name AS "Index",
  close AS "Close",
  (date::timestamp + TIME '16:30') AT TIME ZONE 'Asia/Kolkata' AS "Feature Cutoff UTC",
  news_article_count,
  news_source_count,
  news_relevance_sum,
  news_sentiment_raw,
  news_sentiment_shrunk,
  news_positive_probability,
  news_negative_probability,
  news_neutral_probability,
  news_gdelt_tone,
  news_text_coverage,
  news_source_diversity,
  news_novelty,
  news_embedding,
  (news_article_count >= 2 AND news_source_count >= 2)::integer AS news_reliable,
  news_sentiment_shrunk * (news_article_count >= 2 AND news_source_count >= 2)::integer AS news_sentiment_reliable,
  AVG(news_sentiment_shrunk) OVER (PARTITION BY index_name ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS news_sentiment_shrunk_ewm_3,
  AVG(news_sentiment_shrunk) OVER (PARTITION BY index_name ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS news_sentiment_shrunk_ewm_7,
  AVG(news_sentiment_shrunk) OVER (PARTITION BY index_name ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS news_sentiment_shrunk_ewm_21,
  AVG(news_sentiment_shrunk) OVER (PARTITION BY index_name ORDER BY date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW) AS news_sentiment_shrunk_ewm_63,
  100.0 * (close / NULLIF(lag_1, 0) - 1.0) AS price_return_1d,
  100.0 * (close / NULLIF(lag_5, 0) - 1.0) AS price_return_5d,
  100.0 * (close / NULLIF(lag_21, 0) - 1.0) AS price_return_21d,
  100.0 * (close / NULLIF(lag_63, 0) - 1.0) AS price_return_63d,
  price_volatility_21d,
  volume_abnormal,
  turnover_abnormal,
  breadth_positive,
  participation_rate,
  100.0 * (lead_5 / close - 1.0) AS "Absolute Return 5D %",
  benchmark_5d AS "Benchmark Return 5D %",
  100.0 * (lead_5 / close - 1.0) - benchmark_5d AS "Excess Return 5D %",
  (100.0 * (lead_5 / close - 1.0) - benchmark_5d > 0)::integer AS "Positive Excess 5D",
  100.0 * (lead_21 / close - 1.0) AS "Absolute Return 1M %",
  benchmark_1m AS "Benchmark Return 1M %",
  100.0 * (lead_21 / close - 1.0) - benchmark_1m AS "Excess Return 1M %",
  (100.0 * (lead_21 / close - 1.0) - benchmark_1m > 0)::integer AS "Positive Excess 1M",
  100.0 * (lead_63 / close - 1.0) AS "Absolute Return 3M %",
  benchmark_3m AS "Benchmark Return 3M %",
  100.0 * (lead_63 / close - 1.0) - benchmark_3m AS "Excess Return 3M %",
  (100.0 * (lead_63 / close - 1.0) - benchmark_3m > 0)::integer AS "Positive Excess 3M"
FROM combined
;
