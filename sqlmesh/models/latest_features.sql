MODEL (
  name public.news_latest_features,
  kind VIEW,
  grain ("Date", "Index")
);

SELECT training.*
FROM news_ml.training_frame AS training
JOIN (
  SELECT "Index", MAX("Date") AS latest_date
  FROM news_ml.training_frame
  GROUP BY "Index"
) AS latest
  ON latest."Index" = training."Index" AND latest.latest_date = training."Date"
;
