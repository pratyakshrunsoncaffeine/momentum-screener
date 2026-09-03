MODEL (
  name news_ml.forward_labels,
  kind VIEW,
  grain ("Date", "Index")
);

SELECT
  "Date",
  "Index",
  "Feature Cutoff UTC",
  "Absolute Return 5D %",
  "Benchmark Return 5D %",
  "Excess Return 5D %",
  "Positive Excess 5D",
  "Absolute Return 1M %",
  "Benchmark Return 1M %",
  "Excess Return 1M %",
  "Positive Excess 1M",
  "Absolute Return 3M %",
  "Benchmark Return 3M %",
  "Excess Return 3M %",
  "Positive Excess 3M"
FROM news_ml.training_frame
;
