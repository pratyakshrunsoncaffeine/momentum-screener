from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .news_config import NEWS_HORIZONS, NewsCatalystConfig
from .news_features import inject_training_noise, model_feature_columns, winsorize_features


@dataclass
class FeaturePreprocessor:
    feature_columns: list[str]
    winsor_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    embedding_columns: list[str] = field(default_factory=list)
    passthrough_columns: list[str] = field(default_factory=list)
    svd: Any | None = None

    def fit(self, frame: pd.DataFrame, config: NewsCatalystConfig) -> "FeaturePreprocessor":
        cleaned, limits = winsorize_features(
            frame,
            self.feature_columns,
            config.shrinkage.winsor_lower,
            config.shrinkage.winsor_upper,
        )
        self.winsor_limits = limits
        self.embedding_columns = [column for column in self.feature_columns if column.startswith("embedding_")]
        self.passthrough_columns = [column for column in self.feature_columns if column not in self.embedding_columns]
        self.medians = {
            column: float(pd.to_numeric(cleaned[column], errors="coerce").median())
            if pd.to_numeric(cleaned[column], errors="coerce").notna().any()
            else 0.0
            for column in self.feature_columns
        }
        if len(self.embedding_columns) > max(2, config.shrinkage.embedding_components):
            from sklearn.decomposition import TruncatedSVD

            components = min(
                int(config.shrinkage.embedding_components),
                len(self.embedding_columns) - 1,
                max(1, len(cleaned) - 1),
            )
            matrix = self._numeric(cleaned, self.embedding_columns)
            self.svd = TruncatedSVD(n_components=components, random_state=config.random_seed).fit(matrix)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in self.feature_columns:
            if column not in result:
                result[column] = np.nan
            low, high = self.winsor_limits.get(column, (-np.inf, np.inf))
            result[column] = pd.to_numeric(result[column], errors="coerce").clip(low, high).fillna(
                self.medians.get(column, 0.0)
            )
        output = self._numeric(result, self.passthrough_columns)
        if self.embedding_columns:
            embeddings = self._numeric(result, self.embedding_columns)
            if self.svd is not None:
                reduced = self.svd.transform(embeddings)
                reduced_frame = pd.DataFrame(
                    reduced,
                    index=result.index,
                    columns=[f"embedding_svd_{index}" for index in range(reduced.shape[1])],
                )
                output = pd.concat([output, reduced_frame], axis=1)
            else:
                output = pd.concat([output, embeddings], axis=1)
        return output.astype(float)

    def _numeric(self, frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: pd.to_numeric(frame[column], errors="coerce").fillna(self.medians.get(column, 0.0))
                for column in columns
            },
            index=frame.index,
        )


@dataclass
class HorizonModels:
    horizon: str
    days: int
    excess_ridge: Any
    excess_lightgbm: Any
    absolute_lightgbm: Any
    positive_classifier: Any
    probability_calibrator: Any | None
    price_only_model: Any
    lightgbm_weight: float
    noise_enabled: bool


@dataclass
class NewsModelBundle:
    model_version: str
    trained_at_utc: str
    preprocessor: FeaturePreprocessor
    horizons: dict[str, HorizonModels]
    metrics: pd.DataFrame
    status: str
    feature_columns: list[str]


def chronological_forward_split(
    frame: pd.DataFrame,
    embargo_days: int = 63,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, pd.DataFrame]:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["Date"], errors="coerce").dropna().unique()).sort_values()
    if len(dates) < max(embargo_days * 2 + 30, 100):
        raise ValueError("Insufficient dated history for an embargoed train/validation/test split.")
    train_boundary = max(int(len(dates) * train_fraction), 1)
    validation_boundary = max(int(len(dates) * (train_fraction + validation_fraction)), train_boundary + 1)
    train_dates = dates[:train_boundary]
    validation_dates = dates[min(train_boundary + embargo_days, validation_boundary) : validation_boundary]
    test_dates = dates[min(validation_boundary + embargo_days, len(dates)) :]
    if len(validation_dates) < 10 or len(test_dates) < 10:
        raise ValueError("The requested embargo leaves too little validation or test history.")
    normalized = frame.copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce").dt.normalize()
    return {
        "Train": normalized[normalized["Date"].isin(train_dates)].copy(),
        "Validation": normalized[normalized["Date"].isin(validation_dates)].copy(),
        "Test": normalized[normalized["Date"].isin(test_dates)].copy(),
    }


def train_news_models(
    training_frame: pd.DataFrame,
    output_dir: str | Path,
    config: NewsCatalystConfig | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, object]:
    settings = config or NewsCatalystConfig()
    features = feature_columns or model_feature_columns(training_frame)
    if not features:
        raise ValueError("No numeric point-in-time model features are available.")
    splits = chronological_forward_split(training_frame, embargo_days=max(settings.horizons.values()))
    preprocessor = FeaturePreprocessor(features).fit(splits["Train"], settings)
    transformed = {name: preprocessor.transform(frame) for name, frame in splits.items()}
    horizon_models: dict[str, HorizonModels] = {}
    metric_rows: list[dict[str, object]] = []
    test_predictions: list[pd.DataFrame] = []

    for horizon, days in settings.horizons.items():
        targets = _target_columns(horizon)
        valid_splits = {
            name: frame.dropna(subset=list(targets.values())).copy()
            for name, frame in splits.items()
        }
        if any(frame.empty for frame in valid_splits.values()):
            continue
        matrices = {name: transformed[name].loc[frame.index] for name, frame in valid_splits.items()}
        base_candidate = _fit_candidate(
            matrices, valid_splits, features, preprocessor, settings, horizon, noise=False
        )
        noise_candidate = _fit_candidate(
            matrices, valid_splits, features, preprocessor, settings, horizon, noise=True
        )
        chosen = noise_candidate if noise_candidate["validation_score"] > base_candidate["validation_score"] else base_candidate
        models = HorizonModels(
            horizon=horizon,
            days=int(days),
            excess_ridge=chosen["excess_ridge"],
            excess_lightgbm=chosen["excess_lightgbm"],
            absolute_lightgbm=chosen["absolute_lightgbm"],
            positive_classifier=chosen["positive_classifier"],
            probability_calibrator=chosen["probability_calibrator"],
            price_only_model=chosen["price_only_model"],
            lightgbm_weight=float(chosen["lightgbm_weight"]),
            noise_enabled=bool(chosen["noise_enabled"]),
        )
        horizon_models[horizon] = models
        test_result, metrics = _evaluate_horizon(models, matrices["Test"], valid_splits["Test"], horizon)
        test_result["Horizon"] = horizon
        metrics.update(
            {
                "Horizon": horizon,
                "Days": days,
                "Noise Enabled": models.noise_enabled,
                "Validation Score": chosen["validation_score"],
                "LightGBM Ensemble Weight": models.lightgbm_weight,
            }
        )
        metric_rows.append(metrics)
        test_predictions.append(test_result)

    if not horizon_models:
        raise ValueError("No horizon had enough mature labels to train.")
    metrics_frame = pd.DataFrame(metric_rows)
    status = _promotion_status(metrics_frame)
    now = pd.Timestamp.now(tz="UTC")
    version = f"news-{now.strftime('%Y%m%dT%H%M%SZ')}"
    bundle = NewsModelBundle(
        model_version=version,
        trained_at_utc=now.isoformat(),
        preprocessor=preprocessor,
        horizons=horizon_models,
        metrics=metrics_frame,
        status=status,
        feature_columns=features,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "news_model_bundle.joblib"
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Model persistence requires joblib from requirements-ml.txt.") from exc
    joblib.dump(bundle, artifact)
    metrics_frame.to_csv(destination / "news_model_metrics.csv", index=False)
    predictions = pd.concat(test_predictions, ignore_index=True) if test_predictions else pd.DataFrame()
    if not predictions.empty:
        predictions["Model Version"] = version
    predictions.to_csv(destination / "news_test_predictions.csv", index=False)
    return {"bundle": bundle, "metrics": metrics_frame, "test_predictions": predictions, "artifact": artifact}


def load_news_model(path: str | Path) -> NewsModelBundle:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Loading the news model requires joblib from requirements-ml.txt.") from exc
    bundle = joblib.load(path)
    if not isinstance(bundle, NewsModelBundle):
        raise TypeError("The saved artifact is not a NewsModelBundle.")
    return bundle


def predict_news_catalysts(
    bundle: NewsModelBundle,
    latest_features: pd.DataFrame,
) -> pd.DataFrame:
    if latest_features.empty:
        return pd.DataFrame()
    matrix = bundle.preprocessor.transform(latest_features)
    rows: list[pd.DataFrame] = []
    for horizon, models in bundle.horizons.items():
        ridge = models.excess_ridge.predict(matrix)
        lightgbm = models.excess_lightgbm.predict(matrix)
        expected_excess = models.lightgbm_weight * lightgbm + (1.0 - models.lightgbm_weight) * ridge
        expected_absolute = models.absolute_lightgbm.predict(matrix)
        raw_probability = models.positive_classifier.predict_proba(matrix)[:, 1]
        probability = (
            models.probability_calibrator.predict(raw_probability)
            if models.probability_calibrator is not None
            else raw_probability
        )
        frame = pd.DataFrame(
            {
                "As Of Date": pd.to_datetime(latest_features["Date"], errors="coerce").dt.date,
                "Index": latest_features["Index"].astype(str).values,
                "Horizon": horizon,
                "Expected Excess Return %": expected_excess,
                "Expected Absolute Return %": expected_absolute,
                "Probability Positive Excess %": probability * 100.0,
                "Confidence %": np.abs(probability - 0.5) * 200.0,
                "Model Version": bundle.model_version,
                "Model Status": bundle.status,
                "Noise Enabled": models.noise_enabled,
            }
        )
        frame["Signal"] = np.where(frame["Expected Excess Return %"] >= 0, "Tailwind", "Headwind")
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    result["Rank"] = result.groupby("Horizon")["Expected Excess Return %"].rank(
        ascending=False, method="first"
    ).astype(int)
    return result.sort_values(["Horizon", "Rank"]).reset_index(drop=True)


def explain_model_features(
    bundle: NewsModelBundle,
    latest_features: pd.DataFrame,
    horizon: str,
    top_n: int = 8,
) -> pd.DataFrame:
    models = bundle.horizons[horizon]
    matrix = bundle.preprocessor.transform(latest_features)
    try:
        import shap

        values = shap.TreeExplainer(models.excess_lightgbm).shap_values(matrix)
        contributions = np.asarray(values)
    except (ImportError, ValueError, TypeError):
        importance = np.asarray(getattr(models.excess_lightgbm, "feature_importances_", np.ones(matrix.shape[1])))
        centered = matrix - matrix.median(axis=0)
        contributions = centered.to_numpy() * (importance / max(float(importance.sum()), 1.0))
    rows: list[dict[str, object]] = []
    for row_position, (_, source) in enumerate(latest_features.iterrows()):
        order = np.argsort(np.abs(contributions[row_position]))[::-1][:top_n]
        for rank, column_position in enumerate(order, start=1):
            rows.append(
                {
                    "Index": source["Index"],
                    "Horizon": horizon,
                    "Driver Rank": rank,
                    "Feature": matrix.columns[column_position],
                    "Contribution": float(contributions[row_position, column_position]),
                }
            )
    return pd.DataFrame(rows)


def catalyst_headlines(
    predictions: pd.DataFrame,
    articles: pd.DataFrame,
    links: pd.DataFrame,
    horizon: str,
    top_per_index: int = 5,
) -> pd.DataFrame:
    selected = predictions[predictions["Horizon"].eq(horizon)].copy()
    merged = links.merge(articles, on="Article ID", how="inner").merge(
        selected[["Index", "Expected Excess Return %", "Signal", "Model Version"]], on="Index", how="inner"
    )
    if merged.empty:
        return merged
    sentiment = pd.to_numeric(merged.get("Sentiment Score"), errors="coerce").fillna(0.0)
    direction = np.sign(pd.to_numeric(merged["Expected Excess Return %"], errors="coerce")).replace(0, 1)
    merged["Catalyst Contribution"] = pd.to_numeric(merged["Relevance"], errors="coerce").fillna(0.0) * sentiment * direction
    merged["Catalyst Magnitude"] = merged["Catalyst Contribution"].abs()
    merged = merged.sort_values(["Index", "Catalyst Magnitude"], ascending=[True, False])
    return merged.groupby("Index", observed=True).head(top_per_index).reset_index(drop=True)


def _fit_candidate(
    matrices: dict[str, pd.DataFrame],
    splits: dict[str, pd.DataFrame],
    original_features: list[str],
    preprocessor: FeaturePreprocessor,
    config: NewsCatalystConfig,
    horizon: str,
    noise: bool,
) -> dict[str, object]:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    targets = _target_columns(horizon)
    train_matrix = matrices["Train"]
    train_rows = splits["Train"]
    if noise:
        noisy_raw = inject_training_noise(train_rows, original_features, config)
        noisy_matrix = preprocessor.transform(noisy_raw)
        train_matrix = pd.concat([train_matrix, noisy_matrix], ignore_index=True)
        train_rows = pd.concat([train_rows, train_rows], ignore_index=True)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(train_matrix, train_rows[targets["excess"]])
    common = {
        "n_estimators": 300,
        "learning_rate": 0.035,
        "num_leaves": 15,
        "max_depth": 5,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.50,
        "reg_lambda": 2.0,
        "random_state": config.random_seed,
        "verbosity": -1,
    }
    excess = LGBMRegressor(**common).fit(train_matrix, train_rows[targets["excess"]])
    absolute = LGBMRegressor(**common).fit(train_matrix, train_rows[targets["absolute"]])
    classifier = LGBMClassifier(**common).fit(train_matrix, train_rows[targets["positive"]].astype(int))
    price_columns = [
        column
        for column in train_matrix.columns
        if column.startswith(("price_", "market_", "volume_", "turnover_", "breadth_", "participation_"))
    ]
    if not price_columns:
        price_columns = train_matrix.columns.tolist()
    price_only = LGBMRegressor(**common).fit(train_matrix[price_columns], train_rows[targets["excess"]])
    setattr(price_only, "news_price_columns_", price_columns)

    validation = matrices["Validation"]
    validation_rows = splits["Validation"]
    ridge_prediction = ridge.predict(validation)
    lightgbm_prediction = excess.predict(validation)
    ridge_mae = mean_absolute_error(validation_rows[targets["excess"]], ridge_prediction)
    lightgbm_mae = mean_absolute_error(validation_rows[targets["excess"]], lightgbm_prediction)
    lightgbm_weight = float(ridge_mae / max(ridge_mae + lightgbm_mae, 1e-12))
    ensemble = lightgbm_weight * lightgbm_prediction + (1.0 - lightgbm_weight) * ridge_prediction
    raw_probability = classifier.predict_proba(validation)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_probability, validation_rows[targets["positive"]].astype(int)
    )
    calibrated = calibrator.predict(raw_probability)
    auc = _safe_auc(validation_rows[targets["positive"]], calibrated)
    validation_mae = mean_absolute_error(validation_rows[targets["excess"]], ensemble)
    score = -float(validation_mae) + max(float(auc) - 0.5, 0.0)
    return {
        "excess_ridge": ridge,
        "excess_lightgbm": excess,
        "absolute_lightgbm": absolute,
        "positive_classifier": classifier,
        "probability_calibrator": calibrator,
        "price_only_model": price_only,
        "lightgbm_weight": lightgbm_weight,
        "noise_enabled": noise,
        "validation_score": score,
    }


def _evaluate_horizon(
    models: HorizonModels,
    matrix: pd.DataFrame,
    rows: pd.DataFrame,
    horizon: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score

    targets = _target_columns(horizon)
    ridge = models.excess_ridge.predict(matrix)
    tree = models.excess_lightgbm.predict(matrix)
    expected = models.lightgbm_weight * tree + (1.0 - models.lightgbm_weight) * ridge
    price_columns = getattr(models.price_only_model, "news_price_columns_", matrix.columns.tolist())
    price_only = models.price_only_model.predict(matrix[price_columns])
    raw_probability = models.positive_classifier.predict_proba(matrix)[:, 1]
    probability = (
        models.probability_calibrator.predict(raw_probability)
        if models.probability_calibrator is not None
        else raw_probability
    )
    result = rows[["Date", "Index", targets["absolute"], targets["excess"], targets["positive"]]].copy()
    result = result.rename(
        columns={
            targets["absolute"]: "Actual Absolute Return %",
            targets["excess"]: "Actual Excess Return %",
            targets["positive"]: "Actual Positive Excess",
        }
    )
    result["Predicted Excess Return %"] = expected
    result["Predicted Positive Probability"] = probability
    result["Price-Only Predicted Excess Return %"] = price_only
    rank_ic = _daily_rank_ic(result, "Actual Excess Return %", "Predicted Excess Return %")
    price_rank_ic = _daily_rank_ic(result, "Actual Excess Return %", "Price-Only Predicted Excess Return %")
    top_five = result.sort_values(["Date", "Predicted Excess Return %"], ascending=[True, False]).groupby("Date").head(5)
    portfolio = top_five.groupby("Date", observed=True)["Actual Excess Return %"].mean() - 0.20
    curve = (1.0 + portfolio / 100.0).cumprod()
    drawdown = (curve / curve.cummax() - 1.0) * 100.0
    sharpe = 0.0
    if portfolio.std(ddof=1) > 0:
        sharpe = float(portfolio.mean() / portfolio.std(ddof=1) * np.sqrt(252.0 / max(models.days, 1)))
    return result, {
        "Test MAE": float(mean_absolute_error(result["Actual Excess Return %"], expected)),
        "Test AUC": _safe_auc(result["Actual Positive Excess"], probability),
        "Directional Accuracy": float(accuracy_score(result["Actual Positive Excess"], probability >= 0.5)),
        "Rank IC": rank_ic,
        "Price-Only Rank IC": price_rank_ic,
        "Rank IC Improvement": rank_ic - price_rank_ic,
        "Top Five Mean Excess %": float(top_five["Actual Excess Return %"].mean()),
        "Top Five Hit Rate %": float((top_five["Actual Excess Return %"] > 0).mean() * 100.0),
        "Theoretical Excess After Cost %": float(portfolio.mean()),
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown %": float(drawdown.min()) if not drawdown.empty else 0.0,
        "Observations": int(len(result)),
    }


def _target_columns(horizon: str) -> dict[str, str]:
    return {
        "absolute": f"Absolute Return {horizon} %",
        "excess": f"Excess Return {horizon} %",
        "positive": f"Positive Excess {horizon}",
    }


def _daily_rank_ic(frame: pd.DataFrame, actual: str, predicted: str) -> float:
    values = frame.groupby("Date", observed=True).apply(
        lambda group: group[actual].corr(group[predicted], method="spearman") if len(group) >= 3 else np.nan,
        include_groups=False,
    )
    return float(values.mean()) if values.notna().any() else 0.0


def _safe_auc(actual: pd.Series, probability: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    values = pd.to_numeric(actual, errors="coerce").dropna().astype(int)
    if values.nunique() < 2:
        return 0.5
    aligned = np.asarray(probability)[actual.notna().to_numpy()]
    return float(roc_auc_score(values, aligned))


def _promotion_status(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "Experimental"
    improved = (pd.to_numeric(metrics["Rank IC"], errors="coerce") > 0) & (
        pd.to_numeric(metrics["Rank IC Improvement"], errors="coerce") > 0
    )
    return "Validated" if int(improved.sum()) >= 2 else "Experimental"
