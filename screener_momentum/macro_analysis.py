"""Release-aware macro sensitivities and chronological scenario evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .macro_data import CATALOGUE


@dataclass(frozen=True)
class MacroCorrelationConfig:
    years: int = 15
    horizon: int = 63
    excess: bool = False
    bootstrap_samples: int = 300
    seed: int = 42
    relationship: str = "forward"
    lag: int = 0

    def __post_init__(self):
        if not 5 <= self.years <= 20 or self.horizon not in (21, 63):
            raise ValueError("Use 5-20 years and a 21- or 63-session horizon.")
        if self.relationship not in {"forward", "same_period"} or self.lag not in (0, 1, 3, 6):
            raise ValueError("Invalid relationship or lag.")


def feature_value(history, definition, transform=None):
    """Exact calendar lags prevent missing periods becoming shorter growth windows."""
    history = history.sort_values("period").drop_duplicates("period", keep="last")
    if history.empty:
        return np.nan
    last = history.iloc[-1]
    transform = transform or definition.transform
    if transform == "level":
        return float(last.value)
    periods = history.period.dt.to_period("Q" if definition.frequency == "Q" else "M")
    lag = (4 if definition.frequency == "Q" else 12) if transform == "yoy" else 1
    previous = history[periods == periods.iloc[-1] - lag]
    if previous.empty or previous.iloc[-1].base_year != last.base_year:
        return np.nan
    old = float(previous.iloc[-1].value)
    if transform == "change":
        return float(last.value) - old
    return (float(last.value) / old - 1) * 100 if old > 0 else np.nan


def feature_history(observations, identifier, as_of, eligible_only=False, transform=None):
    d = CATALOGUE[identifier]
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    obs = observations[(observations.series_id == identifier) & (observations.available_at <= cutoff)].copy()
    if eligible_only:
        obs = obs[obs.eligible == 1]
    if obs.empty:
        return pd.DataFrame(columns=["signal_date", "period", "x", "eligible"])
    if not eligible_only and obs.eligible.eq(1).all():
        return feature_history(observations, identifier, as_of, eligible_only=True, transform=transform)
    result = []
    if eligible_only:
        for release in sorted(obs.available_at.unique()):
            history = obs[obs.available_at <= release].sort_values("available_at").drop_duplicates("period", keep="last")
            result.append({"signal_date": pd.Timestamp(release).normalize(), "period": history.period.max(),
                           "x": feature_value(history, d, transform), "eligible": True})
    else:
        latest = obs.sort_values("available_at").drop_duplicates("period", keep="last").sort_values("period")
        for _, row in latest.iterrows():
            history = latest[latest.period <= row.period]
            # Explicitly exploratory: release-lag proxy is not a historical vintage.
            anchor = row.available_at.normalize() if row.eligible else row.period + pd.Timedelta(days=90 if d.frequency == "Q" else 45)
            result.append({"signal_date": anchor, "period": row.period,
                           "x": feature_value(history, d, transform), "eligible": bool(row.eligible)})
    f = pd.DataFrame(result).dropna(subset=["x"])
    return f[f.signal_date <= cutoff].sort_values("signal_date").drop_duplicates("signal_date", keep="last")


def aligned_returns(features, prices, asset, config, as_of):
    if features.empty or asset not in prices:
        return pd.DataFrame()
    columns = [asset] + (["Nifty 50"] if config.excess and asset != "Nifty 50" else [])
    if not set(columns).issubset(prices):
        return pd.DataFrame()
    # Preserve the benchmark trading calendar: missing asset sessions cannot silently stretch a horizon.
    p = prices.loc[prices.index <= pd.Timestamp(as_of), columns].sort_index()
    rows = []
    for f in features.itertuples(index=False):
        if config.relationship == "same_period":
            period = pd.Timestamp(f.period)
            months = 3 if getattr(f, "frequency", "M") == "Q" else 1
            entry = p.index.searchsorted(period-pd.DateOffset(months=months), side="right")-1
            exit_pos = p.index.searchsorted(period, side="right")-1
        else:
            entry = p.index.searchsorted(f.signal_date, side="right")
            exit_pos = entry + config.horizon
        if exit_pos >= len(p) or entry < 63:
            continue
        start, end = p.iloc[entry], p.iloc[exit_pos]
        if start.isna().any() or end.isna().any() or (start <= 0).any() or (end <= 0).any():
            continue
        returns = (end / start - 1) * 100
        trailing = p[asset].iloc[entry-1] / p[asset].iloc[entry-64] - 1
        if not np.isfinite(trailing):
            continue
        rows.append({"signal_date": f.signal_date, "entry_date": p.index[entry], "exit_date": p.index[exit_pos],
                     "x": f.x, "y": returns[asset] - (returns["Nifty 50"] if len(columns) > 1 else 0),
                     "momentum": trailing * 100, "eligible": f.eligible})
    return pd.DataFrame(rows)


def shrunk_correlation(x, y, trim=True):
    data = np.column_stack([x, y]).astype(float)
    if len(data) < 4 or not np.isfinite(data).all():
        return np.nan
    if trim:
        data = np.clip(data, np.quantile(data, .01, axis=0), np.quantile(data, .99, axis=0))
    std = data.std(axis=0)
    if (std < 1e-12).any():
        return np.nan
    cov = LedoitWolf().fit((data-data.mean(axis=0))/std).covariance_
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def bootstrap_stats(x, y, repeats, seed, block):
    rng = np.random.default_rng(seed)
    n = len(x)
    correlations, nulls = [], []
    observed = abs(shrunk_correlation(x, y))
    for _ in range(repeats):
        starts = rng.integers(0, n, size=int(np.ceil(n/block)))
        indices = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        correlations.append(shrunk_correlation(x[indices], y[indices]))
        # Independently resample contiguous target blocks for an autocorrelation-preserving null.
        starts_y = rng.integers(0, n, size=len(starts))
        iy = np.concatenate([(s + np.arange(block)) % n for s in starts_y])[:n]
        nulls.append(abs(shrunk_correlation(x[indices], y[iy])))
    valid = np.asarray(correlations)[np.isfinite(correlations)]
    low, high = np.quantile(valid, [.05, .95]) if len(valid) else (np.nan, np.nan)
    return low, high, (1 + np.sum(np.asarray(nulls) >= observed)) / (repeats + 1)


def adjust_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        result[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order)+1))[::-1])[::-1])
    return result


def analysis_features(observations, identifier, config, as_of):
    d = CATALOGUE[identifier]
    features = feature_history(observations, identifier, as_of)
    if config.lag and not features.empty:
        keyed = features.assign(key=features.period.dt.to_period(d.frequency)).drop_duplicates("key", keep="last").set_index("key").x
        features["x"] = [keyed.get(p-config.lag, np.nan) for p in features.period.dt.to_period(d.frequency)]
        features = features.dropna(subset=["x"])
    features = features[features.signal_date >= pd.Timestamp(as_of)-pd.DateOffset(years=config.years)].copy()
    features["frequency"] = d.frequency
    return features


def analyze(observations, prices, identifiers, config, as_of, progress=None, checkpoint=None):
    rows, pairs = [], {}
    assets = [c for c in prices if c != "Nifty 50"]
    total = len(identifiers) * len(assets)
    for identifier in identifiers:
        d = CATALOGUE[identifier]
        features = analysis_features(observations, identifier, config, as_of)
        for asset in assets:
            if progress:
                progress(len(rows), total, f"{asset}: {d.name}")
            f = aligned_returns(features, prices, asset, config, as_of)
            pairs[(identifier, asset)] = f
            n = len(f)
            minimum = 32 if d.frequency == "Q" else 60
            row = {"Indicator ID": identifier, "Indicator": d.name, "Asset": asset, "Frequency": d.frequency,
                   "Transform": d.transform, "Observations": n, "Status": "Insufficient History", "Correlation": np.nan,
                   "Relationship": config.relationship, "Lag periods": config.lag,
                   "P value": np.nan, "Source": d.source, "History": "Exploratory: revised values / estimated release dates"}
            if n >= minimum and f.x.std() > 0 and f.y.std() > 0:
                x, y = f.x.to_numpy(), f.y.to_numpy()
                rho = shrunk_correlation(x, y)
                lo, hi, pv = bootstrap_stats(x, y, config.bootstrap_samples, config.seed, 4 if d.frequency == "Q" else 6)
                window1, window2 = (12, 20) if d.frequency == "Q" else (36, 60)
                rolling = [shrunk_correlation(x[i-window1:i], y[i-window1:i]) for i in range(window1, n+1)]
                row.update({"Correlation": rho, "Pearson": np.corrcoef(x, y)[0, 1], "Spearman": spearmanr(x, y).statistic,
                            "Untrimmed": shrunk_correlation(x, y, False), "CI low": lo, "CI high": hi, "P value": pv,
                            "Recent correlation": shrunk_correlation(x[-window1:], y[-window1:]),
                            "Long rolling correlation": shrunk_correlation(x[-window2:], y[-window2:]),
                            "Sign stability %": np.mean(np.sign(rolling) == np.sign(rho))*100,
                            "First signal": f.signal_date.min(), "Last signal": f.signal_date.max(),
                            "Status": "Exploratory" if not f.eligible.all() else "Estimated",
                            "History": "Release vintages" if f.eligible.all() else row["History"]})
            rows.append(row)
            if checkpoint:
                checkpoint(pd.DataFrame(rows))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["FDR q"] = adjust_fdr(result["P value"])
        result = result.sort_values("Correlation", ascending=False, na_position="last")
    if progress:
        progress(total, total, "Analysis complete")
    return result, pairs


def tune_ridge(development, columns):
    best_alpha, best_loss = 1., np.inf
    for alpha in [.1, 1., 10., 100., 1000.]:
        losses = []
        for fraction in [.6, .8]:
            split = int(len(development)*fraction)
            val = development.iloc[split:]
            train = development.iloc[:split]
            train = train[train.exit_date < val.signal_date.iloc[0] - pd.offsets.BDay(63)]
            if len(train) < 12:
                continue
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train[columns], train.y)
            losses.append(np.mean(np.abs(model.predict(val[columns])-val.y)))
        if losses and np.mean(losses) < best_loss:
            best_alpha, best_loss = alpha, np.mean(losses)
    if not np.isfinite(best_loss):
        raise ValueError("Insufficient purged validation history.")
    return best_alpha


def scenario_model(observations, prices, identifiers, asset, values, config, as_of):
    if not 1 <= len(identifiers) <= 5:
        raise ValueError("Choose one to five factors.")
    frequencies = {CATALOGUE[i].frequency for i in identifiers}
    if config.relationship != "forward" or config.lag:
        raise ValueError("Scenarios require forward returns and zero additional lag.")
    frames = []
    for identifier in identifiers:
        h = feature_history(observations, identifier, as_of, eligible_only=True)
        if h.empty:
            raise ValueError(f"{CATALOGUE[identifier].name}: dated release vintages are required for scenario forecasts.")
        max_age = 185 if CATALOGUE[identifier].frequency == "Q" else 75
        if (pd.Timestamp(as_of)-h.signal_date.max()).days > max_age:
            raise ValueError(f"{CATALOGUE[identifier].name}: the latest eligible release is stale.")
        frames.append(h[["signal_date", "x"]].rename(columns={"x": identifier}))
    # Predict only when a complete as-of state exists; never backfill later releases.
    dates = pd.DataFrame({"signal_date": sorted(set().union(*(set(f.signal_date) for f in frames)))})
    for identifier, f in zip(identifiers, frames):
        dates = pd.merge_asof(dates, f.sort_values("signal_date"), on="signal_date", direction="backward", tolerance=pd.Timedelta(days=185 if CATALOGUE[identifier].frequency == "Q" else 75))
    dates = dates.dropna().set_index("signal_date").resample("QE" if "Q" in frequencies else "ME").last().dropna().reset_index()
    dates = dates[dates.signal_date <= pd.Timestamp(as_of)]
    dates = dates[dates.signal_date >= pd.Timestamp(as_of)-pd.DateOffset(years=config.years)]
    features = dates[["signal_date"]].assign(x=0., eligible=True)
    aligned = aligned_returns(features, prices, asset, config, as_of)
    if aligned.empty:
        raise ValueError("No matured return labels.")
    data = aligned.merge(dates, on="signal_date").sort_values("signal_date")
    minimum = 48 if "Q" in frequencies else 100
    if len(data) < minimum:
        raise ValueError(f"Need {minimum} release-aligned observations; found {len(data)}.")
    cut = int(len(data)*.8)
    test = data.iloc[cut:]
    development = data.iloc[:cut]
    development = development[development.exit_date < test.signal_date.iloc[0] - pd.offsets.BDay(63)]
    if len(development) < 20:
        raise ValueError("Insufficient training history after the embargo.")
    # Tune within development history only, using expanding chronological folds.
    best_alpha = tune_ridge(development, identifiers)
    model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    model.fit(development[identifiers], development.y)
    predicted = model.predict(test[identifiers])
    baseline = make_pipeline(StandardScaler(), Ridge(alpha=tune_ridge(development, ["momentum"]))).fit(development[["momentum"]], development.y)
    mae = float(np.mean(np.abs(predicted-test.y)))
    mean_mae = float(np.mean(np.abs(test.y-development.y.mean())))
    price_mae = float(np.mean(np.abs(baseline.predict(test[["momentum"]])-test.y)))
    residual = test.y.to_numpy()-predicted
    # Refit on matured history only after freezing held-out evaluation.
    model.fit(data[identifiers], data.y)
    current = pd.DataFrame([{i: float(frames[j].iloc[-1][i]) for j, i in enumerate(identifiers)}])
    proposed = pd.DataFrame([{i: float(values[i]) for i in identifiers}])
    if not np.isfinite(proposed.to_numpy()).all():
        raise ValueError("Scenario values must be finite.")
    estimate, reference = float(model.predict(proposed)[0]), float(model.predict(current)[0])
    lo, hi = np.quantile(residual, [.05, .95])
    outside = any(values[i] < data[i].min() or values[i] > data[i].max() for i in identifiers)
    summary = {"Asset": asset, "Analysis date": str(as_of), "Horizon sessions": config.horizon,
               "Excess return": config.excess, "Factors": ", ".join(identifiers),
               "Scenario values": str(values), "Expected return %": estimate, "Baseline return %": reference,
               "Scenario difference pp": estimate-reference, "90% lower %": estimate+lo, "90% upper %": estimate+hi,
               "MAE": mae, "Mean baseline MAE": mean_mae, "Price baseline MAE": price_mae,
               "Alpha": best_alpha, "Observations": len(data), "Extrapolation": outside,
               "Status": "Validated" if mae < min(mean_mae, price_mae) and not outside else "Experimental"}
    evaluation = test[["signal_date", "entry_date", "exit_date", "y"]].copy()
    evaluation["predicted"] = predicted
    return pd.DataFrame([summary]), evaluation
