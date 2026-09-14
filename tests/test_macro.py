import io
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from screener_momentum.macro_analysis import (
    MacroCorrelationConfig, adjust_fdr, aligned_returns, feature_history,
    analysis_features, analyze, market_pairs, feature_value, scenario_model, shrunk_correlation,
)
from screener_momentum.macro_data import CATALOGUE, MacroStore, clean_observations, parse_mospi


def observations(identifier="pfce_real", n=48):
    dates = pd.date_range("2000-03-31", periods=n, freq="QE" if CATALOGUE[identifier].frequency == "Q" else "ME")
    return pd.DataFrame({"series_id": identifier, "period": dates,
        "available_at": dates + pd.Timedelta(days=60), "retrieved_at": pd.Timestamp("2026-01-01"),
        "value": 100*np.power(1.02, np.arange(n)), "vintage": "original", "eligible": 1,
        "source": "https://mospi.gov.in/", "base_year": "2011"})


class MacroTests(unittest.TestCase):
    def test_market_returns_no_release_delay_and_lag(self):
        days = pd.bdate_range("2020-01-01", periods=800)
        levels = 100*np.exp(np.cumsum(np.random.default_rng(9).normal(0, .01, len(days))))
        obs = pd.DataFrame({"series_id": "brent_crude", "period": days,
            "value": levels, "base_year": "daily market", "retrieved_at": pd.Timestamp("2025-01-01")})
        prices = pd.DataFrame({"Test": levels, "Nifty 50": levels}, index=days)
        config = MacroCorrelationConfig(market_frequency="D")
        f = market_pairs(obs, prices, "brent_crude", "Test", config, days[-1])
        np.testing.assert_allclose(f.x, f.y)
        self.assertEqual(f.signal_date.iloc[0], days[1])
        delayed = market_pairs(obs, prices, "brent_crude", "Test", MacroCorrelationConfig(market_frequency="D", market_lag=1), days[-1])
        np.testing.assert_allclose(delayed.x, f.x.iloc[:-1])
        self.assertTrue((delayed.signal_date < delayed.exit_date).all())
        obs["base_year"] = "source definition"
        self.assertTrue(market_pairs(obs, prices, "brent_crude", "Test", config, days[-1]).empty)

    def test_two_and_three_year_minimums(self):
        for identifier, n in [("pfce_real", 8), ("cpi_headline", 24)]:
            f = pd.DataFrame({"x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float),
                "signal_date": pd.date_range("2020-01-01", periods=n), "eligible": False})
            with patch("screener_momentum.macro_analysis.aligned_returns", return_value=f):
                two, _ = analyze(observations(identifier), pd.DataFrame(columns=["Test"]), [identifier],
                    MacroCorrelationConfig(bootstrap_samples=10), "2025-01-01")
                three, _ = analyze(observations(identifier), pd.DataFrame(columns=["Test"]), [identifier],
                    MacroCorrelationConfig(minimum_years=3, bootstrap_samples=10), "2025-01-01")
            self.assertTrue(np.isfinite(two.Correlation.iloc[0]))
            self.assertEqual(three.Status.iloc[0], "Insufficient History")
            self.assertEqual(two["Sample warning"].iloc[0], "Short history; unstable estimate")
            self.assertTrue(pd.isna(two["Long rolling correlation"].iloc[0]))

    def test_missing_factor_is_not_short_history(self):
        result, _ = analyze(observations().iloc[:0], pd.DataFrame(columns=["Test"]),
            ["repo_rate", "cpi_headline"], MacroCorrelationConfig(), "2025-01-01")
        statuses = result.set_index("Indicator ID").Status
        self.assertEqual(statuses["repo_rate"], "Manual Import Required")
        self.assertEqual(statuses["cpi_headline"], "Factor Data Unavailable")

    def test_mospi_fiscal_quarters(self):
        rows = [{"year": "2024-25", "quarter": q, "constant_price": v}
                for q, v in [("Q1", 100), ("Q4", 110)]]
        result = parse_mospi(rows, CATALOGUE["pfce_real"])
        self.assertEqual(list(result.index), [pd.Timestamp("2024-06-30"), pd.Timestamp("2025-03-31")])

    def test_native_lag_and_chart_features(self):
        obs = observations(n=20)
        a = analysis_features(obs, "pfce_real", MacroCorrelationConfig(), "2006-01-01")
        b = analysis_features(obs, "pfce_real", MacroCorrelationConfig(lag=1), "2006-01-01")
        self.assertEqual(len(b), len(a)-1)
        np.testing.assert_allclose(b.x, a.x.iloc[:-1])
        self.assertTrue(b.frequency.eq("Q").all())

    def test_same_quarter_alignment(self):
        days = pd.bdate_range("2000-01-01", "2002-01-01")
        prices = pd.DataFrame({"Test": np.arange(len(days))+100.}, index=days)
        features = pd.DataFrame({"period": [pd.Timestamp("2001-06-30")],
            "signal_date": [pd.Timestamp("2001-08-30")], "frequency": ["Q"], "x": [2.], "eligible": [False]})
        result = aligned_returns(features, prices, "Test", MacroCorrelationConfig(relationship="same_period"), days[-1])
        self.assertEqual(result.entry_date.iloc[0], pd.Timestamp("2001-03-30"))
        self.assertEqual(result.exit_date.iloc[0], pd.Timestamp("2001-06-29"))

    def test_scenario_pipeline_and_no_same_day_entry(self):
        obs = observations("repo_rate", 240)
        rng = np.random.default_rng(8)
        obs["value"] = 5+rng.normal(0, .4, len(obs))
        days = pd.bdate_range("1999-01-01", "2020-03-01")
        prices = pd.DataFrame({"Test": 100*np.exp(np.cumsum(rng.normal(.0002, .01, len(days))))}, index=days)
        summary, evaluation = scenario_model(obs, prices, ["repo_rate"], "Test", {"repo_rate": .2},
            MacroCorrelationConfig(years=20, horizon=21), "2020-02-28")
        self.assertTrue((evaluation.entry_date > evaluation.signal_date).all())
        self.assertTrue((evaluation.exit_date <= pd.Timestamp("2020-02-28")).all())
        self.assertTrue(np.isfinite(summary["Expected return %"].iloc[0]))

    def test_pfce_exact_quarter_growth(self):
        f = observations(n=8)
        self.assertAlmostEqual(feature_value(f, CATALOGUE["pfce_real"]), (1.02**4-1)*100)
        self.assertTrue(np.isnan(feature_value(f.drop(index=3), CATALOGUE["pfce_real"])))

    def test_base_break(self):
        f = observations(n=8)
        f.loc[7, "base_year"] = "2022"
        self.assertTrue(np.isnan(feature_value(f, CATALOGUE["pfce_real"])))

    def test_rate_changes_are_percentage_points(self):
        f = observations("repo_rate", n=2)
        f["value"] = [5., 5.25]
        self.assertEqual(feature_value(f, CATALOGUE["repo_rate"]), .25)

    def test_future_revision_cannot_change_prior_feature(self):
        f = observations(n=12)
        cutoff = f.available_at.iloc[7]
        before = feature_history(f, "pfce_real", cutoff, True)
        revision = f.iloc[[0]].copy()
        revision["value"] = 900
        revision["vintage"] = "revision"
        revision["available_at"] = cutoff+pd.Timedelta(days=200)
        after = feature_history(pd.concat([f, revision]), "pfce_real", cutoff, True)
        pd.testing.assert_frame_equal(before, after)

    def test_quarterly_not_upsampled(self):
        f = observations(n=12)
        h = feature_history(f, "pfce_real", "2026-01-01", True)
        self.assertEqual(len(h), 8)

    def test_revised_only_excluded_from_scenario_history(self):
        f = observations()
        f["eligible"] = 0
        self.assertTrue(feature_history(f, "pfce_real", "2026-01-01", True).empty)

    def test_strict_future_entry_and_matured_exit(self):
        days = pd.bdate_range("2000-01-01", periods=150)
        p = pd.DataFrame({"Test": np.arange(150)+100., "Nifty 50": np.arange(150)+200.}, index=days)
        f = pd.DataFrame({"signal_date": [days[70], days[130]], "x": [1, 2], "eligible": [True, True]})
        result = aligned_returns(f, p, "Test", MacroCorrelationConfig(horizon=21), days[-1])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.entry_date.iloc[0], days[71])
        self.assertEqual(result.exit_date.iloc[0], days[92])

    def test_missing_exit_not_stretched(self):
        days = pd.bdate_range("2000-01-01", periods=150)
        p = pd.DataFrame({"Test": np.arange(150)+100.}, index=days)
        p.loc[days[92], "Test"] = np.nan
        f = pd.DataFrame({"signal_date": [days[70]], "x": [1], "eligible": [True]})
        self.assertTrue(aligned_returns(f, p, "Test", MacroCorrelationConfig(horizon=21), days[-1]).empty)

    def test_shrinkage_sign_and_constants(self):
        x = np.random.default_rng(2).normal(size=300)
        self.assertGreater(shrunk_correlation(x, x), .9)
        self.assertLess(shrunk_correlation(x, -x), -.9)
        self.assertTrue(np.isnan(shrunk_correlation(x, np.ones(300))))

    def test_fdr(self):
        result = adjust_fdr([.01, .04, .03, np.nan])
        np.testing.assert_allclose(result[:3], [.03, .04, .04])

    def test_store_idempotent_and_archive(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            s, t = MacroStore(a), MacroStore(b)
            f = observations(n=8)
            s.put(f)
            s.put(f)
            self.assertEqual(len(s.get()), 8)
            t.restore(s.export())
            pd.testing.assert_frame_equal(s.get(), t.get())

    def test_invalid_import_rejected_atomically(self):
        f = observations(n=8)
        f.loc[0, "value"] = np.inf
        with self.assertRaises(ValueError):
            clean_observations(f)

    def test_duplicate_vintage_rejected(self):
        f = observations(n=8)
        with self.assertRaises(ValueError):
            clean_observations(pd.concat([f, f.iloc[[0]]]))


if __name__ == "__main__":
    unittest.main()
