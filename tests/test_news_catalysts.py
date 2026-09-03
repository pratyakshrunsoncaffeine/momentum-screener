from __future__ import annotations

from datetime import date, datetime
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
import pandas as pd

from screener_momentum.news_config import IST, NewsCatalystConfig, after_news_cutoff, index_model_eligibility
from screener_momentum.news_features import (
    assign_signal_dates,
    build_daily_news_features,
    build_forward_labels,
    inject_training_noise,
    map_articles_to_indices,
)
from screener_momentum.news_model import chronological_forward_split, predict_news_catalysts, train_news_models
from screener_momentum.news_market import calculate_constituent_daily_activity
from screener_momentum.news_pipeline import _challenger_can_replace
from screener_momentum.news_sources import (
    AuthorizedPulseProvider,
    BigQuerySandboxBudgetExceeded,
    GdeltBigQueryProvider,
    _query_term_groups,
    deduplicate_articles,
    normalize_news_frame,
)
from screener_momentum.news_store import (
    LocalNewsResultStore,
    SupabaseNewsResultStore,
    _database_records,
    _record_batches,
)


class NewsCatalystPointInTimeTests(unittest.TestCase):
    def test_cutoff_is_430_pm_ist(self) -> None:
        self.assertFalse(after_news_cutoff(datetime(2026, 8, 31, 16, 29, tzinfo=IST)))
        self.assertTrue(after_news_cutoff(datetime(2026, 8, 31, 16, 30, tzinfo=IST)))

    def test_after_close_and_weekend_news_move_to_next_trading_signal(self) -> None:
        trading_dates = pd.to_datetime(["2026-08-31", "2026-09-01", "2026-09-02"])
        articles = pd.DataFrame(
            {
                "Published At UTC": pd.to_datetime(
                    [
                        "2026-08-31 09:00:00+00:00",
                        "2026-08-31 11:01:00+00:00",
                        "2026-08-29 06:00:00+00:00",
                        None,
                    ],
                    utc=True,
                )
            }
        )

        assigned = assign_signal_dates(articles, trading_dates)

        self.assertEqual(pd.Timestamp(assigned.iloc[0]["Signal Date"]), pd.Timestamp("2026-08-31"))
        self.assertEqual(pd.Timestamp(assigned.iloc[1]["Signal Date"]), pd.Timestamp("2026-09-01"))
        self.assertEqual(pd.Timestamp(assigned.iloc[2]["Signal Date"]), pd.Timestamp("2026-08-31"))
        self.assertTrue(pd.isna(assigned.iloc[3]["Signal Date"]))

    def test_constituent_mapping_uses_membership_valid_on_publication_date(self) -> None:
        article = normalize_news_frame(
            pd.DataFrame(
                [
                    {
                        "Published At UTC": "2026-07-10T10:00:00Z",
                        "Publisher": "Permitted Feed",
                        "Title": "Alpha Industries announces a major order",
                        "URL": "https://example.com/alpha-order",
                    }
                ]
            )
        )
        catalogue = pd.DataFrame(
            {"Category": ["Sectoral", "Sectoral"], "Index": ["Nifty Auto", "Nifty IT"]}
        )
        constituents = pd.DataFrame(
            [
                {
                    "Index": "Nifty Auto",
                    "Ticker": "ALPHA",
                    "Company": "Alpha Industries",
                    "Valid From": "2026-01-01",
                    "Valid To": "2026-12-31",
                },
                {
                    "Index": "Nifty IT",
                    "Ticker": "ALPHA",
                    "Company": "Alpha Industries",
                    "Valid From": "2027-01-01",
                    "Valid To": None,
                },
            ]
        )

        links = map_articles_to_indices(article, catalogue, constituents)

        self.assertIn("Nifty Auto", links["Index"].tolist())
        self.assertNotIn("Nifty IT", links["Index"].tolist())

    def test_semantic_similarity_can_attribute_without_keyword_match(self) -> None:
        article = normalize_news_frame(
            pd.DataFrame(
                [{"Published At UTC": "2026-07-10T10:00:00Z", "Title": "Unrelated wording", "URL": "https://x.test/1"}]
            )
        )
        article["Embedding"] = [np.array([1.0, 0.0, 0.0])]
        catalogue = pd.DataFrame(
            {"Category": ["Thematic"], "Index": ["Nifty New Theme"], "Embedding": [np.array([1.0, 0.0, 0.0])]}
        )

        links = map_articles_to_indices(article, catalogue)

        self.assertEqual(links.iloc[0]["Index"], "Nifty New Theme")
        self.assertIn("semantic", links.iloc[0]["Attribution Reason"])

    def test_special_indices_are_explicitly_unsupported(self) -> None:
        self.assertEqual(index_model_eligibility("India VIX"), (False, "Volatility index"))
        self.assertFalse(index_model_eligibility("Nifty 50 2x Leverage")[0])
        self.assertTrue(index_model_eligibility("Nifty Auto")[0])


class NewsCatalystFeatureTests(unittest.TestCase):
    def test_forward_labels_begin_after_feature_date_and_unmatured_labels_are_null(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=12)
        features = pd.concat(
            [
                pd.DataFrame({"Date": dates, "Index": "Nifty 50", "Close": np.arange(100.0, 112.0)}),
                pd.DataFrame({"Date": dates, "Index": "Nifty IT", "Close": np.arange(100.0, 124.0, 2.0)}),
            ],
            ignore_index=True,
        )

        labelled = build_forward_labels(features, horizons={"5D": 5})
        first_it = labelled[labelled["Index"].eq("Nifty IT")].iloc[0]
        last_it = labelled[labelled["Index"].eq("Nifty IT")].iloc[-1]

        expected = (110.0 / 100.0 - 1.0) * 100.0
        self.assertAlmostEqual(first_it["Absolute Return 5D %"], expected)
        self.assertTrue(pd.isna(last_it["Excess Return 5D %"]))
        self.assertTrue(pd.isna(last_it["Positive Excess 5D"]))

    def test_sentiment_is_shrunk_toward_cross_index_daily_prior(self) -> None:
        dates = pd.bdate_range("2026-06-01", periods=30)
        prices = pd.concat(
            [
                pd.DataFrame({"Date": dates, "Index": "Nifty 50", "Close": 100.0}),
                pd.DataFrame({"Date": dates, "Index": "Nifty Auto", "Close": 100.0}),
            ],
            ignore_index=True,
        )
        articles = normalize_news_frame(
            pd.DataFrame(
                [
                    {"Published At UTC": "2026-06-30T09:00:00Z", "Title": "Auto demand jumps", "URL": "https://x.test/a"},
                    {"Published At UTC": "2026-06-30T09:01:00Z", "Title": "India market weak", "URL": "https://x.test/b"},
                ]
            )
        )
        articles["Sentiment Score"] = [1.0, -1.0]
        articles["Positive Probability"] = [1.0, 0.0]
        articles["Negative Probability"] = [0.0, 1.0]
        articles["Neutral Probability"] = [0.0, 0.0]
        links = pd.DataFrame(
            [
                {"Article ID": articles.iloc[0]["Article ID"], "Index": "Nifty Auto", "Relevance": 1.0},
                {"Article ID": articles.iloc[1]["Article ID"], "Index": "Nifty 50", "Relevance": 1.0},
            ]
        )

        feature_frame = build_daily_news_features(articles, links, prices)
        auto = feature_frame[
            feature_frame["Index"].eq("Nifty Auto") & feature_frame["news_article_count"].gt(0)
        ].iloc[0]

        self.assertLess(auto["news_sentiment_shrunk"], auto["news_sentiment_raw"])

    def test_training_noise_never_changes_identity_dates_or_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "Date": pd.bdate_range("2026-01-01", periods=20),
                "Index": ["Nifty IT"] * 20,
                "news_sentiment_shrunk": np.linspace(-1, 1, 20),
                "price_return_5d": np.linspace(-2, 2, 20),
                "Excess Return 5D %": np.linspace(-3, 3, 20),
                "Positive Excess 5D": [0] * 10 + [1] * 10,
            }
        )

        augmented = inject_training_noise(frame, ["news_sentiment_shrunk", "price_return_5d"])

        pd.testing.assert_series_equal(augmented["Date"], frame["Date"])
        pd.testing.assert_series_equal(augmented["Index"], frame["Index"])
        pd.testing.assert_series_equal(augmented["Excess Return 5D %"], frame["Excess Return 5D %"])
        pd.testing.assert_series_equal(augmented["Positive Excess 5D"], frame["Positive Excess 5D"])

    def test_chronological_split_has_sixty_three_trading_date_embargoes(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=500)
        frame = pd.DataFrame({"Date": dates, "Index": "Nifty IT", "feature": np.arange(len(dates))})

        splits = chronological_forward_split(frame, embargo_days=63)

        train_end = splits["Train"]["Date"].max()
        validation_start = splits["Validation"]["Date"].min()
        validation_end = splits["Validation"]["Date"].max()
        test_start = splits["Test"]["Date"].min()
        self.assertGreater(len(dates[(dates > train_end) & (dates < validation_start)]), 62)
        self.assertGreater(len(dates[(dates > validation_end) & (dates < test_start)]), 62)

    def test_pooled_ridge_lightgbm_path_trains_and_predicts(self) -> None:
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2021-01-01", periods=500)
        rows = []
        for index_position, index_name in enumerate(("Nifty Auto", "Nifty Bank", "Nifty IT", "Nifty Metal")):
            sentiment = rng.normal(0, 1, len(dates))
            price = rng.normal(0, 1, len(dates))
            excess = 0.8 * sentiment + 0.2 * price + rng.normal(0, 0.5, len(dates))
            for position, day in enumerate(dates):
                rows.append(
                    {
                        "Date": day,
                        "Index": index_name,
                        "Close": 100 + index_position + position * 0.02,
                        "news_sentiment_shrunk": sentiment[position],
                        "price_return_5d": price[position],
                        "Absolute Return 5D %": excess[position] + 0.1,
                        "Excess Return 5D %": excess[position],
                        "Positive Excess 5D": int(excess[position] > 0),
                    }
                )
        training = pd.DataFrame(rows)
        settings = NewsCatalystConfig(horizons={"5D": 5})

        with tempfile.TemporaryDirectory() as directory:
            result = train_news_models(
                training,
                directory,
                config=settings,
                feature_columns=["news_sentiment_shrunk", "price_return_5d"],
            )
            predictions = predict_news_catalysts(result["bundle"], training.groupby("Index").tail(1))

        self.assertEqual(set(predictions["Horizon"]), {"5D"})
        self.assertEqual(len(predictions), 4)
        self.assertTrue(result["artifact"].name.endswith(".joblib"))
        self.assertIn(result["bundle"].status, {"Experimental", "Validated"})

    def test_challenger_promotion_requires_quality_and_drawdown_guard(self) -> None:
        champion = pd.DataFrame(
            {
                "horizon": ["5D", "1M", "3M"],
                "rank_ic": [0.03, 0.04, 0.02],
                "price_only_rank_ic": [0.01, 0.01, 0.01],
                "maximum_drawdown_pct": [-8.0, -10.0, -12.0],
            }
        )
        challenger = pd.DataFrame(
            {
                "Horizon": ["5D", "1M", "3M"],
                "Rank IC": [0.06, 0.07, 0.01],
                "Price-Only Rank IC": [0.02, 0.03, 0.02],
                "Maximum Drawdown %": [-9.0, -11.0, -12.0],
            }
        )

        self.assertTrue(_challenger_can_replace(challenger, champion, "Validated"))
        self.assertFalse(_challenger_can_replace(challenger, champion, "Experimental"))
        challenger.loc[0, "Maximum Drawdown %"] = -20.0
        self.assertFalse(_challenger_can_replace(challenger, champion, "Validated"))


class NewsCatalystStorageTests(unittest.TestCase):
    def test_sandbox_planner_samples_large_partitions_below_query_cap(self) -> None:
        class PredictableProvider(GdeltBigQueryProvider):
            def estimate_bytes(self, start, end, query, sample_percent=100.0):
                del start, end, query
                return int(20 * 1024**3 * sample_percent / 100.0)

        provider = PredictableProvider(maximum_query_bytes=5 * 1024**3)
        sample, estimate = provider.plan_sample_percent(
            datetime(2026, 1, 1, tzinfo=IST),
            datetime(2026, 1, 2, tzinfo=IST),
            "india market",
        )

        self.assertLess(sample, 100.0)
        self.assertLessEqual(estimate, 5 * 1024**3)

    def test_sandbox_planner_stops_when_remaining_budget_is_too_small(self) -> None:
        class PredictableProvider(GdeltBigQueryProvider):
            def estimate_bytes(self, start, end, query, sample_percent=100.0):
                del start, end, query
                return int(20 * 1024**3 * sample_percent / 100.0)

        provider = PredictableProvider(
            maximum_query_bytes=5 * 1024**3,
            minimum_sample_percent=1.0,
        )
        with self.assertRaises(BigQuerySandboxBudgetExceeded):
            provider.plan_sample_percent(
                datetime(2026, 1, 1, tzinfo=IST),
                datetime(2026, 1, 2, tzinfo=IST),
                "india market",
                remaining_budget_bytes=50 * 1024**2,
            )

    def test_supabase_select_returns_remote_rows(self) -> None:
        response = Mock()
        response.json.return_value = [{"status": "completed"}]
        response.raise_for_status.return_value = None
        store = SupabaseNewsResultStore("https://example.supabase.co", "service-key")

        with patch("screener_momentum.news_store.requests.get", return_value=response):
            rows = store.select("news_pipeline_jobs")

        self.assertEqual(rows, [{"status": "completed"}])

    def test_supabase_records_serialize_python_dates(self) -> None:
        records = _database_records(
            pd.DataFrame(
                [
                    {
                        "Partition Date": date(2026, 9, 3),
                        "Processed At UTC": pd.Timestamp("2026-09-03T12:00:00Z"),
                    }
                ]
            )
        )

        self.assertEqual(records[0]["partition_date"], "2026-09-03")
        self.assertEqual(records[0]["processed_at_utc"], "2026-09-03T12:00:00+00:00")

    def test_supabase_upsert_batches_large_frames(self) -> None:
        response = Mock(status_code=201, text="")
        response.raise_for_status.return_value = None
        store = SupabaseNewsResultStore(
            "https://example.supabase.co",
            "service-key",
            upsert_batch_rows=2,
            upsert_batch_bytes=100_000,
        )
        frame = pd.DataFrame([{"Article ID": str(value)} for value in range(5)])

        with patch("screener_momentum.news_store.requests.post", return_value=response) as post:
            store.upsert("news_articles", frame, "article_id")

        self.assertEqual(post.call_count, 3)
        self.assertEqual([len(call.kwargs["json"]) for call in post.call_args_list], [2, 2, 1])

    def test_supabase_record_batches_obey_payload_limit(self) -> None:
        records = [{"article_id": str(value), "title": "x" * 40} for value in range(3)]

        batches = _record_batches(records, maximum_rows=10, maximum_bytes=100)

        self.assertEqual([len(batch) for batch in batches], [1, 1, 1])

    def test_deduplication_uses_canonical_urls(self) -> None:
        frame = pd.DataFrame(
            [
                {"Published At UTC": "2026-08-01T09:00:00Z", "Title": "Same", "URL": "https://EXAMPLE.com/a?utm_source=x"},
                {"Published At UTC": "2026-08-01T09:00:00Z", "Title": "Same", "URL": "https://example.com/a"},
            ]
        )
        self.assertEqual(len(deduplicate_articles(frame)), 1)

    def test_gdelt_query_terms_do_not_include_boolean_operators(self) -> None:
        query = '(india OR nifty OR "reserve bank of india") AND (oil OR inflation)'
        groups = _query_term_groups(query)

        self.assertEqual(groups[0], ["reserve bank of india", "india", "nifty"])
        self.assertEqual(groups[1], ["oil", "inflation"])

    def test_pulse_provider_refuses_unauthorized_scraping(self) -> None:
        with self.assertRaises(PermissionError):
            AuthorizedPulseProvider().fetch(
                datetime(2026, 1, 1, tzinfo=IST), datetime(2026, 1, 2, tzinfo=IST), "market"
            )

    def test_local_saved_run_round_trips_identically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalNewsResultStore(Path(directory))
            expected = pd.DataFrame([{"Index": "Nifty IT", "Expected Excess Return %": 1.25}])
            store.save_dashboard({"predictions": expected})
            actual = store.load_dashboard()["predictions"]
        pd.testing.assert_frame_equal(actual, expected)

    def test_constituent_activity_respects_effective_membership_dates(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=12)
        prices = pd.DataFrame(
            {
                "Date": list(dates) * 2,
                "Ticker": ["ALPHA"] * len(dates) + ["BETA"] * len(dates),
                "Close": list(np.linspace(100, 112, len(dates))) + list(np.linspace(100, 94, len(dates))),
                "Volume": [1000 + position * 50 for position in range(len(dates))] * 2,
            }
        )
        constituents = pd.DataFrame(
            [
                {"Index": "Nifty Auto", "Ticker": "ALPHA", "Valid From": dates[0], "Valid To": dates[5]},
                {"Index": "Nifty IT", "Ticker": "ALPHA", "Valid From": dates[6], "Valid To": None},
                {"Index": "Nifty Auto", "Ticker": "BETA", "Valid From": dates[0], "Valid To": None},
            ]
        )

        activity = calculate_constituent_daily_activity(prices, constituents)

        auto_after_change = activity[
            activity["Index"].eq("Nifty Auto") & activity["Date"].ge(dates[6])
        ]
        it_before_change = activity[
            activity["Index"].eq("Nifty IT") & activity["Date"].lt(dates[6])
        ]
        self.assertTrue(auto_after_change["constituent_count"].eq(1).all())
        self.assertTrue(it_before_change.empty)


if __name__ == "__main__":
    unittest.main()
