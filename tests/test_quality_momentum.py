from __future__ import annotations

from datetime import date
import unittest

import numpy as np
import pandas as pd

from screener_momentum.config import QualityMomentumConfig
from screener_momentum.quality_momentum import (
    calculate_quality_price_metrics,
    evaluate_quality_fundamentals,
    parse_security_master,
    parse_surveillance_report,
    trendline_momentum,
    verify_latest_trend,
)


class QualityMomentumTests(unittest.TestCase):
    def test_trendline_score_rewards_a_consistent_uptrend(self) -> None:
        smooth = pd.Series(100.0 * np.exp(np.arange(126) * 0.002))
        score, annualized, r_squared = trendline_momentum(smooth, 126)

        self.assertGreater(score, 0)
        self.assertAlmostEqual(score, annualized, places=2)
        self.assertAlmostEqual(r_squared, 1.0, places=4)

    def test_price_pipeline_keeps_only_top_decile_after_filters(self) -> None:
        dates = pd.bdate_range("2025-06-01", periods=320)
        tickers = [f"S{index}.NS" for index in range(10)]
        closes = pd.DataFrame(
            {
                ticker: 100.0 * np.exp(np.arange(320) * (0.0005 + index * 0.00008))
                for index, ticker in enumerate(tickers)
            },
            index=dates,
        )
        volumes = pd.DataFrame(200_000.0, index=dates, columns=tickers)
        universe = pd.DataFrame(
            {
                "Ticker": [ticker.removesuffix(".NS") for ticker in tickers],
                "YFinance Ticker": tickers,
            }
        )
        config = QualityMomentumConfig(max_rsi=100.0, min_average_traded_value_cr=0.1)

        result = calculate_quality_price_metrics(universe, closes, volumes, config)

        self.assertEqual(int(result["Price Filter Pass"].sum()), 10)
        self.assertEqual(int(result["Price Stage Pass"].sum()), 1)
        self.assertEqual(result.loc[result["Price Stage Pass"], "Ticker"].iloc[0], "S9")

    def test_vertical_surge_is_rejected(self) -> None:
        dates = pd.bdate_range("2025-06-01", periods=320)
        values = np.linspace(100.0, 130.0, 320)
        values[-22:] = np.linspace(105.0, 140.0, 22)
        closes = pd.DataFrame({"SURGE.NS": values}, index=dates)
        volumes = pd.DataFrame({"SURGE.NS": 200_000.0}, index=dates)
        universe = pd.DataFrame({"Ticker": ["SURGE"], "YFinance Ticker": ["SURGE.NS"]})
        config = QualityMomentumConfig(max_rsi=100.0, min_average_traded_value_cr=0.1)

        result = calculate_quality_price_metrics(universe, closes, volumes, config)

        self.assertTrue(bool(result.iloc[0]["Vertical Surge"]))
        self.assertFalse(bool(result.iloc[0]["Price Filter Pass"]))
        self.assertIn("vertical surge", result.iloc[0]["Price Rejection Reasons"])

    def test_quality_verification_passes_complete_clean_company(self) -> None:
        row = {
            "Ticker": "CLEAN",
            "Promoter Pledge %": 5.0,
            "Quarterly Revenue YoY Growth %": 2.0,
            "Quarterly Profit YoY Growth %": -10.0,
            "FII Previous Period": "MAR 2026",
            "FII Latest Period": "JUN 2026",
            "FII Previous Holding %": 10.0,
            "FII Latest Holding %": 11.0,
            "FII Holding Change %": 1.0,
            "DII Previous Period": "MAR 2026",
            "DII Latest Period": "JUN 2026",
            "DII Previous Holding %": 8.0,
            "DII Latest Holding %": 8.5,
            "DII Holding Change %": 0.5,
        }
        master = {"ISIN": "INE123A01011", "NSE Name": "Clean Ltd", "NSE Series": "EQ"}
        surveillance = {
            "Surveillance Codes": "",
            "Surveillance Status": "Clear",
            "Surveillance Reject": False,
            "Surveillance Flag": False,
        }

        result = evaluate_quality_fundamentals(
            row,
            master,
            surveillance,
            QualityMomentumConfig(),
            date(2026, 9, 5),
            surveillance_available=True,
        )

        self.assertTrue(result["Quality Pass"])
        self.assertEqual(result["Quality Status"], "Qualified")
        self.assertEqual(result["Combined FII+DII Change %"], 1.5)

    def test_combined_institutional_fall_of_two_points_is_rejected(self) -> None:
        row = {
            "Ticker": "FALL",
            "Promoter Pledge %": 0.0,
            "Quarterly Revenue YoY Growth %": 1.0,
            "Quarterly Profit YoY Growth %": 1.0,
            "FII Previous Period": "MAR 2026",
            "FII Latest Period": "JUN 2026",
            "FII Previous Holding %": 10.0,
            "FII Latest Holding %": 9.0,
            "FII Holding Change %": -1.0,
            "DII Previous Period": "MAR 2026",
            "DII Latest Period": "JUN 2026",
            "DII Previous Holding %": 8.0,
            "DII Latest Holding %": 7.0,
            "DII Holding Change %": -1.0,
        }
        result = evaluate_quality_fundamentals(
            row,
            {"ISIN": "INE123A01011"},
            {"Surveillance Reject": False, "Surveillance Flag": False},
            QualityMomentumConfig(),
            date(2026, 9, 5),
            surveillance_available=True,
        )

        self.assertFalse(result["Quality Pass"])
        self.assertIn("fell 2.00", result["Quality Rejection Reasons"])

    def test_surveillance_parser_rejects_stage_two_and_flags_stage_one(self) -> None:
        frame = pd.DataFrame(
            {"SYMBOL": ["STAGE1", "STAGE2", "CLEAR"], "SurvInd": ["11", "2", "0"]}
        )
        result = parse_surveillance_report(frame).set_index("Ticker")

        self.assertTrue(bool(result.loc["STAGE1", "Surveillance Flag"]))
        self.assertFalse(bool(result.loc["STAGE1", "Surveillance Reject"]))
        self.assertTrue(bool(result.loc["STAGE2", "Surveillance Reject"]))
        self.assertEqual(result.loc["CLEAR", "Surveillance Status"], "Clear")

    def test_surveillance_parser_handles_current_reg1_shape(self) -> None:
        frame = pd.DataFrame(
            {
                "Symbol": ["CLEAR", "ASMONE", "GSMTWO"],
                "GSM": [100, 100, 2],
                "Long_Term_Additional_Surveillance_Measure (Long Term ASM)": [100, 13, 100],
                "Short_Term_Additional_Surveillance_Measure (Short Term ASM)": [100, 100, 100],
                "ESM": [100, 100, 100],
            }
        )
        result = parse_surveillance_report(frame).set_index("Ticker")

        self.assertEqual(result.loc["CLEAR", "Surveillance Status"], "Clear")
        self.assertTrue(bool(result.loc["ASMONE", "Surveillance Flag"]))
        self.assertFalse(bool(result.loc["ASMONE", "Surveillance Reject"]))
        self.assertTrue(bool(result.loc["GSMTWO", "Surveillance Reject"]))

    def test_security_master_prefers_eq_and_validates_isin(self) -> None:
        frame = pd.DataFrame(
            {
                "TckrSymb": ["ABC", "ABC", "BAD"],
                "SctySrs": ["BE", "EQ", "EQ"],
                "FinInstrmNm": ["ABC BE", "ABC EQ", "Bad"],
                "ISIN": ["INE123A01011", "INE123A01011", "invalid"],
            }
        )
        result = parse_security_master(frame)

        self.assertEqual(result["Ticker"].tolist(), ["ABC"])
        self.assertEqual(result.iloc[0]["NSE Series"], "EQ")

    def test_latest_trend_refreshes_both_moving_averages(self) -> None:
        dates = pd.bdate_range(end="2026-09-04", periods=252)
        latest = pd.DataFrame({"ABC.NS": np.linspace(100.0, 200.0, len(dates))}, index=dates)
        quality = pd.DataFrame(
            {
                "Ticker": ["ABC"],
                "YFinance Ticker": ["ABC.NS"],
                "CMP Rs.": [190.0],
                "Price Date": ["2026-09-03"],
                "SMA50": [250.0],
                "SMA200": [220.0],
                "Quality Pass": [True],
                "Trendline Momentum Score": [10.0],
                "Quality Rejection Reasons": ["passed"],
            }
        )

        result = verify_latest_trend(
            quality, latest, QualityMomentumConfig(), scan_date=date(2026, 9, 5)
        )

        self.assertTrue(bool(result.iloc[0]["Trend Confirmation Pass"]))
        self.assertLess(result.iloc[0]["Verification SMA50"], result.iloc[0]["Verification Price"])
        self.assertLess(result.iloc[0]["Verification SMA200"], result.iloc[0]["Verification SMA50"])


if __name__ == "__main__":
    unittest.main()
