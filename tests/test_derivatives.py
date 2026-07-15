import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from screener_momentum.config import DerivativesSignalConfig
from screener_momentum.derivatives import (
    DailyDerivativesData,
    NseEodDerivativesProvider,
    futures_oi_classification,
    normalize_cash_bhavcopy,
    normalize_fo_bhavcopy,
    scan_derivatives_momentum,
)
from screener_momentum.derivatives_backtest import derivatives_backtest


class DerivativesScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trade_date = date(2026, 7, 14)
        self.universe = pd.DataFrame(
            [
                {"Ticker": "TEST", "Name": "Test Ltd", "YFinance Ticker": "TEST.NS"},
                {"Ticker": "NOFO", "Name": "No Derivatives", "YFinance Ticker": "NOFO.NS"},
            ]
        )

    def test_normalizes_current_nse_report_columns(self) -> None:
        cash_source = pd.DataFrame(
            [{"SYMBOL": " TEST ", " SERIES": " EQ", " DATE1": " 14-Jul-2026", " PREV_CLOSE": 100, " OPEN_PRICE": 101, " HIGH_PRICE": 103, " LOW_PRICE": 99, " CLOSE_PRICE": 102, " TTL_TRD_QNTY": 1000, " TURNOVER_LACS": 10}]
        )
        fo_source = pd.DataFrame(
            [{"TradDt": "2026-07-14", "FinInstrmTp": "STO", "FinInstrmId": 11, "TckrSymb": "TEST", "XpryDt": "2026-07-28", "StrkPric": 100, "OptnTp": "CE", "FinInstrmNm": "TEST26JUL100CE", "OpnPric": 10, "HghPric": 12, "LwPric": 9, "ClsPric": 10.8, "PrvsClsgPric": 10, "UndrlygPric": 102, "OpnIntrst": 10000, "ChngInOpnIntrst": 1000, "TtlTradgVol": 100, "TtlTrfVal": 100000, "NewBrdLotQty": 100}]
        )

        cash = normalize_cash_bhavcopy(cash_source)
        fo = normalize_fo_bhavcopy(fo_source)

        self.assertEqual(cash.iloc[0]["Ticker"], "TEST")
        self.assertEqual(fo.iloc[0]["Instrument"], "OPTSTK")
        self.assertEqual(fo.iloc[0]["Contract"], "TEST26JUL100CE")

    def test_exact_eight_percent_call_and_two_percent_stock_pass(self) -> None:
        daily = self._daily(call_previous=10, call_close=10.8, stock_previous=100, stock_close=102)
        result = scan_derivatives_momentum(self.universe, daily, DerivativesSignalConfig())

        self.assertEqual(result["signals"]["Ticker"].tolist(), ["TEST"])
        self.assertEqual(result["signals"].iloc[0]["Call Return %"], 8.0)
        no_fo = result["rejections"][result["rejections"]["Ticker"].eq("NOFO")]
        self.assertEqual(no_fo.iloc[0]["Rejection Reasons"], "No listed derivatives")

    def test_expiry_rolls_past_contract_with_fewer_than_seven_days(self) -> None:
        daily = self._daily(call_previous=10, call_close=11, stock_previous=100, stock_close=103)
        too_near = daily.derivatives.iloc[0].copy()
        too_near["Expiry"] = self.trade_date + timedelta(days=3)
        too_near["Contract"] = "TEST_NEAR"
        too_near["Strike"] = 103
        daily.derivatives = pd.concat([daily.derivatives, pd.DataFrame([too_near])], ignore_index=True)

        result = scan_derivatives_momentum(self.universe, daily, DerivativesSignalConfig())

        self.assertEqual(result["features"].iloc[0]["Call Contract"], "TEST_CALL")

    def test_atm_selection_wins_over_a_more_active_far_strike(self) -> None:
        daily = self._daily(call_previous=10, call_close=12, stock_previous=100, stock_close=103)
        far_call = daily.derivatives.iloc[0].copy()
        far_call["Instrument ID"] = 99
        far_call["Contract"] = "TEST_FAR_CALL"
        far_call["Strike"] = 130
        far_call["Volume Contracts"] = 10000
        daily.derivatives = pd.concat([daily.derivatives, pd.DataFrame([far_call])], ignore_index=True)

        result = scan_derivatives_momentum(self.universe, daily, DerivativesSignalConfig())

        self.assertEqual(result["features"].iloc[0]["Call Contract"], "TEST_CALL")

    def test_zero_volume_and_open_interest_are_rejected(self) -> None:
        daily = self._daily(call_previous=10, call_close=12, stock_previous=100, stock_close=103)
        daily.derivatives.loc[daily.derivatives["Instrument"].eq("OPTSTK"), "Volume Contracts"] = 0
        daily.derivatives.loc[daily.derivatives["Instrument"].eq("OPTSTK"), "OI Units"] = 0

        result = scan_derivatives_momentum(self.universe, daily, DerivativesSignalConfig())

        reasons = result["features"].iloc[0]["Rejection Reasons"]
        self.assertIn("Call volume below", reasons)
        self.assertIn("Call OI below", reasons)

    def test_exact_contract_previous_close_fallback(self) -> None:
        daily = self._daily(call_previous=10, call_close=11, stock_previous=100, stock_close=103)
        daily.derivatives.loc[daily.derivatives["Instrument"].eq("OPTSTK"), "Previous Close"] = pd.NA
        previous = daily.derivatives.copy()
        previous.loc[previous["Contract"].eq("TEST_CALL"), "Close"] = 10

        result = scan_derivatives_momentum(
            self.universe,
            daily,
            DerivativesSignalConfig(),
            previous_derivatives=previous,
        )

        self.assertEqual(result["signals"].iloc[0]["Call Return %"], 10.0)

    def test_corporate_adjustment_is_rejected(self) -> None:
        daily = self._daily(call_previous=10, call_close=12, stock_previous=100, stock_close=103)
        daily.contracts.loc[0, "Corporate Adjustment"] = "Y"

        result = scan_derivatives_momentum(self.universe, daily, DerivativesSignalConfig())

        self.assertTrue(result["signals"].empty)
        self.assertIn("corporate-action", result["features"].iloc[0]["Rejection Reasons"])

    def test_futures_activity_classification(self) -> None:
        self.assertEqual(futures_oi_classification(2, 100), "Long build-up")
        self.assertEqual(futures_oi_classification(2, -100), "Short covering")
        self.assertEqual(futures_oi_classification(-2, 100), "Short build-up")
        self.assertEqual(futures_oi_classification(-2, -100), "Long unwinding")

    def test_call_threshold_cannot_be_configured_below_eight_percent(self) -> None:
        with self.assertRaises(ValueError):
            scan_derivatives_momentum(
                self.universe,
                self._daily(call_previous=10, call_close=11, stock_previous=100, stock_close=103),
                replace(DerivativesSignalConfig(), min_call_return_pct=7.99),
            )

    def _daily(self, call_previous: float, call_close: float, stock_previous: float, stock_close: float) -> DailyDerivativesData:
        expiry = self.trade_date + timedelta(days=14)
        cash = pd.DataFrame(
            [{"Ticker": "TEST", "Trade Date": self.trade_date, "Previous Close": stock_previous, "Open": stock_previous, "High": stock_close, "Low": stock_previous, "Close": stock_close, "Volume": 10000, "Turnover": 100}]
        )
        derivatives = pd.DataFrame(
            [
                {"Trade Date": self.trade_date, "Instrument": "OPTSTK", "Instrument ID": 11, "Ticker": "TEST", "Expiry": expiry, "Strike": stock_close, "Option Type": "CE", "Contract": "TEST_CALL", "Open": call_previous, "High": call_close, "Low": call_previous, "Close": call_close, "Previous Close": call_previous, "Underlying Price": stock_close, "OI Units": 10000, "OI Change Units": 1000, "Volume Contracts": 100, "Turnover": 100000, "Lot Size": 100},
                {"Trade Date": self.trade_date, "Instrument": "FUTSTK", "Instrument ID": 12, "Ticker": "TEST", "Expiry": expiry, "Strike": 0, "Option Type": "", "Contract": "TEST_FUT", "Open": stock_previous, "High": stock_close, "Low": stock_previous, "Close": stock_close, "Previous Close": stock_previous, "Underlying Price": stock_close, "OI Units": 10000, "OI Change Units": 1000, "Volume Contracts": 100, "Turnover": 100000, "Lot Size": 100},
            ]
        )
        contracts = pd.DataFrame(
            [
                {"Instrument ID": 11, "Ticker": "TEST", "Contract": "TEST_CALL", "Corporate Adjustment": "N"},
                {"Instrument ID": 12, "Ticker": "TEST", "Contract": "TEST_FUT", "Corporate Adjustment": "N"},
            ]
        )
        return DailyDerivativesData(self.trade_date, cash, derivatives, contracts)


class DerivativesBacktestTests(unittest.TestCase):
    def test_cached_download_range_recovers_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = NseEodDerivativesProvider(temp_dir)
            trade_date = date(2026, 7, 14)
            target = Path(temp_dir) / "normalized" / trade_date.strftime("%Y%m%d")
            target.mkdir(parents=True)
            pd.DataFrame([{"Ticker": "TEST", "Close": 100}]).to_csv(target / "cash.csv", index=False)
            pd.DataFrame([{"Ticker": "TEST", "Close": 10}]).to_csv(target / "fo.csv", index=False)
            pd.DataFrame([{"Instrument ID": 1}]).to_csv(target / "contracts.csv", index=False)

            health = provider.download_range(trade_date, trade_date)

            self.assertEqual(health.iloc[0]["Status"], "cached")
            self.assertEqual(provider.available_dates(), [trade_date])

    def test_backtest_enters_after_signal_date_and_recovers_cached_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = NseEodDerivativesProvider(temp_dir)
            universe = pd.DataFrame([{"Ticker": "TEST", "YFinance Ticker": "TEST.NS"}])
            start = date(2026, 1, 2)
            price = 100.0
            for index, timestamp in enumerate(pd.bdate_range(start, periods=15)):
                trade_date = timestamp.date()
                previous = price
                price = previous * 1.03
                expiry = trade_date + timedelta(days=21)
                cash = pd.DataFrame([{"Ticker": "TEST", "Trade Date": trade_date, "Previous Close": previous, "Open": previous * 1.001, "High": price, "Low": previous, "Close": price, "Volume": 10000, "Turnover": 100}])
                option_previous = 10 + index
                fo = pd.DataFrame([
                    {"Trade Date": trade_date, "Instrument": "OPTSTK", "Instrument ID": 11, "Ticker": "TEST", "Expiry": expiry, "Strike": round(price), "Option Type": "CE", "Contract": f"TEST_CALL_{index}", "Open": option_previous, "High": option_previous * 1.10, "Low": option_previous, "Close": option_previous * 1.10, "Previous Close": option_previous, "Underlying Price": price, "OI Units": 10000, "OI Change Units": 1000, "Volume Contracts": 150, "Turnover": 100000, "Lot Size": 100},
                    {"Trade Date": trade_date, "Instrument": "FUTSTK", "Instrument ID": 12, "Ticker": "TEST", "Expiry": expiry, "Strike": 0, "Option Type": "", "Contract": f"TEST_FUT_{index}", "Open": previous, "High": price, "Low": previous, "Close": price, "Previous Close": previous, "Underlying Price": price, "OI Units": 10000, "OI Change Units": 1000, "Volume Contracts": 150, "Turnover": 100000, "Lot Size": 100},
                ])
                contracts = pd.DataFrame([{"Instrument ID": 11, "Ticker": "TEST", "Contract": f"TEST_CALL_{index}", "Corporate Adjustment": "N"}])
                target = Path(temp_dir) / "normalized" / trade_date.strftime("%Y%m%d")
                target.mkdir(parents=True)
                cash.to_csv(target / "cash.csv", index=False)
                fo.to_csv(target / "fo.csv", index=False)
                contracts.to_csv(target / "contracts.csv", index=False)

            result = derivatives_backtest(
                provider,
                universe,
                DerivativesSignalConfig(),
                start,
                (pd.Timestamp(start) + pd.offsets.BDay(14)).date(),
                holding_days=3,
                top_n=1,
            )

            self.assertFalse(result.events.empty)
            self.assertTrue((result.events["Entry Date"] > result.events["Signal Date"]).all())
            self.assertTrue((result.events["Exit Date"] >= result.events["Entry Date"]).all())
            full_curve = result.curve[result.curve["Variant"].eq("Full Derivatives")]
            self.assertGreater(full_curve.iloc[-1]["Portfolio Value"], full_curve.iloc[0]["Portfolio Value"])


if __name__ == "__main__":
    unittest.main()
