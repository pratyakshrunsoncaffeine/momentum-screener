import unittest

import pandas as pd
from bs4 import BeautifulSoup

from screener_momentum.fundamentals import extract_quarterly_results
from screener_momentum.pipeline import (
    prepare_quarterly_results,
    score_quarterly_stock_return_momentum,
)


class QuarterlyResultsTests(unittest.TestCase):
    def test_extracts_qoq_and_yoy_for_all_metrics(self) -> None:
        html = """
        <section id="quarters"><table>
          <thead><tr><th></th><th>Jun 2025</th><th>Mar 2026</th><th>Jun 2026</th><th>TTM</th></tr></thead>
          <tbody>
            <tr><td>Sales +</td><td>100</td><td>120</td><td>150</td><td>500</td></tr>
            <tr><td>Operating Profit +</td><td>20</td><td>30</td><td>45</td><td>130</td></tr>
            <tr><td>Net Profit +</td><td>10</td><td>15</td><td>30</td><td>80</td></tr>
            <tr><td>EPS in Rs</td><td>1</td><td>1.5</td><td>3</td><td>8</td></tr>
          </tbody>
        </table></section>
        """
        result = extract_quarterly_results(BeautifulSoup(html, "lxml"), "JUN 2026")

        self.assertTrue(result["Target Quarter Found"])
        self.assertEqual(result["Previous Quarter"], "MAR 2026")
        self.assertEqual(result["Year Ago Quarter"], "JUN 2025")
        self.assertEqual(result["Sales QoQ Growth %"], 25.0)
        self.assertEqual(result["Sales YoY Growth %"], 50.0)
        self.assertEqual(result["Operating Profit YoY Growth %"], 125.0)
        self.assertEqual(result["Net Profit QoQ Growth %"], 100.0)
        self.assertEqual(result["EPS YoY Growth %"], 200.0)

    def test_ranking_switches_to_the_selected_metric(self) -> None:
        frame = pd.DataFrame(
            [
                {"Ticker": "LOW", "Target Quarter": "JUN 2026", "Target Quarter Found": True, "Sales YoY Growth %": 20, "Operating Profit YoY Growth %": 60},
                {"Ticker": "HIGH", "Target Quarter": "JUN 2026", "Target Quarter Found": True, "Sales YoY Growth %": 50, "Operating Profit YoY Growth %": 10},
                {"Ticker": "MISS", "Target Quarter": "JUN 2026", "Target Quarter Found": False, "Sales YoY Growth %": 99, "Operating Profit YoY Growth %": 99},
            ]
        )

        sales_order = prepare_quarterly_results(frame, "Jun 2026", "Sales")["Ticker"].tolist()
        operating_profit_order = prepare_quarterly_results(frame, "Jun 2026", "Operating Profit")["Ticker"].tolist()

        self.assertEqual(sales_order, ["HIGH", "LOW"])
        self.assertEqual(operating_profit_order, ["LOW", "HIGH"])

    def test_stock_return_momentum_prioritizes_ten_day_return(self) -> None:
        returns = pd.DataFrame(
            [
                {"Ticker": "TEN_DAY", "Earnings 2D Return": 1, "Earnings 5D Return": 1, "Earnings 10D Return": 12},
                {"Ticker": "SHORT_TERM", "Earnings 2D Return": 5, "Earnings 5D Return": 5, "Earnings 10D Return": 4},
            ]
        )

        ranked = score_quarterly_stock_return_momentum(returns)

        self.assertEqual(ranked["Ticker"].tolist(), ["TEN_DAY", "SHORT_TERM"])
        self.assertEqual(ranked["Post-Earnings Stock Return Momentum Rank"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
