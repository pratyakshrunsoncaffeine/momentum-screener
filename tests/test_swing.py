from __future__ import annotations

from datetime import date
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from screener_momentum.swing import (
    SwingConfig, apply_events, build_watchlist, clean_daily, confirmed_structure, download_universe, indicators,
    load_swing_universe, select_picks, session_vwap, snapshot, wilder,
)


def history(size=230):
    index = pd.bdate_range('2025-01-01', periods=size)
    close = np.arange(size) * 0.2 + 100 + np.sin(np.arange(size))
    return pd.DataFrame({'Open': close - .2, 'High': close + 1.0, 'Low': close - 1.0,
                         'Close': close, 'Volume': 1_000_000.0}, index=index)


class SwingTests(unittest.TestCase):
    def test_bse_codes_and_explicit_suffixes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tickers.csv'
            path.write_text('Ticker\n543225\nRELIANCE.NS\n500325.BO\n', encoding='utf-8')
            result = load_swing_universe(path)
        self.assertEqual(result['YFinance Ticker'].tolist(), ['543225.BO','RELIANCE.NS','500325.BO'])

    def test_wilder_seed_and_update(self):
        actual = wilder(pd.Series([np.nan, 1., 2., 3., 6.]), 3)
        self.assertAlmostEqual(actual.iloc[3], 2.)
        self.assertAlmostEqual(actual.iloc[4], 10 / 3)

    def test_volume_reference_excludes_signal_day_and_no_future_leakage(self):
        frame = history()
        frame.iloc[200, frame.columns.get_loc('Volume')] = 2_000_000
        full = indicators(frame)
        truncated = indicators(frame.iloc[:201])
        self.assertAlmostEqual(full.iloc[200].RVOL20, 2.)
        self.assertAlmostEqual(full.iloc[200]['Average Turnover Cr'], (frame.Close.iloc[180:200] * 1_000_000).mean() / 10_000_000)
        pd.testing.assert_series_equal(full.iloc[200], truncated.iloc[-1])

    def test_flat_rsi_is_neutral_and_proxy_is_not_session_vwap(self):
        frame = history()
        frame[['Open', 'High', 'Low', 'Close']] = [100., 101., 99., 100.]
        f = indicators(frame)
        self.assertEqual(f.iloc[-1].RSI14, 50.)
        self.assertEqual(f.iloc[-1]['Daily VWAP Proxy20'], 100.)
        self.assertNotIn('Session VWAP5m', f.columns)

    def test_invalid_or_future_daily_bars_do_not_qualify(self):
        frame = history(3)
        cutoff = frame.index[1].date()
        frame.iloc[1, frame.columns.get_loc('High')] = 1.0
        actual = clean_daily(frame, cutoff)
        self.assertEqual(len(actual), 1)

    def test_session_vwap_requires_completed_regular_session(self):
        index = pd.date_range('2026-10-01 09:15', periods=75, freq='5min', tz='Asia/Kolkata')
        f = pd.DataFrame({'Open':100., 'High':102., 'Low':99., 'Close':101., 'Volume':100.}, index=index)
        result = session_vwap(f, date(2026,10,1))
        self.assertEqual(result['Session VWAP Status'], 'Pass')
        self.assertAlmostEqual(result['Session VWAP5m'], 302 / 3)
        self.assertEqual(session_vwap(f.iloc[:-1], date(2026,10,1))['Session VWAP Status'], 'Unavailable / incomplete')
        self.assertEqual(session_vwap(f, date(2026,9,30))['Session VWAP Status'], 'Unavailable / incomplete')

    def test_missing_vwap_cannot_be_selected_and_industry_cap_is_respected(self):
        f = pd.DataFrame({'Ticker':['A','B','C','D'], 'Industry':['X','X','X','Y'],
                          'Technical Pass':[True]*4, 'Session VWAP Status':['Pass','Pass','Pass','Not checked'],
                          'Setup Score':[95,94,93,99], 'Excess Return20 %':[1,1,1,1]})
        result = select_picks(f, SwingConfig(top_n=3, max_per_industry=2))
        self.assertEqual(result.Ticker.tolist(), ['A','B'])

    def test_known_events_exclude_and_absent_events_remain_unverified(self):
        f = pd.DataFrame({'Ticker':['A','B'], 'Technical Pass':[True,True], 'Rejection Reasons':['','']})
        events = pd.DataFrame({'Ticker':['A.NS'], 'Date':['2026-10-10'], 'Event':['Earnings']})
        actual = apply_events(f, events, date(2026,10,4), SwingConfig())
        self.assertFalse(actual.iloc[0]['Technical Pass'])
        self.assertTrue(actual.iloc[1]['Technical Pass'])
        self.assertIn('Unverified', actual.iloc[1]['Event Status'])
        old_events = events.assign(Date='2026-09-01')
        self.assertTrue(apply_events(f, old_events, date(2026,10,4), SwingConfig())['Technical Pass'].all())

    def test_target_preserves_net_reward_risk_at_maximum_fill(self):
        f = history()
        result = snapshot(f, f, SwingConfig())
        entry, stop, target = (result[k] for k in ['Max Entry','Initial Stop','Target'])
        cost = entry * .003
        self.assertAlmostEqual((target-entry-cost)/(entry-stop+cost), 2.)
        self.assertLessEqual(result['Worst Fill Stop %'], 8.0 + 1e-10)

    def test_stale_and_missing_tickers_receive_coverage_rows(self):
        frame = history()
        current = frame.index[-1].date()
        u = pd.DataFrame({'Ticker':['A','B','C'], 'YFinance Ticker':['A.NS','B.NS','C.NS']})
        with tempfile.TemporaryDirectory() as directory, patch('screener_momentum.swing.fetch_bars', return_value={'A.NS':frame, 'B.NS':frame.iloc[:-1]}):
            _, health = download_universe(u, SwingConfig(), current, current, Path(directory), False, None)
        self.assertEqual(health['Data Status'].tolist(), ['Ready','Stale / inactive','Missing data'])

    def test_watchlist_counts_distinct_sessions_and_does_not_create_today_pass(self):
        f = pd.DataFrame({'Ticker':['A'], 'Name':['Alpha'], 'Industry':['X'], 'Technical Pass':[True], 'Setup':['Fresh breakout'], 'Setup Score':[80.]})
        first = build_watchlist(f, pd.DataFrame(), date(2026,10,1))
        twice = build_watchlist(f, first, date(2026,10,1))
        self.assertEqual(twice.iloc[0]['Sessions Seen'], 1)
        next_day = build_watchlist(f.assign(**{'Technical Pass':False}), twice, date(2026,10,5))
        self.assertEqual(next_day.iloc[0]['Current Status'], 'Monitor — does not qualify today')

    def test_unconfirmed_latest_pivot_cannot_appear(self):
        frame = pd.DataFrame({'Low':[10,9,7,9,10,8,6], 'High':[12,11,10,11,12,10,8]}, index=pd.bdate_range('2026-09-01',periods=7))
        result = confirmed_structure(frame)
        self.assertEqual(result['Last Confirmed Swing Low'], 7.)
        self.assertNotEqual(result['Last Confirmed Swing Low'], 6.)


if __name__ == '__main__':
    unittest.main()
