# Momentum Screener

This project screens NSE stocks from `ticker.csv` in four stages:

1. Append `.NS` to every ticker and fetch prices with `yfinance`.
2. Calculate trailing returns for 5 days, 15 days, 1 month, 2 months, 3 months, 4 months, 6 months, 1 year, 3 years, and 5 years.
3. Build a weighted momentum score and keep only stocks with positive 5 day, 15 day, and 1 month returns.
4. Optionally scrape Screener.in fundamentals and apply market-cap, quarterly revenue growth, annual revenue growth, and promoter holding change filters.

The dashboard also backtests a portfolio invested in the top 10 final companies: 70% split across the top 5 and 30% split across the next 5, compared against Nifty 50 and a Nifty Midcap benchmark where Yahoo Finance data is available.

It also includes independent FII, DII, quarterly-results, post-earnings stock-return, and NSE derivatives momentum workflows. Each long-running scanner writes checkpoints so a Streamlit refresh does not discard completed work.

The portfolio tab uses a monthly walk-forward price backtest. Each month it recalculates momentum using only price data available before that rebalance date, invests for one month, then reinvests the ending capital into the next month's selected portfolio. If fundamentals are applied, the historical backtest uses the current fundamentals-passed universe as a static filter, so it avoids price lookahead but still has current-fundamentals bias.

## Dashboard Flow

Use the dashboard in stages:

1. `Refresh Momentum Data` downloads yfinance prices, shows progress by ticker batches, calculates returns, writes `output/latest/returns.csv`, and immediately shows the momentum table.
2. `Use Saved Momentum` rebuilds momentum from `output/latest/returns.csv` without downloading yfinance data again.
3. `Run Fundamentals on Top 100` scrapes Screener.in only for the top momentum candidates and writes `output/latest/fundamentals_partial.csv` after every completed company.
4. `Skip Fundamentals` builds the final list from momentum only.
5. `Recover Saved Run` reloads saved intermediate CSVs if the app refreshes or Screener.in is slow.

The main checkpoint files are:

- `output/latest/returns.csv`
- `output/latest/momentum.csv`
- `output/latest/fundamentals_partial.csv`
- `output/latest/fundamentals.csv`
- `output/latest/final.csv`
- `output/latest/backtest.csv`
- `output/latest/normalized_backtest.csv`
- `output/latest/walk_forward_periods.csv`
- `output/latest/current_allocation.csv`

## FII And DII Accumulation Screeners

The `FII Accumulation` and `DII Accumulation` tabs run separate but parallel workflows:

1. Scrape the shareholding table for every ticker in `ticker.csv`.
2. Calculate latest quarter FII or DII holding change.
3. Read market capitalization from the same Screener.in page already being scraped.
4. Sort the full scan by market cap from highest to lowest for export.
5. Rank companies by positive institutional holding change.
6. Keep the top accumulation shortlist, default 50 companies.
7. Run the existing momentum score only on that shortlist.
8. Show the final top momentum picks, default 3 companies.

Yahoo Finance is still used for price/momentum data and as a recovery fallback for older saved institutional files that do not have market cap columns.

`Run / Resume FII Scan` and `Run / Resume DII Scan` write checkpoints while they run, so a stopped Streamlit run can continue from the saved rows instead of starting again. `Use Saved FII Scan` and `Use Saved DII Scan` load the newest saved file, including partial checkpoints, and finalize the top accumulation and momentum tables without re-scraping Screener.in.

Checkpoint files:

- `output/latest/fii_partial.csv`
- `output/latest/fii_marketcap_partial.csv`
- `output/latest/fii_all.csv`
- `output/latest/fii_top50.csv`
- `output/latest/fii_momentum.csv`
- `output/latest/fii_final.csv`
- `output/latest/dii_partial.csv`
- `output/latest/dii_marketcap_partial.csv`
- `output/latest/dii_all.csv`
- `output/latest/dii_top50.csv`
- `output/latest/dii_momentum.csv`
- `output/latest/dii_final.csv`

## Quarterly Results Growth Scanner

The `Quarterly Results` tab scans every ticker in `ticker.csv` using the existing Screener.in company-page workflow. Enter the period exactly as a quarterly table label, such as `Jun 2026`. The scanner keeps companies whose table contains that period and calculates both comparisons for Sales, Operating Profit, Net Profit, and EPS:

- QoQ: selected result quarter versus the immediately preceding quarter, for example `Jun 2026` versus `Mar 2026`.
- YoY: selected result quarter versus the same quarter one year earlier, for example `Jun 2026` versus `Jun 2025`.

`Run / Resume Quarterly Scan` retains successful checkpoint rows and continues interrupted work, while retrying saved network failures. `Run Full Scan From Beginning` clears only the quarterly-results and post-earnings checkpoints, then scrapes the complete ticker universe again from ticker one. Use the full restart after Screener.in has updated more company results for the same quarter.

The same restart pattern is available wherever a resumable full scan exists:

- FII and DII full restart buttons clear only their respective institutional checkpoints and derived rankings.
- The derivatives full-range restart clears raw and normalized NSE EOD cache directories only for the selected date range, preserves cached dates outside that range, and rebuilds derived signal and backtest files.

Use the `Rank by YoY growth` control to sort the results by Sales, Operating Profit, Net Profit, or EPS without running the scrape again. The full scan and the matching-quarter subset are saved separately, and checkpoints resume only when they belong to the same requested quarter.

- `output/latest/quarterly_results_partial.csv`
- `output/latest/quarterly_results_all.csv`
- `output/latest/quarterly_results_matching.csv`

### Post-Earnings Stock Return Momentum

Inside the `Quarterly Results` tab, `Post-Earnings Stock Return Momentum` is a separate price-momentum workflow. It uses only companies that reported the selected result quarter, downloads their fresh Yahoo Finance prices, and ranks them from highest to lowest using stock returns over 2, 5, and 10 trading days.

The default weights are 20% for 2 days, 30% for 5 days, and 50% for 10 days. They can be adjusted in the dashboard before running the score. This is a stock-return momentum score, not an earnings-growth score; the earnings filter only determines which companies are eligible.

- `output/latest/quarterly_stock_return_returns.csv`
- `output/latest/quarterly_stock_return_momentum.csv`

## Derivatives Momentum Scanner

The `Derivatives Momentum` tab uses official end-of-day NSE cash-market, F&O bhavcopy, and F&O contract reports. It intersects each date's stock F&O universe with all tickers in `ticker.csv`; companies without listed derivatives are retained in the export with `No listed derivatives` status.

A default options-confirmed equity signal requires all of the following on the same trading day:

- Underlying stock return of at least 2%.
- Selected near-ATM call return of at least 8%. The 8% threshold is a hard minimum; larger gains remain eligible and rank higher.
- Call close of at least Rs. 5.
- Call volume and open interest of at least 100 contracts each.
- A monthly stock-option expiry between 7 and 45 calendar days away.
- Valid current and previous closes with no detected corporate-action distortion.

The signal recommends buying the underlying equity, not the call. Qualifying stocks are ranked by call return, stock return, call OI change, call volume versus its 20-day median, and near-month futures return. Rising call OI is a ranking input rather than a hard bullish rule because it can represent buying or writing. Futures activity is separately labelled as long build-up, short covering, short build-up, long unwinding, or unconfirmed.

Use the tab in this order:

1. Select a date range and click `Download / Resume EOD Data`. Raw reports and normalized daily CSVs are cached by trade date.
2. Choose a cached signal date and click `Run Options-Confirmed Momentum`.
3. Review ranked signals, rejection reasons, the complete F&O eligibility map, and report health.
4. For validation, first backfill the historical range, then run the chronological event study and portfolio simulation.

The backtest enters the stock at the next trading day's open, measures 1, 3, 5, and 10-day forward returns, deducts the configured round-trip cost, and compares stock-only, stock-plus-call, full derivatives, regular momentum, and Nifty 50 results. Events are split chronologically into 60% Train, 20% Validation, and 20% untouched Test periods. A useful strategy should improve the full derivatives Test result over the stock-only baseline after costs, not merely look attractive in training.

Saved derivatives files:

- `output/derivatives_cache/` for raw and normalized daily NSE reports.
- `output/latest/derivatives_contracts.csv`
- `output/latest/derivatives_daily_features.csv`
- `output/latest/derivatives_signals.csv`
- `output/latest/derivatives_rejections.csv`
- `output/latest/derivatives_data_health.csv`
- `output/latest/derivatives_backtest_events.csv`
- `output/latest/derivatives_backtest_curve.csv`
- `output/latest/derivatives_backtest_summary.csv`
- `output/latest/derivatives_event_summary.csv`

## NSE Sector Rotation

The `Sector Rotation` tab compares official NSE sector price indices with Nifty 50. It downloads daily index closes directly from NSE Indices, aligns every selected sector to the latest common trading date, and calculates 1-week, 1-month, 3-month, and 6-month returns and excess returns.

The relative-rotation view uses 3-month excess return as relative strength and the change in monthly excess return as relative momentum. Sectors are classified as Leading, Improving, Weakening, or Lagging, then ranked with 40% weight on 1-month excess return, 35% on 3-month excess return, and 25% on relative-momentum acceleration. This is price-based trend analysis, not a measure of institutional fund flows.

Use `Refresh NSE Sector Data` for a new official download or `Use Saved Sector Data` to recover the most recent completed run. If NSE is temporarily unavailable during a refresh, the dashboard shows a stale-data warning before using saved NSE data. Yahoo Finance is never substituted for this section.

Saved sector files:

- `output/sector_rotation_cache/` for per-index official NSE history.
- `output/latest/sector_rotation_prices.csv`
- `output/latest/sector_rotation_snapshot.csv`
- `output/latest/sector_rotation_health.csv`

## Macro Factor Correlation

The `Correlation` tab scans the complete `ticker.csv` universe against Brent crude, gold, the US 10-year yield, the India 10-year yield, and USD/INR. Every fresh run downloads stock history only through the selected analysis date, so a later run automatically rolls the historical window forward without using dates after the requested cutoff.

Brent, gold, and USD/INR use percentage price changes. Sovereign yields use basis-point changes because percentage returns on yield levels are not economically comparable. The public India 10-year FRED/OECD series is monthly; the dashboard labels that effective frequency even when the other selected factors use daily or weekly observations.

Stock history is requested from Yahoo Finance using each ticker with `.NS` appended. Failed Yahoo batches are retried in smaller groups. Symbols that remain unavailable are attempted through official NSE equity history and are identified in the source-health export. Yahoo closes are corporate-action adjusted; the NSE fallback is a raw close, so its different price basis is disclosed rather than silently blended.

For each factor, the tab shows:

- Stocks with the strongest positive historical relationship.
- Stocks with the strongest inverse relationship, which have tended to benefit when the factor fell.
- Average stock returns and positive-return hit rates when the factor rose or fell.
- Strong-move tail returns, sensitivity, R-squared, and observation counts.
- A cross-factor correlation heatmap, detailed table, factor history, and source-health report.
- Standardized multivariate ridge coefficients that isolate each factor's partial relationship after controlling for the other factors available at the same frequency.
- Normalized factor-versus-stock charts for the five strongest positive and inverse picks.

`Same period` measures contemporaneous co-movement. `Next stock period` compares each factor change with the following stock-return period and is the more relevant research view for a watchlist, but it remains a historical relationship rather than a prediction or guarantee.

Use the `Ranking model` control to switch the top picks between raw correlation and ridge regression. Ridge is the default because crude, gold, yields, and currencies can move together; regularization makes the partial coefficients less unstable than an unpenalized multivariate regression. `Ridge alpha` controls the penalty and is saved with the run.

`Use Saved Correlation` reloads the complete research state, including the stock-price matrix, factor observations, model outputs, settings, and source-health reports. This is sufficient to redraw the comparison charts without downloading Yahoo or NSE data again.

Saved correlation files:

- `output/correlation_cache/` for macro source-level checkpoints.
- `output/latest/correlation_stock_prices.csv`
- `output/latest/correlation_factors.csv`
- `output/latest/correlation_all.csv`
- `output/latest/correlation_top_positive.csv`
- `output/latest/correlation_top_negative.csv`
- `output/latest/correlation_health.csv`
- `output/latest/correlation_stock_health.csv`
- `output/latest/correlation_ridge_top_positive.csv`
- `output/latest/correlation_ridge_top_negative.csv`
- `output/latest/correlation_run_metadata.csv`

## Local Setup

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For a CLI run:

```powershell
.\.venv\Scripts\python.exe run_screener.py --csv ticker.csv --top 100 --final 100
```

Outputs are written to the `output` folder.

## Customization

Most screener knobs are in `screener_momentum/config.py`:

- `DEFAULT_MOMENTUM_WEIGHTS`
- `DEFAULT_POSITIVE_RETURN_FILTERS`
- `FundamentalThresholds`
- `DerivativesSignalConfig`
- `BENCHMARKS`

The Streamlit sidebar also lets you adjust weights and filter thresholds without code edits.
