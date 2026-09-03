# Momentum Screener

This project screens NSE stocks from `ticker.csv` in four stages:

1. Append `.NS` to every ticker and fetch prices with `yfinance`.
2. Calculate trailing returns for 5 days, 15 days, 1 month, 2 months, 3 months, 4 months, 6 months, 1 year, 3 years, and 5 years.
3. Build a weighted momentum score and keep only stocks with positive 5 day, 15 day, and 1 month returns.
4. Optionally scrape Screener.in fundamentals and apply market-cap, quarterly revenue growth, annual revenue growth, and promoter holding change filters.

The dashboard also backtests a portfolio invested in the top 10 final companies: 70% split across the top 5 and 30% split across the next 5, compared against Nifty 50 and a Nifty Midcap benchmark where Yahoo Finance data is available.

It also includes independent FII, DII, quarterly-results, post-earnings stock-return, NSE index momentum, uploaded-stock momentum, macro correlation, and 200DMA opportunity workflows. Each long-running scanner writes checkpoints so a Streamlit refresh does not discard completed work.

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

The same restart pattern is available for the FII and DII full restart buttons, which clear only their respective institutional checkpoints and derived rankings.

Use the `Rank by YoY growth` control to sort the results by Sales, Operating Profit, Net Profit, or EPS without running the scrape again. The full scan and the matching-quarter subset are saved separately, and checkpoints resume only when they belong to the same requested quarter.

- `output/latest/quarterly_results_partial.csv`
- `output/latest/quarterly_results_all.csv`
- `output/latest/quarterly_results_matching.csv`

### Post-Earnings Stock Return Momentum

Inside the `Quarterly Results` tab, `Post-Earnings Stock Return Momentum` is a separate price-momentum workflow. It uses only companies that reported the selected result quarter, downloads their fresh Yahoo Finance prices, and ranks them from highest to lowest using stock returns over 2, 5, and 10 trading days.

The default weights are 20% for 2 days, 30% for 5 days, and 50% for 10 days. They can be adjusted in the dashboard before running the score. This is a stock-return momentum score, not an earnings-growth score; the earnings filter only determines which companies are eligible.

- `output/latest/quarterly_stock_return_returns.csv`
- `output/latest/quarterly_stock_return_momentum.csv`

## NSE Index Momentum

The `Index Momentum` tab uses the complete 128-index catalogue supplied from NSE, covering derivatives-eligible, broad-market, sectoral, strategy, and thematic indices. It downloads official daily index closes from NSE Indices and ranks each available index independently, so a newer index does not force every other index onto a shorter common date range.

## News Catalysts

The `News Catalysts` tab is an after-close research system for 5-day, 1-month, and 3-month NSE index forecasts. It combines point-in-time news, official NSE index closes, dated constituent activity, and the existing index momentum features. The dashboard queues background jobs and always keeps the last successful prediction available if a later job fails.

The data and model stack is deliberately separated from the Streamlit process:

- Filtered five-year GDELT backfills run through the no-card BigQuery Sandbox. Every UTC date is dry-run first, capped at 5 GiB, adaptively sampled when necessary, and checkpointed in Supabase before the next date begins. Incremental runs use GDELT plus explicitly configured, permitted publisher RSS feeds with a 48-hour overlap.
- Supabase PostgreSQL stores articles, dated index attribution, constituents, prices, features, labels, jobs, model versions, predictions, and catalysts. `pgvector` stores sentence embeddings; private Supabase Object Storage stores Parquet partitions and model artifacts.
- SQLMesh builds incremental point-in-time news features and forward labels. News after 4:30 PM IST is assigned to the next trading session.
- FinBERT produces sentiment when headline text is available. GDELT tone fallback is labelled explicitly for metadata-only records.
- Pooled LightGBM regressors/classifiers are ensembled with Ridge. A price-only LightGBM model is the mandatory benchmark.
- Saved catalyst headlines are paired with SHAP feature contributions for the selected horizon.
- Winsorization, empirical-Bayes sentiment shrinkage, embedding compression, minimum source/article features, and exponentially weighted aggregation reduce noise.
- Training-only headline dropout, embedding noise, source masking, relevance jitter, and technical-feature jitter are enabled only if the validation ablation improves. Dates, labels, index identities, split boundaries, and constituent effective dates are never altered.
- Training uses chronological 60%/20%/20% partitions with a 63-trading-day embargo. A model remains `Experimental` unless its untouched-test rank IC is positive and beats the price-only baseline on at least two horizons.

Pulse by Zerodha is not scraped or reverse-engineered. The dashboard provides only a manual Pulse link; an automated adapter remains disabled unless Zerodha supplies an authorized interface.

### News Setup

1. Create a Supabase project and run [`supabase/news_schema.sql`](supabase/news_schema.sql) in its SQL editor.
2. Create a Google Cloud project without linking billing. Open BigQuery once to activate Sandbox, then create a service account with `BigQuery Job User` and `BigQuery Read Session User` roles.
3. Add these GitHub repository secrets: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_DB_HOST`, `SUPABASE_DB_USER`, `SUPABASE_DB_PASSWORD`, `SUPABASE_DB_NAME`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_SERVICE_ACCOUNT_JSON`. The worker intentionally constructs its database connection from these fields so the password is not duplicated inside a URL secret.
4. Optionally add `NEWS_RSS_FEEDS_JSON` as a JSON object containing only feeds whose terms permit automated retrieval.
5. Add the values from [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) to Streamlit Community Cloud secrets. Use a new fine-grained GitHub token limited to this repository with Actions write access.
6. Run `News Catalysts - Historical Backfill` repeatedly until the job reports that all historical partitions are complete. Each run processes at most 90 pending dates and resumes from the durable Supabase ledger. Daily inference runs after 4:30 PM IST on weekdays; retraining runs monthly.

The workflow reserves roughly 10% of Sandbox's monthly 1 TiB query allowance and never relies on `LIMIT` as a cost control. `LIMIT` restricts downloaded rows only; dry runs, partition pruning, adaptive `TABLESAMPLE`, and `maximum_bytes_billed` enforce the free boundary. If the 900 GiB guard is reached, the job moves to `waiting_for_resume` and can continue after the monthly quota resets. BigQuery stores no durable training corpus; Supabase holds the persistent data, so Sandbox's 60-day BigQuery table expiration does not affect saved history.

Heavy worker packages are isolated in `requirements-ml.txt`; normal Streamlit deployment continues to use `requirements.txt`.

The near-term default score is 10% for the 2-day return, 20% for 5 days, 25% for 10 days, 25% for 1 month, 15% for 2 months, and 5% for 3 months. All six weights are adjustable inside the tab and normalized automatically. The complete ranking remains visible, while `Short-Term Positive` identifies indices whose 2-day, 5-day, and 10-day returns are all positive.

Use `Run / Resume Index Scan` to refresh official data, `Recalculate Saved Prices` to apply changed weights without another download, `Run Full Scan From Beginning` to clear only this index cache, or `Use Saved Index Results` to recover the latest ranking.

Saved index files:

- `output/index_momentum_cache/`
- `output/latest/index_momentum_prices.csv`
- `output/latest/index_momentum_ranking.csv`
- `output/latest/index_momentum_health.csv`
- `output/latest/index_momentum_metadata.csv`

## CSV Stock Momentum

After identifying a strong index, upload that index's constituent ticker list in the `CSV Stock Momentum` tab. The CSV may use `Ticker`, `Ticker Name`, `Symbol`, `NSE Symbol`, or a single ticker column. Existing `.NS` suffixes are normalized before Yahoo Finance is queried.

This workflow uses the regular stock momentum weights from the sidebar and preserves the usual positive 5-day, 15-day, and 1-month filter as a flag. Every stock with sufficient data receives a score and rank, including stocks that do not pass the positive-return gate, so the complete constituent set can be compared.

Saved uploaded-stock files:

- `output/latest/custom_stock_universe.csv`
- `output/latest/custom_stock_returns.csv`
- `output/latest/custom_stock_momentum.csv`
- `output/latest/custom_stock_health.csv`

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

## 200DMA Opportunity Finder

The `200DMA Finder` tab scans every stock in `ticker.csv` for a price from 0% through the selected maximum above its 200-day moving average. The default maximum is 10%. Stocks below the average or farther above it are rejected, and qualifying stocks are ranked by the smallest positive distance.

Yahoo Finance adjusted daily closes are downloaded in batches with `.NS` appended to each symbol. Incomplete histories are retried in smaller batches and then attempted through official NSE daily archives. Near-live mode refreshes delayed or intraday Yahoo quotes only for a buffered shortlist; latest completed close remains the fallback. Every row identifies its price source, basis, date, 200DMA, 20-day SMA slope, and rejection reason.

After the proximity calculation, the same action automatically runs the configured Screener.in filters on only the shortlist. `Run / Resume 200DMA Scan` reuses recent prices and completed fundamental checkpoints. `Run Full Scan From Beginning` removes only this feature's saved run and cache before starting again. `Use Saved 200DMA Scan` redraws the complete result and backtest without another network request.

The monthly walk-forward backtest uses only price data before each rebalance, recalculates the 200DMA at every signal date, and selects the closest ten positive-proximity stocks. It uses the existing 70% allocation across the first five and 30% across the next five, fractional shares, monthly compounding, and Nifty benchmarks. Months with no eligible stocks remain in cash. The current fundamental-passed universe is used as a static historical universe, so the dashboard explicitly discloses current-data and survivorship bias.

Saved 200DMA files:

- `output/latest/sma200_prices.csv`
- `output/latest/sma200_universe.csv`
- `output/latest/sma200_candidates.csv`
- `output/latest/sma200_rejected.csv`
- `output/latest/sma200_fundamentals_partial.csv`
- `output/latest/sma200_fundamentals.csv`
- `output/latest/sma200_final.csv`
- `output/latest/sma200_health.csv`
- `output/latest/sma200_backtest_curve.csv`
- `output/latest/sma200_backtest_periods.csv`
- `output/latest/sma200_current_allocation.csv`
- `output/latest/sma200_backtest_summary.csv`

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
- `DEFAULT_INDEX_MOMENTUM_WEIGHTS`
- `FundamentalThresholds`
- `Sma200ScanConfig`
- `BENCHMARKS`

The Streamlit sidebar also lets you adjust weights and filter thresholds without code edits.
