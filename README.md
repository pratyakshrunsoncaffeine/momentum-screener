# Momentum Screener

This project screens NSE stocks from `ticker.csv` in four stages:

1. Append `.NS` to every ticker and fetch prices with `yfinance`.
2. Calculate trailing returns for 5 days, 15 days, 1 month, 2 months, 3 months, 4 months, 6 months, 1 year, 3 years, and 5 years.
3. Build a weighted momentum score and keep only stocks with positive 5 day, 15 day, and 1 month returns.
4. Optionally scrape Screener.in fundamentals and apply market-cap, quarterly revenue growth, annual revenue growth, and promoter holding change filters.

The dashboard also backtests a portfolio invested in the top 10 final companies: 70% split across the top 5 and 30% split across the next 5, compared against Nifty 50 and a Nifty Midcap benchmark where Yahoo Finance data is available.

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

## FII Accumulation Screener

The `FII Accumulation` tab runs a separate workflow:

1. Scrape the shareholding table for every ticker in `ticker.csv`.
2. Calculate latest quarter FII holding change.
3. Add Yahoo Finance market capitalization using `.NS` tickers.
4. Sort the full FII scan by market cap from highest to lowest for export.
5. Rank companies by positive FII holding change.
6. Keep the top FII accumulation shortlist, default 50 companies.
7. Run the existing momentum score only on that shortlist.
8. Show the final top momentum picks, default 3 companies.

Checkpoint files:

- `output/latest/fii_partial.csv`
- `output/latest/fii_all.csv`
- `output/latest/fii_top50.csv`
- `output/latest/fii_momentum.csv`
- `output/latest/fii_final.csv`

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
- `BENCHMARKS`

The Streamlit sidebar also lets you adjust weights and filter thresholds without code edits.
