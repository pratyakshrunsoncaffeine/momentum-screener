# Evening swing rules — version 1.0

Purpose: scan the full user ticker CSV after 8 p.m. IST, identify long-only setups for the next session, and plan trades lasting 5–10 trading sessions. Scores describe setup strength; no probability of a positive next-day or next-week return has been estimated. These exact combinations and thresholds are research hypotheses, not results proved by the linked papers.

## Indicator selection for this horizon

| Family | Indicators considered | Choice and role |
| --- | --- | --- |
| Price structure | Support/resistance, trading-range breaks, retests, engulfing candles, rejection wicks, chart patterns | Primary setup. Use causal 20-session resistance, consolidation range, breakout/retest and structural lows. Display engulfing/rejection flags; avoid subjective automated pattern names. |
| Trend | SMA, EMA, WMA, HMA, DEMA/TEMA, Supertrend, Ichimoku, Parabolic SAR | Choose daily EMA20/EMA50 plus completed-week EMA10. EMA10 guides exits. SMA200 is context only. Supertrend and additional averages largely duplicate trend information and are excluded from the score. |
| Momentum | RSI, MACD, ROC, stochastic, Williams %R, CCI, TSI | Choose Wilder RSI14 and MACD(12,26,9 EMA signal). MACD contributes confirmation points; a new crossover is not mandatory. Return5/10/20 describes speed; avoid adding several correlated oscillators. |
| Trend strength | ADX/+DI/-DI, Aroon | Choose ADX14 and directional indicators for 5 score points, not a mandatory entry gate that can miss early breakouts. ADX strength alone does not imply upward direction. |
| Participation/liquidity | Relative volume, traded value, OBV, CMF, MFI, accumulation/distribution, delivery percentage | Prior-20-session RVOL and average rupee turnover are primary. OBV5 contributes only 5 points. Volume cannot identify institutions or establish why participants traded. MFI overlaps volume and RSI; omit extra weighting. Delivery data is not reliably available from the price feed. |
| Volume-weighted price | Session VWAP, rolling VWAP, anchored VWAP, volume profile | Use 5-minute-bar session VWAP confirmation on the daily shortlist. Also show a distinct 20-day daily typical-price/volume proxy. Volume profile and exact anchored VWAP need consistent intraday/trade data; omit them from the initial rules. |
| Volatility/compression | ATR, Bollinger bandwidth/%B, Keltner channels, historical volatility | ATR14 sets extension, stop, gap and volatility limits. Bollinger(20,2) bandwidth percentile recognizes tight bases; touching the upper band is not an automatic sell signal. |
| Relative leadership | Stock/index price ratio, excess returns, industry breadth | Require positive 20-session return and outperformance of NIFTY over matched sessions. Show rising-trend breadth and setup counts by CSV industry. These are industry groups, not verified NSE sector indices. |
| Targets/exits | Structure, EMA10, ATR chandelier, Fibonacci, oscillator divergence | Structural initial stop, a 2R target after estimated costs, EMA10 close exit, and a 10-session time exit. Show chandelier level as optional context. Fibonacci and retrospective divergence do not receive score points. |

The suite is deliberately compact. Price, averages, MACD and RSI share price inputs; their combined score does not represent independent statistical votes. All comparisons use completed observations and no future-confirmed pivots.

## Data and universe gates

- Attempt every unique symbol in `ticker.csv`: the current shared list contains 2,926, not exactly 3,000.
- Map six-digit numeric BSE codes to Yahoo's `.BO` suffix; map NSE symbols to `.NS`; preserve explicit `.BO` symbols. Unsupported/dummy/right-entitlement tickers remain visible in coverage rather than being replaced by guessed symbols.
- At least 200 valid daily OHLCV bars; latest bar matches the NIFTY benchmark session; latest volume is positive.
- Price at least ₹20; average daily traded value over the **previous** 20 sessions at least ₹5 crore. This is a liquidity filter, not a market-cap filter.
- Latest NIFTY completed bar defines the scan session. Before 4 p.m. IST, today's daily bar is ignored. Run at 8 p.m. and confirm the displayed session date. Weekends/holidays may correctly show the prior trading session.
- Yahoo adjusted OHLC is used consistently for indicators and levels. Adjustment/vendor revisions and inaccurate CSV name/industry mappings remain possible. Corporate actions and exchange restrictions require a final check.
- Every ticker receives a coverage row. Missing, stale, inactive or short-history prices do not qualify and are never silently replaced by an old scan.

## Shared long-only gates

1. Close > EMA20 > EMA50; EMA20 is higher than five sessions ago.
2. The last completed Friday-labelled weekly close is above weekly EMA10, which is rising. A holiday week with its Friday after the data session is conservatively omitted.
3. RSI14 between 52 and 78. This initial band is configurable; 70 is not a universal ceiling for an uptrend.
4. ATR14 / close between 0.8% and 6%; close no more than 3 ATR above EMA20.
5. Twenty-session return > 0 and greater than NIFTY's return over matched dates. Five-session return no greater than 20%, to reduce chasing vertical moves.
6. Close at or above the daily 20-day volume-weighted proxy.
7. Qualify for one price-action setup below. A candle closing strongly means close > open, close in the top 35% of its daily range, and upper wick no larger than 30% of the range.
8. Structural stop distance from the trigger is positive, at most 8% and at most 2.5 ATR. The previous 60-session high must be below the trigger or beyond the planned target. This approximates overhead supply and does not prove the absence of resistance.
9. Reject volume ≥3× with an upper wick ≥40% of the candle range. This is a rejection warning, not proof of institutional distribution.
10. Score at least 60. If supplied, known events in the coming 14 calendar days exclude the stock. Missing event information is labeled unverified.

MACD is scored by default; setting `require_bullish_macd: true` additionally requires MACD > 0 and MACD > its signal line. This optional gate is stricter and can delay entries.

## Four price-action setups

| Setup | Exact initial rule | Structural support for stop |
| --- | --- | --- |
| Fresh breakout | Close above the previous 20-session high; volume ≥1.5× prior-20-session average; strong close; preceding five-session range ≤2.5 ATR | Signal candle low |
| Breakout retest | A breakout above its then-current 20-session high with ≥1.5× RVOL in the previous six sessions; today's low within ±0.5 ATR of that level; close above it with a strong candle and ≥0.8× RVOL | Lowest low of latest three sessions |
| Pullback recovery | Today and yesterday traded within 0.5 ATR of EMA20; today closes above EMA10 with a strong candle and ≥1× RVOL | Lowest low of latest three sessions |
| Tight base near breakout | Close below/at 20-session resistance but within 0.75 ATR; last five-session range ≤2.5 ATR; Bollinger bandwidth in its bottom 35% over a trailing 120-session window | Lowest low of latest five sessions |

A tight base is a conditional watchlist setup: it has not yet broken out. Confirm expansion in price and volume before entering it; an evening ranking alone is insufficient.

## Transparent 100-point score

| Block | Points | Calculation |
| --- | ---: | --- |
| Trend | 20 | 10 daily trend + 5 completed-week trend + 5 ADX≥20 with +DI>-DI |
| Momentum | 20 | Up to 10 for RSI near 65 (linear decline to zero 25 points away); 5 positive MACD histogram; 5 histogram rising since yesterday |
| Structure | 25 | Breakout/retest 25; pullback recovery 22; tight base 18; none 0 |
| Participation | 15 | RVOL scaled from 0.5× to 2× gives 0–10; OBV change over five sessions divided by five times prior average volume, clipped to 0–1, gives 0–5 |
| Relative strength | 10 | 0–10 for 0–10 percentage points of matched 20-session NIFTY outperformance |
| Daily volume-weighted price | 5 | Close ≥ the 20-day daily proxy |
| Market | 5 | NIFTY close > EMA20 > EMA50; otherwise cautious |

Ranks use score, then relative strength, then ticker for deterministic ties. Default final list is at most ten stocks with at most three per CSV industry. No minimum number of picks is forced. An industry with missing metadata is grouped as Unknown. Broad market weakness is displayed even when an individual stock qualifies.

## VWAP verification

The scan fetches 5-minute bars for up to the highest-ranked 200 daily-qualified stocks. Require at least 70 of the expected 75 cash-session bars, a start no later than 9:20 a.m., and a bar at 3:25 p.m., with valid positive aggregate volume. Calculate sum(typical price × bar volume) / sum(bar volume), where typical price = (high + low + close) / 3. The final intraday close must be at or above this estimate.

This is a **bar-based session VWAP estimate**, not exact trade-by-trade VWAP. Missing/incomplete bars cannot pass; remaining candidates outside the refinement limit remain Not checked. The final picks require Pass by default. Daily VWAP proxy20 has a different horizon and is never labeled session VWAP. All tickers are screened with daily data; only the daily shortlist receives intraday refinement.

## Entry, position size and selling rules

- Trigger = signal-day high + 0.05 ATR; for a tight base use the larger of high and resistance, plus the buffer. The setup is valid only for the next session; rescan each evening if untriggered.
- Maximum permitted fill starts at trigger + 0.25 ATR, then is reduced if needed to keep the stop distance within 8% and 2.5 ATR. Skip an opening gap above this price. An entry trigger does not guarantee a fill at that price; recheck the plan using the actual fill.
- Initial stop = structural support minus 0.10 ATR. Reject a plan with a stop too distant; do not move the stop into the structure just to pass the filter.
- Target is calculated from the maximum permitted fill so planned net reward/risk is at least 2 under an explicit estimated 0.30% total round-trip cost. Costs are an assumption, not a current broker fee calculation. Overnight gaps and price limits can create losses beyond the planned stop.
- If using position sizing, choose an account-risk budget first (an initial research setting can be 0.5%–1% per trade). Shares = floor(risk budget / (actual entry − stop + estimated cost per share)), capped by available cash and liquidity. The program produces a watchlist, not orders or portfolio allocations.
- On a held position, exit when the protective stop is breached. After +1R, consider tightening the stop at each evening scan with `max(previous stop, EMA10 − 0.1 ATR)`; never loosen it. This is a separate position-management rule: EMA10 in the output is a reference, not a stateful trailing stop.
- Sell after a daily close below EMA10 using the next executable price, or at the target, or after ten trading sessions from entry. A daily-close exit cannot fill at that day's close when evaluated at 8 p.m.
- Review a trade after three sessions without progress. MACD below signal, declining OBV or a high-volume rejection are review warnings. RSI>70 by itself is not an automatic exit. If news changes the thesis, reassess independently of these technical rules.

## What the literature supports

- [Physical Momentum in the Indian Stock Market (arXiv, 2023)](https://arxiv.org/abs/2302.13245): tests daily, weekly and other timescales on NSE500; relevant motivation for comparing horizons. It does not validate these entry/exit thresholds or a deployable probability model.
- [Momentum, Reversals and Liquidity: Indian Evidence (2023)](https://doi.org/10.1016/j.pacfin.2023.102193): motivation for liquidity and cross-sectional leadership. Its holding-period evidence is not interchangeable with a 5–10-session breakout.
- [Volume-Based Price Momentum in India](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2817832): finds historical volume did not boost momentum-return magnitude in its sample. A volume filter therefore requires an ablation test rather than a presumed benefit.
- [IIM Ahmedabad: The Dynamic Relationship between Price and Trading Volume](https://iima.ac.in/publication/dynamic-relationship-between-price-and-trading-volumeevidence-indian-stock-market): relationships between returns and volume, with weak dynamics in some tests. It provides no universal volume-spike profit threshold.
- [Usefulness of Moving Average Based Trading Rules in India](https://doi.org/10.5539/ijbm.v6n7p199): highlights transaction costs that erased many apparent opportunities.
- [Optimization of MACD and RSI Indicators: Indian Equity Market](https://indianjournals.com/article/ajrbf-5-12-002): reports buy-and-hold outperforming the standard indicators and better results from optimized versions. Optimization on the same sample is not evidence of future performance; retain transparent defaults until tested out of sample.
- [Aggregate News Sentiment and Stock Market Returns in India](https://www.mdpi.com/1911-8074/16/8/376): motivates event awareness, without providing a complete upcoming-events feed.
- [Zerodha Varsity: RSI](https://zerodha.com/varsity/chapter/indicators-part-1/) and [other indicators including ADX and VWAP](https://zerodha.com/varsity/chapter/supplementary-notes-1/): practical definitions, not academic proof of profitability.

User video references were reviewed through their public descriptions and chapter lists; full caption retrieval was unavailable:

- [How to scan stocks for swing trading? — Ft. Manas Arora](https://www.youtube.com/watch?v=PBEyuc9jjMU): its description highlights volume, recent highs, one-/three-month performance and watchlist maintenance. We use RVOL, 20-session resistance, Return20/63 context and a persistent watchlist. Its advice against daily discovery is adapted to the user's explicit 8 p.m. scan: monitor an existing list, retain previous sightings, and avoid forcing fresh trades each day.
- [Swing Trading Analysis Using 3 Technical Indicators](https://www.youtube.com/watch?v=D-0w76MGF5M): its description combines moving averages, ZigZag structure and MACD histogram, listing EMA3/15, ZigZag deviation 3/pivot legs 2 and MACD12/26/9. We retain the trend/structure/momentum hierarchy but choose EMA20/50 for this daily swing horizon. Instead of a mutable ZigZag endpoint, display local swing highs/lows only after two subsequent bars confirm them. These delayed pivots are context, not independent score points; requiring a fresh MA crossover would miss established-trend entries.

Multiple aligned indicators are not independent statistical proof of a high-probability trade. The user-supplied price-action/indicator framework also informs the hierarchy above.

## Watchlist and industry coverage

Retain qualifying technical setups for 30 calendar days after their last sighting. Record first/last seen, number of distinct sessions, current qualification and latest score. Repeating a scan within the same session does not inflate the sightings count. A retained watchlist stock that fails today's rules is explicitly a monitor item and cannot enter today's final list through its history alone.

The original ticker CSV has industry labels for only 763 of 2,926 rows. Where available, local quality/correlation universe metadata fills missing labels without changing the ticker universe or overriding existing labels; the Industry Source column records this. Unknown groups and groups with fewer than five usable stocks are unrankable as pockets. Sector breadth is therefore incomplete and must not be interpreted as an exhaustive sector-rotation study.

## Running and changing the rules

Dashboard: open **Evening Swing Screener**, then **Run evening swing scan**. Same-session histories resume automatically; use Refresh all daily data to fetch again. Saved results retain their original settings.

Terminal: `.venv\Scripts\python.exe run_swing.py` or `./start_swing_scan.ps1`. No recurring task is created; run manually at 8 p.m. IST.

Use `--config path/to/overrides.json` to change any SwingConfig field, for example:

```json
{"top_n": 5, "min_turnover_cr": 10.0, "breakout_rvol": 1.8, "require_bullish_macd": false}
```

Optional known-events file via `--events events.csv`:

```csv
Ticker,Date,Event
EXAMPLE,2026-10-15,Quarterly earnings
```

Outputs in `output/swing/latest/`: picks, complete usable metrics, excluded setups, coverage, industry pockets, benchmark and metadata. Dated runs are archived under `output/swing/runs/`; per-ticker daily history allows recovery after interruptions. Same-session failed symbols remain reported and are skipped on resume; Refresh retries them. Even short-history stocks can reuse same-session cached bars, while still failing the history gate.

Before describing any score as a likelihood, validate it with chronological held-out Indian data, point-in-time ticker universes, next-session executions, 1/5/10-session outcomes, transaction costs and gap/limit behavior. Compare removing RVOL, MACD, RSI and VWAP separately. The current historical ticker file cannot establish survivorship-free performance. Unit checks validate calculations and data handling, not economic profitability.
