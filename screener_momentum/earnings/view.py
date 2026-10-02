"""Earnings probability tab inside the Momentum Screener."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from .engine import (
    DEFAULT_ARCHIVE, DEFAULT_BUNDLE, DEFAULT_TICKERS, IST,
    previous_completed_quarter, score_universe,
)
from .market_cap import DEFAULT_CAPS, load_market_caps, refresh_market_caps

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_RANKINGS = DATA_DIR / "default_rankings.csv"
DEFAULT_SUMMARY = DATA_DIR / "default_summary.json"


@st.cache_data(show_spinner=False)
def _load_default() -> tuple[pd.DataFrame, dict]:
    return pd.read_csv(DEFAULT_RANKINGS), json.loads(DEFAULT_SUMMARY.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Scoring all supplied tickers from first-published filings…")
def _score(target: str, cutoff: str, source_mtimes: tuple[int, int, int]) -> tuple[pd.DataFrame, dict]:
    _ = source_mtimes
    result = score_universe(DEFAULT_TICKERS, DEFAULT_ARCHIVE, DEFAULT_BUNDLE, target, cutoff)
    return result.rows, result.summary

def display_rankings(scored: pd.DataFrame, target: str) -> None:
    st.subheader("Highest estimated probability of higher quarterly net profit")
    st.caption(
        f"Target quarter: {target}. The event is first-reported net profit above the immediately previous quarter."
    )
    left, right, third = st.columns([2.2, 1.3, 1.1])
    with left:
        query = st.text_input("Find a company or ticker", placeholder="For example, SYRMA or TCS")
    with right:
        industries = st.multiselect(
            "Industry filter (display only)",
            sorted(scored["industry"].dropna().unique()),
        )
    with third:
        threshold = st.slider("Minimum up probability", 0, 100, 0, step=5, format="%d%%")

    shown = scored.copy()
    if query:
        contains = (
            shown["ticker"].str.contains(query, case=False, regex=False)
            | shown["company"].str.contains(query, case=False, regex=False)
        )
        shown = shown.loc[contains]
    if industries:
        shown = shown.loc[shown["industry"].isin(industries)]
    shown = shown.loc[shown["probability_up"].ge(threshold / 100)]
    st.caption(f"{len(shown):,} qualifying forecasts · sorted from highest to lowest probability")
    if shown.empty:
        st.info("No scored tickers match these filters.")
        return

    chart = shown.head(20).assign(**{"Probability up (%)": lambda frame: 100 * frame["probability_up"]})
    st.bar_chart(
        chart[["ticker", "Probability up (%)"]],
        x="ticker", y="Probability up (%)", horizontal=True, sort=False,
        color="#3478b6", height=520,
    )
    table = shown[[
        "rank", "ticker", "company", "industry", "probability_up",
        "probability_down", "last_report_period", "last_net_profit_cr",
        "history_quarters", "missing_financial_features",
    ]].copy()
    table["probability_up"] *= 100
    table["probability_down"] *= 100
    table = table.rename(columns={
        "rank": "Rank", "ticker": "Ticker", "company": "Company",
        "industry": "Industry", "probability_up": "Up probability",
        "probability_down": "Down probability", "last_report_period": "Last result",
        "last_net_profit_cr": "Last net profit (₹ cr)",
        "history_quarters": "History quarters", "missing_financial_features": "Missing features",
    })
    st.dataframe(
        table, hide_index=True, width="stretch", height=570,
        column_config={
            "Up probability": st.column_config.ProgressColumn(
                "Up probability", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Down probability": st.column_config.NumberColumn(format="%.1f%%"),
            "Last net profit (₹ cr)": st.column_config.NumberColumn(format="₹%.2f"),
        },
    )
    st.download_button(
        "Download filtered ranking (CSV)",
        data=shown.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"earnings_ranking_{target}.csv", mime="text/csv",
    )


def display_top50_market_caps(scored: pd.DataFrame, target: str) -> None:
    st.subheader("Top 50 forecasts · compare by market cap")
    st.write(
        "The model selects the 50 highest-probability companies first. "
        "Market cap then filters and sorts only that shortlist; it does not change their earnings probabilities."
    )
    top50 = scored.sort_values("rank").head(50).copy()
    if top50.empty:
        st.info("No scored companies are available for this quarter.")
        return
    if st.button("Refresh market caps for these 50", help="Read each company's current Screener profile"):
        with st.spinner("Updating market caps from Screener…"):
            _, errors = refresh_market_caps(top50["ticker"].tolist(), DEFAULT_CAPS)
        if errors:
            st.warning(f"Updated with {len(errors)} profiles unavailable. Earlier saved values were kept where available.")
        st.rerun()

    caps = load_market_caps(DEFAULT_CAPS).rename(columns={"status": "market_cap_status"})
    top50 = top50.merge(caps, on="ticker", how="left", validate="one_to_one")
    coverage = int(top50["market_cap_cr"].notna().sum())
    st.caption(
        f"Market caps available for {coverage} of {len(top50)} companies. "
        "Source: linked Screener company profiles. Values are a display-only snapshot and may differ "
        "from the earnings forecast's information cutoff."
    )
    web_snapshots = int(top50["market_cap_status"].eq("web snapshot").sum())
    if web_snapshots:
        st.caption(f"{web_snapshots} market caps came from cached public page snapshots after direct requests were rate limited.")
    if coverage:
        retrieved = pd.to_datetime(top50.loc[top50["market_cap_cr"].notna(), "retrieved_at_ist"], errors="coerce")
        if retrieved.notna().any():
            st.caption(f"Market-cap snapshot checked: {retrieved.min():%d %b %Y %H:%M}–{retrieved.max():%d %b %Y %H:%M} IST")
    if coverage < len(top50):
        st.warning("Companies without a verified market cap appear last when sorted by market cap. Refresh to retry them.")

    a, b, c = st.columns([1, 1, 1.3])
    with a:
        minimum = st.number_input("Minimum market cap (₹ cr)", min_value=0.0, value=0.0, step=500.0)
    with b:
        maximum = st.number_input("Maximum market cap (₹ cr; 0 = no limit)", min_value=0.0, value=0.0, step=500.0)
    with c:
        order = st.selectbox("Sort the top 50", ["Market cap · highest first", "Earnings probability · highest first"])
    shown = top50.copy()
    if minimum > 0:
        shown = shown.loc[shown["market_cap_cr"].ge(minimum)]
    if maximum > 0:
        shown = shown.loc[shown["market_cap_cr"].le(maximum)]
    if order.startswith("Market cap"):
        shown = shown.sort_values(["market_cap_cr", "rank"], ascending=[False, True], na_position="last")
    else:
        shown = shown.sort_values("rank")
    st.caption(f"{len(shown)} of the top {len(top50)} earnings forecasts match the market-cap range")
    if shown.empty:
        st.info("No companies match this market-cap range.")
        return

    chart = shown.loc[shown["market_cap_cr"].notna()].head(20)
    if not chart.empty:
        chart = chart.assign(**{"Market cap (₹ cr)": chart["market_cap_cr"]})
        st.bar_chart(
            chart, x="ticker", y="Market cap (₹ cr)", horizontal=True,
            sort=False, color="#4ba3bd", height=500,
        )
    table = shown[[
        "rank", "ticker", "company", "industry", "probability_up",
        "market_cap_cr", "retrieved_at_ist", "source_url",
    ]].copy()
    table["probability_up"] *= 100
    table = table.rename(columns={
        "rank": "Probability rank", "ticker": "Ticker", "company": "Company",
        "industry": "Industry", "probability_up": "Up probability",
        "market_cap_cr": "Market cap (₹ cr)", "retrieved_at_ist": "Market-cap checked (IST)",
        "source_url": "Market-cap source",
    })
    st.dataframe(
        table, hide_index=True, width="stretch", height=560,
        column_config={
            "Up probability": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Market cap (₹ cr)": st.column_config.NumberColumn(format="₹%,.0f"),
            "Market-cap source": st.column_config.LinkColumn(display_text="Screener"),
        },
    )
    st.download_button(
        "Download this market-cap shortlist (CSV)",
        data=shown.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"top50_earnings_probability_market_cap_{target}.csv",
        mime="text/csv",
    )


def display_coverage(rows: pd.DataFrame, summary: dict) -> None:
    st.subheader("All supplied tickers and coverage")
    st.write(
        "Every input ticker remains in the output. A ticker without a valid previous-quarter "
        "first filing has a reason instead of an invented probability."
    )
    counts = pd.DataFrame(
        [(reason, count) for reason, count in summary["status_counts"].items()],
        columns=["Result", "Tickers"],
    )
    st.dataframe(counts, hide_index=True, width="stretch")
    unscored = rows.loc[rows["status"].ne("Scored"), [
        "input_row", "ticker", "company", "industry", "reason", "last_report_period",
        "issuer_isin", "basis",
    ]].rename(columns={
        "input_row": "Input row", "ticker": "Ticker", "company": "Company",
        "industry": "Industry", "reason": "Why no score",
        "last_report_period": "Last eligible result", "issuer_isin": "ISIN", "basis": "Basis",
    })
    st.dataframe(unscored, hide_index=True, width="stretch", height=500)
    complete = rows.assign(
        target_period_end=summary["target_period_end"],
        as_of_ist=summary["as_of_ist"],
    )
    st.download_button(
        "Download all 2,926 input rows with scores or reasons (CSV)",
        data=complete.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"all_ticker_earnings_results_{summary['target_period_end']}.csv",
        mime="text/csv",
    )


def display_company(rows: pd.DataFrame) -> None:
    st.subheader("Company view")
    ticker = st.selectbox("Select a ticker", rows["ticker"].tolist(), index=0)
    company = rows.loc[rows["ticker"].eq(ticker)].iloc[0]
    st.markdown(f"### {company['company']} · {ticker}")
    if company["status"] != "Scored":
        st.warning(f"No model forecast: {company['reason']}")
        if company["last_report_period"]:
            st.write(f"Last eligible report: {company['last_report_period']}")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability up", f"{company['probability_up']:.1%}")
    c2.metric("Probability down", f"{company['probability_down']:.1%}")
    c3.metric("Universe rank", f"#{int(company['rank']):,}")
    c4.metric("Last reported profit", f"₹{company['last_net_profit_cr']:,.2f} cr")
    st.progress(float(company["probability_up"]))
    st.write(
        f"Latest eligible quarter: **{company['last_report_period']}** · "
        f"Accounting basis: **{company['basis']}** · "
        f"History: **{int(company['history_quarters'])} quarters** · "
        f"Missing financial features before imputation: **{int(company['missing_financial_features'])}**"
    )
    st.caption(
        f"ISIN {company['issuer_isin']} · latest first-XBRL value available "
        f"{company['last_xbrl_available_ist']}"
    )
    st.info(
        "The industry label is supplied for browsing. It is not an input to this point-in-time "
        "model. The probability concerns net-profit direction, not the share-price reaction."
    )


def render_earnings_dashboard() -> None:
    """Render the saved season immediately; run the model on demand for other cutoffs."""
    default_rows, default_summary = _load_default()
    saved_cutoff = pd.Timestamp(default_summary["as_of_ist"]).tz_convert(IST)
    now = datetime.now(ZoneInfo(IST)).replace(microsecond=0)
    current_target = previous_completed_quarter(now.date()).date()
    saved_target = pd.Timestamp(default_summary["target_period_end"]).date()
    if "earnings_target_date" not in st.session_state:
        st.session_state["earnings_target_date"] = current_target
    if "earnings_cutoff_date" not in st.session_state:
        initial = saved_cutoff if current_target == saved_target else now
        st.session_state["earnings_cutoff_date"] = initial.date()
        st.session_state["earnings_cutoff_time"] = initial.time().replace(microsecond=0)

    st.subheader("Quarterly earnings probability")
    st.caption("Chance that first-reported quarterly net profit rises from the preceding quarter; this is not a stock-return forecast.")
    a, b, c, d = st.columns([1.2, 1.2, 1, 1.2])
    with a:
        target_date = st.date_input("Target quarter end", key="earnings_target_date")
    with b:
        cutoff_date = st.date_input("Information cutoff · IST", key="earnings_cutoff_date")
    with c:
        cutoff_time = st.time_input("Cutoff time · IST", key="earnings_cutoff_time")
    with d:
        st.write("\u00a0")
        run_model = st.button("Run earnings model", type="primary", key="earnings_run_model")
    cutoff = datetime.combine(cutoff_date, cutoff_time, ZoneInfo(IST))
    selection = (target_date.isoformat(), cutoff.isoformat())

    if run_model:
        try:
            mtimes = tuple(path.stat().st_mtime_ns for path in (DEFAULT_TICKERS, DEFAULT_ARCHIVE, DEFAULT_BUNDLE))
            rows, summary = _score(*selection, mtimes)
        except Exception as exc:
            st.error(f"Could not score this season: {exc}")
            return
        st.session_state["earnings_model_result"] = (selection, rows, summary)
    stored = st.session_state.get("earnings_model_result")
    if stored is not None and stored[0] == selection:
        _, rows, summary = stored
    elif selection == (default_summary["target_period_end"], default_summary["as_of_ist"]):
        rows, summary = default_rows, default_summary
    else:
        st.info("Select the quarter and information cutoff, then click Run earnings model. The saved September 2026 snapshot is also available by selecting 30 September 2026 and 2 October 2026, 00:00 IST.")
        return

    scored = rows.loc[rows["status"].eq("Scored")].copy()
    top = scored.sort_values("rank").iloc[0] if not scored.empty else None
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Supplied tickers", f"{summary['input_count']:,}")
    m2.metric("Scored", f"{summary['scored_count']:,}")
    m3.metric("Unscored", f"{summary['unscored_count']:,}")
    m4.metric("Highest up probability", f"{top['probability_up']:.1%}" if top is not None else "—")
    latest = pd.Timestamp(summary["archive_info"]["latest_archived_knowledge_utc"]).tz_convert(IST)
    st.caption(
        f"Forecast as of {summary['as_of_ist']} · latest filing in the bundled archive "
        f"{latest:%d %b %Y %H:%M IST}. New filings require an updated archive."
    )
    views = st.tabs(["Probability ranking", "Top 50 + market cap", "Coverage", "Company", "Method"])
    with views[0]:
        display_rankings(scored, summary["target_period_end"])
    with views[1]:
        display_top50_market_caps(scored, summary["target_period_end"])
    with views[2]:
        display_coverage(rows, summary)
    with views[3]:
        display_company(rows)
    with views[4]:
        st.write(
            "The saved two-layer neural model uses 17 lagged financial and seasonality features "
            "from first-published NSE filings. Its output is calibrated on a separate period. "
            "Industry and market cap are display filters, not model inputs."
        )
        st.write(
            f"Model fit: {summary['model_training_n']:,} examples; probability calibration: "
            f"{summary['model_calibration_n']:,} examples. This archived panel contains "
            f"{summary['archive_info']['eligible_rows']:,} quality-eligible first filings."
        )
        st.caption("The model does not include current news, order books, analyst remarks, earnings beats, or stock-price reactions.")