from __future__ import annotations

from pathlib import Path
from datetime import date
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .swing import SwingConfig, _cache_path, clean_daily, indicators, load_swing_results, run_swing_scan


def render_swing_dashboard(ticker_csv: str | Path, output_dir: str | Path) -> None:
    st.subheader("Evening Swing Screener")
    st.write("Run after 8 p.m. IST to rank long-only setups for the next session and a 5–10-session holding period. Entry is conditional on price reaching the displayed trigger within the maximum entry price.")
    st.caption("Setup scores are configurable research rules, not probabilities. Missing or stale data cannot qualify. Results may legitimately contain no picks.")
    controls = st.columns(4)
    top_n = controls[0].number_input("Maximum picks", 1, 30, 10, key="swing_top_n")
    liquidity = controls[1].number_input("Minimum daily traded value (₹ crore)", 0.1, 1000.0, 5.0, key="swing_liquidity")
    rvol = controls[2].number_input("Breakout volume / prior 20-day average", 1.0, 5.0, 1.5, 0.1, key="swing_rvol")
    score = controls[3].number_input("Minimum setup score", 0.0, 100.0, 60.0, 1.0, key="swing_score")
    with st.expander("VWAP, entry costs and event controls"):
        require_vwap = st.checkbox("Require complete-session VWAP confirmation", True, key="swing_require_vwap")
        st.caption("VWAP is estimated from 5-minute session bars for up to 200 daily-qualified candidates. The 20-day daily price/volume proxy is labeled separately. Unavailable intraday data is never treated as confirmation.")
        cost = st.number_input("Estimated total entry + exit costs (%)", 0.0, 5.0, 0.30, 0.05, key="swing_cost")
        event_upload = st.file_uploader("Optional known-event CSV (Ticker, Date, Event)", type=["csv"], key="swing_events")
        st.caption("Use YYYY-MM-DD dates. Known events within 14 calendar days are excluded. Stocks without an uploaded event remain unverified; check NSE/BSE announcements before entering.")
    config = SwingConfig(top_n=int(top_n), min_turnover_cr=float(liquidity), breakout_rvol=float(rvol), min_score=float(score), require_session_vwap=require_vwap, estimated_roundtrip_cost_pct=float(cost))
    buttons = st.columns(3)
    scan = buttons[0].button("Run evening swing scan", type="primary", key="swing_run")
    load = buttons[1].button("Load saved swing scan", key="swing_load")
    refresh = buttons[2].checkbox("Refresh all daily data", False, key="swing_refresh")
    if scan:
        bar, status = st.progress(0.0), st.empty()

        def progress(done, total, message):
            bar.progress(min(done / max(total, 1), 1.0))
            status.text(message)

        try:
            events_path = None
            if event_upload is not None:
                events_path = Path(output_dir) / "uploaded_events.csv"
                events_path.parent.mkdir(parents=True, exist_ok=True)
                events_path.write_bytes(event_upload.getvalue())
            with st.spinner("Screening the ticker list and checking shortlist VWAP…"):
                result = run_swing_scan(ticker_csv, config, output_dir, resume=not refresh, events_csv=events_path, callback=progress)
            st.session_state["swing_results"] = result
            bar.progress(1.0)
            status.text("Scan finished")
        except Exception as exc:
            st.error(f"Swing scan could not finish: {exc}")
    if load:
        try:
            st.session_state["swing_results"] = load_swing_results(output_dir)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
    result = st.session_state.get("swing_results")
    if result is None and (Path(output_dir) / "metadata.json").exists():
        try:
            result = load_swing_results(output_dir)
        except (OSError, ValueError):
            pass
    if result is not None:
        m = result["metadata"]
        st.caption(f"Data session: {m['session']} • Run: {m['run_time_ist']} • Rule version: {m['rule_version']}. Saved results retain the settings used when scanned; changed controls apply on the next scan.")
        cards = st.columns(4)
        for column, label, value in zip(cards, ["Tickers attempted", "Fresh usable histories", "Technical setups", "Final picks"], [m['universe_count'], m['usable_count'], m['technical_pass_count'], m['final_count']]):
            column.metric(label, f"{value:,}")
        if m["market_regime"] == "Cautious":
            st.warning("NIFTY trend is cautious. Qualified stocks outperform the index, but broader market pressure can still defeat a bullish setup.")
        views = st.tabs(["Top swing setups", "Persistent watchlist", "Industry pockets", "All technical metrics", "Excluded setups", "Data coverage"])
        compact = ["Swing Rank", "Ticker", "Name", "Industry", "Setup", "Setup Score", "Entry Trigger", "Max Entry", "Initial Stop", "Target", "Stop Distance %", "RSI14", "RVOL20", "Excess Return20 %", "Session VWAP5m", "Session VWAP Status", "Event Status", "Why Selected"]
        for tab, key in zip(views, ["picks", "watchlist", "pockets", "metrics", "rejected", "health"]):
            with tab:
                frame = result[key]
                if frame.empty:
                    st.info("No qualifying rows in this view. The scan does not force a shortlist.")
                else:
                    displayed = frame[[c for c in compact if c in frame]] if key == "picks" else frame
                    st.dataframe(displayed, width="stretch", hide_index=True)
                    st.download_button(f"Download {key}", frame.to_csv(index=False).encode("utf-8"), file_name=f"swing_{key}_{m['session']}.csv", mime="text/csv", key=f"swing_download_{key}")
        st.caption(f"Industry classification exists for {m.get('classified_usable_count', 'some')} usable stocks. Pockets use available CSV labels, not official NSE sector-index classifications. Unknown and groups smaller than five stocks are marked unrankable. Check stock identity and exchange restrictions before entry.")
        if not result["picks"].empty:
            with st.expander("Inspect a shortlisted setup"):
                ticker = st.selectbox("Stock", result["picks"]["Ticker"].tolist(), key="swing_chart_ticker")
                pick = result["picks"].set_index("Ticker").loc[ticker]
                cached = _cache_path(Path(output_dir) / "daily_cache", pick["YFinance Ticker"])
                if cached.exists():
                    frame = clean_daily(pd.read_csv(cached, index_col=0, parse_dates=True), date.fromisoformat(m["session"]))
                    data = indicators(frame).tail(90)
                    chart = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.025, row_heights=[.52,.16,.16,.16])
                    chart.add_trace(go.Candlestick(x=data.index, open=data.Open, high=data.High, low=data.Low, close=data.Close, name="Price"), row=1,col=1)
                    for column, color in [("EMA10","#797979"),("EMA20","#3a7bd5"),("EMA50","#a769b7"),("Daily VWAP Proxy20","#b8860b")]:
                        chart.add_trace(go.Scatter(x=data.index, y=data[column], name=column, line=dict(color=color,width=1.2)), row=1,col=1)
                    for label, color in [("Entry Trigger","#238c48"),("Max Entry","#9b8b18"),("Initial Stop","#c34949"),("Target","#2d9d98")]:
                        chart.add_hline(y=float(pick[label]), line_color=color, line_dash="dot", annotation_text=label, row=1,col=1)
                    colors = ["#4c956c" if c >= o else "#bf6666" for c,o in zip(data.Close,data.Open)]
                    chart.add_trace(go.Bar(x=data.index, y=data.Volume, marker_color=colors, name="Volume"),row=2,col=1)
                    chart.add_trace(go.Bar(x=data.index,y=data["MACD Histogram"],name="MACD histogram",marker_color="#a0a0a0"),row=3,col=1)
                    for column in ["MACD","MACD Signal"]:
                        chart.add_trace(go.Scatter(x=data.index,y=data[column],name=column),row=3,col=1)
                    chart.add_trace(go.Scatter(x=data.index,y=data.RSI14,name="RSI14",line_color="#3a7bd5"),row=4,col=1)
                    chart.add_hline(y=50,line_dash="dot",line_color="#999999",row=4,col=1)
                    chart.add_hline(y=70,line_dash="dot",line_color="#999999",row=4,col=1)
                    chart.update_layout(height=720,xaxis_rangeslider_visible=False,template="plotly_white",title=f"{ticker} • {pick['Setup']} • daily bars through {m['session']}",legend=dict(orientation="h",y=1.1))
                    chart.update_yaxes(title_text="₹",row=1,col=1)
                    st.plotly_chart(chart, width="stretch")
                    st.caption("The gold daily VWAP proxy line spans 20 sessions. Session VWAP from 5-minute bars is a separate value in the results table. Entry, stop and target lines show a conditional plan, not a trade already entered.")
    with st.expander("Selected indicators, exact rules and research sources"):
        rules = Path(__file__).resolve().parents[1] / "docs" / "swing_rules.md"
        if rules.exists():
            st.markdown(rules.read_text(encoding="utf-8"))
