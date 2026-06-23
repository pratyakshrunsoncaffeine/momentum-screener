from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import plotly.express as px
import streamlit as st

from screener_momentum.config import (
    DEFAULT_MOMENTUM_WEIGHTS,
    DEFAULT_POSITIVE_RETURN_FILTERS,
    FundamentalThresholds,
    ScreeningConfig,
)
from screener_momentum.pipeline import (
    load_saved_returns,
    output_paths,
    run_fundamentals_screen,
    run_momentum,
    score_and_save_momentum,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_TICKER_FILE = ROOT / "ticker.csv"
OUTPUT_DIR = ROOT / "output" / "latest"


st.set_page_config(
    page_title="Momentum Screener",
    layout="wide",
)


def build_config() -> tuple[ScreeningConfig, str]:
    st.sidebar.header("Inputs")
    uploaded = st.sidebar.file_uploader("Ticker CSV", type=["csv"])
    csv_path = str(DEFAULT_TICKER_FILE)
    if uploaded is not None:
        temp = NamedTemporaryFile(delete=False, suffix=".csv")
        temp.write(uploaded.getbuffer())
        temp.close()
        csv_path = temp.name

    st.sidebar.header("Momentum")
    weights = {}
    for label, default in DEFAULT_MOMENTUM_WEIGHTS.items():
        weights[label] = st.sidebar.number_input(
            label,
            min_value=0.0,
            max_value=1.0,
            value=float(default),
            step=0.05,
        )

    st.sidebar.header("Filters")
    top_for_fundamentals = st.sidebar.number_input("Momentum candidates for fundamentals", 10, 500, 100, 10)
    final_count = st.sidebar.number_input("Final companies", 10, 200, 100, 10)
    min_market_cap = st.sidebar.number_input("Market cap > Cr", 0.0, 1000000.0, 1500.0, 100.0)
    min_qoq_growth = st.sidebar.number_input("Quarterly revenue growth > %", -100.0, 500.0, 10.0, 1.0)
    min_yoy_growth = st.sidebar.number_input("Annual revenue growth > %", -100.0, 500.0, 15.0, 1.0)
    max_promoter_change = st.sidebar.number_input("Promoter holding change < %", 0.0, 100.0, 5.0, 0.5)
    backtest_months = st.sidebar.slider("Backtest months", 1, 36, 6)

    thresholds = FundamentalThresholds(
        min_market_cap_cr=min_market_cap,
        min_quarterly_revenue_growth_pct=min_qoq_growth,
        min_annual_revenue_growth_pct=min_yoy_growth,
        max_promoter_holding_change_pct=max_promoter_change,
    )
    config = ScreeningConfig(
        momentum_weights=weights,
        positive_return_filters=DEFAULT_POSITIVE_RETURN_FILTERS,
        fundamental_thresholds=thresholds,
        top_momentum_for_fundamentals=int(top_for_fundamentals),
        final_count=int(final_count),
        backtest_months=int(backtest_months),
    )
    return config, csv_path


def format_percent_columns(frame: pd.DataFrame) -> pd.DataFrame:
    percent_like = [column for column in frame.columns if "Return" in column or "ret" in column or column.endswith("%")]
    styled = frame.copy()
    for column in percent_like:
        if column in styled.columns:
            styled[column] = pd.to_numeric(styled[column], errors="coerce")
    return styled


def show_download(label: str, frame: pd.DataFrame, file_name: str) -> None:
    if frame.empty:
        return
    st.download_button(
        label=label,
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )


def empty_results(momentum: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "momentum": momentum,
        "fundamentals": pd.DataFrame(),
        "final": momentum.head(config.final_count).copy(),
        "backtest": pd.DataFrame(),
        "holdings": pd.DataFrame(),
        "performance": pd.DataFrame(),
    }


def current_or_saved_momentum(config: ScreeningConfig) -> pd.DataFrame | None:
    results = st.session_state.get("results")
    if results is not None and not results["momentum"].empty:
        return results["momentum"]

    paths = output_paths(OUTPUT_DIR)
    if paths["momentum"].exists():
        return pd.read_csv(paths["momentum"])
    if paths["returns"].exists():
        returns = load_saved_returns(OUTPUT_DIR)
        return score_and_save_momentum(returns, config, output_dir=OUTPUT_DIR)
    return None


def make_progress(label: str):
    st.markdown(label)
    bar = st.progress(0)
    status = st.empty()

    def update(completed: int, total: int, message: str) -> None:
        ratio = 0 if total <= 0 else min(completed / total, 1.0)
        bar.progress(ratio)
        status.write(f"{message} ({completed:,}/{total:,})")

    return update


def load_saved_run(config: ScreeningConfig) -> None:
    try:
        returns = load_saved_returns(OUTPUT_DIR)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    momentum = score_and_save_momentum(returns, config, output_dir=OUTPUT_DIR)
    st.session_state["results"] = empty_results(momentum)
    st.success(f"Loaded saved returns and rebuilt momentum for {len(momentum):,} passing stocks.")


def read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def recover_saved_results(config: ScreeningConfig) -> None:
    paths = output_paths(OUTPUT_DIR)
    momentum = read_csv_if_exists(paths["momentum"])
    if momentum.empty and paths["returns"].exists():
        momentum = score_and_save_momentum(load_saved_returns(OUTPUT_DIR), config, output_dir=OUTPUT_DIR)
    if momentum.empty:
        st.error("No saved momentum or returns file found yet.")
        return

    fundamentals = read_csv_if_exists(paths["fundamentals"])
    if fundamentals.empty:
        fundamentals = read_csv_if_exists(paths["fundamentals_partial"])

    final = read_csv_if_exists(paths["final"])
    if final.empty and not fundamentals.empty and "Fundamental Pass" in fundamentals.columns:
        pass_mask = fundamentals["Fundamental Pass"]
        if pass_mask.dtype == object:
            pass_mask = pass_mask.astype(str).str.lower().isin(("true", "1", "yes"))
        else:
            pass_mask = pass_mask.astype(bool)
        final = fundamentals[pass_mask].copy()
    if final.empty:
        final = momentum.head(config.final_count).copy()

    backtest = read_csv_if_exists(paths["backtest"], index_col=0, parse_dates=True)
    holdings = read_csv_if_exists(paths["holdings"])
    performance = read_csv_if_exists(paths["performance"])
    st.session_state["results"] = {
        "momentum": momentum,
        "fundamentals": fundamentals,
        "final": final,
        "backtest": backtest,
        "holdings": holdings,
        "performance": performance,
    }
    st.success("Recovered saved screener files from output/latest.")


config, csv_path = build_config()

st.title("Momentum Screener")
st.caption("NSE momentum from yfinance, fundamentals from Screener.in, and a top-10 portfolio comparison.")

paths = output_paths(OUTPUT_DIR)
saved_returns_available = paths["returns"].exists()
saved_momentum_available = paths["momentum"].exists()

st.caption(
    f"Saved cache: returns {'available' if saved_returns_available else 'missing'}, "
    f"momentum {'available' if saved_momentum_available else 'missing'}."
)

first, second, third, fourth, fifth, sixth = st.columns([1.2, 1.1, 1.2, 1, 1, 0.8])
refresh_momentum = first.button("Refresh Momentum Data", type="primary", use_container_width=True)
use_saved_momentum = second.button("Use Saved Momentum", use_container_width=True)
run_fundamentals = third.button(f"Run Fundamentals on Top {config.top_momentum_for_fundamentals}", use_container_width=True)
skip_fundamentals = fourth.button("Skip Fundamentals", use_container_width=True)
recover_saved = fifth.button("Recover Saved Run", use_container_width=True)
clear_cache = sixth.button("Clear UI Cache", use_container_width=True)

if clear_cache:
    st.cache_data.clear()
    st.toast("Cache cleared.")

if not Path(csv_path).exists():
    st.error(f"Ticker CSV not found: {csv_path}")
    st.stop()

if refresh_momentum:
    progress = make_progress("Downloading yfinance data")
    momentum = run_momentum(
        csv_path,
        config,
        progress_callback=progress,
        output_dir=OUTPUT_DIR,
        use_saved_returns=False,
    )
    st.session_state["results"] = empty_results(momentum)
    st.success(f"Momentum complete: {len(momentum):,} stocks passed the short-term return filter.")

if use_saved_momentum:
    load_saved_run(config)

if recover_saved:
    recover_saved_results(config)

if skip_fundamentals:
    momentum = current_or_saved_momentum(config)
    if momentum is None:
        st.error("Run momentum first or use saved momentum before skipping fundamentals.")
    else:
        st.session_state["results"] = run_fundamentals_screen(
            momentum,
            config,
            include_fundamentals=False,
            output_dir=OUTPUT_DIR,
        )
        st.success("Built final list directly from momentum and refreshed the portfolio backtest.")

if run_fundamentals:
    momentum = current_or_saved_momentum(config)
    if momentum is None:
        st.error("Run momentum first or use saved momentum before running fundamentals.")
    else:
        progress = make_progress("Scraping Screener.in fundamentals")
        st.session_state["results"] = run_fundamentals_screen(
            momentum,
            config,
            include_fundamentals=True,
            progress_callback=progress,
            output_dir=OUTPUT_DIR,
        )
        st.success("Fundamentals complete and final portfolio backtest refreshed.")

results = st.session_state.get("results")
if results is None:
    st.info("Refresh momentum data, or use saved momentum if a previous run exists.")
    st.stop()

momentum = results["momentum"]
fundamentals = results["fundamentals"]
final = results["final"]
backtest = results["backtest"]
holdings = results["holdings"]
performance = results["performance"]

metric_cols = st.columns(4)
metric_cols[0].metric("Momentum Pass", f"{len(momentum):,}")
metric_cols[1].metric("Fundamental Rows", f"{len(fundamentals):,}" if not fundamentals.empty else "Skipped")
metric_cols[2].metric("Final List", f"{len(final):,}")
metric_cols[3].metric("Top Score", f"{final['Momentum Score'].max():.2f}" if not final.empty else "NA")

tabs = st.tabs(["Final Screener", "Momentum", "Fundamentals", "Portfolio"])

with tabs[0]:
    st.subheader("Final Momentum List")
    st.dataframe(format_percent_columns(final), use_container_width=True, hide_index=True)
    show_download("Download final list", final, "final_momentum_screener.csv")

with tabs[1]:
    st.subheader("Momentum Candidates")
    st.dataframe(format_percent_columns(momentum), use_container_width=True, hide_index=True)
    show_download("Download momentum candidates", momentum, "momentum_candidates.csv")
    st.caption(f"Saved at {paths['momentum']}")

with tabs[2]:
    st.subheader("Screener.in Fundamentals")
    if fundamentals.empty:
        st.info("Fundamentals were skipped for this run.")
    else:
        st.dataframe(format_percent_columns(fundamentals), use_container_width=True, hide_index=True)
        show_download("Download fundamentals", fundamentals, "fundamentals_screen.csv")
        st.caption(f"Partial checkpoints are written to {paths['fundamentals_partial']}")

with tabs[3]:
    st.subheader("1L Portfolio Backtest")
    if backtest.empty:
        st.info("Run the full screener with at least one final company to build the backtest.")
    else:
        chart_frame = backtest.reset_index(names="Date").melt("Date", var_name="Series", value_name="Value")
        fig = px.line(chart_frame, x="Date", y="Value", color="Series")
        fig.update_layout(yaxis_title="Portfolio Value", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            st.markdown("Portfolio allocations")
            st.dataframe(holdings, use_container_width=True, hide_index=True)
        with cols[1]:
            st.markdown("Performance")
            st.dataframe(performance, use_container_width=True, hide_index=True)
