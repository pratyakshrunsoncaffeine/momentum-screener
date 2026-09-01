from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import plotly.express as px
import streamlit as st

from screener_momentum.backtest import current_allocation, performance_summary
from screener_momentum.correlation import (
    FACTOR_CATALOG,
    correlation_matrix,
    normalized_factor_stock_performance,
    select_relationship_leaders,
)
from screener_momentum.config import (
    DEFAULT_POST_EARNINGS_STOCK_RETURN_WEIGHTS,
    DEFAULT_MOMENTUM_WEIGHTS,
    DEFAULT_POSITIVE_RETURN_FILTERS,
    DerivativesSignalConfig,
    FundamentalThresholds,
    ScreeningConfig,
    Sma200ScanConfig,
)
from screener_momentum.index_momentum import (
    DEFAULT_INDEX_MOMENTUM_WEIGHTS,
    INDEX_MOMENTUM_PERIODS,
    index_catalogue_frame,
    normalized_index_performance,
)
from screener_momentum.pipeline import (
    download_derivatives_eod_data,
    finalize_dii_momentum_screen,
    finalize_fii_momentum_screen,
    finalize_quarterly_results_screen,
    load_saved_correlation,
    load_saved_sma200_results,
    load_saved_index_momentum,
    load_saved_custom_stock_momentum,
    load_saved_returns,
    load_saved_derivatives_results,
    output_paths,
    prepare_dii_all,
    prepare_fii_all,
    prepare_quarterly_results,
    run_dii_momentum_screen,
    run_derivatives_backtest_screen,
    run_derivatives_momentum_screen,
    run_fii_momentum_screen,
    run_fundamentals_screen,
    run_quarterly_results_screen,
    run_quarterly_stock_return_momentum,
    run_correlation_screen,
    run_index_momentum_screen,
    rescore_saved_index_momentum,
    reset_index_momentum_scan,
    run_custom_stock_momentum,
    run_sma200_screen,
    run_momentum,
    reset_derivatives_eod_data,
    reset_dii_momentum_scan,
    reset_fii_momentum_scan,
    reset_quarterly_results_scan,
    reset_sma200_scan,
    score_and_save_momentum,
)
from screener_momentum.sma200 import sma200_chart_data


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
        "normalized_backtest": pd.DataFrame(),
        "periods": pd.DataFrame(),
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
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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
    normalized = read_csv_if_exists(paths["normalized_backtest"], index_col=0, parse_dates=True)
    periods = read_csv_if_exists(paths["walk_forward_periods"])
    holdings = read_csv_if_exists(paths["holdings"])
    performance = read_csv_if_exists(paths["performance"])
    st.session_state["results"] = {
        "momentum": momentum,
        "fundamentals": fundamentals,
        "final": final,
        "backtest": backtest,
        "normalized_backtest": normalized,
        "periods": periods,
        "holdings": holdings,
        "performance": performance,
    }
    st.success("Recovered saved screener files from output/latest.")


def latest_fii_source_path(paths: dict[str, Path]) -> Path | None:
    return latest_institutional_source_path(paths, "fii")


def latest_dii_source_path(paths: dict[str, Path]) -> Path | None:
    return latest_institutional_source_path(paths, "dii")


def latest_institutional_source_path(paths: dict[str, Path], prefix: str) -> Path | None:
    candidates = [paths[f"{prefix}_partial"], paths[f"{prefix}_marketcap_partial"], paths[f"{prefix}_all"]]
    existing = [path for path in candidates if path.exists() and path.stat().st_size > 0]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def load_saved_fii_preview() -> dict[str, pd.DataFrame]:
    return load_saved_institutional_preview("fii", prepare_fii_all)


def load_saved_dii_preview() -> dict[str, pd.DataFrame]:
    return load_saved_institutional_preview("dii", prepare_dii_all)


def load_saved_institutional_preview(prefix: str, prepare_func) -> dict[str, pd.DataFrame]:
    paths = output_paths(OUTPUT_DIR)
    source = latest_institutional_source_path(paths, prefix)
    all_scan = read_csv_if_exists(source) if source else pd.DataFrame()
    if not all_scan.empty:
        all_scan = prepare_func(all_scan)
    return {
        f"{prefix}_all": all_scan,
        f"{prefix}_top": read_csv_if_exists(paths[f"{prefix}_top"]),
        f"{prefix}_momentum": read_csv_if_exists(paths[f"{prefix}_momentum"]),
        f"{prefix}_final": read_csv_if_exists(paths[f"{prefix}_final"]),
    }


def recover_saved_fii_results(
    config: ScreeningConfig,
    fii_top_n: int,
    fii_final_n: int,
    progress_callback=None,
) -> None:
    recover_saved_institutional_results(
        prefix="fii",
        label="FII",
        config=config,
        top_n=fii_top_n,
        final_n=fii_final_n,
        finalizer=finalize_fii_momentum_screen,
        state_key="fii_results",
        progress_callback=progress_callback,
    )


def recover_saved_dii_results(
    config: ScreeningConfig,
    dii_top_n: int,
    dii_final_n: int,
    progress_callback=None,
) -> None:
    recover_saved_institutional_results(
        prefix="dii",
        label="DII",
        config=config,
        top_n=dii_top_n,
        final_n=dii_final_n,
        finalizer=finalize_dii_momentum_screen,
        state_key="dii_results",
        progress_callback=progress_callback,
    )


def recover_saved_institutional_results(
    prefix: str,
    label: str,
    config: ScreeningConfig,
    top_n: int,
    final_n: int,
    finalizer,
    state_key: str,
    progress_callback=None,
) -> None:
    paths = output_paths(OUTPUT_DIR)
    source = latest_institutional_source_path(paths, prefix)
    all_scan = read_csv_if_exists(source) if source else pd.DataFrame()
    if all_scan.empty:
        st.error(f"No saved {label} scan files found yet.")
        return

    if "Market Cap Cr" not in all_scan.columns or all_scan["Market Cap Cr"].isna().all():
        st.warning(
            f"The newest saved {label} scan does not contain market cap. Run or resume the {label} scan once with the updated scraper."
        )

    results = finalizer(
        all_scan,
        config=config,
        **{f"{prefix}_top_n": top_n},
        final_n=final_n,
        price_progress_callback=progress_callback,
        output_dir=OUTPUT_DIR,
    )
    if all(frame.empty for frame in results.values()):
        st.error(f"No saved {label} scan files found yet.")
        return
    st.session_state[state_key] = results
    st.success(f"Recovered and finalized saved {label} scan from {source.name}.")


def latest_quarterly_source_path(paths: dict[str, Path]) -> Path | None:
    candidates = [paths["quarterly_results_partial"], paths["quarterly_results_all"]]
    existing = [path for path in candidates if path.exists() and path.stat().st_size > 0]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def load_saved_quarterly_preview(target_period: str) -> dict[str, pd.DataFrame]:
    paths = output_paths(OUTPUT_DIR)
    source = latest_quarterly_source_path(paths)
    all_scan = read_csv_if_exists(source) if source else pd.DataFrame()
    return {
        "quarterly_all": all_scan,
        "quarterly_matching": prepare_quarterly_results(all_scan, target_period=target_period),
    }


def recover_saved_quarterly_results(target_period: str) -> None:
    paths = output_paths(OUTPUT_DIR)
    source = latest_quarterly_source_path(paths)
    all_scan = read_csv_if_exists(source) if source else pd.DataFrame()
    if all_scan.empty:
        st.error("No saved quarterly-results scan files found yet.")
        return
    matching_target = prepare_quarterly_results(all_scan, target_period=target_period, matching_only=False)
    if matching_target.empty:
        st.warning("The saved scan belongs to a different result quarter. Run a new scan for this quarter.")
        return
    results = finalize_quarterly_results_screen(all_scan, target_period=target_period, output_dir=OUTPUT_DIR)
    st.session_state["quarterly_results"] = results
    st.success(f"Recovered saved quarterly results from {source.name}.")


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
    momentum = pd.DataFrame()
    fundamentals = pd.DataFrame()
    final = pd.DataFrame()
    backtest = pd.DataFrame()
    normalized_backtest = pd.DataFrame()
    periods = pd.DataFrame()
    holdings = pd.DataFrame()
    performance = pd.DataFrame()
else:
    momentum = results["momentum"]
    fundamentals = results["fundamentals"]
    final = results["final"]
    backtest = results["backtest"]
    normalized_backtest = results.get("normalized_backtest", pd.DataFrame())
    periods = results.get("periods", pd.DataFrame())
    holdings = results.get("holdings", pd.DataFrame())
    performance = results["performance"]

metric_cols = st.columns(4)
metric_cols[0].metric("Momentum Pass", f"{len(momentum):,}")
metric_cols[1].metric("Fundamental Rows", f"{len(fundamentals):,}" if not fundamentals.empty else "Skipped")
metric_cols[2].metric("Final List", f"{len(final):,}")
metric_cols[3].metric("Top Score", f"{final['Momentum Score'].max():.2f}" if not final.empty else "NA")

tabs = st.tabs(
    [
        "Final Screener",
        "Momentum",
        "Fundamentals",
        "Portfolio",
        "FII Accumulation",
        "DII Accumulation",
        "Quarterly Results",
        "Derivatives Momentum",
        "Index Momentum",
        "CSV Stock Momentum",
        "Correlation",
        "200DMA Finder",
    ]
)

with tabs[0]:
    st.subheader("Final Momentum List")
    if final.empty:
        st.info("No final momentum list available yet.")
    else:
        st.dataframe(format_percent_columns(final), use_container_width=True, hide_index=True)
        show_download("Download final list", final, "final_momentum_screener.csv")

with tabs[1]:
    st.subheader("Momentum Candidates")
    if momentum.empty:
        st.info("No momentum candidates available yet.")
    else:
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
    st.subheader("Walk-Forward Portfolio Backtest")
    investment_amount = st.number_input(
        "Investment amount",
        min_value=500.0,
        value=100000.0,
        step=500.0,
        help="Used to scale allocation and backtest values. Example: enter 7500 for a Rs. 7,500 portfolio.",
    )
    st.caption(
        "Method: walk-forward price backtest with monthly rebalancing. Signals use prices available before each "
        "rebalance date. When fundamentals are applied, this uses the current fundamentals-passed universe, which "
        "still has current-data bias."
    )
    if fundamentals.empty:
        st.warning("Fundamentals are not applied in this run. This portfolio backtest is momentum-only.")

    allocation = current_allocation(final, capital=investment_amount)
    if not allocation.empty:
        st.markdown("Current allocation")
        st.dataframe(allocation, use_container_width=True, hide_index=True)
        show_download("Download current allocation", allocation, "current_allocation.csv")

    if backtest.empty:
        st.info("Run fundamentals or skip fundamentals with at least one final company to build the walk-forward backtest.")
    else:
        scaled_backtest = (backtest / 100000.0) * float(investment_amount)
        normalized_view = normalized_backtest if not normalized_backtest.empty else (backtest / 100000.0) * 100.0
        view = st.radio("Backtest view", ["Actual amount", "Rs. 100 normalized"], horizontal=True)
        chart_source = scaled_backtest if view == "Actual amount" else normalized_view
        chart_frame = chart_source.reset_index(names="Date").melt("Date", var_name="Series", value_name="Value")
        fig = px.line(chart_frame, x="Date", y="Value", color="Series")
        fig.update_layout(yaxis_title="Portfolio Value" if view == "Actual amount" else "Value from Rs. 100", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            st.markdown("Rebalance periods")
            if periods.empty:
                st.info("No period table available for this run.")
            else:
                st.dataframe(periods, use_container_width=True, hide_index=True)
                show_download("Download rebalance periods", periods, "walk_forward_periods.csv")
        with cols[1]:
            st.markdown("Performance")
            scaled_performance = performance_summary(scaled_backtest)
            st.dataframe(scaled_performance if not scaled_performance.empty else performance, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("FII Accumulation + Momentum")
    st.caption(
        "This scanner scrapes FII holding change for the full ticker universe, ranks positive FII accumulation, "
        "then momentum-scores only the top FII shortlist."
    )
    controls = st.columns([1, 1, 1, 1, 1])
    fii_top_n = controls[0].number_input("FII shortlist", min_value=10, max_value=200, value=50, step=5)
    fii_final_n = controls[1].number_input("Final picks", min_value=1, max_value=20, value=3, step=1)
    run_fii = controls[2].button("Run / Resume FII Scan", type="primary", use_container_width=True)
    restart_fii = controls[3].button(
        "Run Full Scan From Beginning",
        key="restart_fii_scan",
        use_container_width=True,
    )
    recover_fii = controls[4].button("Use Saved FII Scan", use_container_width=True)

    if run_fii or restart_fii:
        fii_progress = make_progress("Scraping Screener.in FII holdings and market caps")
        price_progress = make_progress("Fetching shortlist prices")
        try:
            if restart_fii:
                reset_fii_momentum_scan(output_dir=OUTPUT_DIR)
                st.session_state.pop("fii_results", None)
            st.session_state["fii_results"] = run_fii_momentum_screen(
                csv_path,
                config,
                fii_top_n=int(fii_top_n),
                final_n=int(fii_final_n),
                progress_callback=fii_progress,
                price_progress_callback=price_progress,
                output_dir=OUTPUT_DIR,
            )
            st.success("Fresh full FII scan complete." if restart_fii else "FII accumulation scan complete.")
        except Exception as exc:
            st.error(f"FII scan stopped before finalization: {exc}")
            saved_preview = load_saved_fii_preview()
            if not saved_preview["fii_all"].empty:
                st.session_state["fii_results"] = saved_preview
                st.warning("Loaded the latest saved FII checkpoint. Use Saved FII Scan can finalize it.")

    if recover_fii:
        recover_progress = make_progress("Finalizing saved FII scan")
        recover_saved_fii_results(
            config=config,
            fii_top_n=int(fii_top_n),
            fii_final_n=int(fii_final_n),
            progress_callback=recover_progress,
        )

    if "fii_results" not in st.session_state:
        saved_preview = load_saved_fii_preview()
        if not saved_preview["fii_all"].empty:
            st.session_state["fii_results"] = saved_preview

    fii_results = st.session_state.get("fii_results", {})
    fii_all = fii_results.get("fii_all", pd.DataFrame())
    fii_top = fii_results.get("fii_top", pd.DataFrame())
    fii_momentum = fii_results.get("fii_momentum", pd.DataFrame())
    fii_final = fii_results.get("fii_final", pd.DataFrame())
    if not fii_all.empty and "Market Cap Cr" in fii_all.columns:
        fii_all["Market Cap Cr"] = pd.to_numeric(fii_all["Market Cap Cr"], errors="coerce")
        fii_all = fii_all.sort_values("Market Cap Cr", ascending=False, na_position="last").reset_index(drop=True)

    fii_metrics = st.columns(4)
    fii_metrics[0].metric("Companies Scanned", f"{len(fii_all):,}" if not fii_all.empty else "0")
    positive_count = int(pd.to_numeric(fii_all.get("FII Holding Change %", pd.Series(dtype=float)), errors="coerce").gt(0).sum()) if not fii_all.empty else 0
    fii_metrics[1].metric("Positive FII Change", f"{positive_count:,}")
    fii_metrics[2].metric("FII Shortlist", f"{len(fii_top):,}" if not fii_top.empty else "0")
    fii_metrics[3].metric("Final Picks", f"{len(fii_final):,}" if not fii_final.empty else "0")

    fii_tabs = st.tabs(["Final Top Picks", "Top FII Change", "Momentum on FII Shortlist", "All FII Scan"])
    with fii_tabs[0]:
        if fii_final.empty:
            st.info("Run or recover an FII scan to see final picks.")
        else:
            st.dataframe(format_percent_columns(fii_final), use_container_width=True, hide_index=True)
            show_download("Download FII final picks", fii_final, "fii_final.csv")
    with fii_tabs[1]:
        if fii_top.empty:
            st.info("No positive FII shortlist available yet.")
        else:
            st.dataframe(format_percent_columns(fii_top), use_container_width=True, hide_index=True)
            show_download("Download top FII change", fii_top, "fii_top50.csv")
    with fii_tabs[2]:
        if fii_momentum.empty:
            st.info("No momentum-scored FII shortlist available yet.")
        else:
            st.dataframe(format_percent_columns(fii_momentum), use_container_width=True, hide_index=True)
            show_download("Download FII momentum", fii_momentum, "fii_momentum.csv")
    with fii_tabs[3]:
        if fii_all.empty:
            st.info("No full FII scan available yet. Full-universe scans can take a while because Screener.in is scraped company by company.")
        else:
            st.dataframe(format_percent_columns(fii_all), use_container_width=True, hide_index=True)
            show_download("Download all FII scan", fii_all, "fii_all.csv")
            st.caption(f"Partial checkpoints are written to {paths['fii_partial']}. Use Saved FII Scan finalizes the latest saved checkpoint.")

with tabs[5]:
    st.subheader("DII Accumulation + Momentum")
    st.caption(
        "This scanner scrapes DII holding change for the full ticker universe, ranks positive DII accumulation, "
        "then momentum-scores only the top DII shortlist."
    )
    controls = st.columns([1, 1, 1, 1, 1])
    dii_top_n = controls[0].number_input("DII shortlist", min_value=10, max_value=200, value=50, step=5)
    dii_final_n = controls[1].number_input("DII final picks", min_value=1, max_value=20, value=3, step=1)
    run_dii = controls[2].button("Run / Resume DII Scan", type="primary", use_container_width=True)
    restart_dii = controls[3].button(
        "Run Full Scan From Beginning",
        key="restart_dii_scan",
        use_container_width=True,
    )
    recover_dii = controls[4].button("Use Saved DII Scan", use_container_width=True)

    if run_dii or restart_dii:
        dii_progress = make_progress("Scraping Screener.in DII holdings and market caps")
        price_progress = make_progress("Fetching DII shortlist prices")
        try:
            if restart_dii:
                reset_dii_momentum_scan(output_dir=OUTPUT_DIR)
                st.session_state.pop("dii_results", None)
            st.session_state["dii_results"] = run_dii_momentum_screen(
                csv_path,
                config,
                dii_top_n=int(dii_top_n),
                final_n=int(dii_final_n),
                progress_callback=dii_progress,
                price_progress_callback=price_progress,
                output_dir=OUTPUT_DIR,
            )
            st.success("Fresh full DII scan complete." if restart_dii else "DII accumulation scan complete.")
        except Exception as exc:
            st.error(f"DII scan stopped before finalization: {exc}")
            saved_preview = load_saved_dii_preview()
            if not saved_preview["dii_all"].empty:
                st.session_state["dii_results"] = saved_preview
                st.warning("Loaded the latest saved DII checkpoint. Use Saved DII Scan can finalize it.")

    if recover_dii:
        recover_progress = make_progress("Finalizing saved DII scan")
        recover_saved_dii_results(
            config=config,
            dii_top_n=int(dii_top_n),
            dii_final_n=int(dii_final_n),
            progress_callback=recover_progress,
        )

    if "dii_results" not in st.session_state:
        saved_preview = load_saved_dii_preview()
        if not saved_preview["dii_all"].empty:
            st.session_state["dii_results"] = saved_preview

    dii_results = st.session_state.get("dii_results", {})
    dii_all = dii_results.get("dii_all", pd.DataFrame())
    dii_top = dii_results.get("dii_top", pd.DataFrame())
    dii_momentum = dii_results.get("dii_momentum", pd.DataFrame())
    dii_final = dii_results.get("dii_final", pd.DataFrame())
    if not dii_all.empty and "Market Cap Cr" in dii_all.columns:
        dii_all["Market Cap Cr"] = pd.to_numeric(dii_all["Market Cap Cr"], errors="coerce")
        dii_all = dii_all.sort_values("Market Cap Cr", ascending=False, na_position="last").reset_index(drop=True)

    dii_metrics = st.columns(4)
    dii_metrics[0].metric("Companies Scanned", f"{len(dii_all):,}" if not dii_all.empty else "0")
    positive_count = int(pd.to_numeric(dii_all.get("DII Holding Change %", pd.Series(dtype=float)), errors="coerce").gt(0).sum()) if not dii_all.empty else 0
    dii_metrics[1].metric("Positive DII Change", f"{positive_count:,}")
    dii_metrics[2].metric("DII Shortlist", f"{len(dii_top):,}" if not dii_top.empty else "0")
    dii_metrics[3].metric("Final Picks", f"{len(dii_final):,}" if not dii_final.empty else "0")

    dii_tabs = st.tabs(["Final Top Picks", "Top DII Change", "Momentum on DII Shortlist", "All DII Scan"])
    with dii_tabs[0]:
        if dii_final.empty:
            st.info("Run or recover a DII scan to see final picks.")
        else:
            st.dataframe(format_percent_columns(dii_final), use_container_width=True, hide_index=True)
            show_download("Download DII final picks", dii_final, "dii_final.csv")
    with dii_tabs[1]:
        if dii_top.empty:
            st.info("No positive DII shortlist available yet.")
        else:
            st.dataframe(format_percent_columns(dii_top), use_container_width=True, hide_index=True)
            show_download("Download top DII change", dii_top, "dii_top50.csv")
    with dii_tabs[2]:
        if dii_momentum.empty:
            st.info("No momentum-scored DII shortlist available yet.")
        else:
            st.dataframe(format_percent_columns(dii_momentum), use_container_width=True, hide_index=True)
            show_download("Download DII momentum", dii_momentum, "dii_momentum.csv")
    with dii_tabs[3]:
        if dii_all.empty:
            st.info("No full DII scan available yet. Full-universe scans can take a while because Screener.in is scraped company by company.")
        else:
            st.dataframe(format_percent_columns(dii_all), use_container_width=True, hide_index=True)
            show_download("Download all DII scan", dii_all, "dii_all.csv")
            st.caption(f"Partial checkpoints are written to {paths['dii_partial']}. Use Saved DII Scan finalizes the latest saved checkpoint.")

with tabs[6]:
    st.subheader("Quarterly Results Growth Scanner")
    st.caption(
        "Scans the full ticker universe on Screener.in. It keeps companies whose quarterly table contains the "
        "selected result period, then compares Sales, Operating Profit, Net Profit, and EPS with the previous "
        "quarter and the same quarter one year earlier."
    )
    controls = st.columns([1.1, 1.4, 1, 1, 1])
    target_quarter = controls[0].text_input("Result quarter", value="Jun 2026", help="Use the quarter label shown by Screener.in, for example Jun 2026.")
    ranking_metric = controls[1].radio(
        "Rank by YoY growth",
        ["Sales", "Operating Profit", "Net Profit", "EPS"],
        horizontal=True,
    )
    run_quarterly = controls[2].button("Run / Resume Quarterly Scan", type="primary", use_container_width=True)
    restart_quarterly = controls[3].button(
        "Run Full Scan From Beginning",
        key="restart_quarterly_scan",
        use_container_width=True,
    )
    recover_quarterly = controls[4].button("Use Saved Quarterly Scan", use_container_width=True)

    if run_quarterly or restart_quarterly:
        quarterly_progress = make_progress("Scraping Screener.in quarterly results")
        try:
            if restart_quarterly:
                reset_quarterly_results_scan(target_quarter, output_dir=OUTPUT_DIR)
                st.session_state.pop("quarterly_results", None)
            st.session_state["quarterly_results"] = run_quarterly_results_screen(
                csv_path,
                target_period=target_quarter,
                progress_callback=quarterly_progress,
                output_dir=OUTPUT_DIR,
            )
            if restart_quarterly:
                st.success("Fresh full-universe quarterly-results scan complete.")
            else:
                st.success("Quarterly-results scan complete.")
        except Exception as exc:
            st.error(f"Quarterly-results scan stopped before finalization: {exc}")
            saved_preview = load_saved_quarterly_preview(target_quarter)
            if not saved_preview["quarterly_all"].empty:
                st.session_state["quarterly_results"] = saved_preview
                st.warning("Loaded the latest saved quarterly checkpoint. Use Saved Quarterly Scan to finalize it.")

    if recover_quarterly:
        recover_saved_quarterly_results(target_quarter)

    if "quarterly_results" not in st.session_state:
        saved_preview = load_saved_quarterly_preview(target_quarter)
        if not saved_preview["quarterly_all"].empty:
            st.session_state["quarterly_results"] = saved_preview

    quarterly_results = st.session_state.get("quarterly_results", {})
    quarterly_all = quarterly_results.get("quarterly_all", pd.DataFrame())
    ranked_quarterly = prepare_quarterly_results(
        quarterly_all,
        target_period=target_quarter,
        ranking_metric=ranking_metric,
        matching_only=True,
    )
    all_for_quarter = prepare_quarterly_results(
        quarterly_all,
        target_period=target_quarter,
        ranking_metric=ranking_metric,
        matching_only=False,
    )

    quarterly_metrics = st.columns(3)
    quarterly_metrics[0].metric("Companies Scanned", f"{len(all_for_quarter):,}" if not all_for_quarter.empty else "0")
    quarterly_metrics[1].metric("Reported Selected Quarter", f"{len(ranked_quarterly):,}" if not ranked_quarterly.empty else "0")
    quarterly_metrics[2].metric("Current Ranking", f"{ranking_metric} YoY Growth")

    quarterly_tabs = st.tabs(["Ranked Results", "All Quarterly Scan", "Post-Earnings Stock Return Momentum"])
    with quarterly_tabs[0]:
        if ranked_quarterly.empty:
            st.info("Run the scan to find companies with the selected quarter, or load a saved scan for the same quarter.")
        else:
            st.dataframe(format_percent_columns(ranked_quarterly), use_container_width=True, hide_index=True)
            show_download(
                f"Download ranked by {ranking_metric} YoY growth",
                ranked_quarterly,
                f"quarterly_results_{target_quarter.replace(' ', '_')}_{ranking_metric.lower().replace(' ', '_')}_yoy.csv",
            )
    with quarterly_tabs[1]:
        if all_for_quarter.empty:
            st.info("No saved full-universe scan exists for this result quarter yet.")
        else:
            st.dataframe(format_percent_columns(all_for_quarter), use_container_width=True, hide_index=True)
            show_download("Download all quarterly scan", all_for_quarter, f"quarterly_results_all_{target_quarter.replace(' ', '_')}.csv")
            st.caption(
                f"Partial checkpoints are written to {paths['quarterly_results_partial']}. "
                "Run / Resume continues only a checkpoint for the selected result quarter."
            )
    with quarterly_tabs[2]:
        st.caption(
            "Ranks the current stock-price reaction for companies that reported the selected quarter. "
            "It uses 2-day, 5-day, and 10-day stock returns, with 10-day return weighted most heavily. "
            "This is not an earnings-growth score."
        )
        weight_controls = st.columns([1, 1, 1, 1.25])
        two_day_weight = weight_controls[0].number_input(
            "2-day weight",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_POST_EARNINGS_STOCK_RETURN_WEIGHTS["Earnings 2D Return"]),
            step=0.05,
        )
        five_day_weight = weight_controls[1].number_input(
            "5-day weight",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_POST_EARNINGS_STOCK_RETURN_WEIGHTS["Earnings 5D Return"]),
            step=0.05,
        )
        ten_day_weight = weight_controls[2].number_input(
            "10-day weight",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_POST_EARNINGS_STOCK_RETURN_WEIGHTS["Earnings 10D Return"]),
            step=0.05,
        )
        run_stock_return_momentum = weight_controls[3].button(
            "Run Stock Return Momentum",
            type="primary",
            use_container_width=True,
        )
        stock_return_weights = {
            "Earnings 2D Return": float(two_day_weight),
            "Earnings 5D Return": float(five_day_weight),
            "Earnings 10D Return": float(ten_day_weight),
        }

        if run_stock_return_momentum:
            if quarterly_all.empty:
                st.error("Run or recover the quarterly-results scan first.")
            else:
                stock_return_progress = make_progress("Downloading fresh stock returns for quarterly reporters")
                try:
                    st.session_state["quarterly_stock_return_momentum"] = run_quarterly_stock_return_momentum(
                        quarterly_all,
                        target_period=target_quarter,
                        weights=stock_return_weights,
                        progress_callback=stock_return_progress,
                        output_dir=OUTPUT_DIR,
                        price_batch_size=config.price_batch_size,
                    )
                    st.success("Post-earnings stock return momentum is ready.")
                except Exception as exc:
                    st.error(f"Stock return momentum could not be calculated: {exc}")

        if "quarterly_stock_return_momentum" not in st.session_state:
            saved_stock_return_momentum = read_csv_if_exists(paths["quarterly_stock_return_momentum"])
            if not saved_stock_return_momentum.empty:
                st.session_state["quarterly_stock_return_momentum"] = saved_stock_return_momentum

        stock_return_momentum = st.session_state.get("quarterly_stock_return_momentum", pd.DataFrame()).copy()
        if not stock_return_momentum.empty and "Target Quarter" in stock_return_momentum.columns:
            stock_return_momentum = stock_return_momentum[
                stock_return_momentum["Target Quarter"].astype(str).str.strip().str.upper().eq(target_quarter.strip().upper())
            ].copy()
        score_column = "Post-Earnings Stock Return Momentum Score"
        if score_column in stock_return_momentum.columns:
            stock_return_momentum[score_column] = pd.to_numeric(stock_return_momentum[score_column], errors="coerce")
            stock_return_momentum = stock_return_momentum.sort_values(score_column, ascending=False, na_position="last").reset_index(drop=True)

        if stock_return_momentum.empty:
            st.info("Run Stock Return Momentum to rank the selected quarter's reporters from highest to lowest stock-return momentum.")
        else:
            st.dataframe(format_percent_columns(stock_return_momentum), use_container_width=True, hide_index=True)
            show_download(
                "Download post-earnings stock return momentum",
                stock_return_momentum,
                f"post_earnings_stock_return_momentum_{target_quarter.replace(' ', '_')}.csv",
            )
            st.caption(
                f"Saved price returns: {paths['quarterly_stock_return_returns']}. "
                f"Saved ranking: {paths['quarterly_stock_return_momentum']}."
            )

with tabs[7]:
    st.subheader("Options-Confirmed Equity Momentum")
    st.caption(
        "End-of-day continuation signals from official NSE cash and derivatives reports. A signal requires "
        "the stock to rise on the signal day and its selected near-ATM call to gain at least 8%. The result "
        "is an equity-buying signal; it is not a recommendation to buy the option."
    )

    st.markdown("**1. Download Or Resume NSE EOD Data**")
    default_derivatives_end = date.today() - timedelta(days=1)
    download_controls = st.columns([1, 1, 1.25, 1.25, 1])
    derivatives_start = download_controls[0].date_input(
        "Download start",
        value=default_derivatives_end - timedelta(days=45),
        key="derivatives_download_start",
    )
    derivatives_end = download_controls[1].date_input(
        "Download end",
        value=default_derivatives_end,
        max_value=default_derivatives_end,
        key="derivatives_download_end",
    )
    download_derivatives = download_controls[2].button(
        "Download / Resume EOD Data",
        type="primary",
        use_container_width=True,
    )
    restart_derivatives = download_controls[3].button(
        "Run Full Scan From Beginning",
        key="restart_derivatives_download",
        use_container_width=True,
    )
    recover_derivatives = download_controls[4].button(
        "Use Saved Derivatives Run",
        use_container_width=True,
    )

    if download_derivatives or restart_derivatives:
        if derivatives_start > derivatives_end:
            st.error("Download start must be on or before download end.")
        else:
            derivatives_download_progress = make_progress("Downloading official NSE reports by date")
            try:
                if restart_derivatives:
                    reset_derivatives_eod_data(
                        derivatives_start,
                        derivatives_end,
                        output_dir=OUTPUT_DIR,
                    )
                    st.session_state.pop("derivatives_results", None)
                health = download_derivatives_eod_data(
                    derivatives_start,
                    derivatives_end,
                    progress_callback=derivatives_download_progress,
                    output_dir=OUTPUT_DIR,
                )
                saved = st.session_state.get("derivatives_results", load_saved_derivatives_results(OUTPUT_DIR))
                saved["data_health"] = health
                st.session_state["derivatives_results"] = saved
                completed_dates = int(
                    health.get("Status", pd.Series(dtype=str)).astype(str).str.lower().isin(["downloaded", "cached"]).sum()
                )
                prefix = "Fresh NSE EOD range downloaded." if restart_derivatives else "NSE EOD cache updated."
                st.success(f"{prefix} {completed_dates:,} requested dates are available.")
            except Exception as exc:
                st.error(f"NSE EOD download stopped: {exc}. Completed dates remain cached and can be resumed.")

    if recover_derivatives:
        st.session_state["derivatives_results"] = load_saved_derivatives_results(OUTPUT_DIR)
        st.success("Recovered the latest saved derivatives scan and backtest files.")

    st.markdown("**2. Signal Rules And Ranking**")
    rule_controls = st.columns(5)
    derivative_signal_date = rule_controls[0].date_input(
        "Signal date",
        value=default_derivatives_end,
        max_value=default_derivatives_end,
        key="derivatives_signal_date",
    )
    min_stock_return = rule_controls[1].number_input(
        "Minimum stock return %",
        min_value=0.01,
        max_value=25.0,
        value=2.0,
        step=0.25,
        help="The stock must close up by at least this amount on the same day.",
    )
    min_call_return = rule_controls[2].number_input(
        "Minimum call return %",
        min_value=8.0,
        max_value=500.0,
        value=8.0,
        step=1.0,
        help="8% is the hard minimum. Larger call gains remain eligible and rank higher.",
    )
    min_call_price = rule_controls[3].number_input(
        "Minimum call close Rs.",
        min_value=0.0,
        max_value=10000.0,
        value=5.0,
        step=1.0,
    )
    derivative_result_count = rule_controls[4].number_input(
        "Maximum signals",
        min_value=1,
        max_value=500,
        value=100,
        step=10,
    )

    liquidity_controls = st.columns(4)
    min_option_volume = liquidity_controls[0].number_input(
        "Minimum call volume (contracts)",
        min_value=0,
        max_value=1000000,
        value=100,
        step=50,
    )
    min_option_oi = liquidity_controls[1].number_input(
        "Minimum call OI (contracts)",
        min_value=0,
        max_value=1000000,
        value=100,
        step=50,
    )
    min_dte = liquidity_controls[2].number_input(
        "Minimum days to expiry",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
    )
    max_dte = liquidity_controls[3].number_input(
        "Maximum days to expiry",
        min_value=2,
        max_value=90,
        value=45,
        step=1,
    )

    with st.expander("Ranking weights", expanded=False):
        weight_controls = st.columns(5)
        call_return_weight = weight_controls[0].number_input("Call return % weight", 0.0, 100.0, 35.0, 5.0)
        stock_return_weight = weight_controls[1].number_input("Stock return % weight", 0.0, 100.0, 25.0, 5.0)
        call_oi_weight = weight_controls[2].number_input("Call OI change weight", 0.0, 100.0, 15.0, 5.0)
        volume_ratio_weight = weight_controls[3].number_input("Volume ratio weight", 0.0, 100.0, 15.0, 5.0)
        futures_return_weight = weight_controls[4].number_input("Futures return weight", 0.0, 100.0, 10.0, 5.0)

    total_derivative_weight = sum(
        [call_return_weight, stock_return_weight, call_oi_weight, volume_ratio_weight, futures_return_weight]
    )
    settings_valid = max_dte >= min_dte and abs(total_derivative_weight - 100.0) < 0.001
    if max_dte < min_dte:
        st.error("Maximum days to expiry must be greater than or equal to the minimum.")
    if abs(total_derivative_weight - 100.0) >= 0.001:
        st.error(f"Ranking weights must total 100%. Current total: {total_derivative_weight:.1f}%.")

    derivatives_config = DerivativesSignalConfig(
        min_underlying_return_pct=float(min_stock_return),
        min_call_return_pct=float(min_call_return),
        min_call_price=float(min_call_price),
        min_volume_contracts=int(min_option_volume),
        min_open_interest_contracts=int(min_option_oi),
        min_days_to_expiry=int(min_dte),
        max_days_to_expiry=int(max_dte),
        result_count=int(derivative_result_count),
        call_return_weight=float(call_return_weight) / 100.0,
        underlying_return_weight=float(stock_return_weight) / 100.0,
        call_oi_change_weight=float(call_oi_weight) / 100.0,
        call_volume_ratio_weight=float(volume_ratio_weight) / 100.0,
        futures_return_weight=float(futures_return_weight) / 100.0,
    )
    run_derivatives_signal = st.button(
        "Run Options-Confirmed Momentum",
        type="primary",
        disabled=not settings_valid,
    )
    if run_derivatives_signal:
        derivatives_scan_progress = make_progress("Building history, then checking every ticker")
        try:
            scan_results = run_derivatives_momentum_screen(
                csv_path,
                derivatives_config,
                scan_date=derivative_signal_date,
                progress_callback=derivatives_scan_progress,
                output_dir=OUTPUT_DIR,
            )
            existing = st.session_state.get("derivatives_results", load_saved_derivatives_results(OUTPUT_DIR))
            existing.update(scan_results)
            st.session_state["derivatives_results"] = existing
            signal_count = len(scan_results["signals"])
            st.success(f"Derivatives scan complete: {signal_count:,} stocks passed every signal and liquidity rule.")
        except Exception as exc:
            st.error(f"Derivatives signal scan could not finish: {exc}")

    st.markdown("**3. Strategy Validation**")
    st.caption(
        "Signals enter the underlying equity at the next trading day's open. The event study compares the "
        "stock-only, stock-plus-call, full derivatives, and regular momentum variants after costs."
    )
    backtest_controls = st.columns(6)
    derivative_backtest_start = backtest_controls[0].date_input(
        "Backtest start",
        value=default_derivatives_end - timedelta(days=365 * 3),
        key="derivatives_backtest_start",
    )
    derivative_backtest_end = backtest_controls[1].date_input(
        "Backtest end",
        value=default_derivatives_end,
        max_value=default_derivatives_end,
        key="derivatives_backtest_end",
    )
    derivative_holding_days = backtest_controls[2].number_input("Holding days", 1, 20, 5, 1)
    derivative_top_n = backtest_controls[3].number_input("Top concurrent positions", 1, 50, 10, 1)
    derivative_cost = backtest_controls[4].number_input("Round-trip cost %", 0.0, 5.0, 0.30, 0.05)
    run_derivatives_backtest = backtest_controls[5].button(
        "Run Backtest",
        type="primary",
        use_container_width=True,
        disabled=not settings_valid,
    )
    if run_derivatives_backtest:
        if derivative_backtest_start >= derivative_backtest_end:
            st.error("Backtest start must be before backtest end.")
        else:
            derivatives_backtest_progress = make_progress("Walking forward through cached trading dates")
            try:
                backtest_results = run_derivatives_backtest_screen(
                    csv_path,
                    derivatives_config,
                    derivative_backtest_start,
                    derivative_backtest_end,
                    holding_days=int(derivative_holding_days),
                    top_n=int(derivative_top_n),
                    round_trip_cost_pct=float(derivative_cost),
                    progress_callback=derivatives_backtest_progress,
                    output_dir=OUTPUT_DIR,
                )
                existing = st.session_state.get("derivatives_results", load_saved_derivatives_results(OUTPUT_DIR))
                existing.update(backtest_results)
                st.session_state["derivatives_results"] = existing
                st.success("Chronological derivatives event study and portfolio simulation are ready.")
            except Exception as exc:
                st.error(f"Backtest could not finish: {exc}")
                st.info("Download or resume the selected backtest date range first. At least 12 cached trading dates are required.")

    if "derivatives_results" not in st.session_state:
        saved_derivatives = load_saved_derivatives_results(OUTPUT_DIR)
        if any(not frame.empty for frame in saved_derivatives.values()):
            st.session_state["derivatives_results"] = saved_derivatives

    derivatives_results = st.session_state.get("derivatives_results", {})
    derivative_signals = derivatives_results.get("signals", pd.DataFrame())
    derivative_features = derivatives_results.get("features", pd.DataFrame())
    derivative_rejections = derivatives_results.get("rejections", pd.DataFrame())
    derivative_contracts = derivatives_results.get("contracts", pd.DataFrame())
    derivative_health = derivatives_results.get("data_health", pd.DataFrame())
    derivative_events = derivatives_results.get("events", pd.DataFrame())
    derivative_curve = derivatives_results.get("curve", pd.DataFrame())
    derivative_performance = derivatives_results.get("performance", pd.DataFrame())
    derivative_event_summary = derivatives_results.get("event_summary", pd.DataFrame())

    derivative_metrics = st.columns(4)
    eligible_count = int(
        derivative_contracts.get("Listed Derivatives", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1"]).sum()
    ) if not derivative_contracts.empty else 0
    derivative_metrics[0].metric("Ticker Universe", f"{len(derivative_contracts):,}")
    derivative_metrics[1].metric("Listed Stock F&O", f"{eligible_count:,}")
    derivative_metrics[2].metric("F&O Features Checked", f"{len(derivative_features):,}")
    derivative_metrics[3].metric("Qualified Signals", f"{len(derivative_signals):,}")

    derivative_tabs = st.tabs(["Ranked Signals", "Rejected Candidates", "F&O Universe", "Data Health", "Backtest"])
    with derivative_tabs[0]:
        if derivative_signals.empty:
            st.info("No saved qualifying signals yet, or no stock passed every rule on the selected date.")
        else:
            st.dataframe(format_percent_columns(derivative_signals), use_container_width=True, hide_index=True)
            show_download("Download derivatives signals", derivative_signals, "derivatives_signals.csv")
    with derivative_tabs[1]:
        if derivative_rejections.empty:
            st.info("Run a derivatives signal scan to inspect rejection reasons.")
        else:
            st.dataframe(format_percent_columns(derivative_rejections), use_container_width=True, hide_index=True)
            show_download("Download rejected candidates", derivative_rejections, "derivatives_rejections.csv")
    with derivative_tabs[2]:
        if derivative_contracts.empty:
            st.info("Run a derivatives signal scan to map the full ticker universe to NSE stock F&O eligibility.")
        else:
            st.dataframe(derivative_contracts, use_container_width=True, hide_index=True)
            show_download("Download F&O universe", derivative_contracts, "derivatives_contracts.csv")
    with derivative_tabs[3]:
        if derivative_health.empty:
            st.info("Download an NSE EOD date range to see cached, completed, and unavailable report dates.")
        else:
            st.dataframe(derivative_health, use_container_width=True, hide_index=True)
            show_download("Download data health", derivative_health, "derivatives_data_health.csv")
            st.caption(f"Raw and normalized reports are checkpointed under {paths['derivatives_cache']}.")
    with derivative_tabs[4]:
        if derivative_curve.empty:
            st.info("Download a historical EOD range and run the derivatives backtest to see strategy validation.")
        else:
            chart_frame = derivative_curve.copy()
            chart_frame["Date"] = pd.to_datetime(chart_frame["Date"], errors="coerce")
            chart = px.line(
                chart_frame.dropna(subset=["Date"]),
                x="Date",
                y="Portfolio Value",
                color="Variant",
                title="Walk-Forward Portfolio Value",
            )
            st.plotly_chart(chart, use_container_width=True)
        if not derivative_performance.empty:
            st.markdown("**Portfolio Performance**")
            st.dataframe(format_percent_columns(derivative_performance), use_container_width=True, hide_index=True)
        if not derivative_event_summary.empty:
            st.markdown("**Event Study By Chronological Split**")
            st.dataframe(format_percent_columns(derivative_event_summary), use_container_width=True, hide_index=True)
        if not derivative_events.empty:
            st.markdown("**Backtest Events**")
            st.dataframe(format_percent_columns(derivative_events), use_container_width=True, hide_index=True)
            show_download("Download backtest events", derivative_events, "derivatives_backtest_events.csv")
            show_download("Download backtest curve", derivative_curve, "derivatives_backtest_curve.csv")
        st.caption(
            "The Train, Validation, and Test periods are chronological 60% / 20% / 20% splits. Treat the strategy "
            "as promising only when the full derivatives signal improves the untouched Test period over the equity-only baseline after costs."
        )


with tabs[8]:
    st.subheader("NSE Index Momentum")
    st.caption(
        "Recent-return momentum across the NSE index catalogue you supplied. This replaces the relative sector-rotation model and ranks each index on its own price trend."
    )

    index_catalogue = index_catalogue_frame()
    index_categories = index_catalogue["Category"].drop_duplicates().tolist()
    selected_index_categories = st.multiselect(
        "Index categories",
        options=index_categories,
        default=index_categories,
        key="index_momentum_categories",
    )
    category_indices = index_catalogue.loc[
        index_catalogue["Category"].isin(selected_index_categories), "Index"
    ].drop_duplicates().tolist()
    scan_all_category_indices = st.checkbox(
        "Scan every index in the selected categories",
        value=True,
        key="index_momentum_scan_all",
    )
    if scan_all_category_indices:
        selected_indices = category_indices
        st.caption(f"{len(selected_indices):,} indices selected from the shared NSE list.")
    else:
        selected_indices = st.multiselect(
            "Indices to scan",
            options=category_indices,
            default=category_indices[: min(20, len(category_indices))],
            key="index_momentum_indices",
        )

    with st.expander("Index momentum weights", expanded=True):
        st.caption("Weights are normalized automatically, so they do not need to total exactly 1.00.")
        index_weight_columns = st.columns(6)
        index_weights: dict[str, float] = {}
        for column, (label, default_weight) in zip(
            index_weight_columns,
            DEFAULT_INDEX_MOMENTUM_WEIGHTS.items(),
        ):
            index_weights[label] = column.number_input(
                label.replace(" Return %", ""),
                min_value=0.0,
                max_value=1.0,
                value=float(default_weight),
                step=0.05,
                key=f"index_weight_{label}",
            )
        index_weight_total = sum(index_weights.values())
        st.caption(f"Current weight total: {index_weight_total:.2f}")
        if index_weight_total <= 0:
            st.error("At least one index momentum weight must be greater than zero.")

    index_controls = st.columns([1, 1, 1])
    index_end_date = index_controls[0].date_input(
        "Analysis end date",
        value=date.today() - timedelta(days=1),
        max_value=date.today(),
        key="index_momentum_end_date",
    )
    index_history_months = index_controls[1].slider(
        "History window (months)",
        min_value=4,
        max_value=36,
        value=12,
        key="index_momentum_history_months",
    )
    index_chart_days = index_controls[2].slider(
        "Comparison chart days",
        min_value=21,
        max_value=252,
        value=126,
        key="index_momentum_chart_days",
    )

    index_actions = st.columns(4)
    run_index_momentum = index_actions[0].button(
        "Run / Resume Index Scan",
        type="primary",
        use_container_width=True,
    )
    rescore_index_momentum = index_actions[1].button(
        "Recalculate Saved Prices",
        use_container_width=True,
        help="Apply the current weights to saved NSE prices without downloading again.",
    )
    restart_index_momentum = index_actions[2].button(
        "Run Full Scan From Beginning",
        use_container_width=True,
    )
    use_saved_index_momentum = index_actions[3].button(
        "Use Saved Index Results",
        use_container_width=True,
    )

    if run_index_momentum or restart_index_momentum:
        if not selected_indices:
            st.error("Select at least one NSE index.")
        elif index_weight_total <= 0:
            st.error("Set at least one momentum weight above zero.")
        else:
            if restart_index_momentum:
                reset_index_momentum_scan(OUTPUT_DIR)
                st.session_state.pop("index_momentum_results", None)
            index_progress = make_progress("Downloading official NSE index history")
            try:
                st.session_state["index_momentum_results"] = run_index_momentum_screen(
                    selected_indices,
                    end_date=index_end_date,
                    weights=index_weights,
                    history_months=int(index_history_months),
                    progress_callback=index_progress,
                    output_dir=OUTPUT_DIR,
                )
                st.success("NSE index momentum ranking is ready.")
            except Exception as exc:
                st.error(f"Index momentum scan could not complete: {exc}")

    if rescore_index_momentum:
        if index_weight_total <= 0:
            st.error("Set at least one momentum weight above zero.")
        else:
            try:
                st.session_state["index_momentum_results"] = rescore_saved_index_momentum(
                    selected_indices,
                    index_weights,
                    output_dir=OUTPUT_DIR,
                )
                st.success("Saved NSE prices were rescored with the current weights.")
            except Exception as exc:
                st.error(f"Saved index prices could not be rescored: {exc}")

    if use_saved_index_momentum:
        try:
            st.session_state["index_momentum_results"] = load_saved_index_momentum(OUTPUT_DIR)
            st.success("Loaded the saved NSE index momentum run.")
        except FileNotFoundError as exc:
            st.error(str(exc))

    index_results = st.session_state.get("index_momentum_results", {})
    index_ranking = index_results.get("ranking", pd.DataFrame()).copy()
    index_prices = index_results.get("prices", pd.DataFrame()).copy()
    index_health = index_results.get("health", pd.DataFrame()).copy()
    index_metadata = index_results.get("metadata", pd.DataFrame()).copy()

    if index_ranking.empty:
        st.info("Run the NSE index scan or load a saved result to see momentum rankings.")
    else:
        positive_mask = index_ranking.get("Short-Term Positive", pd.Series(False, index=index_ranking.index))
        if positive_mask.dtype == object:
            positive_mask = positive_mask.astype(str).str.lower().isin(("true", "1", "yes"))
        latest_index_date = index_ranking.get("Data Date", pd.Series(["NA"])).astype(str).max()
        index_metrics = st.columns(4)
        index_metrics[0].metric("Indices Ranked", f"{index_ranking['Momentum Score'].notna().sum():,}")
        index_metrics[1].metric("Positive Short-Term", f"{int(positive_mask.sum()):,}")
        index_metrics[2].metric(
            "Top Index",
            str(index_ranking.loc[index_ranking["Momentum Score"].notna(), "Index"].iloc[0])
            if index_ranking["Momentum Score"].notna().any()
            else "NA",
        )
        index_metrics[3].metric("Latest Data Date", latest_index_date)

        index_visuals = st.columns(2)
        top_index_ranking = index_ranking.dropna(subset=["Momentum Score"]).head(20).sort_values("Momentum Score")
        if not top_index_ranking.empty:
            index_bar = px.bar(
                top_index_ranking,
                x="Momentum Score",
                y="Index",
                orientation="h",
                color="Category",
                hover_data=["2D Return %", "5D Return %", "10D Return %", "1M Return %"],
                title="Top 20 NSE Index Momentum Scores",
            )
            index_bar.update_layout(height=max(480, len(top_index_ranking) * 28 + 120))
            index_visuals[0].plotly_chart(index_bar, use_container_width=True)

            heatmap_columns = list(INDEX_MOMENTUM_PERIODS)
            index_heatmap_frame = (
                index_ranking.dropna(subset=["Momentum Score"])
                .head(25)
                .set_index("Index")[heatmap_columns]
                .apply(pd.to_numeric, errors="coerce")
            )
            index_heatmap = px.imshow(
                index_heatmap_frame,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                aspect="auto",
                labels={"color": "Return %"},
                title="Recent Return Heatmap",
            )
            index_heatmap.update_layout(height=max(480, len(index_heatmap_frame) * 26 + 120))
            index_visuals[1].plotly_chart(index_heatmap, use_container_width=True)

        st.markdown("**Complete Index Momentum Ranking**")
        st.dataframe(format_percent_columns(index_ranking), use_container_width=True, hide_index=True)
        show_download("Download index momentum ranking", index_ranking, "index_momentum_ranking.csv")

        chart_indices = index_ranking.dropna(subset=["Momentum Score"]).head(5)["Index"].astype(str).tolist()
        index_performance = normalized_index_performance(
            index_prices,
            chart_indices,
            trading_days=int(index_chart_days),
        )
        if not index_performance.empty:
            index_lines = px.line(
                index_performance,
                x="Date",
                y="Normalized Value",
                color="Index",
                title="Top 5 Index Performance (Start = 100)",
            )
            index_lines.add_hline(y=100, line_width=1, line_dash="dot", line_color="#777777")
            st.plotly_chart(index_lines, use_container_width=True)

        with st.expander("Index data health and saved settings"):
            if not index_metadata.empty:
                st.dataframe(index_metadata, use_container_width=True, hide_index=True)
            if index_health.empty:
                st.info("No per-index health report is available.")
            else:
                st.dataframe(index_health, use_container_width=True, hide_index=True)
                show_download("Download index data health", index_health, "index_momentum_health.csv")



with tabs[9]:
    st.subheader("CSV Stock Momentum")
    st.caption(
        "After identifying a strong index, upload its constituent tickers here. The file may contain Ticker, Ticker Name, Symbol, NSE Symbol, or just one ticker column."
    )
    custom_stock_upload = st.file_uploader(
        "Upload constituent ticker CSV",
        type=["csv"],
        key="custom_stock_momentum_upload",
    )
    custom_weight_frame = pd.DataFrame(
        {
            "Return Period": list(config.momentum_weights),
            "Weight": [config.momentum_weights[label] for label in config.momentum_weights],
            "Positive Filter": [
                "Required" if label in config.positive_return_filters else "No"
                for label in config.momentum_weights
            ],
        }
    )
    with st.expander("Stock momentum model used for this upload"):
        st.dataframe(custom_weight_frame, use_container_width=True, hide_index=True)
        st.caption("Adjust these stock weights from the Momentum controls in the left sidebar.")

    custom_stock_actions = st.columns(2)
    run_custom_stock_scan = custom_stock_actions[0].button(
        "Run Uploaded Stock Momentum",
        type="primary",
        use_container_width=True,
    )
    use_saved_custom_stock = custom_stock_actions[1].button(
        "Use Saved Uploaded-Stock Run",
        use_container_width=True,
    )

    if run_custom_stock_scan:
        if custom_stock_upload is None:
            st.error("Upload a CSV containing the constituent ticker symbols first.")
        else:
            custom_temp = NamedTemporaryFile(delete=False, suffix=".csv")
            custom_temp.write(custom_stock_upload.getbuffer())
            custom_temp.close()
            custom_progress = make_progress("Downloading uploaded stock prices and calculating momentum")
            try:
                st.session_state["custom_stock_momentum_results"] = run_custom_stock_momentum(
                    custom_temp.name,
                    config,
                    progress_callback=custom_progress,
                    output_dir=OUTPUT_DIR,
                )
                st.success("Uploaded constituent stocks have been momentum-scored.")
            except Exception as exc:
                st.error(f"Uploaded stock momentum could not complete: {exc}")

    if use_saved_custom_stock:
        try:
            st.session_state["custom_stock_momentum_results"] = load_saved_custom_stock_momentum(OUTPUT_DIR)
            st.success("Loaded the saved uploaded-stock momentum run.")
        except FileNotFoundError as exc:
            st.error(str(exc))

    custom_results = st.session_state.get("custom_stock_momentum_results", {})
    custom_universe = custom_results.get("universe", pd.DataFrame()).copy()
    custom_ranking = custom_results.get("ranking", pd.DataFrame()).copy()
    custom_returns = custom_results.get("returns", pd.DataFrame()).copy()
    custom_health = custom_results.get("health", pd.DataFrame()).copy()

    if custom_ranking.empty:
        st.info("Upload and run a constituent list, or recover the last saved uploaded-stock momentum run.")
    else:
        custom_pass = custom_ranking.get("Momentum Pass", pd.Series(False, index=custom_ranking.index))
        if custom_pass.dtype == object:
            custom_pass = custom_pass.astype(str).str.lower().isin(("true", "1", "yes"))
        valid_scores = pd.to_numeric(custom_ranking.get("Momentum Score"), errors="coerce")
        custom_metrics = st.columns(4)
        custom_metrics[0].metric("Uploaded Tickers", f"{len(custom_universe):,}")
        custom_metrics[1].metric("Valid Scores", f"{int(valid_scores.notna().sum()):,}")
        custom_metrics[2].metric("Positive Momentum Pass", f"{int(custom_pass.sum()):,}")
        custom_metrics[3].metric(
            "Top Stock",
            str(custom_ranking.loc[valid_scores.notna(), "Ticker"].iloc[0])
            if valid_scores.notna().any()
            else "NA",
        )

        custom_ranked_valid = custom_ranking.loc[valid_scores.notna()].head(20).sort_values("Momentum Score")
        if not custom_ranked_valid.empty:
            custom_chart = px.bar(
                custom_ranked_valid,
                x="Momentum Score",
                y="Ticker",
                orientation="h",
                color="Momentum Pass",
                color_discrete_map={True: "#16845b", False: "#b44343"},
                hover_data=list(config.momentum_weights),
                title="Uploaded Stock Momentum Ranking",
            )
            custom_chart.update_layout(height=max(430, len(custom_ranked_valid) * 28 + 120))
            st.plotly_chart(custom_chart, use_container_width=True)

        custom_tabs = st.tabs(["Momentum Ranking", "All Returns", "Data Health"])
        with custom_tabs[0]:
            st.dataframe(format_percent_columns(custom_ranking), use_container_width=True, hide_index=True)
            show_download("Download uploaded stock momentum", custom_ranking, "custom_stock_momentum.csv")
        with custom_tabs[1]:
            st.dataframe(format_percent_columns(custom_returns), use_container_width=True, hide_index=True)
            show_download("Download uploaded stock returns", custom_returns, "custom_stock_returns.csv")
        with custom_tabs[2]:
            st.dataframe(custom_health, use_container_width=True, hide_index=True)
            show_download("Download uploaded stock data health", custom_health, "custom_stock_health.csv")


with tabs[10]:
    st.subheader("Macro Factor Correlation")
    st.caption(
        "Historical stock-return relationships with crude oil, gold, sovereign yields, and USD/INR. "
        "Price factors use percentage changes; yields use basis-point changes."
    )

    selected_correlation_factors = st.multiselect(
        "Factors",
        options=list(FACTOR_CATALOG),
        default=list(FACTOR_CATALOG),
        key="correlation_factors_selection",
    )
    correlation_controls = st.columns(5)
    correlation_end_date = correlation_controls[0].date_input(
        "Analysis end date",
        value=date.today() - timedelta(days=1),
        max_value=date.today(),
        key="correlation_end_date",
    )
    correlation_lookback = correlation_controls[1].slider(
        "Lookback (years)",
        min_value=1,
        max_value=10,
        value=3,
        key="correlation_lookback",
    )
    correlation_frequency = correlation_controls[2].selectbox(
        "Return frequency",
        options=["Daily", "Weekly", "Monthly"],
        index=1,
        key="correlation_frequency",
    )
    correlation_method = correlation_controls[3].selectbox(
        "Method",
        options=["Pearson", "Spearman"],
        key="correlation_method",
    )
    correlation_relation = correlation_controls[4].selectbox(
        "Relationship",
        options=["Same period", "Next stock period"],
        key="correlation_relation",
    )

    correlation_rules = st.columns(5)
    correlation_min_observations = correlation_rules[0].number_input(
        "Minimum observations",
        min_value=10,
        max_value=1000,
        value=24,
        step=5,
        key="correlation_min_observations",
    )
    correlation_top_n = correlation_rules[1].slider(
        "Stocks per direction",
        min_value=3,
        max_value=30,
        value=10,
        key="correlation_top_n",
    )
    correlation_ridge_alpha = correlation_rules[2].number_input(
        "Ridge alpha",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.5,
        key="correlation_ridge_alpha",
    )
    run_correlation = correlation_rules[3].button(
        "Run Correlation Scan",
        type="primary",
        use_container_width=True,
    )
    use_saved_correlation = correlation_rules[4].button(
        "Use Saved Correlation",
        use_container_width=True,
    )

    if run_correlation:
        if not selected_correlation_factors:
            st.error("Select at least one macro factor.")
        else:
            correlation_progress = make_progress("Downloading as-of market history and calculating relationships")
            try:
                st.session_state["correlation_results"] = run_correlation_screen(
                    ticker_csv=csv_path,
                    factors=selected_correlation_factors,
                    end_date=correlation_end_date,
                    lookback_years=int(correlation_lookback),
                    frequency=correlation_frequency,
                    method=correlation_method,
                    relation=correlation_relation,
                    min_observations=int(correlation_min_observations),
                    top_n=int(correlation_top_n),
                    ridge_alpha=float(correlation_ridge_alpha),
                    progress_callback=correlation_progress,
                    output_dir=OUTPUT_DIR,
                )
                st.success("Full-universe correlation scan and saved-run checkpoint are ready.")
            except Exception as exc:
                st.error(f"Correlation scan could not complete: {exc}")

    if use_saved_correlation:
        try:
            st.session_state["correlation_results"] = load_saved_correlation(OUTPUT_DIR)
            st.success("Loaded the saved prices, factors, correlation results, and ridge model.")
        except FileNotFoundError as exc:
            st.error(str(exc))

    correlation_results = st.session_state.get("correlation_results", {})
    correlation_all = correlation_results.get("results", pd.DataFrame()).copy()
    correlation_factors = correlation_results.get("factors", pd.DataFrame()).copy()
    correlation_health = correlation_results.get("health", pd.DataFrame()).copy()
    correlation_stock_health = correlation_results.get("stock_health", pd.DataFrame()).copy()
    correlation_prices = correlation_results.get("prices", pd.DataFrame()).copy()
    correlation_metadata = correlation_results.get("metadata", pd.DataFrame()).copy()
    correlation_stale = bool(correlation_results.get("stale", False))

    if correlation_all.empty:
        st.info("Run the full-universe scan or recover a completed correlation run.")
    else:
        available_factors = correlation_all["Factor"].dropna().astype(str).unique().tolist()
        if correlation_stale:
            latest_data_date = correlation_all.get("Data End", pd.Series(["unknown"])).max()
            st.warning(f"Saved correlation results are being shown through {latest_data_date}.")
        if not correlation_metadata.empty:
            saved_row = correlation_metadata.iloc[0]
            st.caption(
                f"Saved run: {saved_row.get('Saved At UTC', 'unknown')} | "
                f"{saved_row.get('Requested Frequency', 'unknown')} | "
                f"{saved_row.get('Relationship', 'unknown')} | "
                f"Ridge alpha {saved_row.get('Ridge Alpha', 'unknown')}"
            )

        summary_columns = st.columns(4)
        summary_columns[0].metric("Stocks Analyzed", f"{correlation_all['Ticker'].nunique():,}")
        summary_columns[1].metric("Factors Available", f"{len(available_factors):,}")
        summary_columns[2].metric(
            "Stock-Factor Pairs",
            f"{len(correlation_all):,}",
        )
        summary_columns[3].metric(
            "Latest Data Date",
            str(correlation_all.get("Data End", pd.Series(["NA"])).max()),
        )

        display_factor = st.selectbox(
            "Factor to inspect",
            options=available_factors,
            key="correlation_display_factor",
        )
        ranking_basis = st.segmented_control(
            "Ranking model",
            options=["Ridge Regression", "Correlation"],
            default="Ridge Regression",
            key="correlation_ranking_basis",
        )
        ranking_metric = "Ridge Coefficient" if ranking_basis == "Ridge Regression" else "Correlation"
        if ranking_metric not in correlation_all.columns or correlation_all[ranking_metric].notna().sum() == 0:
            ranking_metric = "Correlation"
            if ranking_basis == "Ridge Regression":
                st.warning("This in-memory run predates ridge regression. Load Saved Correlation to upgrade it without downloading again.")
        current_results = correlation_all.loc[correlation_all["Factor"].eq(display_factor)].copy()
        positive_all, inverse_all = select_relationship_leaders(
            correlation_all,
            top_n=int(correlation_top_n),
            ranking_metric=ranking_metric,
        )
        positive_view = positive_all.loc[positive_all["Factor"].eq(display_factor)].copy()
        inverse_view = inverse_all.loc[inverse_all["Factor"].eq(display_factor)].copy()

        if display_factor == "India 10Y Yield" and correlation_frequency != "Monthly":
            st.info(
                "India 10Y uses the public FRED/OECD monthly series, so its effective frequency is Monthly. "
                "Other factors use the frequency saved with this run."
            )

        chart_columns = st.columns(2)
        if not positive_view.empty:
            positive_chart = px.bar(
                positive_view.sort_values(ranking_metric),
                x=ranking_metric,
                y="Ticker",
                orientation="h",
                color="Strong Rise Avg Return %",
                color_continuous_scale="Greens",
                hover_data=[
                    "Name",
                    "Industry",
                    "Avg Return When Factor Rises %",
                    "Positive Hit Rate When Factor Rises %",
                    "Correlation",
                    "Ridge Coefficient",
                    "Observations",
                ],
                title=f"Rises With {display_factor} ({ranking_basis})",
            )
            positive_chart.update_layout(height=max(390, len(positive_view) * 32 + 120))
            chart_columns[0].plotly_chart(positive_chart, use_container_width=True)
        else:
            chart_columns[0].info("No positive relationships met the observation requirement.")

        if not inverse_view.empty:
            inverse_chart = px.bar(
                inverse_view.sort_values(ranking_metric, ascending=False),
                x=ranking_metric,
                y="Ticker",
                orientation="h",
                color="Strong Fall Avg Return %",
                color_continuous_scale="Blues",
                hover_data=[
                    "Name",
                    "Industry",
                    "Avg Return When Factor Falls %",
                    "Positive Hit Rate When Factor Falls %",
                    "Correlation",
                    "Ridge Coefficient",
                    "Observations",
                ],
                title=f"Benefits When {display_factor} Falls ({ranking_basis})",
            )
            inverse_chart.update_layout(height=max(390, len(inverse_view) * 32 + 120))
            chart_columns[1].plotly_chart(inverse_chart, use_container_width=True)
        else:
            chart_columns[1].info("No inverse relationships met the observation requirement.")

        comparison_columns = st.columns(2)
        positive_performance = normalized_factor_stock_performance(
            correlation_prices,
            correlation_factors,
            display_factor,
            positive_view.head(5).get("Ticker", pd.Series(dtype=str)).astype(str).tolist(),
        )
        if positive_performance.empty:
            comparison_columns[0].info("Saved stock prices are required for the positive top-five comparison chart.")
        else:
            positive_lines = px.line(
                positive_performance,
                x="Date",
                y="Normalized Value",
                color="Series",
                line_dash="Type",
                title=f"{display_factor} And Top 5 Positive Picks (Start = 100)",
            )
            positive_lines.add_hline(y=100, line_width=1, line_dash="dot", line_color="#777777")
            comparison_columns[0].plotly_chart(positive_lines, use_container_width=True)

        inverse_performance = normalized_factor_stock_performance(
            correlation_prices,
            correlation_factors,
            display_factor,
            inverse_view.head(5).get("Ticker", pd.Series(dtype=str)).astype(str).tolist(),
        )
        if inverse_performance.empty:
            comparison_columns[1].info("Saved stock prices are required for the inverse top-five comparison chart.")
        else:
            inverse_lines = px.line(
                inverse_performance,
                x="Date",
                y="Normalized Value",
                color="Series",
                line_dash="Type",
                title=f"{display_factor} And Top 5 Inverse Picks (Start = 100)",
            )
            inverse_lines.add_hline(y=100, line_width=1, line_dash="dot", line_color="#777777")
            comparison_columns[1].plotly_chart(inverse_lines, use_container_width=True)
        st.caption(
            "Comparison lines use the common available window and normalize every series to 100. "
            "Ridge coefficients are standardized partial relationships after controlling for the other factors at the same frequency."
        )

        matrix = correlation_matrix(correlation_all, stocks_per_factor=5)
        if not matrix.empty:
            correlation_heatmap = px.imshow(
                matrix,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                zmin=-1,
                zmax=1,
                aspect="auto",
                labels={"color": "Correlation"},
                title="Cross-Factor Correlation Matrix For Leading And Inverse Stocks",
            )
            correlation_heatmap.update_layout(height=max(460, 22 * len(matrix) + 150))
            st.plotly_chart(correlation_heatmap, use_container_width=True)

        direction_tabs = st.tabs(["Factor Rises", "Factor Falls", "All Relationships", "Factor History", "Data Health"])
        with direction_tabs[0]:
            st.dataframe(format_percent_columns(positive_view), use_container_width=True, hide_index=True)
            show_download(
                f"Download stocks that rise with {display_factor}",
                positive_view,
                f"{display_factor.lower().replace('/', '_').replace(' ', '_')}_positive.csv",
            )
        with direction_tabs[1]:
            st.dataframe(format_percent_columns(inverse_view), use_container_width=True, hide_index=True)
            show_download(
                f"Download stocks that benefit when {display_factor} falls",
                inverse_view,
                f"{display_factor.lower().replace('/', '_').replace(' ', '_')}_inverse.csv",
            )
        with direction_tabs[2]:
            st.dataframe(format_percent_columns(current_results), use_container_width=True, hide_index=True)
            show_download("Download all stock-factor relationships", correlation_all, "correlation_all.csv")
        with direction_tabs[3]:
            factor_history_view = correlation_factors.loc[
                correlation_factors.get("Factor", pd.Series(dtype=str)).eq(display_factor)
            ].copy()
            if factor_history_view.empty:
                st.info("No saved factor history is available for this factor.")
            else:
                factor_history_view["Date"] = pd.to_datetime(factor_history_view["Date"], errors="coerce")
                history_chart = px.line(
                    factor_history_view.dropna(subset=["Date"]),
                    x="Date",
                    y="Level",
                    title=f"{display_factor} Historical Level",
                )
                st.plotly_chart(history_chart, use_container_width=True)
                st.dataframe(factor_history_view, use_container_width=True, hide_index=True)
        with direction_tabs[4]:
            st.markdown("**Macro Factor Sources**")
            st.dataframe(correlation_health, use_container_width=True, hide_index=True)
            show_download("Download correlation data health", correlation_health, "correlation_health.csv")
            st.markdown("**Stock Price Sources**")
            if correlation_stock_health.empty:
                st.info("No per-stock source report was saved for this run.")
            else:
                source_counts = (
                    correlation_stock_health["Price Source"].value_counts().rename_axis("Price Source").reset_index(name="Stocks")
                )
                st.dataframe(source_counts, use_container_width=True, hide_index=True)
                with st.expander("Inspect all stock price sources"):
                    st.dataframe(correlation_stock_health, use_container_width=True, hide_index=True)
                show_download(
                    "Download stock price source health",
                    correlation_stock_health,
                    "correlation_stock_health.csv",
                )

        st.caption(
            "Correlation describes historical co-movement, not causation or a guaranteed future reaction. "
            "Same-period results are contemporaneous; Next stock period compares a factor move with the following stock-return period."
        )


with tabs[11]:
    st.subheader("200DMA Opportunity Finder")
    st.caption(
        "Find fundamentally strong stocks trading just above their 200-day moving average. "
        "The scan ranks the closest positive proximity first and runs Screener.in fundamentals automatically."
    )

    sma_controls = st.columns(4)
    sma_max_proximity = sma_controls[0].slider(
        "Maximum distance above 200DMA",
        min_value=1.0,
        max_value=20.0,
        value=10.0,
        step=0.5,
        format="%.1f%%",
        key="sma200_max_proximity",
    )
    sma_price_mode = sma_controls[1].selectbox(
        "Current price mode",
        options=["Near-live shortlist", "Latest close"],
        key="sma200_price_mode",
        help="Near-live mode refreshes delayed/intraday Yahoo quotes only for a buffered shortlist.",
    )
    sma_backtest_months = sma_controls[2].slider(
        "Walk-forward months",
        min_value=1,
        max_value=36,
        value=int(config.backtest_months),
        key="sma200_backtest_months",
    )
    sma_investment_amount = sma_controls[3].number_input(
        "Investment amount",
        min_value=500.0,
        value=100000.0,
        step=500.0,
        key="sma200_investment_amount",
    )

    sma_actions = st.columns(3)
    run_sma200 = sma_actions[0].button(
        "Run / Resume 200DMA Scan",
        type="primary",
        use_container_width=True,
    )
    restart_sma200 = sma_actions[1].button(
        "Run Full Scan From Beginning",
        key="restart_sma200_scan",
        use_container_width=True,
    )
    use_saved_sma200 = sma_actions[2].button(
        "Use Saved 200DMA Scan",
        use_container_width=True,
    )

    sma_scan_config = Sma200ScanConfig(
        max_distance_pct=float(sma_max_proximity),
        price_mode=str(sma_price_mode),
        backtest_months=int(sma_backtest_months),
        price_batch_size=int(config.price_batch_size),
    )

    if run_sma200 or restart_sma200:
        if restart_sma200:
            reset_sma200_scan(OUTPUT_DIR)
            st.session_state.pop("sma200_results", None)
        price_progress = make_progress("Downloading or recovering daily price history")
        calculation_progress = make_progress("Calculating 200-day moving averages")
        quote_progress = make_progress("Refreshing near-live shortlist quotes")
        fundamental_progress = make_progress("Scraping fundamentals for proximity candidates")
        try:
            st.session_state["sma200_results"] = run_sma200_screen(
                ticker_csv=csv_path,
                scan_config=sma_scan_config,
                fundamental_thresholds=config.fundamental_thresholds,
                price_progress_callback=price_progress,
                calculation_progress_callback=calculation_progress,
                quote_progress_callback=quote_progress,
                fundamental_progress_callback=fundamental_progress,
                output_dir=OUTPUT_DIR,
                resume=not restart_sma200,
            )
            st.session_state["sma200_saved_mode"] = False
            st.success("200DMA prices, proximity filter, fundamentals, and walk-forward backtest are ready.")
        except Exception as exc:
            st.error(f"200DMA scan could not complete: {exc}")

    if use_saved_sma200:
        try:
            st.session_state["sma200_results"] = load_saved_sma200_results(OUTPUT_DIR)
            st.session_state["sma200_saved_mode"] = True
            st.success("Loaded the saved 200DMA scan without downloading or scraping again.")
        except FileNotFoundError as exc:
            st.error(str(exc))

    sma_results = st.session_state.get("sma200_results", {})
    sma_prices = sma_results.get("prices", pd.DataFrame()).copy()
    sma_universe = sma_results.get("universe", pd.DataFrame()).copy()
    sma_candidates = sma_results.get("candidates", pd.DataFrame()).copy()
    sma_rejected = sma_results.get("rejected", pd.DataFrame()).copy()
    sma_fundamentals = sma_results.get("fundamentals", pd.DataFrame()).copy()
    sma_final = sma_results.get("final", pd.DataFrame()).copy()
    sma_health = sma_results.get("health", pd.DataFrame()).copy()
    sma_backtest = sma_results.get("backtest", pd.DataFrame()).copy()
    sma_normalized = sma_results.get("normalized_backtest", pd.DataFrame()).copy()
    sma_periods = sma_results.get("periods", pd.DataFrame()).copy()
    sma_performance = sma_results.get("performance", pd.DataFrame()).copy()
    sma_metadata = sma_results.get("metadata", pd.DataFrame()).copy()

    if sma_universe.empty and sma_final.empty:
        st.info("Run the 200DMA scan or load a saved scan to see opportunities.")
    else:
        if st.session_state.get("sma200_saved_mode", False):
            saved_at = sma_metadata.get("Saved At UTC", pd.Series(["unknown"])).iloc[0]
            st.warning(f"Saved data is being shown from {saved_at}. Run the scan to refresh prices and fundamentals.")

        latest_sma_date = sma_universe.get("SMA Date", pd.Series(["NA"])).astype(str).max()
        sma_metrics = st.columns(4)
        sma_metrics[0].metric("Universe", f"{len(sma_universe):,}")
        sma_metrics[1].metric("Near 200DMA", f"{len(sma_candidates):,}")
        sma_metrics[2].metric("Fundamental Pass", f"{len(sma_final):,}")
        sma_metrics[3].metric("Latest SMA Date", latest_sma_date)

        sma_result_tabs = st.tabs(
            ["Final Opportunities", "Proximity Candidates", "Rejected & Health", "Walk-Forward Backtest"]
        )

        with sma_result_tabs[0]:
            if sma_final.empty:
                st.info("No stocks passed both the positive-proximity and fundamental filters.")
            else:
                st.dataframe(format_percent_columns(sma_final), use_container_width=True, hide_index=True)
                show_download("Download final 200DMA opportunities", sma_final, "sma200_final.csv")

                chart_ticker = st.selectbox(
                    "Stock to inspect",
                    options=sma_final["YFinance Ticker"].astype(str).tolist(),
                    format_func=lambda value: value.removesuffix(".NS"),
                    key="sma200_chart_ticker",
                )
                chart_data = sma200_chart_data(
                    sma_prices,
                    chart_ticker,
                    window_days=int(sma_scan_config.window_days),
                )
                if chart_data.empty:
                    st.info("Saved daily history is unavailable for this stock chart.")
                else:
                    sma_chart = px.line(
                        chart_data,
                        x="Date",
                        y="Value",
                        color="Series",
                        title=f"{chart_ticker.removesuffix('.NS')} Price And 200DMA",
                    )
                    sma_chart.update_layout(yaxis_title="Price (Rs.)", xaxis_title="")
                    st.plotly_chart(sma_chart, use_container_width=True)

        with sma_result_tabs[1]:
            if sma_candidates.empty:
                st.info("No stocks are currently between 0% and the selected distance above their 200DMA.")
            else:
                st.dataframe(format_percent_columns(sma_candidates), use_container_width=True, hide_index=True)
                show_download("Download all proximity candidates", sma_candidates, "sma200_candidates.csv")
            with st.expander("Automatic Screener.in fundamental results"):
                if sma_fundamentals.empty:
                    st.info("No fundamentals were available for this scan.")
                else:
                    st.dataframe(
                        format_percent_columns(sma_fundamentals),
                        use_container_width=True,
                        hide_index=True,
                    )
                    show_download("Download 200DMA fundamentals", sma_fundamentals, "sma200_fundamentals.csv")

        with sma_result_tabs[2]:
            rejected_tabs = st.tabs(["Rejected Stocks", "Data Health"])
            with rejected_tabs[0]:
                if sma_rejected.empty:
                    st.info("No rejected rows were saved.")
                else:
                    st.dataframe(format_percent_columns(sma_rejected), use_container_width=True, hide_index=True)
                    show_download("Download rejected 200DMA stocks", sma_rejected, "sma200_rejected.csv")
            with rejected_tabs[1]:
                if sma_health.empty:
                    st.info("No data-health report was saved.")
                else:
                    st.dataframe(sma_health, use_container_width=True, hide_index=True)
                    show_download("Download 200DMA data health", sma_health, "sma200_health.csv")

        with sma_result_tabs[3]:
            st.caption(
                "Walk-forward 200DMA price backtest using the current fundamentals universe. Signals use the "
                "previous trading day, rebalance monthly, and select the ten closest positive-proximity stocks. "
                "Current fundamentals introduce survivorship and current-data bias."
            )
            dynamic_allocation = current_allocation(sma_final, capital=float(sma_investment_amount))
            if not dynamic_allocation.empty:
                st.markdown("**Current Allocation**")
                st.dataframe(dynamic_allocation, use_container_width=True, hide_index=True)
                show_download(
                    "Download current 200DMA allocation",
                    dynamic_allocation,
                    "sma200_current_allocation.csv",
                )

            if sma_backtest.empty:
                st.info("No walk-forward periods had enough saved data and eligible stocks.")
            else:
                scaled_sma_backtest = (sma_backtest / 100000.0) * float(sma_investment_amount)
                normalized_sma_view = (
                    sma_normalized
                    if not sma_normalized.empty
                    else (sma_backtest / 100000.0) * 100.0
                )
                sma_view = st.radio(
                    "Backtest view",
                    ["Actual amount", "Rs. 100 normalized"],
                    horizontal=True,
                    key="sma200_backtest_view",
                )
                sma_chart_source = (
                    scaled_sma_backtest if sma_view == "Actual amount" else normalized_sma_view
                )
                sma_curve_frame = sma_chart_source.reset_index(names="Date").melt(
                    "Date",
                    var_name="Series",
                    value_name="Value",
                )
                sma_curve_chart = px.line(
                    sma_curve_frame,
                    x="Date",
                    y="Value",
                    color="Series",
                )
                sma_curve_chart.update_layout(
                    yaxis_title="Portfolio Value" if sma_view == "Actual amount" else "Value from Rs. 100",
                    xaxis_title="",
                )
                st.plotly_chart(sma_curve_chart, use_container_width=True)

                sma_backtest_columns = st.columns(2)
                with sma_backtest_columns[0]:
                    st.markdown("**Monthly Rebalance Periods**")
                    st.dataframe(sma_periods, use_container_width=True, hide_index=True)
                    show_download(
                        "Download 200DMA rebalance periods",
                        sma_periods,
                        "sma200_backtest_periods.csv",
                    )
                with sma_backtest_columns[1]:
                    st.markdown("**Performance**")
                    scaled_summary = performance_summary(scaled_sma_backtest)
                    st.dataframe(
                        scaled_summary if not scaled_summary.empty else sma_performance,
                        use_container_width=True,
                        hide_index=True,
                    )
