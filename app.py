from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import plotly.express as px
import streamlit as st

from screener_momentum.backtest import current_allocation, performance_summary
from screener_momentum.config import (
    DEFAULT_POST_EARNINGS_STOCK_RETURN_WEIGHTS,
    DEFAULT_MOMENTUM_WEIGHTS,
    DEFAULT_POSITIVE_RETURN_FILTERS,
    DerivativesSignalConfig,
    FundamentalThresholds,
    ScreeningConfig,
)
from screener_momentum.pipeline import (
    download_derivatives_eod_data,
    finalize_dii_momentum_screen,
    finalize_fii_momentum_screen,
    finalize_quarterly_results_screen,
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
    controls = st.columns([1, 1, 1, 1])
    fii_top_n = controls[0].number_input("FII shortlist", min_value=10, max_value=200, value=50, step=5)
    fii_final_n = controls[1].number_input("Final picks", min_value=1, max_value=20, value=3, step=1)
    run_fii = controls[2].button("Run / Resume FII Scan", type="primary", use_container_width=True)
    recover_fii = controls[3].button("Use Saved FII Scan", use_container_width=True)

    if run_fii:
        fii_progress = make_progress("Scraping Screener.in FII holdings and market caps")
        price_progress = make_progress("Fetching shortlist prices")
        try:
            st.session_state["fii_results"] = run_fii_momentum_screen(
                csv_path,
                config,
                fii_top_n=int(fii_top_n),
                final_n=int(fii_final_n),
                progress_callback=fii_progress,
                price_progress_callback=price_progress,
                output_dir=OUTPUT_DIR,
            )
            st.success("FII accumulation scan complete.")
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
    controls = st.columns([1, 1, 1, 1])
    dii_top_n = controls[0].number_input("DII shortlist", min_value=10, max_value=200, value=50, step=5)
    dii_final_n = controls[1].number_input("DII final picks", min_value=1, max_value=20, value=3, step=1)
    run_dii = controls[2].button("Run / Resume DII Scan", type="primary", use_container_width=True)
    recover_dii = controls[3].button("Use Saved DII Scan", use_container_width=True)

    if run_dii:
        dii_progress = make_progress("Scraping Screener.in DII holdings and market caps")
        price_progress = make_progress("Fetching DII shortlist prices")
        try:
            st.session_state["dii_results"] = run_dii_momentum_screen(
                csv_path,
                config,
                dii_top_n=int(dii_top_n),
                final_n=int(dii_final_n),
                progress_callback=dii_progress,
                price_progress_callback=price_progress,
                output_dir=OUTPUT_DIR,
            )
            st.success("DII accumulation scan complete.")
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
    controls = st.columns([1.1, 1.4, 1, 1])
    target_quarter = controls[0].text_input("Result quarter", value="Jun 2026", help="Use the quarter label shown by Screener.in, for example Jun 2026.")
    ranking_metric = controls[1].radio(
        "Rank by YoY growth",
        ["Sales", "Operating Profit", "Net Profit", "EPS"],
        horizontal=True,
    )
    run_quarterly = controls[2].button("Run / Resume Quarterly Scan", type="primary", use_container_width=True)
    recover_quarterly = controls[3].button("Use Saved Quarterly Scan", use_container_width=True)

    if run_quarterly:
        quarterly_progress = make_progress("Scraping Screener.in quarterly results")
        try:
            st.session_state["quarterly_results"] = run_quarterly_results_screen(
                csv_path,
                target_period=target_quarter,
                progress_callback=quarterly_progress,
                output_dir=OUTPUT_DIR,
            )
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
    download_controls = st.columns([1, 1, 1.25, 1])
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
    recover_derivatives = download_controls[3].button(
        "Use Saved Derivatives Run",
        use_container_width=True,
    )

    if download_derivatives:
        if derivatives_start > derivatives_end:
            st.error("Download start must be on or before download end.")
        else:
            derivatives_download_progress = make_progress("Downloading official NSE reports by date")
            try:
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
                st.success(f"NSE EOD cache updated. {completed_dates:,} requested dates are available.")
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
