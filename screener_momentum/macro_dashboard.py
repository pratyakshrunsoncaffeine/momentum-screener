"""Streamlit surface for macro data, sensitivities and saved runs."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from .macro_analysis import MacroCorrelationConfig, aligned_returns, analysis_features, analyze, market_pairs, scenario_model
from .macro_data import CATALOGUE, MacroStore, PublicMacroProvider, clean_observations, save_csv
from .correlation import MacroFactorProvider
from .index_momentum import NSE_INDEX_CATALOGUE
from .sector_rotation import NseSectorIndexProvider


def available_indices():
    return sorted(set([name for category, name in NSE_INDEX_CATALOGUE if category in {"Sectoral", "Thematic"}]
                      + ["Nifty Bank", "Nifty Financial Services"]))


def load_prices(store, assets, start, end, progress, full=False):
    path = store.root / "macro_prices.csv"
    saved = pd.read_csv(path, parse_dates=["Date"]) if path.exists() else pd.DataFrame()
    previous = saved.pivot(index="Date", columns="Asset", values="Close") if not saved.empty else pd.DataFrame()
    pieces, health = [], []
    for i, asset in enumerate(list(dict.fromkeys(["Nifty 50", *assets]))):
        progress(i, len(assets)+1, asset)
        try:
            old = previous[asset].dropna() if asset in previous else pd.Series(dtype=float)
            recent = not old.empty and old.index.max() >= pd.Timestamp(end)-pd.Timedelta(days=4) and old.index.min() <= pd.Timestamp(start)+pd.Timedelta(days=10)
            if recent and not full:
                series = old
            elif asset.endswith(".NS"):
                series = MacroFactorProvider._fetch_yahoo(asset, start, end)
            else:
                f = NseSectorIndexProvider(store.root / "indices").fetch_index(asset, start, end)
                series = f.set_index("Date").Close
            if series.empty:
                raise ValueError("No prices returned")
            if len(old):
                series = pd.concat([old, series]).groupby(level=0).last().sort_index()
            pieces.append(series.rename(asset))
            health.append({"Asset": asset, "Status": "Loaded", "Last date": series.index.max(), "Source": "Yahoo adjusted" if asset.endswith(".NS") else "NSE price index"})
        except Exception as exc:
            if asset in previous:
                pieces.append(previous[asset])
            health.append({"Asset": asset, "Status": "Cached fallback" if asset in previous else "Unavailable", "Message": str(exc)[:250]})
        if pieces:
            current = pd.concat(pieces, axis=1)
            all_prices = previous.combine_first(current)
            all_prices.update(current)
            save_csv(all_prices.rename_axis("Date").reset_index().melt("Date", var_name="Asset", value_name="Close").dropna(), path)
        save_csv(pd.DataFrame(health), store.root / "macro_price_health.csv")
    if not pieces:
        return pd.DataFrame()
    prices = pd.concat(pieces, axis=1).sort_index()
    if "Nifty 50" not in prices or prices["Nifty 50"].dropna().empty:
        raise ValueError("Nifty 50 calendar unavailable. Download again or restore a saved run.")
    return prices.loc[prices["Nifty 50"].notna()]


def import_panel(store):
    with st.expander("Import official data"):
        identifier = st.selectbox("Indicator", list(CATALOGUE), format_func=lambda x: CATALOGUE[x].name, key="macro_import_id")
        st.link_button("Official source", CATALOGUE[identifier].source)
        upload = st.file_uploader("Official CSV or Excel", type=["csv", "xlsx", "xls"], key="macro_import")
        if upload:
            try:
                if upload.name.lower().endswith("csv"):
                    raw = pd.read_csv(upload)
                else:
                    workbook = pd.ExcelFile(upload)
                    sheet = st.selectbox("Worksheet", workbook.sheet_names)
                    header = st.number_input("Header row (zero based)", 0, 100, 0, key="macro_import_header")
                    raw = pd.read_excel(workbook, sheet_name=sheet, header=int(header))
                raw.columns = raw.columns.astype(str)
                st.dataframe(raw.head(10), hide_index=True)
                period = st.selectbox("Period-end date column", raw.columns, key="macro_period")
                value = st.selectbox("Value column", raw.columns, key="macro_value")
                release = st.selectbox("Publication date column", ["Not available", *raw.columns], key="macro_release")
                base = st.text_input("Base year / unit definition", key="macro_base")
                original = st.checkbox("Values are the original values published on the supplied release dates", key="macro_original")
                now = pd.Timestamp.now("UTC").tz_localize(None)
                dates = pd.to_datetime(raw[period], errors="coerce")
                frequency = "Q" if CATALOGUE[identifier].frequency == "Q" else "M"
                dates = dates.dt.to_period(frequency).dt.to_timestamp(frequency)
                frame = pd.DataFrame({"series_id": identifier, "period": dates,
                    "value": pd.to_numeric(raw[value].astype(str).str.replace(",", "", regex=False), errors="coerce"),
                    "available_at": now if release == "Not available" else pd.to_datetime(raw[release], errors="coerce").dt.normalize()+pd.Timedelta(hours=23, minutes=59, seconds=59),
                    "retrieved_at": now, "vintage": hashlib.sha256(upload.getvalue()).hexdigest()[:16],
                    "eligible": int(original and release != "Not available"), "base_year": base,
                    "source": CATALOGUE[identifier].source})
                if not base:
                    st.info("Enter the official base year and units before importing.")
                elif st.button("Validate and import", key="macro_import_run"):
                    store.put(clean_observations(frame))
                    st.success(f"Imported {len(frame):,} observations.")
            except Exception as exc:
                st.error(str(exc))


def relationship_view(store, result, config, as_of):
    if result.empty:
        st.info("No saved relationships yet.")
        return
    tab1, tab2, tab3 = st.tabs(["Indicator Explorer", "Index / Stock Explorer", "All Relationships"])
    with tab1:
        choice = st.selectbox("Macro indicator", result.Indicator.unique(), key="macro_by_indicator")
        view = result[result.Indicator == choice]
        show_rankings(view, "Asset")
    with tab2:
        asset = st.selectbox("Index or stock", result.Asset.unique(), key="macro_by_asset")
        view = result[result.Asset == asset]
        show_rankings(view, "Indicator")
    with tab3:
        st.dataframe(result, hide_index=True)
        st.download_button("Download relationships", result.to_csv(index=False), "macro_relationships.csv")
        matrix = result.pivot(index="Indicator", columns="Asset", values="Correlation").dropna(how="all")
        if not matrix.empty:
            st.plotly_chart(px.imshow(matrix, zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto"), use_container_width=True)
    with st.expander("Relationship charts"):
        indicator = st.selectbox("Chart factor", list(result["Indicator ID"].unique()), format_func=lambda x: CATALOGUE[x].name, key="macro_chart_factor")
        asset = st.selectbox("Chart asset", result.Asset.unique(), key="macro_chart_asset")
        path = store.root / "macro_prices.csv"
        if path.exists():
            p = pd.read_csv(path, parse_dates=["Date"]).pivot(index="Date", columns="Asset", values="Close")
            if "Nifty 50" in p:
                p = p.loc[p["Nifty 50"].notna()]
            if CATALOGUE[indicator].frequency == "D":
                f = market_pairs(store.get(), p, indicator, asset, config, as_of)
            else:
                h = analysis_features(store.get(), indicator, config, as_of)
                f = aligned_returns(h, p, asset, config, as_of)
            if not f.empty:
                st.plotly_chart(px.scatter(f, x="x", y="y", hover_data=["signal_date", "entry_date", "exit_date"], labels={"x": CATALOGUE[indicator].name+" ("+CATALOGUE[indicator].transform+")", "y": "Forward return %" if config.relationship == "forward" else "Same-period return %"}), use_container_width=True)
                st.line_chart(f.set_index("signal_date")[["x"]])
                st.line_chart(f.set_index("signal_date")[["y"]])
                window = 12 if CATALOGUE[indicator].frequency == "Q" else 36
                st.line_chart(f.set_index("signal_date").x.rolling(window).corr(f.set_index("signal_date").y).rename("Rolling Pearson"))


def show_rankings(view, label):
    st.dataframe(view[[c for c in [label, "Correlation", "Pearson", "Spearman", "Observations", "Sampling", "Status", "Reason"] if c in view]], hide_index=True)
    if "Sample warning" in view and view["Sample warning"].fillna("").ne("").any():
        st.warning("Some correlations use short histories and may be unstable.")
    diagnostics = view[view.Correlation.isna() | view.Correlation.eq(0)]
    if not diagnostics.empty:
        with st.expander("Missing data and zero correlations", expanded=view.Correlation.fillna(0).eq(0).all()):
            columns = [label, "Observations", "Required observations", "Status", "Reason"]
            st.dataframe(diagnostics[[c for c in columns if c in diagnostics]], hide_index=True)
    positive, negative = st.columns(2)
    for container, mask, title, ascending in [(positive, view.Correlation > 0, "Positive relationships", False), (negative, view.Correlation < 0, "Negative relationships", True)]:
        with container:
            st.markdown(f"**{title}**")
            subset = view[mask].sort_values("Correlation", ascending=ascending).head(10)
            if subset.empty:
                st.info("No positive correlations." if not ascending else "No negative correlations.")
            else:
                st.plotly_chart(px.bar(subset, x="Correlation", y=label, orientation="h", hover_data=["Observations", "Status"]), use_container_width=True)
                st.dataframe(subset[[label, "Correlation", "Observations", "Status"]], hide_index=True)


def render_macro_dashboard(output_dir):
    st.subheader("Macroeconomic Correlation")
    store = MacroStore(Path(output_dir) / "macro")
    controls = st.columns(3)
    years = controls[0].slider("History (years)", 5, 20, 15, key="macro_years")
    horizon = controls[1].selectbox("Forward return horizon", [63, 21], format_func=lambda x: "3 months" if x == 63 else "1 month", key="macro_horizon")
    excess = controls[2].toggle("Excess return versus Nifty 50", key="macro_excess")
    as_of = st.date_input("Analysis date", date.today(), max_value=date.today(), key="macro_asof")
    config = MacroCorrelationConfig(years=years, horizon=horizon, excess=excess)
    advanced = st.columns(2)
    relationship = advanced[0].selectbox("Relationship timing", ["forward", "same_period"], format_func=lambda x: "Subsequent returns" if x == "forward" else "Same-period movement (retrospective)", key="macro_timing")
    lag = advanced[1].selectbox("Additional factor lag (native reporting periods)", [0, 1, 3, 6], key="macro_lag")
    minimum_years = st.selectbox("Minimum matched history (years)", [2, 3], key="macro_minimum_years")
    market_frequency = st.selectbox("Market-price sampling", ["W-FRI", "D"], format_func=lambda x: "Weekly" if x == "W-FRI" else "Daily", key="macro_market_sampling")
    market_lag = st.selectbox("Market-factor lag (selected periods)", [0, 1, 4], key="macro_market_lag")
    st.caption("Market correlations are retrospective, not tradable same-day predictions. Global and Indian closing times differ. Gold and other World Bank commodities remain monthly-average series.")
    config = MacroCorrelationConfig(years=years, horizon=horizon, excess=excess, relationship=relationship, lag=lag, minimum_years=minimum_years, market_frequency=market_frequency, market_lag=market_lag)
    assets = st.multiselect("Sectoral and thematic indices", available_indices(), default=["Nifty FMCG", "Nifty Consumer Durables", "Nifty Bank"], key="macro_indices")
    ticker = st.text_input("Stock ticker (optional)", placeholder="RELIANCE.NS", key="macro_ticker").strip().upper()
    if ticker and not re.fullmatch(r"[A-Z0-9&_-]+\.NS", ticker):
        st.error("Enter the full NSE ticker, for example RELIANCE.NS.")
        return
    if ticker:
        assets.append(ticker)
    expanded = st.toggle("Full factor catalogue", key="macro_full_catalogue")
    core = ["brent_crude", "usd_inr", "us_10y_yield", "gold", "cpi_headline", "iip_total", "pfce_real", "real_gdp"]
    identifiers = st.multiselect("Macro indicators", list(CATALOGUE) if expanded else core, default=["brent_crude", "usd_inr", "us_10y_yield"], format_func=lambda x: CATALOGUE[x].name, key="macro_factors")
    actions = st.columns(3)
    run = actions[0].button("Run / Resume Macroeconomic Scan", key="macro_run")
    full = actions[1].button("Run Full Scan From Beginning", key="macro_full")
    saved = actions[2].button("Use Saved Macro Run", key="macro_saved")
    result_path = store.root / "macro_relationships.csv"
    manifest_path = store.root / "macro_manifest.csv"
    if (run or full) and identifiers and assets:
        bar, status = st.progress(0), st.empty()
        def progress(done, total, message):
            bar.progress(min(done/max(total, 1), 1.))
            status.text(f"{done:,} / {total:,}: {message}")
        try:
            start = (pd.Timestamp(as_of)-pd.DateOffset(years=years+1)).date()
            PublicMacroProvider(store).refresh(identifiers, start, as_of, progress, full)
            prices = load_prices(store, assets, start, as_of, progress, full)
            if prices.empty:
                raise ValueError("No market history available; the previous run is retained.")
            obs = store.get()
            result, _ = analyze(obs, prices, identifiers, config, as_of, progress,
                                checkpoint=lambda f: save_csv(f, store.root / "macro_relationships_partial.csv"))
            if result.empty or result.Observations.max() == 0:
                raise ValueError("No aligned observations. Inspect Data Coverage; the previous run is retained.")
            save_csv(result, result_path)
            save_csv(pd.DataFrame([{**asdict(config), "as_of": str(as_of), "assets": json.dumps(assets), "identifiers": json.dumps(identifiers), "version": 1, "input_hash": hashlib.sha256(obs.to_csv(index=False).encode()).hexdigest()}]), manifest_path)
            st.session_state["macro_result"] = result
            st.success("Macro run saved.")
        except Exception as exc:
            st.error(str(exc))
    if saved:
        if result_path.exists():
            st.session_state["macro_result"] = pd.read_csv(result_path)
        else:
            st.info("No saved run. Restore an archive or run the scanner.")
    result = st.session_state.get("macro_result", pd.DataFrame())
    views = st.tabs(["Relationships", "Scenario Analysis", "Data Coverage", "Saved Data"])
    with views[0]:
        st.caption("Positive/negative associations describe historical relationships. Revised histories use estimated release lags and are exploratory.")
        if manifest_path.exists() and not result.empty:
            meta = pd.read_csv(manifest_path).iloc[0]
            st.caption(f"Saved analysis: {meta['as_of']} | {meta['horizon']} sessions | Excess returns: {meta['excess']}")
            view_config = MacroCorrelationConfig(years=int(meta['years']), horizon=int(meta['horizon']), excess=str(meta['excess']).lower() == 'true', relationship=str(meta.get('relationship', 'forward')), lag=int(meta.get('lag', 0)))
            view_config = replace(view_config, market_frequency=str(meta.get('market_frequency', 'W-FRI')), market_lag=int(meta.get('market_lag', 0)))
            if 'market_frequency' not in meta.index:
                st.warning("Saved run uses the old market alignment. Run a new scan before interpreting oil, FX or yield correlations.")
            relationship_view(store, result, view_config, pd.Timestamp(meta['as_of']).date())
        else:
            relationship_view(store, result, config, as_of)
    with views[1]:
        st.caption("Enter hypothetical currently available macro values. Return estimates start after the scenario date. Original release vintages are required.")
        selected = st.multiselect("Scenario factors (up to five)", identifiers, max_selections=5, format_func=lambda x: CATALOGUE[x].name, key="macro_scenario_factors")
        target = st.selectbox("Scenario index / stock", assets or ["Nifty FMCG"], key="macro_scenario_target")
        values = {i: st.number_input(f"{CATALOGUE[i].name}: {CATALOGUE[i].transform} (%)", value=10., key="macro_scenario_"+i) for i in selected}
        if st.button("Calculate Scenario", key="macro_scenario_run"):
            try:
                prices = pd.read_csv(store.root / "macro_prices.csv", parse_dates=["Date"]).pivot(index="Date", columns="Asset", values="Close")
                prices = prices.loc[prices["Nifty 50"].notna()]
                summary, evaluation = scenario_model(store.get(), prices, selected, target, values, config, as_of)
                save_csv(summary, store.root / "macro_scenario.csv")
                save_csv(evaluation, store.root / "macro_evaluation.csv")
                st.dataframe(summary, hide_index=True)
                st.line_chart(evaluation.set_index("signal_date")[["y", "predicted"]])
            except Exception as exc:
                st.warning(str(exc))
        for file in ["macro_scenario.csv", "macro_evaluation.csv"]:
            if (store.root / file).exists():
                st.download_button("Download " + file, (store.root / file).read_bytes(), file, key=file)
    with views[2]:
        coverage = store.coverage()
        st.dataframe(coverage, hide_index=True)
        st.download_button("Download coverage catalogue", coverage.to_csv(index=False), "macro_catalogue.csv")
        for file in ["macro_health.csv", "macro_price_health.csv"]:
            if (store.root / file).exists():
                st.dataframe(pd.read_csv(store.root / file), hide_index=True)
        import_panel(store)
    with views[3]:
        st.caption("Saved files on Streamlit may be lost after a restart or redeployment. Keep a recovery archive for restoration.")
        st.download_button("Download Recovery Archive", store.export(), "macro_recovery.zip", "application/zip")
        archive = st.file_uploader("Restore Macro Archive", type=["zip"], key="macro_restore")
        if archive and st.button("Restore Saved Data", key="macro_restore_run"):
            try:
                store.restore(archive.getvalue())
                st.success("Archive restored. Select Use Saved Macro Run to load results.")
            except Exception as exc:
                st.error(str(exc))
