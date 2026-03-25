import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta  # noqa: F401

from portfolio import (
    get_close_prices,
    build_portfolio_returns,
    cumulative_returns,
    annualised_return,
    annualised_volatility,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    drawdown_recovery_days,
    var,
    cvar,
    portfolio_beta,
    portfolio_alpha,
    portfolio_correlation,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Portfolio Analyser", page_icon="📈", layout="wide")
st.title("📈 Portfolio Analyser")

# ---------------------------------------------------------------------------
# Sidebar — analysis settings only
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
start_date  = st.sidebar.date_input("Start date", value=date(2023, 1, 1))
end_date    = st.sidebar.date_input("End date",   value=date.today())
benchmark   = st.sidebar.text_input("Benchmark ticker", value="SPY")
rfr         = st.sidebar.number_input("Risk-free rate (%)", value=4.0, step=0.25) / 100
roll_window = st.sidebar.number_input("Rolling vol. window (days)", min_value=5, max_value=252, value=21, step=1)
mc_sims     = st.sidebar.number_input("Monte Carlo simulations", min_value=100, max_value=2000, value=500, step=100)
mc_days     = st.sidebar.number_input("Monte Carlo horizon (days)", min_value=21, max_value=1260, value=252, step=21)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "portfolios" not in st.session_state:
    st.session_state.portfolios = {
        "My Portfolio": {
            "MSFT": 0.10, "GOOGL": 0.10, "NVDA": 0.10, "AVGO": 0.10, "CEG": 0.10,
            "NRG": 0.10, "MELI": 0.10, "AMZN": 0.10, "NFLX": 0.10, "JPM": 0.10,
        }
    }

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_builder, tab_analysis = st.tabs(["Portfolio Builder", "Analysis"])

# ===========================================================================
# TAB 1 — Portfolio Builder
# ===========================================================================
with tab_builder:
    st.header("Build your portfolios")

    # --- Add / remove portfolios ---
    col_add, col_del = st.columns(2)
    with col_add:
        new_name = st.text_input("New portfolio name", placeholder="e.g. Balanced")
        if st.button("Add portfolio", use_container_width=True) and new_name:
            if new_name not in st.session_state.portfolios:
                st.session_state.portfolios[new_name] = {}
                st.rerun()
            else:
                st.warning("Name already exists.")
    with col_del:
        del_name = st.selectbox("Remove portfolio", ["—"] + list(st.session_state.portfolios))
        if st.button("Remove portfolio", use_container_width=True) and del_name != "—":
            st.session_state.portfolios.pop(del_name, None)
            st.rerun()

    st.divider()

    # --- One sub-tab per portfolio ---
    if not st.session_state.portfolios:
        st.info("No portfolios yet — add one above.")
    else:
        port_tabs = st.tabs(list(st.session_state.portfolios.keys()))
        for ptab, port_name in zip(port_tabs, list(st.session_state.portfolios.keys())):
            with ptab:
                weights = st.session_state.portfolios[port_name]

                # Add ticker
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    new_ticker = st.text_input("Ticker", key=f"tick_{port_name}",
                                               placeholder="e.g. TSLA").upper().strip()
                with c2:
                    new_weight = st.number_input("Weight (%)", min_value=0.01, max_value=100.0,
                                                 value=10.0, step=5.0, key=f"w_{port_name}")
                with c3:
                    st.write("")
                    st.write("")
                    if st.button("Add", key=f"add_{port_name}", use_container_width=True) and new_ticker:
                        weights[new_ticker] = new_weight / 100
                        st.session_state.portfolios[port_name] = weights
                        st.rerun()

                if not weights:
                    st.info("No tickers yet.")
                else:
                    total = sum(weights.values())
                    left, right = st.columns([3, 2])

                    with left:
                        st.markdown("**Current allocations**")
                        rows = [{"Ticker": t, "Weight (%)": round(w / total * 100, 2)}
                                for t, w in weights.items()]
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                        if abs(total - 1.0) > 0.01:
                            st.warning(f"Weights sum to {total*100:.1f}% — will be normalised on run.")

                        rc1, rc2 = st.columns([3, 1])
                        with rc1:
                            tick_del = st.selectbox("Remove ticker", ["—"] + list(weights),
                                                    key=f"del_{port_name}")
                        with rc2:
                            st.write("")
                            st.write("")
                            if st.button("Remove", key=f"rem_{port_name}", use_container_width=True) \
                                    and tick_del != "—":
                                weights.pop(tick_del, None)
                                st.session_state.portfolios[port_name] = weights
                                st.rerun()

                    with right:
                        st.markdown("**Allocation pie**")
                        fig_pie, ax_pie = plt.subplots(figsize=(3.5, 3.5))
                        sizes  = [w / total for w in weights.values()]
                        ax_pie.pie(sizes, labels=list(weights.keys()), autopct="%1.1f%%",
                                   startangle=90, textprops={"fontsize": 9})
                        ax_pie.axis("equal")
                        plt.tight_layout()
                        st.pyplot(fig_pie)
                        plt.close(fig_pie)

    st.divider()
    if st.button("▶ Run Analysis", type="primary", use_container_width=True):
        st.session_state["run_analysis"] = True
        st.rerun()

# ===========================================================================
# TAB 2 — Analysis
# ===========================================================================
with tab_analysis:
    if not st.session_state.get("run_analysis"):
        st.info("Configure your portfolios in **Portfolio Builder**, then click **▶ Run Analysis**.")
        st.stop()

    portfolios = {k: v for k, v in st.session_state.portfolios.items() if v}
    if not portfolios:
        st.error("Add at least one portfolio with tickers before running.")
        st.stop()

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    with st.spinner("Downloading prices…"):
        all_tickers = list({t for w in portfolios.values() for t in w} | {benchmark})
        try:
            prices = get_close_prices(all_tickers, start_str, end_str)
        except Exception as e:
            st.error(f"Failed to download prices: {e}")
            st.stop()

    benchmark_returns = prices[benchmark].pct_change().dropna()

    metrics       = {}
    cum_ret_all   = {}
    daily_ret_all = {}

    for name, weights in portfolios.items():
        ret = build_portfolio_returns(prices, weights)
        cum = cumulative_returns(ret)
        cum_ret_all[name]   = cum
        daily_ret_all[name] = ret

        beta  = portfolio_beta(ret, benchmark_returns)
        alpha = portfolio_alpha(ret, benchmark_returns, beta, rfr)
        rec   = drawdown_recovery_days(ret)

        metrics[name] = {
            "Ann. Return (%)":                round(annualised_return(ret) * 100, 2),
            "Ann. Volatility (%)":            round(annualised_volatility(ret) * 100, 2),
            "Sharpe Ratio":                   round(sharpe_ratio(ret, rfr), 4),
            "Sortino Ratio":                  round(sortino_ratio(ret, rfr), 4),
            "Calmar Ratio":                   round(calmar_ratio(ret), 4),
            "Max Drawdown (%)":               round(max_drawdown(ret) * 100, 2),
            "Max DD Recovery (days)":         rec if rec is not None else "Not recovered",
            "VaR 95% (%)":                    round(var(ret, 0.95) * 100, 2),
            "CVaR 95% (%)":                   round(cvar(ret, 0.95) * 100, 2),
            f"Beta (vs. {benchmark})":        round(beta, 4),
            f"Correlation (vs. {benchmark})": round(portfolio_correlation(ret, benchmark_returns), 4),
            f"Alpha (%) (vs. {benchmark})":   round(alpha * 100, 2),
            "Cumulative Return (%)":          round(cum.iloc[-1] * 100, 2),
        }

    metrics_df = pd.DataFrame(metrics).T
    cols = list(metrics_df.columns)
    cols.insert(1, cols.pop(cols.index("Cumulative Return (%)")))
    metrics_df = metrics_df[cols]
    n = len(daily_ret_all)

    # -----------------------------------------------------------------------
    # Metrics table
    # -----------------------------------------------------------------------
    st.subheader("📊 Metrics")
    st.dataframe(metrics_df, use_container_width=True)

    # -----------------------------------------------------------------------
    # Cumulative returns
    # -----------------------------------------------------------------------
    st.subheader("📈 Cumulative Returns")
    benchmark_cum = cumulative_returns(benchmark_returns)
    fig_cum = go.Figure()
    for name, cr in cum_ret_all.items():
        fig_cum.add_trace(go.Scatter(x=cr.index, y=(cr * 100).round(2), mode="lines", name=name))
    fig_cum.add_trace(go.Scatter(
        x=benchmark_cum.index, y=(benchmark_cum * 100).round(2),
        mode="lines", name=benchmark,
        line=dict(color="gray", dash="dash"),
    ))
    fig_cum.add_hline(y=0, line=dict(color="red", dash="dash", width=1))
    fig_cum.update_layout(
        title=f"Cumulative Returns ({start_str} → {end_str})",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # -----------------------------------------------------------------------
    # Rolling annualised volatility
    # -----------------------------------------------------------------------
    fig_vol = go.Figure()
    for name, daily_ret in daily_ret_all.items():
        roll_vol = daily_ret.rolling(roll_window).std() * np.sqrt(252) * 100
        fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.round(2), mode="lines", name=name))
    bm_roll_vol = benchmark_returns.rolling(roll_window).std() * np.sqrt(252) * 100
    fig_vol.add_trace(go.Scatter(
        x=bm_roll_vol.index, y=bm_roll_vol.round(2),
        mode="lines", name=benchmark,
        line=dict(color="gray", dash="dash"),
    ))
    fig_vol.update_layout(
        title=f"Rolling Annualised Volatility ({roll_window}-day window)",
        xaxis_title="Date",
        yaxis_title="Annualised Volatility (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # -----------------------------------------------------------------------
    # Underwater chart
    # -----------------------------------------------------------------------
    st.subheader("🌊 Underwater Chart (Drawdown from Peak)")
    fig_uw = make_subplots(rows=n, cols=1, shared_xaxes=False,
                           subplot_titles=[f"{nm} — Underwater Chart" for nm in daily_ret_all])
    for i, (name, daily_ret) in enumerate(daily_ret_all.items(), start=1):
        cum      = (1 + daily_ret).cumprod()
        peak     = cum.cummax()
        drawdown = (cum - peak) / peak * 100

        max_dd      = drawdown.min()
        max_dd_date = drawdown.idxmin()

        fig_uw.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.round(2),
            fill="tozeroy", fillcolor="rgba(220,20,60,0.35)",
            line=dict(color="darkred", width=0.8),
            name=name, showlegend=False,
        ), row=i, col=1)
        fig_uw.add_hline(y=0, line=dict(color="black", dash="dash", width=0.8), row=i, col=1)
        fig_uw.add_annotation(
            x=max_dd_date, y=max_dd,
            text=f"Max DD: {max_dd:.1f}%",
            showarrow=True, arrowhead=2, arrowcolor="darkred",
            font=dict(color="darkred", size=9),
            row=i, col=1,
        )
        fig_uw.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=i, col=1)
        fig_uw.update_xaxes(title_text="Date", row=i, col=1)

    fig_uw.update_layout(height=400 * n)
    st.plotly_chart(fig_uw, use_container_width=True)

    # -----------------------------------------------------------------------
    # Monthly returns heatmap
    # -----------------------------------------------------------------------
    st.subheader("🗓️ Monthly Returns Heatmap")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    col_labels = month_labels + ["Ann.", "Cum."]

    def build_heatmap_pivot(daily_ret):
        monthly     = (1 + daily_ret).resample("ME").prod() - 1
        monthly_pct = monthly * 100
        pivot = monthly_pct.groupby([monthly_pct.index.year, monthly_pct.index.month]).first()
        pivot = pivot.unstack(level=1)
        pivot.columns = [month_labels[m - 1] for m in pivot.columns]
        pivot = pivot.reindex(columns=month_labels)
        annual = monthly_pct.groupby(monthly_pct.index.year).apply(
            lambda x: (1 + x / 100).prod() - 1
        ) * 100
        pivot["Ann."] = annual
        monthly_sorted = monthly_pct.sort_index()
        cum_factors = (1 + monthly_sorted / 100).cumprod()
        cum_by_year = cum_factors.groupby(cum_factors.index.year).last()
        pivot["Cum."] = (cum_by_year - 1) * 100
        return pivot

    def draw_heatmap_plotly(name, daily_ret):
        pivot = build_heatmap_pivot(daily_ret)
        z     = pivot.values
        text  = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.1f}%")(z))
        fig = go.Figure(go.Heatmap(
            z=z,
            x=col_labels,
            y=[str(yr) for yr in pivot.index],
            colorscale="RdYlGn",
            zmid=0,
            zmin=-10,
            zmax=10,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="Return (%)"),
            hovertemplate="Year: %{y}<br>Period: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=f"{name} — Monthly Returns (%)", font=dict(size=14)),
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
            height=max(250, 60 * len(pivot.index) + 100),
            margin=dict(l=60, r=60, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    heatmap_items = list(daily_ret_all.items()) + [(benchmark, benchmark_returns)]
    for name, daily_ret in heatmap_items:
        draw_heatmap_plotly(name, daily_ret)

    # -----------------------------------------------------------------------
    # Consistency table
    # -----------------------------------------------------------------------
    st.subheader("📅 Consistency")
    bm_monthly_pct = ((1 + benchmark_returns).resample("ME").prod() - 1) * 100
    consistency_rows = {}
    for name, daily_ret in daily_ret_all.items():
        monthly = (1 + daily_ret).resample("ME").prod() - 1
        monthly_pct = monthly * 100
        aligned = monthly_pct.align(bm_monthly_pct, join="inner")
        port_aligned, bm_aligned = aligned
        consistency_rows[name] = {
            "Positive Months":            int((monthly_pct > 0).sum()),
            "Negative Months":            int((monthly_pct < 0).sum()),
            "Best Month (%)":             round(monthly_pct.max(), 2),
            "Worst Month (%)":            round(monthly_pct.min(), 2),
            f"Months above {benchmark}":   int((port_aligned > bm_aligned).sum()),
            f"Months below {benchmark}":   int((port_aligned < bm_aligned).sum()),
        }
    st.dataframe(pd.DataFrame(consistency_rows).T, use_container_width=True)

    # -----------------------------------------------------------------------
    # Top-performing stock per month
    # -----------------------------------------------------------------------
    st.subheader("🏆 Top-Performing Stock per Month")

    # All portfolio tickers (excluding benchmark)
    port_tickers = list({t for w in portfolios.values() for t in w})
    ticker_prices = prices[port_tickers]
    monthly_ticker = ticker_prices.resample("ME").last().pct_change().dropna() * 100

    top_rows = []
    for dt, row in monthly_ticker.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        best_ticker = valid.idxmax()
        top_rows.append({
            "Month": dt.strftime("%b %Y"),
            "Top Stock": best_ticker,
            "Return (%)": round(valid[best_ticker], 2),
            "Worst Stock": valid.idxmin(),
            "Worst Return (%)": round(valid.min(), 2),
        })

    if top_rows:
        top_df = pd.DataFrame(top_rows).set_index("Month")

        def color_return(val):
            color = "green" if val > 0 else "red"
            return f"color: {color}"

        st.dataframe(
            top_df.style
                .map(color_return, subset=["Return (%)", "Worst Return (%)"])
                .format({"Return (%)": "{:+.2f}%", "Worst Return (%)": "{:+.2f}%"}),
            use_container_width=True,
        )

    # -----------------------------------------------------------------------
    # Monte Carlo
    # -----------------------------------------------------------------------
    st.subheader("🎲 Monte Carlo Simulation")
    rng = np.random.default_rng(42)
    fig_mc = make_subplots(rows=n, cols=1,
                           subplot_titles=[f"{nm} — Monte Carlo" for nm in daily_ret_all])
    for i, (name, daily_ret) in enumerate(daily_ret_all.items(), start=1):
        r       = daily_ret.dropna().values
        sampled = rng.choice(r, size=(mc_sims, mc_days), replace=True)
        paths   = np.cumprod(1 + sampled, axis=1) - 1

        final_values = paths[:, -1] * 100
        p5   = np.percentile(paths,  5, axis=0) * 100
        p25  = np.percentile(paths, 25, axis=0) * 100
        p50  = np.percentile(paths, 50, axis=0) * 100
        p75  = np.percentile(paths, 75, axis=0) * 100
        p95  = np.percentile(paths, 95, axis=0) * 100
        days = np.arange(1, mc_days + 1)

        # Batch all simulation paths into one trace with None separators
        xs, ys = [], []
        for path in paths:
            xs.extend(days.tolist() + [None])
            ys.extend((path * 100).tolist() + [None])
        fig_mc.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="steelblue", width=0.5),
            opacity=0.08, name="Simulations", showlegend=(i == 1),
        ), row=i, col=1)

        # Confidence bands (5-95, 25-75)
        fig_mc.add_trace(go.Scatter(x=days, y=p95, mode="lines", line=dict(width=0), showlegend=False), row=i, col=1)
        fig_mc.add_trace(go.Scatter(x=days, y=p5, fill="tonexty",
            fillcolor="rgba(70,130,180,0.15)", line=dict(width=0),
            name="5–95th pct", showlegend=(i == 1)), row=i, col=1)
        fig_mc.add_trace(go.Scatter(x=days, y=p75, mode="lines", line=dict(width=0), showlegend=False), row=i, col=1)
        fig_mc.add_trace(go.Scatter(x=days, y=p25, fill="tonexty",
            fillcolor="rgba(70,130,180,0.30)", line=dict(width=0),
            name="25–75th pct", showlegend=(i == 1)), row=i, col=1)

        # Median line
        fig_mc.add_trace(go.Scatter(x=days, y=p50, mode="lines",
            line=dict(color="navy", width=2), name="Median", showlegend=(i == 1)), row=i, col=1)

        # Zero reference line
        fig_mc.add_hline(y=0, line=dict(color="black", dash="dash", width=0.8), row=i, col=1)

        # Annotations for final percentile values
        for pct_val, lbl in [(p5[-1], "5th"), (p50[-1], "50th"), (p95[-1], "95th")]:
            fig_mc.add_annotation(
                x=mc_days, y=pct_val,
                text=f"{pct_val:.1f}%", showarrow=False,
                xanchor="left", font=dict(color="navy", size=9),
                row=i, col=1,
            )

        fig_mc.update_yaxes(title_text="Cumulative Return (%)", row=i, col=1)
        fig_mc.update_xaxes(title_text="Trading Days", row=i, col=1)

    fig_mc.update_layout(height=500 * n)
    st.plotly_chart(fig_mc, use_container_width=True)

    # -----------------------------------------------------------------------
    # Monte Carlo — Student-t
    # -----------------------------------------------------------------------
    st.subheader("🎲 Monte Carlo Simulation (Student-t)")
    rng_t = np.random.default_rng(42)
    fig_mct = make_subplots(rows=n, cols=1,
                            subplot_titles=[f"{nm} — Monte Carlo (Student-t)" for nm in daily_ret_all])
    for i, (name, daily_ret) in enumerate(daily_ret_all.items(), start=1):
        r = daily_ret.dropna().values
        mu, sigma = r.mean(), r.std(ddof=1)
        df_t, loc_t, scale_t = scipy.stats.t.fit(r)

        sampled = scipy.stats.t.rvs(df=df_t, loc=loc_t, scale=scale_t,
                                     size=(mc_sims, mc_days), random_state=rng_t)
        paths = np.cumprod(1 + sampled, axis=1) - 1

        p5  = np.percentile(paths,  5, axis=0) * 100
        p25 = np.percentile(paths, 25, axis=0) * 100
        p50 = np.percentile(paths, 50, axis=0) * 100
        p75 = np.percentile(paths, 75, axis=0) * 100
        p95 = np.percentile(paths, 95, axis=0) * 100
        days = np.arange(1, mc_days + 1)

        xs, ys = [], []
        for path in paths:
            xs.extend(days.tolist() + [None])
            ys.extend((path * 100).tolist() + [None])
        fig_mct.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="darkorange", width=0.5),
            opacity=0.08, name="Simulations", showlegend=(i == 1),
        ), row=i, col=1)

        fig_mct.add_trace(go.Scatter(x=days, y=p95, mode="lines", line=dict(width=0), showlegend=False), row=i, col=1)
        fig_mct.add_trace(go.Scatter(x=days, y=p5, fill="tonexty",
            fillcolor="rgba(255,140,0,0.15)", line=dict(width=0),
            name="5–95th pct", showlegend=(i == 1)), row=i, col=1)
        fig_mct.add_trace(go.Scatter(x=days, y=p75, mode="lines", line=dict(width=0), showlegend=False), row=i, col=1)
        fig_mct.add_trace(go.Scatter(x=days, y=p25, fill="tonexty",
            fillcolor="rgba(255,140,0,0.30)", line=dict(width=0),
            name="25–75th pct", showlegend=(i == 1)), row=i, col=1)

        fig_mct.add_trace(go.Scatter(x=days, y=p50, mode="lines",
            line=dict(color="saddlebrown", width=2), name="Median", showlegend=(i == 1)), row=i, col=1)

        fig_mct.add_hline(y=0, line=dict(color="black", dash="dash", width=0.8), row=i, col=1)

        for pct_val, lbl in [(p5[-1], "5th"), (p50[-1], "50th"), (p95[-1], "95th")]:
            fig_mct.add_annotation(
                x=mc_days, y=pct_val,
                text=f"{pct_val:.1f}%", showarrow=False,
                xanchor="left", font=dict(color="saddlebrown", size=9),
                row=i, col=1,
            )

        fig_mct.add_annotation(
            x=0.01, y=0.98, xref="paper", yref="paper",
            text=f"Fitted ν = {df_t:.1f}", showarrow=False,
            font=dict(size=10, color="saddlebrown"),
            xanchor="left", yanchor="top",
            row=i, col=1,
        )

        fig_mct.update_yaxes(title_text="Cumulative Return (%)", row=i, col=1)
        fig_mct.update_xaxes(title_text="Trading Days", row=i, col=1)

    fig_mct.update_layout(height=500 * n)
    st.plotly_chart(fig_mct, use_container_width=True)

    # -----------------------------------------------------------------------
    # Monte Carlo — Cauchy
    # -----------------------------------------------------------------------
    st.subheader("🎲 Monte Carlo Simulation (Cauchy)")
    rng_c = np.random.default_rng(42)
    fig_mcc = make_subplots(rows=n, cols=1,
                            subplot_titles=[f"{nm} — Monte Carlo (Cauchy)" for nm in daily_ret_all])
    for i, (name, daily_ret) in enumerate(daily_ret_all.items(), start=1):
        r = daily_ret.dropna().values
        loc_c, scale_c = scipy.stats.cauchy.fit(r)

        sampled = scipy.stats.cauchy.rvs(loc=loc_c, scale=scale_c,
                                          size=(mc_sims, mc_days), random_state=rng_c)
        sampled = np.clip(sampled, -0.5, 0.5)
        paths = np.cumprod(1 + sampled, axis=1) - 1

        p5  = np.percentile(paths,  5, axis=0) * 100
        p25 = np.percentile(paths, 25, axis=0) * 100
        p50 = np.percentile(paths, 50, axis=0) * 100
        p75 = np.percentile(paths, 75, axis=0) * 100
        p95 = np.percentile(paths, 95, axis=0) * 100
        days = np.arange(1, mc_days + 1)

        xs, ys = [], []
        for path in paths:
            xs.extend(days.tolist() + [None])
            ys.extend((path * 100).tolist() + [None])
        fig_mcc.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="purple", width=0.5),
            opacity=0.08, name="Simulations", showlegend=(i == 1),
        ), row=i, col=1)

        fig_mcc.add_trace(go.Scatter(x=days, y=p95, mode="lines", line=dict(width=0), showlegend=False), row=i, col=1)
        fig_mcc.add_trace(go.Scatter(x=days, y=p5, fill="tonexty",
            fillcolor="rgba(128,0,128,0.15)", line=dict(width=0),
            name="5–95th pct", showlegend=(i == 1)), row=i, col=1)
        fig_mcc.add_trace(go.Scatter(x=days, y=p75, mode="lines", line=dict(width=0), showlegend=False), row=i, col=1)
        fig_mcc.add_trace(go.Scatter(x=days, y=p25, fill="tonexty",
            fillcolor="rgba(128,0,128,0.30)", line=dict(width=0),
            name="25–75th pct", showlegend=(i == 1)), row=i, col=1)

        fig_mcc.add_trace(go.Scatter(x=days, y=p50, mode="lines",
            line=dict(color="indigo", width=2), name="Median", showlegend=(i == 1)), row=i, col=1)

        fig_mcc.add_hline(y=0, line=dict(color="black", dash="dash", width=0.8), row=i, col=1)

        for pct_val in [p5[-1], p50[-1], p95[-1]]:
            fig_mcc.add_annotation(
                x=mc_days, y=pct_val,
                text=f"{pct_val:.1f}%", showarrow=False,
                xanchor="left", font=dict(color="indigo", size=9),
                row=i, col=1,
            )

        fig_mcc.add_annotation(
            x=0.01, y=0.98, xref="paper", yref="paper",
            text=f"Fitted loc={loc_c:.4f}, scale={scale_c:.4f}", showarrow=False,
            font=dict(size=10, color="indigo"),
            xanchor="left", yanchor="top",
            row=i, col=1,
        )

        fig_mcc.update_yaxes(title_text="Cumulative Return (%)", row=i, col=1)
        fig_mcc.update_xaxes(title_text="Trading Days", row=i, col=1)

    fig_mcc.update_layout(height=500 * n)
    st.plotly_chart(fig_mcc, use_container_width=True)
