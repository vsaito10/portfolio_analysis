import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
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
    fig_cum, ax_cum = plt.subplots(figsize=(12, 5))
    for name, cr in cum_ret_all.items():
        ax_cum.plot(cr.index, cr * 100, label=name)
    ax_cum.set_title(f"Cumulative Returns ({start_str} → {end_str})")
    ax_cum.set_xlabel("Date")
    ax_cum.set_ylabel("Cumulative Return (%)")
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_cum)
    plt.close(fig_cum)

    # -----------------------------------------------------------------------
    # Underwater chart
    # -----------------------------------------------------------------------
    st.subheader("🌊 Underwater Chart (Drawdown from Peak)")
    fig_uw, axes_uw = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes_uw = [axes_uw]

    for ax, (name, daily_ret) in zip(axes_uw, daily_ret_all.items()):
        cum      = (1 + daily_ret).cumprod()
        peak     = cum.cummax()
        drawdown = (cum - peak) / peak * 100

        ax.fill_between(drawdown.index, drawdown, 0, color="crimson", alpha=0.55)
        ax.plot(drawdown.index, drawdown, color="darkred", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        max_dd      = drawdown.min()
        max_dd_date = drawdown.idxmin()
        ax.annotate(
            f"Max DD: {max_dd:.1f}%",
            xy=(max_dd_date, max_dd),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=9,
            color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
        )
        ax.set_title(f"{name} — Underwater Chart (Drawdown from Peak)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    plt.tight_layout()
    st.pyplot(fig_uw)
    plt.close(fig_uw)

    # -----------------------------------------------------------------------
    # Monthly returns heatmap
    # -----------------------------------------------------------------------
    st.subheader("🗓️ Monthly Returns Heatmap")
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    col_labels = month_labels + ["Ann."]
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    cmap = plt.get_cmap("RdYlGn")

    fig_m, axes_m = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes_m = [axes_m]

    for ax, (name, daily_ret) in zip(axes_m, daily_ret_all.items()):
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

        im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(13))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{name} — Monthly Returns (%)", fontsize=13, fontweight="bold")
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                val = pivot.values[row, col]
                if not np.isnan(val):
                    color = "white" if abs(val) > 6 else "black"
                    ax.text(col, row, f"{val:.1f}%", ha="center", va="center",
                            fontsize=9, color=color)
        plt.colorbar(im, ax=ax, label="Return (%)", fraction=0.02, pad=0.02)

    plt.tight_layout()
    st.pyplot(fig_m)
    plt.close(fig_m)

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
    fig_mc, axes_mc = plt.subplots(n, 1, figsize=(14, 5 * n))
    if n == 1:
        axes_mc = [axes_mc]

    for ax, (name, daily_ret) in zip(axes_mc, daily_ret_all.items()):
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

        for i in range(mc_sims):
            ax.plot(days, paths[i] * 100, color="steelblue", alpha=0.04, linewidth=0.5)

        ax.fill_between(days, p5,  p95, alpha=0.15, color="steelblue", label="5–95th pct")
        ax.fill_between(days, p25, p75, alpha=0.30, color="steelblue", label="25–75th pct")
        ax.plot(days, p50, color="navy", linewidth=2, label="Median")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        for pct_val, lbl in [(p5[-1], "5th"), (p50[-1], "50th"), (p95[-1], "95th")]:
            ax.annotate(f"{pct_val:.1f}%", xy=(mc_days, pct_val),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="navy")

        ax.set_title(
            f"{name} — Monte Carlo ({mc_sims} paths, {mc_days} days)  |  "
            f"5th: {np.percentile(final_values,5):.1f}%  "
            f"Median: {np.median(final_values):.1f}%  "
            f"95th: {np.percentile(final_values,95):.1f}%",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_mc)
    plt.close(fig_mc)
