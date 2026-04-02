import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta  # noqa: F401

from portfolio import (
    get_ohlc_data,
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
    hill_estimator,
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
    st.session_state.portfolios = {"My Portfolio": []}

# ---------------------------------------------------------------------------
# Heatmap helpers (used in both Analysis and Stock Analysis tabs)
# ---------------------------------------------------------------------------
_month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_col_labels = _month_labels + ["Ann.", "Cum."]

def build_heatmap_pivot(daily_ret):
    monthly     = (1 + daily_ret).resample("ME").prod() - 1
    monthly_pct = monthly * 100
    pivot = monthly_pct.groupby([monthly_pct.index.year, monthly_pct.index.month]).first()
    pivot = pivot.unstack(level=1)
    pivot.columns = [_month_labels[m - 1] for m in pivot.columns]
    pivot = pivot.reindex(columns=_month_labels)
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
        x=_col_labels,
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

# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------
def _portfolio_weights(purchases: list[dict]) -> dict[str, float]:
    """Derive normalised weights using remaining cost basis (net shares × avg buy price)."""
    buy_shares: dict[str, float] = {}
    buy_cost:   dict[str, float] = {}
    sell_shares: dict[str, float] = {}
    for p in purchases:
        t = p["ticker"]
        if p.get("type", "buy") == "buy":
            buy_shares[t] = buy_shares.get(t, 0.0) + p["shares"]
            buy_cost[t]   = buy_cost.get(t, 0.0)   + p["total"]
        else:
            sell_shares[t] = sell_shares.get(t, 0.0) + p["shares"]
    net: dict[str, float] = {}
    for t, bs in buy_shares.items():
        ns = bs - sell_shares.get(t, 0.0)
        if ns > 0 and bs > 0:
            net[t] = ns * (buy_cost[t] / bs)
    s = sum(net.values())
    return {t: v / s for t, v in net.items()} if s > 0 else {}

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_builder, tab_analysis, tab_stocks = st.tabs(["Portfolio Builder", "Analysis", "Stock Analysis"])

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
                st.session_state.portfolios[new_name] = []
                st.rerun()
            else:
                st.warning("Name already exists.")
    with col_del:
        del_name = st.selectbox("Remove portfolio", ["—"] + list(st.session_state.portfolios))
        if st.button("Remove portfolio", use_container_width=True) and del_name != "—":
            st.session_state.portfolios.pop(del_name, None)
            st.rerun()

    st.divider()

    if not st.session_state.portfolios:
        st.info("No portfolios yet — add one above.")
    else:
        port_tabs = st.tabs(list(st.session_state.portfolios.keys()))
        for ptab, port_name in zip(port_tabs, list(st.session_state.portfolios.keys())):
            with ptab:
                purchases = st.session_state.portfolios[port_name]

                # ── Add purchase form ────────────────────────────────────────
                st.markdown("#### Add Purchase")
                r1c1, r1c2 = st.columns([2, 2])
                with r1c1:
                    new_ticker = st.text_input(
                        "Ticker", key=f"tick_{port_name}", placeholder="e.g. AAPL",
                    ).upper().strip()
                with r1c2:
                    new_date = st.date_input(
                        "Purchase date", key=f"date_{port_name}", value=date.today(),
                    )

                r2c1, r2c2, r2c3 = st.columns([2, 2, 2])
                with r2c1:
                    new_shares = st.number_input(
                        "Number of shares", key=f"shares_{port_name}",
                        min_value=0.0001, value=1.0, step=1.0, format="%.4g",
                    )
                with r2c2:
                    new_price = st.number_input(
                        "Price per share", key=f"price_{port_name}",
                        min_value=0.0001, value=100.0, step=0.01, format="%.2f",
                    )
                with r2c3:
                    st.metric("Total invested", f"${new_shares * new_price:,.2f}")

                if st.button("Add Purchase", key=f"add_{port_name}", use_container_width=True) \
                        and new_ticker:
                    purchases.append({
                        "type":   "buy",
                        "ticker": new_ticker,
                        "date":   new_date.strftime("%Y-%m-%d"),
                        "shares": new_shares,
                        "price":  new_price,
                        "total":  round(new_shares * new_price, 2),
                    })
                    st.rerun()

                st.divider()

                # ── Add sale form ────────────────────────────────────────────
                st.markdown("#### Add Sale")
                s1c1, s1c2 = st.columns([2, 2])
                with s1c1:
                    sell_ticker = st.text_input(
                        "Ticker", key=f"sell_tick_{port_name}", placeholder="e.g. AAPL",
                    ).upper().strip()
                with s1c2:
                    sell_date = st.date_input(
                        "Sale date", key=f"sell_date_{port_name}", value=date.today(),
                    )

                s2c1, s2c2, s2c3 = st.columns([2, 2, 2])
                with s2c1:
                    sell_shares = st.number_input(
                        "Number of shares", key=f"sell_shares_{port_name}",
                        min_value=0.0001, value=1.0, step=1.0, format="%.4g",
                    )
                with s2c2:
                    sell_price = st.number_input(
                        "Sale price per share", key=f"sell_price_{port_name}",
                        min_value=0.0001, value=100.0, step=0.01, format="%.2f",
                    )
                with s2c3:
                    st.metric("Total received", f"${sell_shares * sell_price:,.2f}")

                if st.button("Add Sale", key=f"addsell_{port_name}", use_container_width=True) \
                        and sell_ticker:
                    purchases.append({
                        "type":   "sell",
                        "ticker": sell_ticker,
                        "date":   sell_date.strftime("%Y-%m-%d"),
                        "shares": sell_shares,
                        "price":  sell_price,
                        "total":  round(sell_shares * sell_price, 2),
                    })
                    st.rerun()

                st.divider()

                # ── Allocation summary + pie ─────────────────────────────────
                if not purchases:
                    st.info("No transactions yet — add one above.")
                else:
                    st.markdown("#### Allocation (net open positions)")
                    _buy_s: dict[str, float] = {}
                    _buy_c: dict[str, float] = {}
                    _sell_s: dict[str, float] = {}
                    for p in purchases:
                        t = p["ticker"]
                        if p.get("type", "buy") == "buy":
                            _buy_s[t] = _buy_s.get(t, 0.0) + p["shares"]
                            _buy_c[t] = _buy_c.get(t, 0.0) + p["total"]
                        else:
                            _sell_s[t] = _sell_s.get(t, 0.0) + p["shares"]
                    totals_by_ticker: dict[str, float] = {}
                    net_shares_by_ticker: dict[str, float] = {}
                    avg_price_by_ticker: dict[str, float] = {}
                    for t, bs in _buy_s.items():
                        ns = bs - _sell_s.get(t, 0.0)
                        if ns > 0 and bs > 0:
                            avg = _buy_c[t] / bs
                            totals_by_ticker[t]     = ns * avg
                            net_shares_by_ticker[t] = ns
                            avg_price_by_ticker[t]  = avg
                    total_portfolio = sum(totals_by_ticker.values())

                    if totals_by_ticker:
                        # Fetch latest close price for open tickers
                        open_tickers = list(totals_by_ticker.keys())
                        latest_prices: dict[str, float] = {}
                        _price_start = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
                        _price_end   = date.today().strftime("%Y-%m-%d")
                        try:
                            with st.spinner("Fetching latest prices…"):
                                _px = get_close_prices(open_tickers, _price_start, _price_end)
                            for t in open_tickers:
                                if t in _px.columns and not _px[t].dropna().empty:
                                    latest_prices[t] = float(_px[t].dropna().iloc[-1])
                        except Exception:
                            pass  # latest_prices stays empty; column shows N/A

                        left, right = st.columns([3, 2])
                        with left:
                            alloc_rows = []
                            for t, v in sorted(totals_by_ticker.items(), key=lambda x: -x[1]):
                                avg = avg_price_by_ticker[t]
                                cur = latest_prices.get(t)
                                ret = ((cur - avg) / avg * 100) if cur else None
                                alloc_rows.append({
                                    "Ticker":        t,
                                    "Shares":        round(net_shares_by_ticker[t], 2),
                                    "Avg Buy ($)":   avg,
                                    "Net Value ($)": v,
                                    "Weight (%)":    v / total_portfolio * 100,
                                    "Last ($)":      cur,
                                    "Return (%)":    ret,
                                })

                            def _color_return(val):
                                if isinstance(val, (int, float)):
                                    return ("color: #2e7d32; font-weight: bold" if val >= 0
                                            else "color: #c62828; font-weight: bold")
                                return ""

                            _fmt = {
                                "Avg Buy ($)":   "{:.2f}",
                                "Last ($)":      "{:.2f}",
                                "Return (%)":    "{:.2f}",
                                "Net Value ($)": "{:,.2f}",
                                "Weight (%)":    "{:.2f}",
                            }
                            st.dataframe(
                                pd.DataFrame(alloc_rows)
                                  .style
                                  .map(_color_return, subset=["Return (%)"])
                                  .format(_fmt, na_rep="N/A"),
                                use_container_width=True,
                                hide_index=True,
                            )
                            st.caption(f"Portfolio total: **${total_portfolio:,.2f}**")

                        with right:
                            fig_pie, ax_pie = plt.subplots(figsize=(3.5, 3.5))
                            ax_pie.pie(
                                list(totals_by_ticker.values()),
                                labels=list(totals_by_ticker.keys()),
                                autopct="%1.1f%%",
                                startangle=90,
                                textprops={"fontsize": 9},
                            )
                            ax_pie.axis("equal")
                            plt.tight_layout()
                            st.pyplot(fig_pie)
                            plt.close(fig_pie)
                    else:
                        st.info("All positions are closed (net value = 0).")

                    st.divider()

                    # ── Portfolio value evolution ────────────────────────────
                    st.markdown("#### Portfolio Value Evolution")

                    sorted_txns = sorted(purchases, key=lambda p: p["date"])
                    evo_dates, evo_values, evo_colors, evo_labels = [], [], [], []
                    running_total = 0.0
                    for p in sorted_txns:
                        is_buy = p.get("type", "buy") == "buy"
                        running_total += p["total"] if is_buy else -p["total"]
                        evo_dates.append(p["date"])
                        evo_values.append(round(running_total, 2))
                        evo_colors.append("#2e7d32" if is_buy else "#c62828")
                        evo_labels.append(
                            f"{'Buy' if is_buy else 'Sell'} {p['shares']:g} {p['ticker']}"
                            f" @ ${p['price']:.2f}"
                        )

                    fig_evo = go.Figure()
                    fig_evo.add_trace(go.Scatter(
                        x=evo_dates,
                        y=evo_values,
                        mode="lines+markers",
                        line=dict(shape="hv", color="steelblue", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(70,130,180,0.12)",
                        marker=dict(size=10, color=evo_colors,
                                    line=dict(color="white", width=1.5)),
                        customdata=evo_labels,
                        hovertemplate=(
                            "<b>%{customdata}</b><br>"
                            "Date: %{x}<br>"
                            "Net Invested: $%{y:,.2f}<extra></extra>"
                        ),
                    ))
                    fig_evo.add_hline(y=0, line=dict(color="black", dash="dash", width=0.8))
                    fig_evo.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Net Invested ($)",
                        yaxis_tickprefix="$",
                        yaxis_tickformat=",.0f",
                        hovermode="x unified",
                        height=350,
                        margin=dict(t=20),
                    )
                    st.plotly_chart(fig_evo, use_container_width=True)

                    # Summary metrics + realized P&L
                    total_bought = sum(p["total"] for p in purchases if p.get("type","buy")=="buy")
                    total_sold   = sum(p["total"] for p in purchases if p.get("type","buy")!="buy")
                    net_deployed = total_bought - total_sold

                    # Realized P&L via average cost method (chronological)
                    _avg_cost:    dict[str, float] = {}
                    _pos_shares:  dict[str, float] = {}
                    realized_pnl      = 0.0
                    cost_basis_sold   = 0.0
                    for p in sorted(purchases, key=lambda x: x["date"]):
                        t = p["ticker"]
                        if p.get("type", "buy") == "buy":
                            prev_s = _pos_shares.get(t, 0.0)
                            prev_c = _avg_cost.get(t, 0.0)
                            new_s  = prev_s + p["shares"]
                            _avg_cost[t]   = (prev_s * prev_c + p["total"]) / new_s
                            _pos_shares[t] = new_s
                        else:
                            avg  = _avg_cost.get(t, 0.0)
                            realized_pnl    += (p["price"] - avg) * p["shares"]
                            cost_basis_sold += avg * p["shares"]
                            _pos_shares[t]   = _pos_shares.get(t, 0.0) - p["shares"]

                    realized_pct = (realized_pnl / cost_basis_sold * 100
                                    if cost_basis_sold > 0 else 0.0)
                    pnl_sign = "+" if realized_pnl >= 0 else ""

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Total Bought",    f"${total_bought:,.2f}")
                    m2.metric("Total Sold",      f"${total_sold:,.2f}")
                    m3.metric("Net Deployed",    f"${net_deployed:,.2f}")
                    m4.metric("Realized P&L",
                              f"{pnl_sign}${realized_pnl:,.2f}",
                              delta=f"{realized_pct:+.2f}% on sold positions")
                    m5.metric("Realized Return", f"{realized_pct:+.2f}%",
                              delta=f"{pnl_sign}${realized_pnl:,.2f}")

                    st.divider()

                    # ── Transaction history ──────────────────────────────────
                    st.markdown("#### Transaction History")

                    # Sort by date; build running share balance per ticker
                    sorted_idx = sorted(range(len(purchases)),
                                        key=lambda i: purchases[i]["date"])
                    running: dict[str, float] = {}
                    history_rows = []
                    for i in sorted_idx:
                        p    = purchases[i]
                        is_buy = p.get("type", "buy") == "buy"
                        sign = 1.0 if is_buy else -1.0
                        running[p["ticker"]] = running.get(p["ticker"], 0.0) + sign * p["shares"]
                        history_rows.append({
                            "_orig_idx":       i,
                            "Date":            p["date"],
                            "Type":            "Buy" if is_buy else "Sell",
                            "Ticker":          p["ticker"],
                            "Shares":          p["shares"],
                            "Price ($)":       p["price"],
                            "Total ($)":       p["total"],
                            "Balance (shares)": round(running[p["ticker"]], 4),
                        })

                    df_hist = pd.DataFrame(history_rows).drop(columns=["_orig_idx"])
                    df_hist.index = range(1, len(df_hist) + 1)

                    def _style_type(val):
                        return ("color: #2e7d32; font-weight: bold" if val == "Buy"
                                else "color: #c62828; font-weight: bold")

                    st.dataframe(
                        df_hist.style
                            .map(_style_type, subset=["Type"])
                            .format({
                                "Price ($)":        "{:,.2f}",
                                "Total ($)":        "{:,.2f}",
                                "Shares":           "{:,.4g}",
                                "Balance (shares)": "{:,.4g}",
                            }),
                        use_container_width=True,
                    )

                    # Remove a specific transaction
                    remove_opts = [
                        f"#{sorted_idx[j] + 1} — "
                        f"{'Buy' if purchases[sorted_idx[j]].get('type','buy') == 'buy' else 'Sell'}"
                        f"  {purchases[sorted_idx[j]]['ticker']}"
                        f"  |  {purchases[sorted_idx[j]]['date']}"
                        f"  |  {purchases[sorted_idx[j]]['shares']:g}"
                        f" @ ${purchases[sorted_idx[j]]['price']:.2f}"
                        for j in range(len(sorted_idx))
                    ]
                    rc1, rc2 = st.columns([4, 1])
                    with rc1:
                        to_remove = st.selectbox(
                            "Remove transaction", ["—"] + remove_opts, key=f"del_{port_name}",
                        )
                    with rc2:
                        st.write("")
                        st.write("")
                        if st.button("Remove", key=f"rem_{port_name}", use_container_width=True) \
                                and to_remove != "—":
                            orig_idx = sorted_idx[remove_opts.index(to_remove)]
                            purchases.pop(orig_idx)
                            st.rerun()

    st.divider()
    if st.button("▶ Run Analysis", type="primary", use_container_width=True):
        st.session_state["run_analysis"] = True
        st.rerun()

# ===========================================================================
# TAB 3 — Stock Analysis  (rendered before TAB 2 so st.stop() in TAB 2
#          does not prevent this tab from being built)
# ===========================================================================
with tab_stocks:
    st.header("Stock Analysis")

    all_port_tickers = sorted({
        p["ticker"] for purchases in st.session_state.portfolios.values() for p in purchases
    })
    if not all_port_tickers:
        st.info("Add tickers to your portfolios in **Portfolio Builder** first.")
    else:
        selected_ticker = st.selectbox("Select stock", all_port_tickers)

        start_str_sa = start_date.strftime("%Y-%m-%d")
        end_str_sa   = end_date.strftime("%Y-%m-%d")

        with st.spinner(f"Downloading {selected_ticker} data…"):
            try:
                ohlc = get_ohlc_data(selected_ticker, start_str_sa, end_str_sa)
            except Exception as e:
                st.error(f"Failed to download data for {selected_ticker}: {e}")
                ohlc = None

        if ohlc is not None and ohlc.empty:
            st.warning(f"No data found for {selected_ticker} in the selected date range.")
            ohlc = None

        if ohlc is not None:
            # Flatten MultiIndex columns if present (yfinance sometimes returns them)
            if isinstance(ohlc.columns, pd.MultiIndex):
                ohlc.columns = ohlc.columns.get_level_values(0)

            # --- Metrics Table ---
            st.subheader("📊 Metrics")
            _daily_ret_sa = ohlc["Close"].pct_change().dropna()
            _cum_sa       = (1 + _daily_ret_sa).cumprod() - 1

            try:
                _bm_prices  = get_close_prices([benchmark], start_str_sa, end_str_sa)
                _bm_ret_sa  = _bm_prices[benchmark].pct_change().dropna()
            except Exception:
                _bm_ret_sa = None

            _beta  = portfolio_beta(_daily_ret_sa, _bm_ret_sa)  if _bm_ret_sa is not None else None
            _alpha = portfolio_alpha(_daily_ret_sa, _bm_ret_sa, _beta, rfr) if _bm_ret_sa is not None else None
            _corr  = portfolio_correlation(_daily_ret_sa, _bm_ret_sa) if _bm_ret_sa is not None else None
            _rec   = drawdown_recovery_days(_daily_ret_sa)

            metrics_sa = {
                "Metric": [
                    "Annual Return (%)",
                    "Cumulative Return (%)",
                    f"CAGR {selected_ticker} (%)",
                    f"CAGR {benchmark} (%)",
                    "Annual Volatility (%)",
                    "Max Drawdown (%)",
                    "Max DD Recovery (days)",
                    f"Beta (vs. {benchmark})",
                    f"Alpha (%) (vs. {benchmark})",
                    f"Correlation (vs. {benchmark})",
                ],
                "Value": [
                    round(annualised_return(_daily_ret_sa) * 100, 2),
                    round(_cum_sa.iloc[-1] * 100, 2),
                    round(annualised_return(_daily_ret_sa) * 100, 2),
                    round(annualised_return(_bm_ret_sa) * 100, 2) if _bm_ret_sa is not None else "N/A",
                    round(annualised_volatility(_daily_ret_sa) * 100, 2),
                    round(max_drawdown(_daily_ret_sa) * 100, 2),
                    _rec if _rec is not None else "Not recovered",
                    round(_beta, 4)  if _beta  is not None else "N/A",
                    round(_alpha * 100, 2) if _alpha is not None else "N/A",
                    round(_corr, 4)  if _corr  is not None else "N/A",
                ],
            }
            st.dataframe(pd.DataFrame(metrics_sa).set_index("Metric"), use_container_width=True)

            # --- Candlestick + Volume ---
            # --- RSI (14-period) ---
            rsi_period = 14
            delta  = ohlc["Close"].diff()
            gain   = delta.clip(lower=0).rolling(rsi_period).mean()
            loss   = (-delta.clip(upper=0)).rolling(rsi_period).mean()
            rs     = gain / loss
            rsi    = 100 - (100 / (1 + rs))

            fig_candle = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                row_heights=[0.60, 0.20, 0.20],
                vertical_spacing=0.03,
                subplot_titles=("", "Volume", f"RSI ({rsi_period})"),
            )

            fig_candle.add_trace(go.Candlestick(
                x=ohlc.index,
                open=ohlc["Open"],
                high=ohlc["High"],
                low=ohlc["Low"],
                close=ohlc["Close"],
                name=selected_ticker,
                increasing_line_color="limegreen",
                decreasing_line_color="crimson",
            ), row=1, col=1)

            # --- Bollinger Bands (20-day SMA, ±2σ and ±3σ) ---
            bb_window = 20
            sma = ohlc["Close"].rolling(bb_window).mean()
            std = ohlc["Close"].rolling(bb_window).std()
            ub2 = sma + 2 * std
            lb2 = sma - 2 * std
            ub3 = sma + 3 * std
            lb3 = sma - 3 * std

            # 3σ band (outer, lighter fill)
            fig_candle.add_trace(go.Scatter(
                x=ohlc.index, y=ub3, mode="lines",
                line=dict(color="rgba(255,165,0,0.4)", width=1, dash="dot"),
                name="BB +3σ", legendgroup="bb3", showlegend=True,
            ), row=1, col=1)
            fig_candle.add_trace(go.Scatter(
                x=ohlc.index, y=lb3, mode="lines",
                line=dict(color="rgba(255,165,0,0.4)", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(255,165,0,0.06)",
                name="BB −3σ", legendgroup="bb3", showlegend=True,
            ), row=1, col=1)

            # 2σ band (inner, stronger fill)
            fig_candle.add_trace(go.Scatter(
                x=ohlc.index, y=ub2, mode="lines",
                line=dict(color="rgba(30,144,255,0.7)", width=1.2),
                name="BB +2σ", legendgroup="bb2", showlegend=True,
            ), row=1, col=1)
            fig_candle.add_trace(go.Scatter(
                x=ohlc.index, y=lb2, mode="lines",
                line=dict(color="rgba(30,144,255,0.7)", width=1.2),
                fill="tonexty", fillcolor="rgba(30,144,255,0.10)",
                name="BB −2σ", legendgroup="bb2", showlegend=True,
            ), row=1, col=1)

            # Middle band (SMA)
            fig_candle.add_trace(go.Scatter(
                x=ohlc.index, y=sma, mode="lines",
                line=dict(color="rgba(30,144,255,1)", width=1.5, dash="dash"),
                name=f"SMA {bb_window}",
            ), row=1, col=1)

            colors = ["limegreen" if c >= o else "crimson"
                      for c, o in zip(ohlc["Close"], ohlc["Open"])]
            fig_candle.add_trace(go.Bar(
                x=ohlc.index,
                y=ohlc["Volume"],
                name="Volume",
                marker_color=colors,
                showlegend=False,
            ), row=2, col=1)

            fig_candle.add_trace(go.Scatter(
                x=rsi.index, y=rsi.round(2),
                mode="lines",
                line=dict(color="darkorange", width=1.5),
                name=f"RSI {rsi_period}",
                hovertemplate="%{x|%Y-%m-%d}<br>RSI: %{y:.1f}<extra></extra>",
            ), row=3, col=1)
            fig_candle.add_hline(y=70, line=dict(color="red",   width=1, dash="dash"), row=3, col=1)
            fig_candle.add_hline(y=30, line=dict(color="green", width=1, dash="dash"), row=3, col=1)
            fig_candle.add_hrect(y0=70, y1=100, fillcolor="red",   opacity=0.05, line_width=0, row=3, col=1)
            fig_candle.add_hrect(y0=0,  y1=30,  fillcolor="green", opacity=0.05, line_width=0, row=3, col=1)
            fig_candle.update_yaxes(range=[0, 100], title_text=f"RSI ({rsi_period})", row=3, col=1)

            fig_candle.update_layout(
                title=f"{selected_ticker} — Candlestick ({start_str_sa} → {end_str_sa})",
                xaxis_rangeslider_visible=False,
                yaxis_title="Price (USD)",
                yaxis2_title="Volume",
                hovermode="x unified",
                height=750,
            )
            st.plotly_chart(fig_candle, use_container_width=True)

            # --- Cumulative Return ---
            st.subheader("📈 Cumulative Return")
            cum_ret_sa = (1 + ohlc["Close"].pct_change()).cumprod() - 1

            bm_cum_sa = (1 + _bm_ret_sa).cumprod() - 1 if _bm_ret_sa is not None else None

            fig_cum_sa = go.Figure()
            fig_cum_sa.add_trace(go.Scatter(
                x=cum_ret_sa.index, y=(cum_ret_sa * 100).round(2),
                mode="lines", name=selected_ticker,
            ))
            if bm_cum_sa is not None:
                fig_cum_sa.add_trace(go.Scatter(
                    x=bm_cum_sa.index, y=(bm_cum_sa * 100).round(2),
                    mode="lines", name=benchmark,
                ))
            fig_cum_sa.add_hline(y=0, line=dict(color="red", dash="dash", width=1))
            fig_cum_sa.update_layout(
                title=f"Cumulative Return ({start_str_sa} → {end_str_sa})",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_cum_sa, use_container_width=True)

            # --- Momentum ---
            st.subheader("Momentum (10-day)")
            mom_period = 10
            momentum   = ohlc["Close"].diff(mom_period)

            mom_colors = ["limegreen" if v >= 0 else "crimson" for v in momentum]
            fig_mom = go.Figure()
            fig_mom.add_trace(go.Bar(
                x=momentum.index, y=momentum.round(4),
                marker_color=mom_colors,
                name="Momentum",
                hovertemplate="%{x|%Y-%m-%d}<br>Momentum: %{y:.4f}<extra></extra>",
            ))
            fig_mom.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))
            fig_mom.update_layout(
                xaxis_title="Date",
                yaxis_title=f"Close(t) − Close(t−{mom_period})",
                hovermode="x unified",
                height=300,
                margin=dict(t=20),
            )
            st.plotly_chart(fig_mom, use_container_width=True)

            # --- Rolling Volatility ---
            st.subheader("Rolling Annualised Volatility (21-day)")
            daily_ret_sa = _daily_ret_sa
            roll_vol_sa  = daily_ret_sa.rolling(21).std() * np.sqrt(252) * 100

            fig_vol_sa = go.Figure()
            fig_vol_sa.add_trace(go.Scatter(
                x=roll_vol_sa.index, y=roll_vol_sa.round(2),
                mode="lines", fill="tozeroy",
                fillcolor="rgba(30,144,255,0.12)",
                line=dict(color="steelblue", width=1.5),
                name="Rolling Vol",
                hovertemplate="%{x|%Y-%m-%d}<br>Vol: %{y:.2f}%<extra></extra>",
            ))
            fig_vol_sa.update_layout(
                xaxis_title="Date",
                yaxis_title="Annualised Volatility (%)",
                hovermode="x unified",
                height=300,
                margin=dict(t=20),
            )
            st.plotly_chart(fig_vol_sa, use_container_width=True)

            # --- Underwater Chart ---
            st.subheader("🌊 Underwater Chart (Drawdown from Peak)")
            cum_uw   = (1 + daily_ret_sa).cumprod()
            peak_uw  = cum_uw.cummax()
            dd_uw    = (cum_uw - peak_uw) / peak_uw * 100
            max_dd_val  = dd_uw.min()
            max_dd_date = dd_uw.idxmin()

            fig_uw_sa = go.Figure()
            fig_uw_sa.add_trace(go.Scatter(
                x=dd_uw.index, y=dd_uw.round(2),
                fill="tozeroy", fillcolor="rgba(220,20,60,0.35)",
                line=dict(color="darkred", width=0.8),
                name=selected_ticker, showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>",
            ))
            fig_uw_sa.add_hline(y=0, line=dict(color="black", dash="dash", width=0.8))
            fig_uw_sa.add_annotation(
                x=max_dd_date, y=max_dd_val,
                text=f"Max DD: {max_dd_val:.1f}%",
                showarrow=True, arrowhead=2, arrowcolor="darkred",
                font=dict(color="darkred", size=9),
            )
            fig_uw_sa.update_layout(
                title=f"{selected_ticker} — Underwater Chart ({start_str_sa} → {end_str_sa})",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                yaxis_ticksuffix="%",
                hovermode="x unified",
                height=350,
                margin=dict(t=40),
            )
            st.plotly_chart(fig_uw_sa, use_container_width=True)

            # --- Monthly Returns Heatmap ---
            st.subheader("Monthly Returns Heatmap")
            draw_heatmap_plotly(selected_ticker, daily_ret_sa)

            # --- Summary stats ---
            st.subheader("Summary")
            first_close = ohlc["Close"].iloc[0]
            last_close  = ohlc["Close"].iloc[-1]
            period_ret  = (last_close / first_close - 1) * 100

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Last Close",   f"${last_close:.2f}")
            c2.metric("Period Return", f"{period_ret:+.2f}%")
            c3.metric("52w High",     f"${ohlc['High'].max():.2f}")
            c4.metric("52w Low",      f"${ohlc['Low'].min():.2f}")
            c5.metric("Avg Volume",   f"{int(ohlc['Volume'].mean()):,}")

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
        all_tickers = list(
            {p["ticker"] for purchases in portfolios.values() for p in purchases} | {benchmark}
        )
        try:
            prices = get_close_prices(all_tickers, start_str, end_str)
        except Exception as e:
            st.error(f"Failed to download prices: {e}")
            st.stop()

    benchmark_returns = prices[benchmark].pct_change().dropna()

    metrics       = {}
    cum_ret_all   = {}
    daily_ret_all = {}

    for name, purchases in portfolios.items():
        weights = _portfolio_weights(purchases)
        if not weights:
            continue
        ret = build_portfolio_returns(prices, weights)
        cum = cumulative_returns(ret)
        cum_ret_all[name]   = cum
        daily_ret_all[name] = ret

        beta  = portfolio_beta(ret, benchmark_returns)
        alpha = portfolio_alpha(ret, benchmark_returns, beta, rfr)
        rec   = drawdown_recovery_days(ret)

        bm_cagr = annualised_return(benchmark_returns)
        metrics[name] = {
            "Ann. Return (%)":                round(annualised_return(ret) * 100, 2),
            "CAGR (%)":                       round(annualised_return(ret) * 100, 2),
            f"CAGR {benchmark} (%)":          round(bm_cagr * 100, 2),
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
    corr_col = f"Correlation (vs. {benchmark})"
    if corr_col in cols:
        cols.append(cols.pop(cols.index(corr_col)))
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
    port_tickers = list({p["ticker"] for purchases in portfolios.values() for p in purchases})
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
    # Stock correlation heatmap
    # -----------------------------------------------------------------------
    st.subheader("🔗 Stock Correlation")
    for port_name, purchases in portfolios.items():
        tickers = list({p["ticker"] for p in purchases})
        if len(tickers) < 2:
            st.caption(f"{port_name}: need at least 2 tickers for correlation.")
            continue

        corr = prices[tickers].pct_change().dropna().corr()
        z    = corr.values
        text = np.vectorize(lambda v: f"{v:.2f}")(z)

        fig_corr = go.Figure(go.Heatmap(
            z=z,
            x=tickers,
            y=tickers,
            zmin=-1, zmax=1, zmid=0,
            colorscale="RdYlGn",
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="Correlation"),
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
        ))
        fig_corr.update_layout(
            title=dict(text=f"{port_name} — Stock Correlation", font=dict(size=14)),
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
            height=max(300, 55 * len(tickers) + 100),
            margin=dict(l=60, r=60, t=60, b=60),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

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

    # -----------------------------------------------------------------------
    # Hill Estimator — Extreme Value Analysis
    # -----------------------------------------------------------------------
    st.subheader("⛰️ Hill Estimator — Extreme Tail Risk")
    st.caption(
        "The Hill estimator ξ measures the heaviness of the loss tail. "
        "Higher ξ → heavier tail → greater probability of extreme losses. "
        "The Hill plot shows ξ across different numbers of upper-order statistics (k); "
        "a stable plateau region indicates a reliable estimate."
    )

    k_max_hill = st.slider(
        "Max k (order statistics)", min_value=10, max_value=300, value=150, step=10,
        help="Number of largest losses used in the Hill plot.",
    )

    fig_hill = go.Figure()
    hill_summary = {}

    for name, daily_ret in daily_ret_all.items():
        df_hill = hill_estimator(daily_ret, k_max=k_max_hill)
        fig_hill.add_trace(go.Scatter(
            x=df_hill["k"],
            y=df_hill["xi"].round(4),
            mode="lines",
            name=name,
            hovertemplate="k=%{x}<br>ξ=%{y:.4f}<extra>" + name + "</extra>",
        ))

        # Stable estimate: median over the middle 40 % of k range (ignores noisy tails)
        lo = int(len(df_hill) * 0.30)
        hi = int(len(df_hill) * 0.70)
        xi_stable = df_hill["xi"].iloc[lo:hi].median()

        if xi_stable < 0.25:
            risk_label = "Thin tail — low extreme-event risk"
            risk_color = "green"
        elif xi_stable < 0.50:
            risk_label = "Moderate tail — some extreme-event risk"
            risk_color = "orange"
        else:
            risk_label = "Heavy tail — high extreme-event risk (variance may be infinite)"
            risk_color = "red"

        hill_summary[name] = {
            "Tail Index ξ (stable)": round(xi_stable, 4),
            "Tail exponent α = 1/ξ": round(1 / xi_stable, 4) if xi_stable != 0 else float("nan"),
            "Risk Assessment": risk_label,
        }

    # Reference lines
    fig_hill.add_hline(y=0.25, line=dict(color="green",  dash="dot", width=1),
                       annotation_text="ξ = 0.25 (thin/moderate boundary)",
                       annotation_position="bottom right")
    fig_hill.add_hline(y=0.50, line=dict(color="orange", dash="dot", width=1),
                       annotation_text="ξ = 0.50 (moderate/heavy boundary)",
                       annotation_position="bottom right")

    fig_hill.update_layout(
        title="Hill Plot — Tail Index ξ vs Number of Order Statistics (k)",
        xaxis_title="k (number of largest losses)",
        yaxis_title="ξ (tail index)",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig_hill, use_container_width=True)

    summary_df = pd.DataFrame(hill_summary).T

    def _color_risk(val):
        if "Heavy" in str(val):
            return "color: red; font-weight: bold"
        if "Moderate" in str(val):
            return "color: darkorange; font-weight: bold"
        return "color: green; font-weight: bold"

    st.dataframe(
        summary_df.style.map(_color_risk, subset=["Risk Assessment"]),
        use_container_width=True,
    )

