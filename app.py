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
    st.plotly_chart(fig, width='stretch')

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


def _net_shares(purchases: list[dict]) -> dict[str, float]:
    """Net open shares per ticker (buys − sells)."""
    net: dict[str, float] = {}
    for p in purchases:
        sign = 1.0 if p.get("type", "buy") == "buy" else -1.0
        net[p["ticker"]] = net.get(p["ticker"], 0.0) + sign * p["shares"]
    return {t: s for t, s in net.items() if s > 1e-9}


def _close_price_on(ticker: str, date_str: str) -> float | None:
    """Close price of `ticker` on the first trading day on/after `date_str`."""
    end = (pd.Timestamp(date_str) + timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        px = get_close_prices([ticker], date_str, end)
    except Exception:
        return None
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    if ticker in px.columns:
        s = px[ticker].dropna()
        if not s.empty:
            return float(s.iloc[0])
    return None

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
        if st.button("Add portfolio", width='stretch') and new_name:
            if new_name not in st.session_state.portfolios:
                st.session_state.portfolios[new_name] = []
                st.rerun()
            else:
                st.warning("Name already exists.")
    with col_del:
        del_name = st.selectbox("Remove portfolio", ["—"] + list(st.session_state.portfolios))
        if st.button("Remove portfolio", width='stretch') and del_name != "—":
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

                # ── Input mode selector ──────────────────────────────────────
                mode = st.radio(
                    "Input mode",
                    ["Detailed", "Simple"],
                    horizontal=True,
                    key=f"mode_{port_name}",
                    help=(
                        "Detailed: enter shares, price and date for each buy/sell. "
                        "Simple: enter just a ticker and a date — the app fetches the "
                        "close price on that date and invests a fixed amount per position."
                    ),
                )
                st.divider()

                if mode == "Detailed":
                    # ── Add purchase form ────────────────────────────────────
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

                    if st.button("Add Purchase", key=f"add_{port_name}", width='stretch') \
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

                    # ── Add sale form ────────────────────────────────────────
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

                    if st.button("Add Sale", key=f"addsell_{port_name}", width='stretch') \
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

                else:
                    # ── Simple mode: ticker + date only ──────────────────────
                    st.markdown("#### Add Position (simple)")
                    st.caption(
                        "Enter a ticker and a date. The app fetches the close price "
                        "on that date and buys the configured amount per position."
                    )
                    invest_amt = st.number_input(
                        "Investment per position ($)", key=f"simp_amt_{port_name}",
                        min_value=0.01, value=5000.0, step=100.0, format="%.2f",
                    )

                    sp1, sp2 = st.columns([2, 2])
                    with sp1:
                        simp_ticker = st.text_input(
                            "Ticker", key=f"simp_tick_{port_name}", placeholder="e.g. AAPL",
                        ).upper().strip()
                    with sp2:
                        simp_date = st.date_input(
                            "Start date", key=f"simp_date_{port_name}", value=date.today(),
                        )

                    if st.button("Add Position", key=f"simp_add_{port_name}", width='stretch') \
                            and simp_ticker:
                        with st.spinner(f"Fetching {simp_ticker} price…"):
                            px = _close_price_on(simp_ticker, simp_date.strftime("%Y-%m-%d"))
                        if px is None or px <= 0:
                            st.error(
                                f"Could not fetch a price for {simp_ticker} on/after "
                                f"{simp_date:%Y-%m-%d}. Check the ticker and date."
                            )
                        else:
                            shares = invest_amt / px
                            purchases.append({
                                "type":   "buy",
                                "ticker": simp_ticker,
                                "date":   simp_date.strftime("%Y-%m-%d"),
                                "shares": shares,
                                "price":  round(px, 4),
                                "total":  round(shares * px, 2),
                            })
                            st.success(
                                f"Bought ${invest_amt:,.2f} of {simp_ticker} "
                                f"({shares:.4g} shares @ ${px:,.2f})."
                            )
                            st.rerun()

                    st.divider()

                    # ── Simple sale: sell full position by ticker + date ─────
                    st.markdown("#### Close Position (simple)")
                    st.caption(
                        "Select an open position and a date. The app sells all its "
                        "shares at the close price on that date."
                    )
                    open_positions = _net_shares(purchases)
                    ss1, ss2 = st.columns([2, 2])
                    with ss1:
                        simp_sell_ticker = st.selectbox(
                            "Position to close", ["—"] + list(open_positions.keys()),
                            key=f"simp_sell_tick_{port_name}",
                        )
                    with ss2:
                        simp_sell_date = st.date_input(
                            "Sale date", key=f"simp_sell_date_{port_name}", value=date.today(),
                        )

                    if st.button("Close Position", key=f"simp_sell_{port_name}", width='stretch') \
                            and simp_sell_ticker != "—":
                        with st.spinner(f"Fetching {simp_sell_ticker} price…"):
                            px = _close_price_on(simp_sell_ticker, simp_sell_date.strftime("%Y-%m-%d"))
                        if px is None or px <= 0:
                            st.error(
                                f"Could not fetch a price for {simp_sell_ticker} on/after "
                                f"{simp_sell_date:%Y-%m-%d}. Check the date."
                            )
                        else:
                            shares = open_positions[simp_sell_ticker]
                            purchases.append({
                                "type":   "sell",
                                "ticker": simp_sell_ticker,
                                "date":   simp_sell_date.strftime("%Y-%m-%d"),
                                "shares": shares,
                                "price":  round(px, 4),
                                "total":  round(shares * px, 2),
                            })
                            st.success(
                                f"Sold {shares:.4g} shares of {simp_sell_ticker} "
                                f"@ ${px:,.2f} (${shares * px:,.2f})."
                            )
                            st.rerun()

                st.divider()

                # ── Allocation summary + pie ─────────────────────────────────
                if not purchases:
                    st.info("No transactions yet — add one above.")
                else:
                    head_l, head_r = st.columns([4, 1])
                    with head_l:
                        st.markdown("#### Allocation (net open positions)")
                    with head_r:
                        if st.button("🗑️ Reset portfolio", key=f"reset_{port_name}",
                                     width='stretch', help="Remove all transactions from this portfolio"):
                            st.session_state.portfolios[port_name] = []
                            st.rerun()
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
                                width='stretch',
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
                    evo_tickers = sorted({p["ticker"] for p in sorted_txns})
                    first_date  = pd.Timestamp(sorted_txns[0]["date"])
                    today_ts    = pd.Timestamp(date.today())

                    evo_prices = None
                    try:
                        with st.spinner("Computing portfolio value history…"):
                            evo_prices = get_close_prices(
                                evo_tickers,
                                first_date.strftime("%Y-%m-%d"),
                                (today_ts + timedelta(days=1)).strftime("%Y-%m-%d"),
                            )
                    except Exception as e:
                        st.warning(f"Could not fetch price history: {e}")

                    if evo_prices is not None and not evo_prices.empty:
                        if isinstance(evo_prices.columns, pd.MultiIndex):
                            evo_prices.columns = evo_prices.columns.get_level_values(0)
                        evo_prices = evo_prices.ffill()

                        # Build cumulative shares-held & cost-basis time series
                        shares_held = pd.DataFrame(0.0, index=evo_prices.index, columns=evo_tickers)
                        cost_basis  = pd.Series(0.0, index=evo_prices.index)
                        for p in sorted_txns:
                            sign    = 1.0 if p.get("type", "buy") == "buy" else -1.0
                            tx_date = pd.Timestamp(p["date"])
                            mask    = shares_held.index >= tx_date
                            if p["ticker"] in shares_held.columns:
                                shares_held.loc[mask, p["ticker"]] += sign * p["shares"]
                            cost_basis.loc[mask] += sign * p["total"]

                        # Market value = Σ (shares × close price)
                        market_value = (shares_held * evo_prices[evo_tickers]).sum(axis=1)

                        fig_evo = go.Figure()
                        fig_evo.add_trace(go.Scatter(
                            x=market_value.index, y=market_value.round(2),
                            mode="lines", name="Market Value",
                            line=dict(color="steelblue", width=2),
                            fill="tozeroy", fillcolor="rgba(70,130,180,0.18)",
                            hovertemplate="%{x|%Y-%m-%d}<br>Value: $%{y:,.2f}<extra></extra>",
                        ))
                        fig_evo.add_trace(go.Scatter(
                            x=cost_basis.index, y=cost_basis.round(2),
                            mode="lines", name="Net Invested (cost basis)",
                            line=dict(color="darkorange", width=1.5, dash="dash"),
                            hovertemplate="%{x|%Y-%m-%d}<br>Cost: $%{y:,.2f}<extra></extra>",
                        ))

                        # Buy/Sell markers on the market-value line
                        marker_x, marker_y, marker_colors, marker_labels = [], [], [], []
                        for p in sorted_txns:
                            tx_date = pd.Timestamp(p["date"])
                            pos     = min(market_value.index.searchsorted(tx_date),
                                          len(market_value.index) - 1)
                            marker_x.append(market_value.index[pos])
                            marker_y.append(market_value.iloc[pos])
                            is_buy  = p.get("type", "buy") == "buy"
                            marker_colors.append("#2e7d32" if is_buy else "#c62828")
                            marker_labels.append(
                                f"{'Buy' if is_buy else 'Sell'} {p['shares']:g} {p['ticker']}"
                                f" @ ${p['price']:.2f}"
                            )
                        fig_evo.add_trace(go.Scatter(
                            x=marker_x, y=marker_y,
                            mode="markers", name="Transactions",
                            marker=dict(size=10, color=marker_colors,
                                        line=dict(color="white", width=1.5)),
                            customdata=marker_labels,
                            hovertemplate="<b>%{customdata}</b><br>%{x|%Y-%m-%d}<extra></extra>",
                        ))

                        fig_evo.add_hline(y=0, line=dict(color="gray", dash="dash", width=0.8))
                        fig_evo.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            yaxis_tickprefix="$",
                            yaxis_tickformat=",.0f",
                            hovermode="x unified",
                            height=400,
                            margin=dict(t=20),
                            legend=dict(orientation="h", yanchor="bottom",
                                        y=1.02, xanchor="right", x=1),
                        )
                        st.plotly_chart(fig_evo, width='stretch')

                        # ── % variation of portfolio vs. cost basis ──────────
                        st.markdown("#### Portfolio Return (%) — Market Value vs. Cost Basis")
                        pct_return = pd.Series(np.nan, index=market_value.index)
                        valid = cost_basis > 0
                        pct_return.loc[valid] = (
                            (market_value.loc[valid] - cost_basis.loc[valid])
                            / cost_basis.loc[valid] * 100
                        )
                        pct_return = pct_return.dropna()

                        if not pct_return.empty:
                            last_pct = pct_return.iloc[-1]
                            line_clr = "#2e7d32" if last_pct >= 0 else "#c62828"
                            fill_clr = ("rgba(46,125,50,0.18)" if last_pct >= 0
                                        else "rgba(198,40,40,0.18)")

                            fig_pct = go.Figure()
                            fig_pct.add_trace(go.Scatter(
                                x=pct_return.index, y=pct_return.round(2),
                                mode="lines", name="Return (%)",
                                line=dict(color=line_clr, width=2),
                                fill="tozeroy", fillcolor=fill_clr,
                                hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:+.2f}%<extra></extra>",
                            ))
                            fig_pct.add_hline(y=0, line=dict(color="gray",
                                                              dash="dash", width=0.8))
                            fig_pct.update_layout(
                                xaxis_title="Date",
                                yaxis_title="Return (%)",
                                yaxis_ticksuffix="%",
                                hovermode="x unified",
                                height=300,
                                margin=dict(t=20),
                            )
                            st.plotly_chart(fig_pct, width='stretch')

                            pmin = pct_return.min()
                            pmax = pct_return.max()
                            pc1, pc2, pc3 = st.columns(3)
                            pc1.metric("Current Return", f"{last_pct:+.2f}%")
                            pc2.metric("Peak Return",    f"{pmax:+.2f}%",
                                       delta=f"on {pct_return.idxmax().strftime('%Y-%m-%d')}")
                            pc3.metric("Lowest Return",  f"{pmin:+.2f}%",
                                       delta=f"on {pct_return.idxmin().strftime('%Y-%m-%d')}")
                    else:
                        st.info("No price data available to build the evolution chart.")

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
                        width='stretch',
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
                        if st.button("Remove", key=f"rem_{port_name}", width='stretch') \
                                and to_remove != "—":
                            orig_idx = sorted_idx[remove_opts.index(to_remove)]
                            purchases.pop(orig_idx)
                            st.rerun()

    st.divider()
    if st.button("▶ Run Analysis", type="primary", width='stretch'):
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

        # Volatility drag: arithmetic ann. return − geometric ann. return (CAGR).
        # Exact form, not the σ²/2 approximation — works for fat-tailed series too.
        arith_ann_return = ret.mean() * 252
        geom_ann_return  = annualised_return(ret)
        vol_drag         = arith_ann_return - geom_ann_return

        metrics[name] = {
            "Ann. Return (%)":                round(annualised_return(ret) * 100, 2),
            "CAGR (%)":                       round(annualised_return(ret) * 100, 2),
            f"CAGR {benchmark} (%)":          round(bm_cagr * 100, 2),
            "Ann. Volatility (%)":            round(annualised_volatility(ret) * 100, 2),
            "Volatility Drag (%)":            round(vol_drag * 100, 2),
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
    st.dataframe(metrics_df, width='stretch')

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
    st.plotly_chart(fig_cum, width='stretch')

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
    st.plotly_chart(fig_vol, width='stretch')

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
    st.plotly_chart(fig_uw, width='stretch')

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
    st.dataframe(pd.DataFrame(consistency_rows).T, width='stretch')

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
            width='stretch',
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
        st.plotly_chart(fig_corr, width='stretch')

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
    st.plotly_chart(fig_mc, width='stretch')

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
    st.plotly_chart(fig_mct, width='stretch')

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
    st.plotly_chart(fig_mcc, width='stretch')

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
    st.plotly_chart(fig_hill, width='stretch')

    summary_df = pd.DataFrame(hill_summary).T

    def _color_risk(val):
        if "Heavy" in str(val):
            return "color: red; font-weight: bold"
        if "Moderate" in str(val):
            return "color: darkorange; font-weight: bold"
        return "color: green; font-weight: bold"

    st.dataframe(
        summary_df.style.map(_color_risk, subset=["Risk Assessment"]),
        width='stretch',
    )

