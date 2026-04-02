import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime


def get_ohlc_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLC data for a single ticker."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return data[["Open", "High", "Low", "Close", "Volume"]].dropna()


def get_close_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")


def build_portfolio_returns(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Compute daily weighted portfolio returns."""
    tickers = list(weights.keys())
    w = pd.Series(weights)
    w = w / w.sum()  # normalise to 1
    daily_returns = prices[tickers].pct_change().dropna()
    portfolio_returns = (daily_returns * w).sum(axis=1)
    return portfolio_returns


def build_portfolio_returns_dynamic(
    prices: pd.DataFrame,
    slices: list[dict],
    fallback_start: str,
    fallback_end: str,
) -> pd.Series:
    """
    Build daily portfolio returns for a time-sliced allocation.

    slices : list of {"start": str|None, "end": str|None, "weights": {ticker: float}}
             start/end None → fallback_start / fallback_end
    Slices are processed in order; overlapping days keep the first occurrence.
    """
    if not slices:
        return pd.Series(dtype=float)

    all_dates = prices.index
    parts: list[pd.Series] = []

    for sl in slices:
        s = sl.get("start") or fallback_start
        e = sl.get("end")   or fallback_end
        w = {t: v for t, v in sl.get("weights", {}).items() if t in prices.columns}
        if not w:
            continue

        # Include one trading day of context before the slice start so
        # pct_change() produces a return on the first day of the slice.
        start_pos = all_dates.searchsorted(pd.Timestamp(s))
        ctx_pos   = max(0, start_pos - 1)
        end_pos   = all_dates.searchsorted(pd.Timestamp(e), side="right")
        slice_prices = prices.iloc[ctx_pos:end_pos][list(w.keys())]
        if slice_prices.empty:
            continue

        ret = build_portfolio_returns(slice_prices, w)
        ret = ret.loc[pd.Timestamp(s): pd.Timestamp(e)]
        parts.append(ret)

    if not parts:
        return pd.Series(dtype=float)

    combined = pd.concat(parts)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


def annualised_return(returns: pd.Series) -> float:
    n_years = len(returns) / 252
    total = (1 + returns).prod()
    return total ** (1 / n_years) - 1 if n_years > 0 else float("nan")


def annualised_volatility(returns: pd.Series) -> float:
    return returns.std() * (252 ** 0.5)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / 252
    vol = annualised_volatility(returns)
    return (annualised_return(excess) / vol) if vol != 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    return drawdown.min()


def calmar_ratio(returns: pd.Series) -> float:
    """Compute Calmar Ratio: annualised return / absolute max drawdown."""
    md = abs(max_drawdown(returns))
    return annualised_return(returns) / md if md != 0 else float("nan")


def drawdown_recovery_days(returns: pd.Series) -> int | None:
    """Trading days to recover from the maximum drawdown trough. None if not yet recovered."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak

    trough_date = drawdown.idxmin()
    peak_level = peak[trough_date]

    after_trough = cum[trough_date:]
    recovered = after_trough[after_trough >= peak_level]

    if recovered.empty:
        return None

    return len(cum[trough_date : recovered.index[0]]) - 1


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute Sortino Ratio: (Rp - Rf) / downside deviation."""
    excess = returns - risk_free_rate / 252
    downside = excess[excess < 0].std() * (252 ** 0.5)
    return annualised_return(excess) / downside if downside != 0 else float("nan")


def var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: worst daily loss not exceeded with given confidence."""
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR (Expected Shortfall): mean loss beyond the VaR threshold."""
    r = returns.dropna()
    threshold = np.percentile(r, (1 - confidence) * 100)
    return -r[r <= threshold].mean()


def portfolio_correlation(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute Pearson correlation between portfolio and benchmark daily returns."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def portfolio_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute beta of portfolio returns against a benchmark using OLS covariance."""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    cov_matrix = aligned.cov()
    return cov_matrix.iloc[0, 1] / cov_matrix.iloc[1, 1]


def portfolio_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    beta: float,
    risk_free_rate: float = 0.0428,
) -> float:
    """Compute Jensen's Alpha (annualised): α = Rp - [Rf + β × (Rm - Rf)]."""
    rp = annualised_return(returns)
    rm = annualised_return(benchmark_returns)
    return rp - (risk_free_rate + beta * (rm - risk_free_rate))


def hill_estimator(returns: pd.Series, k_max: int | None = None) -> pd.DataFrame:
    """
    Compute the Hill estimator for the tail index across a range of k values.

    The Hill estimator ξ̂(k) = (1/k) Σ_{i=1}^{k} log(X_(i) / X_(k+1))
    where X_(1) ≥ X_(2) ≥ ... ≥ X_(n) are the ordered losses (positive values).

    A larger ξ (closer to 1) signals a heavier tail and higher extreme-event risk.
    Rule of thumb:  ξ < 0.25 → thin tail
                    0.25 ≤ ξ < 0.5 → moderate tail
                    ξ ≥ 0.5 → heavy tail (variance may be infinite)

    Parameters
    ----------
    returns : daily return series
    k_max   : maximum number of upper-order statistics to consider
               (defaults to min(200, n//4))

    Returns
    -------
    DataFrame with columns ['k', 'xi'] where xi = Hill tail-index estimate
    """
    losses = -returns.dropna()
    losses = losses[losses > 0].sort_values(ascending=False).values
    n = len(losses)

    if k_max is None:
        k_max = min(200, n // 4)
    k_max = max(k_max, 10)

    ks, xis = [], []
    for k in range(1, k_max + 1):
        if k >= n:
            break
        xi = np.mean(np.log(losses[:k]) - np.log(losses[k]))
        ks.append(k)
        xis.append(xi)

    return pd.DataFrame({"k": ks, "xi": xis})


def compare_portfolios(
    portfolios: dict[str, dict[str, float]],
    start: str,
    end: str,
    risk_free_rate: float = 0.0,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """
    Compare multiple portfolios and return a metrics DataFrame.

    Parameters
    ----------
    portfolios     : dict mapping portfolio name -> {ticker: weight}
    start          : start date string, e.g. "2020-01-01"
    end            : end date string,   e.g. "2024-12-31"
    risk_free_rate : annual risk-free rate (default 0.0)
    benchmark      : ticker used as market benchmark for beta (default "SPY")
    """
    all_tickers = list({t for w in portfolios.values() for t in w} | {benchmark})
    prices = get_close_prices(all_tickers, start, end)
    benchmark_returns = prices[benchmark].pct_change().dropna()

    metrics = {}
    cum_returns_all = {}
    daily_returns_all = {}

    for name, weights in portfolios.items():
        ret = build_portfolio_returns(prices, weights)
        cum = cumulative_returns(ret)
        cum_returns_all[name] = cum
        daily_returns_all[name] = ret

        beta = portfolio_beta(ret, benchmark_returns)
        alpha = portfolio_alpha(ret, benchmark_returns, beta, risk_free_rate)

        metrics[name] = {
            "Ann. Return (%)":                  round(annualised_return(ret) * 100, 2),
            "Cumulative Return (%)":            round(cum.iloc[-1] * 100, 2),
            "Ann. Volatility (%)":              round(annualised_volatility(ret) * 100, 2),
            "Sharpe Ratio":                     round(sharpe_ratio(ret, risk_free_rate), 4),
            "Sortino Ratio":                    round(sortino_ratio(ret, risk_free_rate), 4),
            "Max Drawdown (%)":                 round(max_drawdown(ret) * 100, 2),
            "Calmar Ratio":                     round(calmar_ratio(ret), 4),
            "Max DD Recovery (days)":           drawdown_recovery_days(ret) or "Not recovered",
            "VaR 95% (%)":                      round(var(ret, 0.95) * 100, 2),
            "CVaR 95% (%)":                     round(cvar(ret, 0.95) * 100, 2),
            f"Beta (vs. {benchmark})":          round(beta, 4),
            f"Correlation (vs. {benchmark})":   round(portfolio_correlation(ret, benchmark_returns), 4),
            f"Alpha (%) (vs. {benchmark})":     round(alpha * 100, 2),
        }

    metrics_df = pd.DataFrame(metrics).T
    plot_cumulative_returns(cum_returns_all, start, end)
    plot_underwater(daily_returns_all)
    plot_monthly_returns(daily_returns_all)
    plot_monte_carlo(daily_returns_all)
    return metrics_df


def plot_monthly_returns(portfolio_returns: dict[str, pd.Series]):
    """Plot a monthly returns heatmap (year × month) for each portfolio."""
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    col_labels = month_labels + ["Ann."]

    n = len(portfolio_returns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    cmap = plt.get_cmap("RdYlGn")

    for ax, (name, daily_ret) in zip(axes, portfolio_returns.items()):
        monthly = (1 + daily_ret).resample("ME").prod() - 1
        monthly_pct = monthly * 100

        pivot = monthly_pct.groupby([monthly_pct.index.year, monthly_pct.index.month]).first()
        pivot = pivot.unstack(level=1)  # columns = months (1..12)
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
    plt.show()


def plot_monte_carlo(
    portfolio_returns: dict[str, pd.Series],
    n_simulations: int = 500,
    n_days: int = 252,
):
    """Monte Carlo simulation: project each portfolio forward using bootstrapped daily returns."""
    n = len(portfolio_returns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, (name, daily_ret) in zip(axes, portfolio_returns.items()):
        r = daily_ret.dropna().values

        # Bootstrap: sample with replacement from historical daily returns
        sampled = rng.choice(r, size=(n_simulations, n_days), replace=True)
        paths = np.cumprod(1 + sampled, axis=1) - 1  # shape (n_simulations, n_days)

        final_values = paths[:, -1] * 100
        p5  = np.percentile(paths, 5,  axis=0) * 100
        p25 = np.percentile(paths, 25, axis=0) * 100
        p50 = np.percentile(paths, 50, axis=0) * 100
        p75 = np.percentile(paths, 75, axis=0) * 100
        p95 = np.percentile(paths, 95, axis=0) * 100

        days = np.arange(1, n_days + 1)

        # Draw a thin line per simulation (low alpha)
        for i in range(n_simulations):
            ax.plot(days, paths[i] * 100, color="steelblue", alpha=0.04, linewidth=0.5)

        # Confidence bands
        ax.fill_between(days, p5,  p95, alpha=0.15, color="steelblue", label="5–95th pct")
        ax.fill_between(days, p25, p75, alpha=0.30, color="steelblue", label="25–75th pct")
        ax.plot(days, p50, color="navy", linewidth=2, label="Median")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        # Annotate final percentile values
        for pct_val, label in [(p5[-1], "5th"), (p50[-1], "50th"), (p95[-1], "95th")]:
            ax.annotate(f"{pct_val:.1f}%", xy=(n_days, pct_val),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="navy")

        # VaR annotation
        var_val = np.percentile(final_values, 5)
        ax.set_title(
            f"{name} — Monte Carlo ({n_simulations} paths, {n_days} days)   "
            f"|   5th pct final: {var_val:.1f}%   Median: {np.median(final_values):.1f}%   "
            f"95th pct final: {np.percentile(final_values, 95):.1f}%",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_cumulative_returns(cum_returns: dict[str, pd.Series], start: str, end: str):
    plt.figure(figsize=(12, 6))
    for name, cr in cum_returns.items():
        plt.plot(cr.index, cr * 100, label=name)
    plt.title(f"Portfolio Cumulative Returns ({start} to {end})")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_underwater(daily_returns: dict[str, pd.Series]):
    """Plot the underwater (drawdown) chart for each portfolio.

    Shows the percentage distance from the running peak at every point in time.
    Values are always <= 0; the shaded area highlights how deep and how long
    each portfolio stays below its previous high-water mark.
    """
    n = len(daily_returns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (name, ret) in zip(axes, daily_returns.items()):
        cum = (1 + ret).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak) / peak * 100  # percentage, always <= 0

        ax.fill_between(drawdown.index, drawdown, 0, color="crimson", alpha=0.55)
        ax.plot(drawdown.index, drawdown, color="darkred", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

        max_dd = drawdown.min()
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
    plt.show()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    START = "2025-01-01"
    END = datetime.today().strftime("%Y-%m-%d")

    # --- US portfolios vs SPY ---
    us_portfolios = {
        "US Portfolio": {
            "MSFT": 0.10,
            "GOOGL": 0.10,
            "NVDA": 0.10,
            "AVGO": 0.10,
            "CEG": 0.10,
            "NRG": 0.10,
            "MELI": 0.10,
            "AMZN": 0.10,
            "NFLX": 0.10,
            "JPM": 0.10,
        }
    }

    us_results = compare_portfolios(us_portfolios, start=START, end=END,
                                    risk_free_rate=0.04, benchmark="SPY")

    print("\n=== US Portfolio Comparison ===\n")
    print(us_results.to_string())
