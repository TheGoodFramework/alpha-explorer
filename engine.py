"""
Data fetching + backtesting engine for WQ101 Alpha Explorer.
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from operators import returns as calc_returns, vwap_approx


# ── S&P 100 Tickers ───────────────────────────────────────────────

SP100 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOG", "GS", "HD", "HON", "IBM", "INTC",
    "INTU", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS",
    "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
    "XOM",
]

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def fetch_data(tickers=None, start="2020-01-01", end=None, use_cache=True):
    """Fetch OHLCV data for tickers, with local caching."""
    if tickers is None:
        tickers = SP100
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_key = f"{'_'.join(sorted(tickers[:5]))}_{len(tickers)}_{start}_{end}"
    cache_file = os.path.join(CACHE_DIR, f"{hash(cache_key) & 0xFFFFFFFF:08x}.pkl")

    if use_cache and os.path.exists(cache_file):
        mod_time = os.path.getmtime(cache_file)
        age_hours = (datetime.now().timestamp() - mod_time) / 3600
        if age_hours < 24:
            try:
                return pd.read_pickle(cache_file)
            except Exception:
                pass

    # Download from yfinance
    raw = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False)

    if len(tickers) == 1:
        data = {
            "open": raw[["Open"]].rename(columns={"Open": tickers[0]}),
            "high": raw[["High"]].rename(columns={"High": tickers[0]}),
            "low": raw[["Low"]].rename(columns={"Low": tickers[0]}),
            "close": raw[["Close"]].rename(columns={"Close": tickers[0]}),
            "volume": raw[["Volume"]].rename(columns={"Volume": tickers[0]}),
        }
    else:
        data = {
            "open": raw.xs("Open", level=1, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw["Open"],
            "high": raw.xs("High", level=1, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw["High"],
            "low": raw.xs("Low", level=1, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw["Low"],
            "close": raw.xs("Close", level=1, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw["Close"],
            "volume": raw.xs("Volume", level=1, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw["Volume"],
        }

    data["vwap"] = vwap_approx(data["high"], data["low"], data["close"])
    data["returns"] = calc_returns(data["close"])

    # Cache
    try:
        pd.to_pickle(data, cache_file)
    except Exception:
        pass

    return data


def backtest_alpha(alpha_signals, forward_returns, top_pct=0.1, bottom_pct=0.1):
    """
    Backtest an alpha signal using long/short decile portfolio.

    Args:
        alpha_signals: DataFrame (dates x tickers) of alpha scores
        forward_returns: DataFrame (dates x tickers) of next-day returns
        top_pct: go long top percentile
        bottom_pct: go short bottom percentile

    Returns:
        dict with equity curve, metrics, holdings
    """
    # Align
    signals = alpha_signals.dropna(how="all")
    fwd = forward_returns.shift(-1)  # next day returns
    common_dates = signals.index.intersection(fwd.index)
    signals = signals.loc[common_dates]
    fwd = fwd.loc[common_dates]

    # Rank each day
    ranks = signals.rank(axis=1, pct=True)

    # Long top decile, short bottom decile
    long_mask = ranks >= (1 - top_pct)
    short_mask = ranks <= bottom_pct

    # Equal-weight portfolio returns
    n_long = long_mask.sum(axis=1).replace(0, np.nan)
    n_short = short_mask.sum(axis=1).replace(0, np.nan)

    long_ret = (fwd * long_mask).sum(axis=1) / n_long
    short_ret = (fwd * short_mask).sum(axis=1) / n_short
    port_ret = (long_ret - short_ret) / 2  # dollar-neutral

    port_ret = port_ret.dropna()

    if len(port_ret) < 20:
        return None

    # Equity curve
    equity = (1 + port_ret).cumprod()

    # Metrics
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (equity / equity.cummax() - 1).min()
    win_rate = (port_ret > 0).mean()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Monthly returns
    monthly = port_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Drawdown series
    drawdown = equity / equity.cummax() - 1

    # Turnover (approximate)
    turnover = (long_mask.astype(int).diff().abs().sum(axis=1) / 2).mean()

    # Last day holdings
    last_date = signals.index[-1]
    last_ranks = ranks.loc[last_date].dropna().sort_values(ascending=False)
    top_holdings = last_ranks.head(10)
    bottom_holdings = last_ranks.tail(10)

    return {
        "equity": equity,
        "daily_returns": port_ret,
        "drawdown": drawdown,
        "monthly_returns": monthly,
        "metrics": {
            "Annual Return": f"{ann_ret:.2%}",
            "Annual Volatility": f"{ann_vol:.2%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Win Rate": f"{win_rate:.2%}",
            "Calmar Ratio": f"{calmar:.2f}",
            "Avg Daily Turnover": f"{turnover:.1f} stocks",
            "Total Days": str(len(port_ret)),
        },
        "top_holdings": top_holdings,
        "bottom_holdings": bottom_holdings,
    }


def run_alpha(alpha_func, data):
    """Compute alpha signals and run backtest."""
    signals = alpha_func(data)
    fwd_returns = data["returns"]
    return backtest_alpha(signals, fwd_returns)
