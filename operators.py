"""
WorldQuant 101 Formulaic Alphas — Operator Library
All operators work on DataFrames where rows=dates, cols=tickers.
"""

import numpy as np
import pandas as pd


# ── Cross-Sectional Operators ──────────────────────────────────────

def rank(df):
    """Cross-sectional percentile rank [0,1] across all stocks at each timestamp."""
    return df.rank(axis=1, pct=True)


def scale(df, a=1):
    """Rescale so sum(abs(x)) = a at each timestamp."""
    return df.mul(a).div(df.abs().sum(axis=1), axis=0)


# ── Time-Series Operators ──────────────────────────────────────────

def delta(df, d):
    """x(t) - x(t-d)."""
    d = int(d)
    return df - df.shift(d)


def delay(df, d):
    """Value of x from d days ago."""
    d = int(d)
    return df.shift(d)


def correlation(x, y, d):
    """Rolling Pearson correlation over d days."""
    d = int(d)
    return x.rolling(d, min_periods=max(d // 2, 2)).corr(y)


def covariance(x, y, d):
    """Rolling covariance over d days."""
    d = int(d)
    return x.rolling(d, min_periods=max(d // 2, 2)).cov(y)


def ts_min(df, d):
    """Rolling minimum over d days."""
    d = int(d)
    return df.rolling(d, min_periods=max(d // 2, 1)).min()


def ts_max(df, d):
    """Rolling maximum over d days."""
    d = int(d)
    return df.rolling(d, min_periods=max(d // 2, 1)).max()


def ts_argmax(df, d):
    """Index of max value in rolling d-day window (0 = today)."""
    d = int(d)
    return df.rolling(d, min_periods=max(d // 2, 1)).apply(
        lambda x: x.argmax(), raw=True
    )


def ts_argmin(df, d):
    """Index of min value in rolling d-day window (0 = today)."""
    d = int(d)
    return df.rolling(d, min_periods=max(d // 2, 1)).apply(
        lambda x: x.argmin(), raw=True
    )


def ts_rank(df, d):
    """Time-series rank of current value within past d days (percentile)."""
    d = int(d)
    return df.rolling(d, min_periods=max(d // 2, 1)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def sum_ts(df, d):
    """Rolling sum over d days."""
    d = int(d)
    return df.rolling(d, min_periods=max(d // 2, 1)).sum()


def product(df, d):
    """Rolling product over d days."""
    d = int(d)
    return df.rolling(d, min_periods=max(d // 2, 1)).apply(
        lambda x: np.prod(x), raw=True
    )


def stddev(df, d):
    """Rolling standard deviation over d days."""
    d = int(d)
    return df.rolling(d, min_periods=max(d // 2, 2)).std()


def decay_linear(df, d):
    """Weighted moving average with linearly decaying weights [d, d-1, ..., 1]."""
    d = int(d)
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()

    def _wma(x):
        if len(x) < d:
            return np.nan
        return np.dot(x[-d:], weights)

    return df.rolling(d, min_periods=d).apply(_wma, raw=True)


# ── Math Operators ─────────────────────────────────────────────────

def signed_power(df, a):
    """sign(x) * |x|^a."""
    return np.sign(df) * np.abs(df) ** a


def log(df):
    """Natural logarithm."""
    return np.log(df.replace(0, np.nan))


def abs_val(df):
    """Absolute value."""
    return df.abs()


def sign(df):
    """Sign function: -1, 0, or 1."""
    return np.sign(df)


# ── Helpers ────────────────────────────────────────────────────────

def adv(volume, d):
    """Average daily volume over d days."""
    d = int(d)
    return volume.rolling(d, min_periods=max(d // 2, 1)).mean()


def returns(close):
    """Daily close-to-close returns."""
    return close.pct_change()


def vwap_approx(high, low, close):
    """Approximate VWAP as typical price (H+L+C)/3 when true VWAP unavailable."""
    return (high + low + close) / 3
