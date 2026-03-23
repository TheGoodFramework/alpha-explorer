"""
WorldQuant 101 Formulaic Alphas — Implementations
Each alpha is a function that takes a data dict and returns a DataFrame of signals.
"""

from operators import (
    rank, scale, delta, delay, correlation, covariance,
    ts_min, ts_max, ts_argmax, ts_argmin, ts_rank,
    sum_ts, product, stddev, decay_linear,
    signed_power, log, abs_val, sign, adv, returns, vwap_approx,
)
import numpy as np


# ── Alpha Registry ─────────────────────────────────────────────────

ALPHA_CATALOG = {}


def register(num, name, category, description, formula):
    """Decorator to register an alpha."""
    def wrapper(fn):
        ALPHA_CATALOG[num] = {
            "num": num,
            "name": name,
            "category": category,
            "description": description,
            "formula": formula,
            "func": fn,
        }
        return fn
    return wrapper


# ── Data Helper ────────────────────────────────────────────────────

def _get(d):
    """Extract common fields from data dict."""
    o = d["open"]
    h = d["high"]
    l = d["low"]
    c = d["close"]
    v = d["volume"]
    vw = d.get("vwap", vwap_approx(h, l, c))
    ret = d.get("returns", returns(c))
    return o, h, l, c, v, vw, ret


# ── Alpha Implementations ─────────────────────────────────────────

@register(1, "Reversal Argmax", "mean-reversion",
    "Sells stocks near recent highs with high recent volatility — classic short-term reversal.",
    "(rank(Ts_ArgMax(SignedPower(close/delay(close,1)-1, 2), 5)) - 0.5)")
def alpha001(d):
    o, h, l, c, v, vw, ret = _get(d)
    inner = signed_power(c / delay(c, 1) - 1, 2)
    return rank(ts_argmax(inner, 5)) - 0.5


@register(2, "Volume-Price Divergence", "momentum",
    "Buys when price drops correlate with rising volume — capitulation signal.",
    "-1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)")
def alpha002(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * correlation(rank(delta(log(v), 2)), rank((c - o) / o), 6)


@register(3, "Volume-Open Divergence", "momentum",
    "Measures divergence between volume rank and open price rank over 10 days.",
    "-1 * correlation(rank(open), rank(volume), 10)")
def alpha003(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * correlation(rank(o), rank(v), 10)


@register(4, "Low Rank Momentum", "momentum",
    "Sells stocks with persistently low prices — trend following on lows.",
    "-1 * Ts_Rank(rank(low), 9)")
def alpha004(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * ts_rank(rank(l), 9)


@register(5, "VWAP vs Close Momentum", "momentum",
    "Buys stocks where VWAP diverges upward from close relative to volume — institutional buying signal.",
    "(rank(open - sum(vwap, 10)/10) * (-1) * abs(rank(close - vwap)))")
def alpha005(d):
    o, h, l, c, v, vw, ret = _get(d)
    return rank(o - sum_ts(vw, 10) / 10) * (-1) * abs_val(rank(c - vw))


@register(6, "Open-Volume Correlation", "momentum",
    "Measures negative correlation between open price and volume — contrarian.",
    "-1 * correlation(open, volume, 10)")
def alpha006(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * correlation(o, v, 10)


@register(8, "Open-Returns Lag", "mean-reversion",
    "Buys on delayed sum of returns weighted by open price — smoothed reversal.",
    "-1 * rank(sum(open, 5) * sum(returns, 5) - delay(sum(open, 5) * sum(returns, 5), 10))")
def alpha008(d):
    o, h, l, c, v, vw, ret = _get(d)
    x = sum_ts(o, 5) * sum_ts(ret, 5)
    return -1 * rank(x - delay(x, 10))


@register(9, "Close Momentum", "momentum",
    "If close near recent min, buy momentum of delta(close). Otherwise mean-revert.",
    "(ts_min(delta(close,1), 5) > 0 ? delta(close,1) : ... ) conditional momentum")
def alpha009(d):
    o, h, l, c, v, vw, ret = _get(d)
    d1 = delta(c, 1)
    cond_pos = ts_min(d1, 5) > 0
    cond_neg = ts_max(d1, 5) < 0
    result = d1.copy()
    result[cond_pos] = d1[cond_pos]
    result[cond_neg] = d1[cond_neg]
    result[~cond_pos & ~cond_neg] = -1 * d1[~cond_pos & ~cond_neg]
    return result


@register(11, "VWAP-Close Volume", "mean-reversion",
    "Combines VWAP-close spread with volume changes — detects price/volume dislocations.",
    "((rank(ts_max(vwap-close, 3)) + rank(ts_min(vwap-close, 3))) * rank(delta(volume, 3)))")
def alpha011(d):
    o, h, l, c, v, vw, ret = _get(d)
    spread = vw - c
    return (rank(ts_max(spread, 3)) + rank(ts_min(spread, 3))) * rank(delta(v, 3))


@register(12, "Volume-Close Signal", "momentum",
    "Sells when volume rises and close drops — distribution detection.",
    "sign(delta(volume, 1)) * (-1 * delta(close, 1))")
def alpha012(d):
    o, h, l, c, v, vw, ret = _get(d)
    return sign(delta(v, 1)) * (-1 * delta(c, 1))


@register(13, "Close-Rank Covariance", "momentum",
    "Covariance between rank of close and rank of volume — institutional flow signal.",
    "-1 * rank(covariance(rank(close), rank(volume), 5))")
def alpha013(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * rank(covariance(rank(c), rank(v), 5))


@register(14, "Returns-Volume Correlation", "mean-reversion",
    "Combines recent returns with volume-open correlation for reversal timing.",
    "(-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)")
def alpha014(d):
    o, h, l, c, v, vw, ret = _get(d)
    return (-1 * rank(delta(ret, 3))) * correlation(o, v, 10)


@register(15, "High-Volume Correlation", "momentum",
    "Ranks the rolling correlation between high prices and volume.",
    "-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)")
def alpha015(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * sum_ts(rank(correlation(rank(h), rank(v), 3)), 3)


@register(16, "High-Volume Covariance", "momentum",
    "Ranks covariance of high and volume — captures attention-driven moves.",
    "-1 * rank(covariance(rank(high), rank(volume), 5))")
def alpha016(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * rank(covariance(rank(h), rank(v), 5))


@register(17, "Close-Volume TS Rank", "momentum",
    "Time-series rank of close combined with delta and volume momentum.",
    "(-1 * rank(ts_rank(close, 10)) * rank(delta(delta(close,1), 1)) * rank(ts_rank(volume/adv20, 5)))")
def alpha017(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv20 = adv(v, 20)
    return (-1 * rank(ts_rank(c, 10)) * rank(delta(delta(c, 1), 1)) * rank(ts_rank(v / adv20, 5)))


@register(18, "Close-Open Correlation", "mean-reversion",
    "Correlates close-stddev spread with open — finds mean-reverting volatility patterns.",
    "-1 * rank(stddev(abs(close-open), 5) + (close-open) + correlation(close, open, 10))")
def alpha018(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * rank(stddev(abs_val(c - o), 5) + (c - o) + correlation(c, o, 10))


@register(19, "Returns Momentum", "momentum",
    "Combines recent returns sign with return acceleration — trend strength signal.",
    "(-1 * sign(close - delay(close,7) + delta(close,7))) * (1 + rank(1 + sum(returns, 250)))")
def alpha019(d):
    o, h, l, c, v, vw, ret = _get(d)
    return (-1 * sign(c - delay(c, 7) + delta(c, 7))) * (1 + rank(1 + sum_ts(ret, 250)))


@register(20, "Open-High-Low-Close", "mean-reversion",
    "Combines open-high and open-low-close spreads — gap and range signal.",
    "-1 * rank(open - delay(high, 1)) * rank(open - delay(close, 1)) * rank(open - delay(low, 1))")
def alpha020(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * rank(o - delay(h, 1)) * rank(o - delay(c, 1)) * rank(o - delay(l, 1))


@register(21, "Close vs MA Signal", "mean-reversion",
    "Mean-reversion based on close relative to 8-day average and 2-day average.",
    "Conditional: if mean(close,8)+std(close,8) < mean(close,2) => -1, else if mean(close,2) < mean(close,8)-std => 1, else volume comparison")
def alpha021(d):
    o, h, l, c, v, vw, ret = _get(d)
    ma8 = sum_ts(c, 8) / 8
    ma2 = sum_ts(c, 2) / 2
    std8 = stddev(c, 8)
    adv20 = adv(v, 20)
    cond1 = (ma8 + std8) < ma2
    cond2 = ma2 < (ma8 - std8)
    result = c.copy() * 0
    result[cond1] = -1
    result[cond2] = 1
    rest = ~cond1 & ~cond2
    result[rest] = (-1 * (v[rest] / adv20[rest]).clip(-1, 1))
    return result


@register(22, "High-Volume-Close", "momentum",
    "Correlation between high and volume weighted by close delta and stddev.",
    "-1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))")
def alpha022(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * delta(correlation(h, v, 5), 5) * rank(stddev(c, 20))


@register(23, "High Price Reversal", "mean-reversion",
    "If SMA of high is above close, sell — overbought reversal.",
    "sum(high,20)/20 < high ? -1 * delta(high,2) : 0")
def alpha023(d):
    o, h, l, c, v, vw, ret = _get(d)
    ma_h = sum_ts(h, 20) / 20
    result = c.copy() * 0
    cond = ma_h < h
    result[cond] = -1 * delta(h, 2)[cond]
    return result


@register(24, "Close vs SMA Trend", "momentum",
    "If close SMA is declining, sell; trend-following on moving averages.",
    "delta(sum(close,100)/100, 100)/delay(close,100) < -0.05 ? -1*(close-ts_min(close,100)) : -1*delta(close,3)")
def alpha024(d):
    o, h, l, c, v, vw, ret = _get(d)
    sma_chg = delta(sum_ts(c, 100) / 100, 100) / delay(c, 100)
    cond = sma_chg < -0.05
    result = -1 * delta(c, 3)
    result[cond] = (-1 * (c - ts_min(c, 100)))[cond]
    return result


@register(25, "VWAP-ADV-Returns", "momentum",
    "Ranks VWAP decay weighted by ADV and return momentum — institutional flow.",
    "rank(-1 * returns * adv20 * vwap * (high - close))")
def alpha025(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv20 = adv(v, 20)
    return rank(-1 * ret * adv20 * vw * (h - c))


@register(26, "Volume-High Correlation", "momentum",
    "Measures deterioration of volume-high correlation combined with TS rank.",
    "-1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3)")
def alpha026(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * ts_max(correlation(ts_rank(v, 5), ts_rank(h, 5), 5), 3)


@register(28, "ADV-Close-Volume", "mean-reversion",
    "Volume-weighted correlation between ADV and low, scaled by VWAP-close spread.",
    "scale(correlation(adv20, low, 5) + (high+low)/2 - close)")
def alpha028(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv20 = adv(v, 20)
    return scale(correlation(adv20, l, 5) + (h + l) / 2 - c)


@register(29, "Returns Reversal", "mean-reversion",
    "Multi-day product of returns scaled by volume — cumulative reversal.",
    "min(product(rank(rank(scale(log(sum(ts_min(rank(-1*rank(delta(close,1))),2),1))))), 1), 5) + ts_rank(delay(-1*returns, 6), 5)")
def alpha029(d):
    o, h, l, c, v, vw, ret = _get(d)
    inner = rank(rank(scale(log(sum_ts(ts_min(rank(-1 * rank(delta(c, 1))), 2), 1)))))
    x = inner.rolling(5, min_periods=1).min()
    return x + ts_rank(delay(-1 * ret, 6), 5)


@register(30, "Volume-Close Momentum", "momentum",
    "Smoothed close-open spread weighted by volume — gap momentum.",
    "((1 - rank(sign(close - delay(close,1)) + sign(delay(close,1) - delay(close,2)) + sign(delay(close,2) - delay(close,3)))) * sum(volume, 5)) / sum(volume, 20)")
def alpha030(d):
    o, h, l, c, v, vw, ret = _get(d)
    s = sign(c - delay(c, 1)) + sign(delay(c, 1) - delay(c, 2)) + sign(delay(c, 2) - delay(c, 3))
    return ((1 - rank(s)) * sum_ts(v, 5)) / sum_ts(v, 20)


@register(33, "Open-Close Spread", "mean-reversion",
    "Simple open-close rank inverted — gap reversal.",
    "rank(-1 * (1 - open/close))")
def alpha033(d):
    o, h, l, c, v, vw, ret = _get(d)
    return rank(-1 * (1 - o / c))


@register(34, "Returns-Close Stddev", "mean-reversion",
    "Close deviation from mean combined with returns delta — volatility reversal.",
    "rank((-1 * rank(stddev(returns,2)/stddev(returns,5))) + (-1 * delta(close,1)))")
def alpha034(d):
    o, h, l, c, v, vw, ret = _get(d)
    return rank(-1 * rank(stddev(ret, 2) / stddev(ret, 5)) + (-1 * delta(c, 1)))


@register(35, "VWAP-Close-Volume TS Rank", "momentum",
    "Multi-factor signal: TS rank of volume * close rank * returns.",
    "Ts_Rank(volume, 32) * (1-Ts_Rank(close+high-low, 16)) * (1-Ts_Rank(returns, 32))")
def alpha035(d):
    o, h, l, c, v, vw, ret = _get(d)
    return ts_rank(v, 32) * (1 - ts_rank(c + h - l, 16)) * (1 - ts_rank(ret, 32))


@register(37, "Open-Close Correlation", "momentum",
    "Correlation between open change and close change — co-movement signal.",
    "rank(correlation(delay(open-close, 1), close, 200)) + rank(open-close)")
def alpha037(d):
    o, h, l, c, v, vw, ret = _get(d)
    return rank(correlation(delay(o - c, 1), c, 200)) + rank(o - c)


@register(38, "High-Close TS Rank", "momentum",
    "Sells stocks ranked high on close/high ratio — overbought detection.",
    "-1 * Ts_Rank(close/open, 10) * rank(close/open)")
def alpha038(d):
    o, h, l, c, v, vw, ret = _get(d)
    ratio = c / o
    return -1 * ts_rank(ratio, 10) * rank(ratio)


@register(40, "High-Volume Correlation", "momentum",
    "Sells when high-volume correlation is strong and increasing.",
    "-1 * rank(stddev(high, 10)) * correlation(high, volume, 10)")
def alpha040(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * rank(stddev(h, 10)) * correlation(h, v, 10)


@register(41, "High-Low-VWAP", "momentum",
    "Powered high-low-VWAP spread — range-based momentum.",
    "((high * low) ** 0.5) - vwap")
def alpha041(d):
    o, h, l, c, v, vw, ret = _get(d)
    return ((h * l) ** 0.5) - vw


@register(42, "VWAP-Close vs High-Low", "mean-reversion",
    "VWAP-close divergence relative to range — spread mean-reversion.",
    "rank(vwap - close) / rank(vwap + close)")
def alpha042(d):
    o, h, l, c, v, vw, ret = _get(d)
    return rank(vw - c) / rank(vw + c)


@register(43, "Volume-Close TS Rank", "momentum",
    "Volume acceleration combined with close delta — breakout signal.",
    "ts_rank(volume/adv20, 20) * ts_rank(-1*delta(close,7), 8)")
def alpha043(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv20 = adv(v, 20)
    return ts_rank(v / adv20, 20) * ts_rank(-1 * delta(c, 7), 8)


@register(44, "High-Volume Correlation Drop", "mean-reversion",
    "Sells when correlation between high and volume rank drops.",
    "-1 * correlation(high, rank(volume), 5)")
def alpha044(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * correlation(h, rank(v), 5)


@register(45, "Close-Volume-VWAP", "momentum",
    "Close-delay correlation weighted by volume-VWAP delta.",
    "-1 * rank(sum(delay(close,5), 20)/20) * correlation(close, volume, 2) * rank(correlation(sum(close,5), sum(close,20), 2))")
def alpha045(d):
    o, h, l, c, v, vw, ret = _get(d)
    return (-1 * rank(sum_ts(delay(c, 5), 20) / 20) *
            correlation(c, v, 2) *
            rank(correlation(sum_ts(c, 5), sum_ts(c, 20), 2)))


@register(46, "Close SMA Trend", "momentum",
    "Conditional: if 20-day SMA is trending, go with trend. Otherwise delta reversal.",
    "Close > delay(close,20) and delta trend conditional")
def alpha046(d):
    o, h, l, c, v, vw, ret = _get(d)
    inner = (delay(c, 20) - delay(c, 10)) / 10 - (delay(c, 10) - c) / 10
    cond = inner > 0.25
    result = -1 * delta(c, 1)
    result[cond] = (-1 * (c - delay(c, 1)))[cond]
    return result


@register(49, "Close Trend Conditional", "momentum",
    "If close near 20-day SMA, sell delta. Otherwise maintain.",
    "Conditional on close trend relative to 20d window")
def alpha049(d):
    o, h, l, c, v, vw, ret = _get(d)
    chg = (delay(c, 20) - delay(c, 10)) / 10 - (delay(c, 10) - c) / 10
    cond = chg < -0.1
    result = c.copy() * 0 + 1
    result[cond] = 1
    result[~cond] = -1 * delta(c, 1)
    return result


@register(50, "Volume-VWAP Correlation", "momentum",
    "Sells when volume-VWAP correlation peaks — exhaustion signal.",
    "-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)")
def alpha050(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * ts_max(rank(correlation(rank(v), rank(vw), 5)), 5)


@register(51, "Close SMA Conditional", "mean-reversion",
    "If close SMA is declining strongly, mean-revert. Else sell delta.",
    "Conditional close SMA-based")
def alpha051(d):
    o, h, l, c, v, vw, ret = _get(d)
    chg = (delay(c, 20) - delay(c, 10)) / 10 - (delay(c, 10) - c) / 10
    cond = chg < -0.05
    result = -1 * delta(c, 1)
    result[cond] = 1
    return result


@register(52, "Low-Volume-Returns", "mean-reversion",
    "Combines low rank with volume-returns momentum — oversold bounce.",
    "((-1*ts_min(low,5) + delay(ts_min(low,5),5)) * rank((sum(returns,240) - sum(returns,20))/220)) * ts_rank(volume, 5)")
def alpha052(d):
    o, h, l, c, v, vw, ret = _get(d)
    return ((-1 * ts_min(l, 5) + delay(ts_min(l, 5), 5)) *
            rank((sum_ts(ret, 240) - sum_ts(ret, 20)) / 220) *
            ts_rank(v, 5))


@register(53, "High-Low-Close Delta", "momentum",
    "Change in high-to-close ratio relative to range — directional strength.",
    "-1 * delta((high - close) / (close - low + 1e-8), 9)")
def alpha053(d):
    o, h, l, c, v, vw, ret = _get(d)
    return -1 * delta((h - c) / (c - l + 1e-8), 9)


@register(54, "Open-Close-Low Power", "mean-reversion",
    "Low-close-open power combo — detects range compression before breakout.",
    "(-1 * (low - close) * open**5) / ((low - high) * close**5 + 1e-8)")
def alpha054(d):
    o, h, l, c, v, vw, ret = _get(d)
    return (-1 * (l - c) * o**5) / ((l - h).replace(0, 1e-8) * c**5 + 1e-8)


@register(55, "High-Low-Close-Volume", "mean-reversion",
    "Correlation between high-low spread and volume, ranked by close position.",
    "-1 * correlation(rank((close - ts_min(low,12)) / (ts_max(high,12) - ts_min(low,12)+1e-8)), rank(volume), 6)")
def alpha055(d):
    o, h, l, c, v, vw, ret = _get(d)
    rng = ts_max(h, 12) - ts_min(l, 12) + 1e-8
    pos = (c - ts_min(l, 12)) / rng
    return -1 * correlation(rank(pos), rank(v), 6)


@register(60, "Close-Low-High Scale", "mean-reversion",
    "Scaled close position in range combined with volume — intraday range reversion.",
    "-1 * scale(rank(2*scale(rank((close-low)/(high-low+1e-8) - 0.5)) - scale(rank(ts_argmax(close, 10)))))")
def alpha060(d):
    o, h, l, c, v, vw, ret = _get(d)
    pos = (c - l) / (h - l + 1e-8)
    inner = 2 * scale(rank(pos - 0.5)) - scale(rank(ts_argmax(c, 10)))
    return -1 * scale(rank(inner))


@register(61, "VWAP-ADV Correlation", "momentum",
    "VWAP rank vs ADV correlation — institutional flow detection.",
    "rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, adv180, 18))")
def alpha061(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv180 = adv(v, 180)
    left = rank(vw - ts_min(vw, 16))
    right = rank(correlation(vw, adv180, 18))
    return (left < right).astype(float)


@register(62, "Open-ADV-High-Low Correlation", "momentum",
    "Open-ADV correlation vs high-low-open correlation — multi-factor flow.",
    "rank(correlation(vwap, sum(adv20, 22), 10)) < rank(rank(open) * rank(open - (high-low)/(high+1e-8)) * rank(open))")
def alpha062(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv20 = adv(v, 20)
    left = rank(correlation(vw, sum_ts(adv20, 22), 10))
    right = rank(rank(o) * rank(o - (h - l) / (h + 1e-8)) * rank(o))
    return (left < right).astype(float) * -1


@register(64, "Open-ADV Correlation Decay", "momentum",
    "Linearly decayed open-ADV correlation vs close-VWAP correlation.",
    "rank(correlation(sum(open*0.178+low*0.822, 13), sum(adv120, 13), 17)) < rank(delta(open*0.352+vwap*0.648, 1))")
def alpha064(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv120 = adv(v, 120)
    x = sum_ts(o * 0.178 + l * 0.822, 13)
    y = sum_ts(adv120, 13)
    left = rank(correlation(x, y, 17))
    right = rank(delta(o * 0.352 + vw * 0.648, 1))
    return (left < right).astype(float) * -1


@register(65, "Open-VWAP-ADV Correlation", "momentum",
    "Open-VWAP combo vs ADV correlation — institutional order flow.",
    "rank(correlation(open*0.00817+vwap*0.99183, sum(adv60,9), 6)) < rank(open - ts_min(open, 14))")
def alpha065(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv60 = adv(v, 60)
    x = o * 0.00817 + vw * 0.99183
    left = rank(correlation(x, sum_ts(adv60, 9), 6))
    right = rank(o - ts_min(o, 14))
    return (left < right).astype(float) * -1


@register(68, "High-ADV TS Rank", "momentum",
    "Time-series rank of high-ADV correlation combined with close delta.",
    "ts_rank(correlation(rank(high), rank(adv15), 9), 14) < rank(delta(close*0.518+close*0.482, 1))")
def alpha068(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv15 = adv(v, 15)
    left = ts_rank(correlation(rank(h), rank(adv15), 9), 14)
    right = rank(delta(c, 1))
    return (left < right).astype(float) * -1


@register(74, "Close-ADV-VWAP Correlation", "momentum",
    "Volume-close correlation vs ADV-VWAP correlation — two-factor flow.",
    "rank(correlation(close, sum(adv30, 37), 15)) < rank(correlation(rank(high*0.0261+vwap*0.9739), rank(volume), 11))")
def alpha074(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv30 = adv(v, 30)
    left = rank(correlation(c, sum_ts(adv30, 37), 15))
    right = rank(correlation(rank(h * 0.0261 + vw * 0.9739), rank(v), 11))
    return (left < right).astype(float) * -1


@register(75, "VWAP-Volume-Low Correlation", "momentum",
    "VWAP-volume correlation vs low-ADV correlation — multi-timeframe flow.",
    "rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12))")
def alpha075(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv50 = adv(v, 50)
    left = rank(correlation(vw, v, 4))
    right = rank(correlation(rank(l), rank(adv50), 12))
    return (left < right).astype(float)


@register(83, "Range-Volume-VWAP", "mean-reversion",
    "Delayed range/price ratio times volume rank, normalized — range reversion.",
    "(rank(delay((high-low)/(sum(close,5)/5), 2)) * rank(rank(volume))) / ((high-low)/(sum(close,5)/5) / (vwap-close+1e-8))")
def alpha083(d):
    o, h, l, c, v, vw, ret = _get(d)
    rng = (h - l) / (sum_ts(c, 5) / 5 + 1e-8)
    numer = rank(delay(rng, 2)) * rank(rank(v))
    denom = rng / (vw - c + 1e-8)
    return numer / denom.replace(0, np.nan)


@register(85, "High-Close-Volume Correlation", "momentum",
    "High rank-close delta correlation combined with close-ADV correlation.",
    "rank(correlation(high*0.876+close*0.124, adv30, 10))^rank(correlation(ts_rank(high*0.876+vwap*0.124, 4), ts_rank(volume, 10), 7))")
def alpha085(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv30 = adv(v, 30)
    x1 = rank(correlation(h * 0.876 + c * 0.124, adv30, 10))
    x2 = rank(correlation(ts_rank(h * 0.876 + vw * 0.124, 4), ts_rank(v, 10), 7))
    return x1 ** x2


@register(86, "Close-Open-ADV TS Rank", "mean-reversion",
    "Close-open-ADV relationship in time-series rank — gap filling detector.",
    "ts_rank(correlation(close, sum(adv20, 15), 6), 20) < rank(open + close - 2*vwap) * -1")
def alpha086(d):
    o, h, l, c, v, vw, ret = _get(d)
    adv20 = adv(v, 20)
    left = ts_rank(correlation(c, sum_ts(adv20, 15), 6), 20)
    right = rank(o + c - 2 * vw) * -1
    return (left < right).astype(float) * -1


@register(101, "Close-Open / Range", "mean-reversion",
    "Simple ratio of close-open gap to range — the simplest alpha in the paper.",
    "(close - open) / ((high - low) + 0.001)")
def alpha101(d):
    o, h, l, c, v, vw, ret = _get(d)
    return (c - o) / ((h - l) + 0.001)


def compute_alpha(num, data):
    """Compute a specific alpha by number."""
    return ALPHA_CATALOG[num]["func"](data)


def list_alphas():
    """Return list of available alphas."""
    return [
        {k: v for k, v in info.items() if k != "func"}
        for info in sorted(ALPHA_CATALOG.values(), key=lambda x: x["num"])
    ]
