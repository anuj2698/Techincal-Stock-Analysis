#!/usr/bin/env python3
"""Unified stock analysis engine — indicators, patterns, order blocks, and recommendation."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime


# ---------------------------------------------------------------------------
# Core Indicators
# ---------------------------------------------------------------------------

def sma(xs: list[float], n: int) -> float | None:
    if len(xs) < n:
        return None
    return statistics.fmean(xs[-n:])


def ema(xs: list[float], n: int) -> float | None:
    if len(xs) < n:
        return None
    k = 2 / (n + 1)
    e = xs[0]
    for x in xs[1:]:
        e = x * k + e * (1 - k)
    return e


def atr(candles: list[list[float]], n: int = 14) -> float:
    if len(candles) < n + 1:
        return 0.0
    trs: list[float] = []
    for i in range(-n, 0):
        h, l_val, pc = float(candles[i][2]), float(candles[i][3]), float(candles[i - 1][4])
        trs.append(max(h - l_val, abs(h - pc), abs(l_val - pc)))
    return statistics.fmean(trs) if trs else 0.0


def rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    gains = losses = 0.0
    for i in range(-period, 0):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100 - (100 / (1 + rs))


def rsi_series(closes: list[float], period: int = 14) -> list[float]:
    if len(closes) <= period:
        return []
    result = []
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        result.append(100.0)
    else:
        result.append(100 - (100 / (1 + avg_gain / avg_loss)))
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        if avg_loss == 0:
            result.append(100.0)
        else:
            result.append(100 - (100 / (1 + avg_gain / avg_loss)))
    return result


def macd(closes: list[float]) -> dict | None:
    if len(closes) < 35:
        return None
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    offset = 26 - 12
    macd_line = [ema12[i + offset] - ema26[i] for i in range(len(ema26))]
    signal = _ema_series(macd_line, 9)
    sig_offset = len(macd_line) - len(signal)
    histogram = [macd_line[i + sig_offset] - signal[i] for i in range(len(signal))]

    current_macd = macd_line[-1]
    current_signal = signal[-1]
    prev_macd = macd_line[-2]
    prev_signal = signal[-2]

    crossover = "none"
    if prev_macd <= prev_signal and current_macd > current_signal:
        crossover = "bullish"
    elif prev_macd >= prev_signal and current_macd < current_signal:
        crossover = "bearish"

    return {
        "macd_line": round(current_macd, 4),
        "signal_line": round(current_signal, 4),
        "histogram": round(histogram[-1], 4),
        "prev_histogram": round(histogram[-2], 4) if len(histogram) > 1 else 0,
        "crossover": crossover,
        "above_zero": current_macd > 0,
    }


def _ema_series(xs: list[float], n: int) -> list[float]:
    if len(xs) < n:
        return []
    k = 2 / (n + 1)
    result = [statistics.fmean(xs[:n])]
    for i in range(n, len(xs)):
        result.append(xs[i] * k + result[-1] * (1 - k))
    return result


def bollinger_bands(closes: list[float], n: int = 20, k: float = 2.0) -> dict | None:
    if len(closes) < n:
        return None
    import math
    window = [float(x) for x in closes[-n:]]
    mid = statistics.fmean(window)
    variance = sum((x - mid) ** 2 for x in window) / (len(window) - 1)
    std = math.sqrt(variance)
    upper = mid + k * std
    lower = mid - k * std
    bandwidth = (upper - lower) / mid * 100 if mid else 0
    last = closes[-1]
    pct_b = (last - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    squeeze = bandwidth < 10
    return {
        "upper": round(upper, 2),
        "middle": round(mid, 2),
        "lower": round(lower, 2),
        "bandwidth": round(bandwidth, 2),
        "pct_b": round(pct_b, 3),
        "squeeze": squeeze,
    }


def detect_rsi_divergence(closes: list[float], period: int = 14, lookback: int = 40) -> dict:
    result = {"bullish": False, "bearish": False, "bullish_note": "", "bearish_note": ""}
    rsi_vals = rsi_series(closes, period)
    if len(rsi_vals) < lookback:
        return result

    price_window = closes[-lookback:]
    rsi_window = rsi_vals[-lookback:]

    price_lows = []
    rsi_lows = []
    price_highs = []
    rsi_highs = []
    win = 3
    for i in range(win, len(price_window) - win):
        seg = price_window[i - win: i + win + 1]
        if price_window[i] == min(seg):
            price_lows.append((i, price_window[i], rsi_window[i]))
        if price_window[i] == max(seg):
            price_highs.append((i, price_window[i], rsi_window[i]))

    if len(price_lows) >= 2:
        prev, curr = price_lows[-2], price_lows[-1]
        if curr[1] < prev[1] and curr[2] > prev[2]:
            result["bullish"] = True
            result["bullish_note"] = "Price made lower low but RSI made higher low — momentum diverging bullishly"

    if len(price_highs) >= 2:
        prev, curr = price_highs[-2], price_highs[-1]
        if curr[1] > prev[1] and curr[2] < prev[2]:
            result["bearish"] = True
            result["bearish_note"] = "Price made higher high but RSI made lower high — momentum weakening"

    return result


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_weekly(candles: list[list[float]]) -> list[list[float]]:
    buckets: dict[tuple[int, int], list] = defaultdict(list)
    for row in candles:
        dt = datetime.fromtimestamp(int(row[0]), tz=UTC)
        iy, iw, _ = dt.isocalendar()
        buckets[(iy, iw)].append(row)
    out: list[list[float]] = []
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda r: r[0])
        ts0 = int(rows[0][0])
        o = float(rows[0][1])
        h = max(float(r[2]) for r in rows)
        l_val = min(float(r[3]) for r in rows)
        c = float(rows[-1][4])
        v = sum(float(r[5]) for r in rows if len(r) > 5)
        out.append([ts0, o, h, l_val, c, v])
    return out


def resample_monthly(candles: list[list[float]]) -> list[list[float]]:
    buckets: dict[tuple[int, int], list] = defaultdict(list)
    for row in candles:
        dt = datetime.fromtimestamp(int(row[0]), tz=UTC)
        buckets[(dt.year, dt.month)].append(row)
    out: list[list[float]] = []
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda r: r[0])
        ts0 = int(rows[0][0])
        o = float(rows[0][1])
        h = max(float(r[2]) for r in rows)
        l_val = min(float(r[3]) for r in rows)
        c = float(rows[-1][4])
        v = sum(float(r[5]) for r in rows if len(r) > 5)
        out.append([ts0, o, h, l_val, c, v])
    return out


# ---------------------------------------------------------------------------
# Volume-Confirmed Order Blocks
# ---------------------------------------------------------------------------

def find_order_blocks(candles: list[list[float]], lookback: int = 60) -> list[dict]:
    blocks: list[dict] = []
    if len(candles) < 20:
        return blocks
    n = len(candles)
    start = max(5, n - lookback)

    for i in range(start, n - 3):
        o, h, l_val, c = float(candles[i][1]), float(candles[i][2]), float(candles[i][3]), float(candles[i][4])
        v = float(candles[i][5]) if len(candles[i]) > 5 else 0
        body = abs(c - o)
        rng = h - l_val
        if rng <= 0:
            continue
        if body / rng < 0.40:
            continue

        vol_window = [float(candles[j][5]) for j in range(max(0, i - 20), i) if len(candles[j]) > 5]
        avg_vol = statistics.fmean(vol_window) if vol_window else 0
        vol_confirmed = avg_vol > 0 and v >= avg_vol * 1.3

        is_bearish_candle = c < o
        is_bullish_candle = c > o

        if is_bearish_candle:
            hi_after = max(float(candles[j][4]) for j in range(i + 1, min(i + 6, n)))
            if hi_after > c * 1.012:
                zone_low = l_val
                zone_high = min(o, c) + body * 0.3
                tested = any(float(candles[j][3]) <= zone_high for j in range(i + 1, n))
                broken = any(float(candles[j][4]) < zone_low for j in range(i + 1, n))
                status = "broken" if broken else ("tested" if tested else "fresh")
                blocks.append({
                    "type": "demand",
                    "zone_low": round(zone_low, 2),
                    "zone_high": round(zone_high, 2),
                    "volume_confirmed": vol_confirmed,
                    "volume": round(v, 0),
                    "avg_volume": round(avg_vol, 0),
                    "status": status,
                    "bar_index": i,
                    "strength": _ob_strength(vol_confirmed, status, body / rng),
                })

        if is_bullish_candle:
            lo_after = min(float(candles[j][4]) for j in range(i + 1, min(i + 6, n)))
            if lo_after < c * 0.988:
                zone_low = max(o, c) - body * 0.3
                zone_high = h
                tested = any(float(candles[j][2]) >= zone_low for j in range(i + 1, n))
                broken = any(float(candles[j][4]) > zone_high for j in range(i + 1, n))
                status = "broken" if broken else ("tested" if tested else "fresh")
                blocks.append({
                    "type": "supply",
                    "zone_low": round(zone_low, 2),
                    "zone_high": round(zone_high, 2),
                    "volume_confirmed": vol_confirmed,
                    "volume": round(v, 0),
                    "avg_volume": round(avg_vol, 0),
                    "status": status,
                    "bar_index": i,
                    "strength": _ob_strength(vol_confirmed, status, body / rng),
                })

    blocks = [b for b in blocks if b["status"] != "broken"]
    blocks.sort(key=lambda b: b["strength"], reverse=True)
    return blocks[:6]


def _ob_strength(vol_confirmed: bool, status: str, body_ratio: float) -> int:
    score = 0
    if vol_confirmed:
        score += 3
    if status == "fresh":
        score += 3
    elif status == "tested":
        score += 1
    if body_ratio >= 0.65:
        score += 2
    elif body_ratio >= 0.50:
        score += 1
    return score


def find_fair_value_gaps(candles: list[list[float]], lookback: int = 30) -> list[dict]:
    gaps: list[dict] = []
    n = len(candles)
    start = max(0, n - lookback)
    for i in range(start + 2, n):
        h1 = float(candles[i - 2][2])
        l3 = float(candles[i][3])
        if l3 > h1:
            gaps.append({
                "type": "bullish_fvg",
                "gap_low": round(h1, 2),
                "gap_high": round(l3, 2),
                "bar_index": i - 1,
                "filled": any(float(candles[j][3]) <= h1 for j in range(i, n)),
            })
        h3 = float(candles[i][2])
        l1 = float(candles[i - 2][3])
        if l1 > h3:
            gaps.append({
                "type": "bearish_fvg",
                "gap_low": round(h3, 2),
                "gap_high": round(l1, 2),
                "bar_index": i - 1,
                "filled": any(float(candles[j][2]) >= l1 for j in range(i, n)),
            })
    return [g for g in gaps if not g["filled"]][-4:]


# ---------------------------------------------------------------------------
# Support & Resistance Zones
# ---------------------------------------------------------------------------

def find_support_resistance(candles: list[list[float]], window: int = 5) -> list[dict]:
    if len(candles) < window * 2 + 3:
        return []

    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    volumes = [float(c[5]) if len(c) > 5 else 0 for c in candles]
    n = len(candles)
    levels: list[tuple[float, int, float]] = []

    for i in range(window, n - window):
        h_window = highs[i - window: i + window + 1]
        l_window = lows[i - window: i + window + 1]
        if highs[i] == max(h_window):
            levels.append((highs[i], i, volumes[i]))
        if lows[i] == min(l_window):
            levels.append((lows[i], i, volumes[i]))

    if not levels:
        return []

    levels.sort(key=lambda x: x[0])
    last_price = float(candles[-1][4])

    zones: list[dict] = []
    cluster: list[tuple[float, int, float]] = [levels[0]]
    threshold_pct = 0.015

    for lev in levels[1:]:
        if abs(lev[0] - cluster[0][0]) / cluster[0][0] <= threshold_pct:
            cluster.append(lev)
        else:
            zones.append(_build_zone(cluster, last_price, n))
            cluster = [lev]
    zones.append(_build_zone(cluster, last_price, n))

    supports = sorted([z for z in zones if z["type"] == "support"], key=lambda z: z["touches"] + z["recency_score"], reverse=True)[:3]
    resistances = sorted([z for z in zones if z["type"] == "resistance"], key=lambda z: z["touches"] + z["recency_score"], reverse=True)[:3]
    combined = supports + resistances
    combined.sort(key=lambda z: z["level"], reverse=True)
    return combined


def _build_zone(cluster: list[tuple[float, int, float]], last_price: float, total_bars: int) -> dict:
    prices = [c[0] for c in cluster]
    indices = [c[1] for c in cluster]
    vols = [c[2] for c in cluster]
    level = statistics.fmean(prices)
    most_recent = max(indices)
    recency = max(0, 10 - (total_bars - most_recent) // 10)
    distance_pct = (last_price - level) / level * 100 if level else 0
    return {
        "level": round(level, 2),
        "zone_low": round(min(prices), 2),
        "zone_high": round(max(prices), 2),
        "touches": len(cluster),
        "recency_score": recency,
        "avg_volume_at_level": round(statistics.fmean(vols), 0) if vols else 0,
        "type": "support" if distance_pct > 0 else "resistance",
        "distance_pct": round(distance_pct, 2),
    }


# ---------------------------------------------------------------------------
# Candlestick Patterns
# ---------------------------------------------------------------------------

def detect_candlestick_patterns(candles: list[list[float]], sr_zones: list[dict] | None = None) -> list[dict]:
    patterns: list[dict] = []
    if len(candles) < 5:
        return patterns

    last = len(candles) - 1
    for i in range(max(2, last - 4), last + 1):
        o, h, l_val, c = float(candles[i][1]), float(candles[i][2]), float(candles[i][3]), float(candles[i][4])
        body = abs(c - o)
        rng = h - l_val
        if rng == 0:
            continue
        body_ratio = body / rng
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l_val

        near_level = _near_sr_level(c, sr_zones) if sr_zones else None
        bars_ago = last - i

        if body_ratio < 0.1 and rng > 0:
            patterns.append({"name": "Doji", "type": "neutral", "bar_index": i, "bars_ago": bars_ago, "near_level": near_level})

        if c > o and lower_wick >= body * 2 and upper_wick < body * 0.5 and body_ratio > 0.15:
            patterns.append({"name": "Hammer", "type": "bullish", "bar_index": i, "bars_ago": bars_ago, "near_level": near_level})

        if c < o and upper_wick >= body * 2 and lower_wick < body * 0.5 and body_ratio > 0.15:
            patterns.append({"name": "Shooting Star", "type": "bearish", "bar_index": i, "bars_ago": bars_ago, "near_level": near_level})

        if i > 0:
            po, pc = float(candles[i - 1][1]), float(candles[i - 1][4])
            p_body = abs(pc - po)

            if pc < po and c > o and o <= pc and c >= po and body > p_body * 0.8:
                patterns.append({"name": "Bullish Engulfing", "type": "bullish", "bar_index": i, "bars_ago": bars_ago, "near_level": near_level})

            if pc > po and c < o and o >= pc and c <= po and body > p_body * 0.8:
                patterns.append({"name": "Bearish Engulfing", "type": "bearish", "bar_index": i, "bars_ago": bars_ago, "near_level": near_level})

            if pc < po and c > o and o < pc and c > (po + pc) / 2 and c < po:
                patterns.append({"name": "Piercing Line", "type": "bullish", "bar_index": i, "bars_ago": bars_ago, "near_level": near_level})

            if pc > po and c < o and o > pc and c < (po + pc) / 2 and c > po:
                patterns.append({"name": "Dark Cloud Cover", "type": "bearish", "bar_index": i, "bars_ago": bars_ago, "near_level": near_level})

        if i > 1:
            po2, pc2 = float(candles[i - 2][1]), float(candles[i - 2][4])
            po1, pc1 = float(candles[i - 1][1]), float(candles[i - 1][4])
            b1 = abs(pc1 - po1)
            r1 = float(candles[i - 1][2]) - float(candles[i - 1][3])

            if pc2 < po2 and c > o and b1 < body * 0.4 and r1 > 0 and b1 / r1 < 0.3 and c > (po2 + pc2) / 2:
                patterns.append({"name": "Morning Star", "type": "bullish", "bar_index": i, "bars_ago": bars_ago, "near_level": near_level})

            if pc2 > po2 and c < o and b1 < body * 0.4 and r1 > 0 and b1 / r1 < 0.3 and c < (po2 + pc2) / 2:
                patterns.append({"name": "Evening Star", "type": "bearish", "bar_index": i, "bars_ago": bars_ago, "near_level": near_level})

    return patterns


def _near_sr_level(price: float, sr_zones: list[dict]) -> dict | None:
    for z in sr_zones:
        if abs(price - z["level"]) / z["level"] <= 0.02:
            return {"level": z["level"], "type": z["type"]}
    return None


# ---------------------------------------------------------------------------
# Chart Patterns (enhanced with volume)
# ---------------------------------------------------------------------------

def detect_chart_patterns(candles: list[list[float]]) -> list[dict]:
    patterns: list[dict] = []
    if len(candles) < 35:
        return patterns

    closes = [float(c[4]) for c in candles]
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    volumes = [float(c[5]) if len(c) > 5 else 0 for c in candles]
    last = closes[-1]

    atr_val = atr(candles, 14)
    vol20 = sma(volumes, 20) if volumes else 0

    w = min(30, len(candles) - 1)
    hh, ll = max(highs[-w:]), min(lows[-w:])
    band_pct = abs(hh - ll) / ll * 100 if ll else 0

    if band_pct <= 10:
        patterns.append({
            "name": "Volatility Compression",
            "note": f"30-day range {band_pct:.1f}% — tight consolidation, breakout likely",
            "bias": "neutral",
            "breakout_target_up": round(hh + (hh - ll), 2),
            "breakout_target_down": round(ll - (hh - ll), 2),
        })

    m = len(candles)
    if m >= 35:
        hl_seg = [(highs[i], lows[i]) for i in range(m - 35, m)]
        h_slope = _linreg_slope([a for a, _ in hl_seg])
        l_slope = _linreg_slope([b for _, b in hl_seg])

        if h_slope < 0 and l_slope > 0 and band_pct <= 22:
            apex = last
            height = hh - ll
            patterns.append({
                "name": "Symmetrical Triangle",
                "note": "Converging trendlines — breakout direction will set trend",
                "bias": "neutral",
                "breakout_target_up": round(apex + height, 2),
                "breakout_target_down": round(apex - height, 2),
            })

        highs_tail = [a for a, _ in hl_seg]
        if abs(h_slope) < abs(l_slope) * 0.45 and l_slope > 0 and max(highs_tail[:10]) >= max(highs_tail[-10:]) * 0.98:
            resistance = max(highs_tail)
            height = resistance - min([b for _, b in hl_seg])
            vol_rising = volumes and sma(volumes[-5:], 5) and vol20 and sma(volumes[-5:], 5) > vol20
            patterns.append({
                "name": "Ascending Triangle",
                "note": f"Flat resistance near {resistance:.2f} with rising lows" + (" — volume rising" if vol_rising else ""),
                "bias": "bullish",
                "breakout_target_up": round(resistance + height, 2),
                "volume_confirmed": bool(vol_rising),
            })

        if h_slope < 0 and l_slope > atr_val / last * -0.2:
            support = min([b for _, b in hl_seg])
            height = max(highs_tail) - support
            patterns.append({
                "name": "Descending Triangle",
                "note": f"Declining highs into support near {support:.2f}",
                "bias": "bearish",
                "breakout_target_down": round(support - height, 2),
            })

    pk, _ = _local_extrema(highs[-80:], win=4)
    if len(pk) >= 3:
        last_peaks = [p for _, p in pk[-3:]]
        if _pct(last_peaks[-1], last_peaks[-2]) < 2.8:
            if last < min(last_peaks) * 0.985:
                neckline = min(lows[-(80 - pk[-2][0]):]) if pk else ll
                patterns.append({
                    "name": "Double Top",
                    "note": f"Resistance cluster near {last_peaks[-1]:.2f} — broke below neckline",
                    "bias": "bearish",
                    "breakout_target_down": round(neckline - (last_peaks[-1] - neckline), 2),
                })

    peaks, troughs = _local_extrema(highs[-100:], win=5)
    if len(peaks) >= 3:
        p3 = [peaks[-3][1], peaks[-2][1], peaks[-1][1]]
        if p3[0] < p3[1] and p3[2] < p3[1] and abs(p3[0] - p3[2]) / p3[1] < 0.04:
            if last < p3[2] * 0.98:
                neckline = min(lows[-60:]) if len(lows) >= 60 else ll
                patterns.append({
                    "name": "Head & Shoulders",
                    "note": "Central peak higher than shoulders — bearish reversal pattern",
                    "bias": "bearish",
                    "breakout_target_down": round(neckline - (p3[1] - neckline), 2),
                })

    if len(closes) >= 60:
        seg = closes[-120:] if len(closes) >= 120 else closes
        lhs = statistics.fmean(seg[:20])
        mid = min(seg[:len(seg) - 15])
        rhs = statistics.fmean(seg[-30:-10])
        handle = statistics.fmean(seg[-8:])
        if lhs > mid * 0.97 and rhs > mid * 1.06 and handle < rhs * 0.988 and handle > mid * 1.03:
            patterns.append({
                "name": "Cup & Handle",
                "note": "Rounded base with handle pullback — bullish continuation",
                "bias": "bullish",
                "breakout_target_up": round(max(highs[-30:]) + (max(highs[-30:]) - mid), 2),
            })

    if len(closes) >= 30:
        ret_10 = _pct(closes[-10], closes[-20])
        ret_prior = _pct(closes[-20], closes[-30])
        if abs(ret_prior) >= 12 and abs(ret_10) <= 9:
            direction = "bullish" if closes[-30] < closes[-20] else "bearish"
            patterns.append({
                "name": "Flag / Pennant",
                "note": f"Impulse then orderly drift — {'bullish' if direction == 'bullish' else 'bearish'} continuation expected",
                "bias": direction,
            })

    return patterns


def _local_extrema(xs: list[float], win: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    peaks: list[tuple[int, float]] = []
    troughs: list[tuple[int, float]] = []
    n = len(xs)
    for i in range(win, n - win):
        window = xs[i - win: i + win + 1]
        if xs[i] == max(window):
            peaks.append((i, xs[i]))
        if xs[i] == min(window):
            troughs.append((i, xs[i]))
    return peaks, troughs


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return abs(a - b) / b * 100.0


def _linreg_slope(y: list[float]) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = statistics.fmean(y)
    num = sum((i - x_mean) * (yi - y_mean) for i, yi in enumerate(y))
    den = sum((i - x_mean) ** 2 for i in range(n)) or 1e-12
    return num / den


# ---------------------------------------------------------------------------
# Leading Indicators — Market Structure, VWAP, Pivots, Fibonacci
# ---------------------------------------------------------------------------

def detect_market_structure(candles: list[list[float]], window: int = 5) -> dict:
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]
    n = len(candles)

    if n < window * 4:
        return {"structure": "transitional", "swing_points": [], "bos_events": [], "choch_events": [], "last_bos": None, "last_choch": None}

    peaks, _ = _local_extrema(highs, window)
    _, troughs = _local_extrema(lows, window)

    raw_points = []
    for idx, val in peaks:
        raw_points.append({"type": "swing_high", "index": idx, "price": val})
    for idx, val in troughs:
        raw_points.append({"type": "swing_low", "index": idx, "price": val})
    raw_points.sort(key=lambda p: (p["index"], 0 if p["type"] == "swing_low" else 1))

    # Deduplicate: keep only alternating high/low sequence
    swing_points = []
    for pt in raw_points:
        if not swing_points or swing_points[-1]["type"] != pt["type"]:
            swing_points.append(pt)
        elif pt["type"] == "swing_high" and pt["price"] > swing_points[-1]["price"]:
            swing_points[-1] = pt
        elif pt["type"] == "swing_low" and pt["price"] < swing_points[-1]["price"]:
            swing_points[-1] = pt

    # Label HH/HL/LH/LL
    prev_high = None
    prev_low = None
    for sp in swing_points:
        if sp["type"] == "swing_high":
            if prev_high is not None:
                sp["label"] = "HH" if sp["price"] > prev_high else "LH"
            else:
                sp["label"] = "HH"
            prev_high = sp["price"]
        else:
            if prev_low is not None:
                sp["label"] = "HL" if sp["price"] > prev_low else "LL"
            else:
                sp["label"] = "HL"
            prev_low = sp["price"]

    # Detect BOS and CHoCH
    bos_events = []
    choch_events = []
    recent_highs = [sp for sp in swing_points if sp["type"] == "swing_high"]
    recent_lows = [sp for sp in swing_points if sp["type"] == "swing_low"]

    # Determine prevailing structure from last 4 swing points
    last_labels = [sp["label"] for sp in swing_points[-6:]]
    bullish_labels = sum(1 for lb in last_labels if lb in ("HH", "HL"))
    bearish_labels = sum(1 for lb in last_labels if lb in ("LH", "LL"))
    prevailing = "bullish" if bullish_labels > bearish_labels else "bearish" if bearish_labels > bullish_labels else "transitional"

    for i in range(len(swing_points)):
        sp = swing_points[i]
        if sp["type"] == "swing_high" and i > 0:
            # Check if any subsequent candle close broke above this swing high
            for j in range(sp["index"] + 1, min(sp["index"] + window * 4, n)):
                if closes[j] > sp["price"]:
                    event = {"direction": "bullish", "break_level": round(sp["price"], 2), "bar_index": j, "close": round(closes[j], 2)}
                    # Is this a CHoCH? (bullish break during bearish structure)
                    labels_before = [s["label"] for s in swing_points[:i+1] if s["type"] in ("swing_high", "swing_low")][-4:]
                    bear_count = sum(1 for lb in labels_before if lb in ("LH", "LL"))
                    if bear_count >= 3:
                        event["note"] = f"Bearish structure broken — bullish BOS above {sp['price']:.2f}"
                        choch_events.append(event)
                    else:
                        bos_events.append(event)
                    break

        elif sp["type"] == "swing_low" and i > 0:
            for j in range(sp["index"] + 1, min(sp["index"] + window * 4, n)):
                if closes[j] < sp["price"]:
                    event = {"direction": "bearish", "break_level": round(sp["price"], 2), "bar_index": j, "close": round(closes[j], 2)}
                    labels_before = [s["label"] for s in swing_points[:i+1] if s["type"] in ("swing_high", "swing_low")][-4:]
                    bull_count = sum(1 for lb in labels_before if lb in ("HH", "HL"))
                    if bull_count >= 3:
                        event["note"] = f"Bullish structure broken — bearish BOS below {sp['price']:.2f}"
                        choch_events.append(event)
                    else:
                        bos_events.append(event)
                    break

    # Determine current structure
    if len(swing_points) >= 4:
        last_highs = [sp for sp in swing_points if sp["type"] == "swing_high"][-2:]
        last_lows = [sp for sp in swing_points if sp["type"] == "swing_low"][-2:]
        hh_seq = len(last_highs) == 2 and last_highs[-1]["label"] == "HH"
        hl_seq = len(last_lows) == 2 and last_lows[-1]["label"] == "HL"
        lh_seq = len(last_highs) == 2 and last_highs[-1]["label"] == "LH"
        ll_seq = len(last_lows) == 2 and last_lows[-1]["label"] == "LL"
        if hh_seq and hl_seq:
            structure = "bullish"
        elif lh_seq and ll_seq:
            structure = "bearish"
        else:
            structure = "transitional"
    else:
        structure = "transitional"

    # Keep only recent events (last 5)
    bos_events = bos_events[-5:]
    choch_events = choch_events[-3:]
    formatted_points = [{"type": sp["type"], "label": sp.get("label", ""), "index": sp["index"], "price": round(sp["price"], 2)} for sp in swing_points]

    return {
        "structure": structure,
        "swing_points": formatted_points[-8:],
        "bos_events": bos_events,
        "choch_events": choch_events,
        "last_bos": bos_events[-1] if bos_events else None,
        "last_choch": choch_events[-1] if choch_events else None,
    }


def compute_vwap(candles: list[list[float]]) -> dict | None:
    if len(candles) < 5:
        return None
    cum_tp_vol = 0.0
    cum_vol = 0.0
    vwap_values = []
    for c in candles:
        h, l_val, cl = float(c[2]), float(c[3]), float(c[4])
        v = float(c[5]) if len(c) > 5 else 0
        tp = (h + l_val + cl) / 3.0
        cum_tp_vol += tp * v
        cum_vol += v
        vwap_values.append(cum_tp_vol / cum_vol if cum_vol > 0 else tp)

    if cum_vol == 0:
        return None

    vwap_val = vwap_values[-1]

    # Standard deviation bands
    cum_tp_vol2 = 0.0
    cum_vol2 = 0.0
    variance_sum = 0.0
    for i, c in enumerate(candles):
        h, l_val, cl = float(c[2]), float(c[3]), float(c[4])
        v = float(c[5]) if len(c) > 5 else 0
        tp = (h + l_val + cl) / 3.0
        cum_tp_vol2 += tp * v
        cum_vol2 += v
        vwap_at_bar = cum_tp_vol2 / cum_vol2 if cum_vol2 > 0 else tp
        variance_sum += v * (tp - vwap_at_bar) ** 2

    stdev = math.sqrt(variance_sum / cum_vol) if cum_vol > 0 else 0

    last = float(candles[-1][4])
    if last > vwap_val * 1.001:
        position = "above"
    elif last < vwap_val * 0.999:
        position = "below"
    else:
        position = "at"

    upper_1 = vwap_val + stdev
    lower_1 = vwap_val - stdev
    upper_2 = vwap_val + 2 * stdev
    lower_2 = vwap_val - 2 * stdev

    if last > upper_2:
        band_zone = "above_2sd"
    elif last > upper_1:
        band_zone = "above_1sd"
    elif last < lower_2:
        band_zone = "below_2sd"
    elif last < lower_1:
        band_zone = "below_1sd"
    else:
        band_zone = "inside"

    return {
        "vwap": round(vwap_val, 2),
        "upper_1": round(upper_1, 2),
        "lower_1": round(lower_1, 2),
        "upper_2": round(upper_2, 2),
        "lower_2": round(lower_2, 2),
        "price_position": position,
        "band_zone": band_zone,
    }


def pivot_points(candles: list[list[float]], method: str = "standard", timeframe: str = "positional") -> dict | None:
    if len(candles) < 5:
        return None

    if timeframe == "short_term":
        c = candles[-1]
        h, l_val, cl = float(c[2]), float(c[3]), float(c[4])
        period = "daily"
    else:
        weekly = resample_weekly(candles)
        if len(weekly) < 2:
            return None
        w = weekly[-2]
        h, l_val, cl = float(w[2]), float(w[3]), float(w[4])
        period = "weekly"

    pp = (h + l_val + cl) / 3.0
    rng = h - l_val

    if method == "fibonacci":
        r1 = pp + 0.382 * rng
        r2 = pp + 0.618 * rng
        r3 = pp + rng
        s1 = pp - 0.382 * rng
        s2 = pp - 0.618 * rng
        s3 = pp - rng
    else:
        r1 = 2 * pp - l_val
        s1 = 2 * pp - h
        r2 = pp + rng
        s2 = pp - rng
        r3 = h + 2 * (pp - l_val)
        s3 = l_val - 2 * (h - pp)

    last = float(candles[-1][4])
    levels_map = {"R3": r3, "R2": r2, "R1": r1, "PP": pp, "S1": s1, "S2": s2, "S3": s3}
    nearest = min(levels_map.items(), key=lambda kv: abs(kv[1] - last))

    return {
        "method": method,
        "period": period,
        "pp": round(pp, 2),
        "r1": round(r1, 2), "r2": round(r2, 2), "r3": round(r3, 2),
        "s1": round(s1, 2), "s2": round(s2, 2), "s3": round(s3, 2),
        "price_vs_pp": "above" if last > pp else "below",
        "nearest_level": {"label": nearest[0], "price": round(nearest[1], 2), "distance_pct": round((last - nearest[1]) / nearest[1] * 100, 2)},
    }


def fibonacci_retracement(candles: list[list[float]], lookback: int = 50) -> dict | None:
    if len(candles) < 10:
        return None
    subset = candles[-lookback:] if len(candles) >= lookback else candles
    highs = [float(c[2]) for c in subset]
    lows = [float(c[3]) for c in subset]

    win = max(2, min(5, len(subset) // 10))
    peaks, _ = _local_extrema(highs, win)
    _, troughs = _local_extrema(lows, win)

    if not peaks or not troughs:
        return None

    highest = max(peaks, key=lambda p: p[1])
    lowest = min(troughs, key=lambda p: p[1])

    swing_high = highest[1]
    swing_low = lowest[1]
    swing_high_idx = highest[0]
    swing_low_idx = lowest[0]

    if swing_high == swing_low:
        return None

    trend = "up" if swing_low_idx < swing_high_idx else "down"
    rng = swing_high - swing_low
    ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    if trend == "up":
        levels = [{"ratio": r, "price": round(swing_high - r * rng, 2), "label": f"{r*100:.1f}%"} for r in ratios]
    else:
        levels = [{"ratio": r, "price": round(swing_low + r * rng, 2), "label": f"{r*100:.1f}%"} for r in ratios]

    last = float(candles[-1][4])
    current_near = None
    for fl in levels:
        if abs(last - fl["price"]) / fl["price"] <= 0.015:
            current_near = {"label": fl["label"], "price": fl["price"], "distance_pct": round((last - fl["price"]) / fl["price"] * 100, 2)}
            break

    golden_low = swing_high - 0.618 * rng if trend == "up" else swing_low + 0.382 * rng
    golden_high = swing_high - 0.382 * rng if trend == "up" else swing_low + 0.618 * rng
    in_golden = golden_low <= last <= golden_high

    return {
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "swing_high_index": swing_high_idx,
        "swing_low_index": swing_low_idx,
        "trend": trend,
        "levels": levels,
        "current_near": current_near,
        "in_golden_zone": in_golden,
    }


# ---------------------------------------------------------------------------
# Swing Analysis (enhanced)
# ---------------------------------------------------------------------------

def swing_analysis(sym: str, candles: list[list[float]]) -> dict | None:
    w = resample_weekly(candles)
    m = resample_monthly(candles)
    if len(w) < 12 or len(m) < 4 or len(candles) < 45:
        return None

    cw = [float(row[4]) for row in w]
    cm = [float(row[4]) for row in m]
    cd = [float(row[4]) for row in candles]
    last = cd[-1]

    w_rsi = rsi(cw)
    d_rsi = rsi(cd)
    if w_rsi is None:
        return None

    sma10_w = sma(cw, 10)
    sma20_w = sma(cw, 20)
    sma6_m = sma(cm, min(6, len(cm)))

    macd_data = macd(cd)

    wh = max(float(x[2]) for x in w[-8:])
    wl = min(float(x[3]) for x in w[-8:])
    bw_pct = abs(wh - wl) / (last or 1e-9) * 100

    m_up = sma6_m and last >= sma6_m * 0.98
    m_dn = sma6_m and last <= sma6_m * 1.02

    notes: list[str] = []
    score = 0
    direction = "neutral"

    if m_up and sma20_w and last >= sma20_w * 0.97 and cw[-1] < cw[-2] and w_rsi < 52:
        score += 28
        direction = "long"
        notes.append("Monthly above 6M mean; pullback into weekly SMA20 (continuation dip)")

    if m_up and w_rsi <= 43 and cw[-1] > (float(w[-1][2]) + float(w[-1][3])) / 2:
        score += 24
        direction = direction if direction != "neutral" else "long"
        notes.append("Weekly RSI softened; close recovered above midpoint")

    if m_up and bw_pct <= 14 and sma10_w and last > sma10_w:
        score += 18
        direction = direction if direction != "neutral" else "long"
        notes.append("Multi-week compression above SMA10")

    hh12 = max(cw[-12:])
    if m_up and last >= hh12 * 0.993 and cw[-1] >= cw[-2]:
        score += 22
        direction = "long"
        notes.append("Pressing 12-week highs (expansion potential)")

    if m_dn and sma20_w and last <= sma20_w * 1.03 and cw[-2] > cw[-1] and w_rsi > 46:
        score += 26
        direction = "short"
        notes.append("Below monthly SMA6, weekly rollover from resistance")

    if macd_data:
        if macd_data["crossover"] == "bullish":
            score += 12
            direction = direction if direction != "neutral" else "long"
            notes.append("MACD bullish crossover")
        elif macd_data["crossover"] == "bearish":
            score += 12
            direction = direction if direction != "neutral" else "short"
            notes.append("MACD bearish crossover")

    div = detect_rsi_divergence(cd)
    if div["bullish"]:
        score += 10
        direction = direction if direction != "neutral" else "long"
        notes.append("Bullish RSI divergence on daily")
    if div["bearish"]:
        score += 10
        direction = direction if direction != "neutral" else "short"
        notes.append("Bearish RSI divergence on daily")

    atr_d = statistics.fmean(float(x[2]) - float(x[3]) for x in candles[-10:])

    if direction == "short":
        stop = float(w[-1][2]) + 0.75 * atr_d
        t1 = last - max(atr_d * 2.8, bw_pct / 100 * last * 0.5 or atr_d)
        t2 = last - max(atr_d * 5.0, float(w[-1][2]) - float(w[-1][3]) + atr_d)
    else:
        stop = float(w[-1][3]) - 0.75 * atr_d
        t1 = last + max(atr_d * 2.6, bw_pct / 100 * last * 0.45 or atr_d)
        t2 = last + max(atr_d * 4.8, float(w[-1][2]) - float(w[-1][3]) + atr_d)

    risk = abs(last - stop)
    reward = abs(t1 - last)
    rr = reward / risk if risk else None

    # Normalize score to 0-100 based on directional max
    # Bullish max: 28+24+18+22+12+10 = 114, Bearish max: 26+12+10 = 48
    dir_max = 114 if direction == "long" else 48 if direction == "short" else 80
    normalized = min(100, round(score / dir_max * 100))
    if normalized >= 65:
        conviction = "High"
    elif normalized >= 40:
        conviction = "Medium"
    else:
        conviction = "Low"

    return {
        "bias": direction,
        "score": normalized,
        "conviction": conviction,
        "last": round(last, 2),
        "weekly_rsi": round(w_rsi, 2),
        "daily_rsi": round(d_rsi, 2) if d_rsi else None,
        "monthly_vs_sma6": round((last / sma6_m - 1) * 100, 2) if sma6_m else None,
        "range8w_pct": round(bw_pct, 2),
        "stop": round(stop, 2),
        "target1": round(t1, 2),
        "target2": round(t2, 2),
        "rr_approx": round(rr, 2) if rr else None,
        "ideas": notes,
        "macd": macd_data,
    }


# ---------------------------------------------------------------------------
# Recommendation Engine — Price-Level Based Action Plan
# ---------------------------------------------------------------------------

def _gather_levels(
    last: float,
    sr_zones: list[dict],
    order_blocks: list[dict],
    sma20: float | None,
    sma50: float | None,
    sma200: float | None,
    ema8: float | None = None,
    oi_data: dict | None = None,
    vwap_data: dict | None = None,
    pivot_data: dict | None = None,
    fib_data: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Collect all actionable levels, split into above/below CMP, ranked by strength."""
    levels: list[dict] = []

    for z in sr_zones:
        strength = z["touches"] * 2 + z["recency_score"]
        levels.append({"price": z["level"], "source": f"S/R zone ({z['touches']} touches)", "strength": strength})

    for b in order_blocks:
        mid = (b["zone_low"] + b["zone_high"]) / 2
        strength = b["strength"] + (3 if b["volume_confirmed"] else 0)
        tag = "demand" if b["type"] == "demand" else "supply"
        vol_tag = ", vol confirmed" if b["volume_confirmed"] else ""
        levels.append({
            "price": mid,
            "zone_low": b["zone_low"],
            "zone_high": b["zone_high"],
            "source": f"{tag} zone {b['zone_low']:.2f}–{b['zone_high']:.2f}{vol_tag}",
            "strength": strength,
        })

    if ema8:
        levels.append({"price": ema8, "source": "EMA 8", "strength": 4})
    for label, val in [("SMA 20", sma20), ("SMA 50", sma50), ("SMA 200", sma200)]:
        if val:
            levels.append({"price": val, "source": label, "strength": 3})

    if oi_data:
        avg_oi = (oi_data.get("call_wall_oi", 0) + oi_data.get("put_wall_oi", 0)) / 2
        cw = oi_data.get("call_wall")
        pw = oi_data.get("put_wall")
        mp = oi_data.get("max_pain")
        if cw:
            s = 6 if oi_data.get("call_wall_oi", 0) > avg_oi * 1.5 else 4
            levels.append({"price": cw, "source": f"Call OI Wall ({oi_data['call_wall_oi']:,.0f})", "strength": s})
        if pw:
            s = 6 if oi_data.get("put_wall_oi", 0) > avg_oi * 1.5 else 4
            levels.append({"price": pw, "source": f"Put OI Wall ({oi_data['put_wall_oi']:,.0f})", "strength": s})
        if mp:
            levels.append({"price": mp, "source": "Max Pain", "strength": 3})

    if vwap_data:
        levels.append({"price": vwap_data["vwap"], "source": "VWAP", "strength": 5})
        levels.append({"price": vwap_data["upper_1"], "source": "VWAP +1SD", "strength": 3})
        levels.append({"price": vwap_data["lower_1"], "source": "VWAP -1SD", "strength": 3})

    if pivot_data:
        levels.append({"price": pivot_data["pp"], "source": f"Pivot PP ({pivot_data['period']})", "strength": 5})
        for lk in ("r1", "r2", "r3", "s1", "s2", "s3"):
            val = pivot_data.get(lk)
            if val:
                levels.append({"price": val, "source": f"Pivot {lk.upper()}", "strength": 4})

    if fib_data:
        for fl in fib_data.get("levels", []):
            if fl["ratio"] in (0.382, 0.5, 0.618):
                levels.append({"price": fl["price"], "source": f"Fib {fl['label']}", "strength": 5 if fl["ratio"] == 0.618 else 4})
            elif fl["ratio"] in (0.236, 0.786):
                levels.append({"price": fl["price"], "source": f"Fib {fl['label']}", "strength": 3})

    above = sorted([l for l in levels if l["price"] > last * 1.003], key=lambda l: l["price"])
    below = sorted([l for l in levels if l["price"] < last * 0.997], key=lambda l: l["price"], reverse=True)

    for group in [above, below]:
        seen = set()
        deduped = []
        for l in group:
            key = round(l["price"], 0)
            if key not in seen:
                seen.add(key)
                deduped.append(l)
        group.clear()
        group.extend(deduped)

    above.sort(key=lambda l: l["strength"], reverse=True)
    below.sort(key=lambda l: l["strength"], reverse=True)

    return above, below


TIMEFRAME_CONFIG = {
    "short_term": {"sr_window": 3, "ob_lookback": 20, "fvg_lookback": 10, "max_dist_pct": 0.03, "sl_atr_mult": 1.0, "use_ema8": True, "label": "Short Term (1-2 Weeks)", "structure_window": 3, "pivot_method": "standard", "fib_lookback": 30},
    "positional": {"sr_window": 5, "ob_lookback": 60, "fvg_lookback": 30, "max_dist_pct": 0.12, "sl_atr_mult": 1.5, "use_ema8": False, "label": "Positional (1-2 Months)", "structure_window": 5, "pivot_method": "fibonacci", "fib_lookback": 50},
}


def generate_recommendation(
    candles: list[list[float]],
    sym: str,
    order_blocks: list[dict],
    sr_zones: list[dict],
    chart_patterns: list[dict],
    candlestick_patterns: list[dict],
    swing: dict | None,
    timeframe: str = "positional",
    entry_price: float | None = None,
    oi_data: dict | None = None,
    market_structure: dict | None = None,
    vwap_data: dict | None = None,
    pivot_data: dict | None = None,
    fib_data: dict | None = None,
) -> dict:
    cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["positional"])

    if len(candles) < 20:
        return {"status": "INSUFFICIENT DATA", "signals": {}, "action_plan": {}, "indicators": {}}

    closes = [float(c[4]) for c in candles]
    volumes = [float(c[5]) if len(c) > 5 else 0 for c in candles]
    last = closes[-1]

    ema8 = ema(closes, 8) if cfg.get("use_ema8") else None
    sma20 = sma(closes, 20)
    sma50 = sma(closes, min(50, len(closes))) if not cfg.get("use_ema8") else None
    sma200 = sma(closes, min(200, len(closes))) if not cfg.get("use_ema8") else None
    atr14 = atr(candles, 14)
    d_rsi = rsi(closes)
    macd_data = macd(closes)
    bb = bollinger_bands(closes)
    div = detect_rsi_divergence(closes)
    vol20 = sma(volumes, 20)
    vol5 = sma(volumes, 5)
    rvol = volumes[-1] / vol20 if vol20 and vol20 > 0 else None

    # --- Gather all price levels, filtered by max distance ---
    max_dist = cfg["max_dist_pct"]
    sl_mult = cfg["sl_atr_mult"]
    above_raw, below_raw = _gather_levels(last, sr_zones, order_blocks, sma20, sma50, sma200, ema8=ema8, oi_data=oi_data, vwap_data=vwap_data, pivot_data=pivot_data, fib_data=fib_data)
    above = [l for l in above_raw if (l["price"] - last) / last <= max_dist]
    below = [l for l in below_raw if (last - l["price"]) / last <= max_dist]

    # Combine all levels into one sorted list for relative lookups
    all_levels = above_raw + below_raw
    all_by_price = sorted(all_levels, key=lambda l: l["price"])

    sell_level = above[0] if above else None
    buy_level = below[0] if below else None
    buy_price = buy_level["price"] if buy_level else None
    sell_price = sell_level["price"] if sell_level else None

    def _levels_above(ref, exclude=None):
        """All technical levels above ref, excluding a specific price."""
        return [l for l in all_by_price
                if l["price"] > ref * 1.005
                and (not exclude or abs(l["price"] - exclude) / exclude > 0.005)]

    def _levels_below(ref, exclude=None):
        """All technical levels below ref, excluding a specific price."""
        return [l for l in reversed(all_by_price)
                if l["price"] < ref * 0.995
                and (not exclude or abs(l["price"] - exclude) / exclude > 0.005)]

    if buy_level:
        # SL: strongest level below buy level, buffered by ATR
        sl_candidates = _levels_below(buy_price)
        if sl_candidates:
            buy_sl = sl_candidates[0]["price"] - sl_mult * atr14 * 0.5
        else:
            buy_sl = buy_price - sl_mult * atr14
        # Targets: levels above buy level (not CMP), excluding sell entry
        tgt_candidates = _levels_above(buy_price, exclude=sell_price)
        buy_targets = [l["price"] for l in tgt_candidates[:2]]
        if not buy_targets:
            buy_targets = [buy_price + 2 * atr14, buy_price + 3.5 * atr14]
    else:
        buy_sl = last - sl_mult * atr14
        buy_targets = []

    if sell_level:
        # SL: strongest level above sell level, buffered by ATR
        sl_candidates = _levels_above(sell_price)
        if sl_candidates:
            sell_sl = sl_candidates[0]["price"] + sl_mult * atr14 * 0.5
        else:
            sell_sl = sell_price + sl_mult * atr14
        # Targets: levels below sell level (not CMP), excluding buy entry
        tgt_candidates = _levels_below(sell_price, exclude=buy_price)
        sell_targets = [l["price"] for l in tgt_candidates[:2]]
        if not sell_targets:
            sell_targets = [sell_price - 2 * atr14, sell_price - 3.5 * atr14]
    else:
        sell_sl = last + sl_mult * atr14
        sell_targets = []

    # Determine current status
    near_threshold = 0.01
    status = "NO TRADE"
    status_note = "Price is between key levels — wait for it to reach a buy or sell level"

    if buy_level and abs(last - buy_level["price"]) / last <= near_threshold:
        status = "BUY ZONE"
        status_note = f"Price is near buy level at {buy_level['price']:.2f} ({buy_level['source']})"
    elif sell_level and abs(last - sell_level["price"]) / last <= near_threshold:
        status = "SELL ZONE"
        status_note = f"Price is near sell level at {sell_level['price']:.2f} ({sell_level['source']})"

    action_plan = {
        "status": status,
        "status_note": status_note,
        "buy": {
            "level": round(buy_level["price"], 2) if buy_level else None,
            "source": buy_level["source"] if buy_level else None,
            "sl": round(buy_sl, 2),
            "targets": [round(t, 2) for t in buy_targets],
        },
        "sell": {
            "level": round(sell_level["price"], 2) if sell_level else None,
            "source": sell_level["source"] if sell_level else None,
            "sl": round(sell_sl, 2),
            "targets": [round(t, 2) for t in sell_targets],
        },
    }

    # --- Signals (kept for context) ---
    signals = _compute_signals(
        last, closes, volumes, sma20, sma50, sma200, atr14, d_rsi,
        macd_data, bb, div, vol20, vol5, rvol, order_blocks, sr_zones,
        chart_patterns, candlestick_patterns, oi_data=oi_data,
        market_structure=market_structure, vwap_data=vwap_data,
        pivot_data=pivot_data, fib_data=fib_data,
    )

    # --- Position Guidance ---
    position_guidance = _position_guidance(
        last, buy_level, sell_level, buy_sl, sell_sl,
        buy_targets, sell_targets, sma20, d_rsi, macd_data, chart_patterns,
        entry_price=entry_price, sr_zones=sr_zones, atr14=atr14, sl_mult=sl_mult,
    )

    return {
        "action_plan": action_plan,
        "signals": signals,
        "position_guidance": position_guidance,
        "indicators": {
            "ema8": round(ema8, 2) if ema8 else None,
            "sma20": round(sma20, 2) if sma20 else None,
            "sma50": round(sma50, 2) if sma50 else None,
            "sma200": round(sma200, 2) if sma200 else None,
            "atr14": round(atr14, 2),
            "rsi": round(d_rsi, 2) if d_rsi else None,
            "macd": macd_data,
            "bollinger": bb,
            "rvol": round(rvol, 2) if rvol else None,
            "divergence": div,
            "vwap": vwap_data,
            "pivot_points": pivot_data,
            "fibonacci": fib_data,
        },
    }


def _compute_signals(
    last, closes, volumes, sma20, sma50, sma200, atr14, d_rsi,
    macd_data, bb, div, vol20, vol5, rvol, order_blocks, sr_zones,
    chart_patterns, candlestick_patterns, oi_data=None,
    market_structure=None, vwap_data=None, pivot_data=None, fib_data=None,
) -> dict:
    signals: dict[str, dict] = {}

    # TREND
    trend_score = 50
    trend_notes = []
    if sma20 and sma50:
        if last > sma20 > sma50:
            trend_score += 25; trend_notes.append("Price > SMA20 > SMA50 — strong uptrend")
        elif last > sma20 and sma20 < sma50:
            trend_score += 10; trend_notes.append("Above SMA20, SMA20 below SMA50 — early recovery")
        elif last < sma20 < sma50:
            trend_score -= 25; trend_notes.append("Price < SMA20 < SMA50 — strong downtrend")
        elif last < sma20 and sma20 > sma50:
            trend_score -= 10; trend_notes.append("Below SMA20, SMA20 above SMA50 — pullback")
    if sma200:
        if last > sma200: trend_score += 10; trend_notes.append("Above 200 SMA — long-term bullish")
        else: trend_score -= 10; trend_notes.append("Below 200 SMA — long-term bearish")
    if macd_data:
        if macd_data["crossover"] == "bullish": trend_score += 15; trend_notes.append("MACD bullish crossover")
        elif macd_data["crossover"] == "bearish": trend_score -= 15; trend_notes.append("MACD bearish crossover")
        elif macd_data["above_zero"] and macd_data["histogram"] > macd_data["prev_histogram"]: trend_score += 5; trend_notes.append("MACD above zero, rising histogram")
        elif not macd_data["above_zero"] and macd_data["histogram"] < macd_data["prev_histogram"]: trend_score -= 5; trend_notes.append("MACD below zero, falling histogram")
    if vwap_data:
        if vwap_data["price_position"] == "above":
            trend_score += 5; trend_notes.append(f"Above VWAP ({vwap_data['vwap']}) — buyers in control")
        elif vwap_data["price_position"] == "below":
            trend_score -= 5; trend_notes.append(f"Below VWAP ({vwap_data['vwap']}) — sellers in control")
        if vwap_data["band_zone"] == "above_2sd":
            trend_score -= 3; trend_notes.append("Above VWAP +2SD — overextended")
        elif vwap_data["band_zone"] == "below_2sd":
            trend_score += 3; trend_notes.append("Below VWAP -2SD — mean reversion likely")
    if pivot_data:
        if last > pivot_data["pp"]:
            trend_score += 3; trend_notes.append(f"Above pivot ({pivot_data['pp']}) — bullish bias")
        else:
            trend_score -= 3; trend_notes.append(f"Below pivot ({pivot_data['pp']}) — bearish bias")
    signals["trend"] = {"score": max(0, min(100, trend_score)), "notes": trend_notes}

    # MOMENTUM
    mom_score = 50
    mom_notes = []
    if d_rsi is not None:
        if d_rsi < 30: mom_score += 20; mom_notes.append(f"RSI {d_rsi:.1f} — oversold")
        elif d_rsi < 40: mom_score += 5; mom_notes.append(f"RSI {d_rsi:.1f} — approaching oversold")
        elif d_rsi > 70: mom_score -= 20; mom_notes.append(f"RSI {d_rsi:.1f} — overbought")
        elif d_rsi > 60: mom_score -= 5; mom_notes.append(f"RSI {d_rsi:.1f} — elevated")
        else: mom_notes.append(f"RSI {d_rsi:.1f} — neutral")
    if div["bullish"]: mom_score += 20; mom_notes.append(div["bullish_note"])
    if div["bearish"]: mom_score -= 20; mom_notes.append(div["bearish_note"])
    if bb and bb["squeeze"]: mom_notes.append("Bollinger squeeze — expansion expected")
    if fib_data and fib_data.get("in_golden_zone"):
        if fib_data["trend"] == "up":
            mom_score += 8; mom_notes.append("In Fibonacci golden zone (38.2%-61.8%) of uptrend — support zone")
        else:
            mom_score -= 8; mom_notes.append("In Fibonacci golden zone (38.2%-61.8%) of downtrend — resistance zone")
    signals["momentum"] = {"score": max(0, min(100, mom_score)), "notes": mom_notes}

    # VOLUME
    vol_score = 50
    vol_notes = []
    if rvol is not None:
        if rvol >= 2.0: vol_score += 15; vol_notes.append(f"RVOL {rvol:.2f}x — very high")
        elif rvol >= 1.3: vol_score += 8; vol_notes.append(f"RVOL {rvol:.2f}x — above average")
        elif rvol <= 0.5: vol_score -= 10; vol_notes.append(f"RVOL {rvol:.2f}x — weak")
    if vol5 and vol20 and vol20 > 0:
        if vol5 > vol20 * 1.2: vol_score += 10; vol_notes.append("5d vol > 20d avg — accumulation")
        elif vol5 < vol20 * 0.8: vol_score -= 5; vol_notes.append("5d vol < 20d avg — fading interest")
    signals["volume"] = {"score": max(0, min(100, vol_score)), "notes": vol_notes}

    # STRUCTURE
    struct_score = 50
    struct_notes = []
    nearby_sup = [z for z in sr_zones if z["type"] == "support" and abs(z["distance_pct"]) <= 3]
    nearby_res = [z for z in sr_zones if z["type"] == "resistance" and abs(z["distance_pct"]) <= 3]
    if nearby_sup:
        best = max(nearby_sup, key=lambda z: z["touches"])
        struct_score += min(15, best["touches"] * 4)
        struct_notes.append(f"Near support at {best['level']:.2f} ({best['touches']} touches)")
    if nearby_res:
        best = max(nearby_res, key=lambda z: z["touches"])
        struct_score -= min(15, best["touches"] * 4)
        struct_notes.append(f"Near resistance at {best['level']:.2f} ({best['touches']} touches)")
    if market_structure:
        ms = market_structure
        if ms["structure"] == "bullish":
            struct_score += 10; struct_notes.append("Market structure bullish (HH/HL sequence)")
        elif ms["structure"] == "bearish":
            struct_score -= 10; struct_notes.append("Market structure bearish (LH/LL sequence)")
        last_bos = ms.get("last_bos")
        if last_bos:
            recency = len(closes) - 1 - last_bos.get("bar_index", 0)
            if recency <= 5:
                if last_bos["direction"] == "bullish":
                    struct_score += 8; struct_notes.append(f"Recent bullish BOS above {last_bos['break_level']}")
                else:
                    struct_score -= 8; struct_notes.append(f"Recent bearish BOS below {last_bos['break_level']}")
        last_choch = ms.get("last_choch")
        if last_choch:
            recency = len(closes) - 1 - last_choch.get("bar_index", 0)
            if recency <= 10:
                if last_choch["direction"] == "bullish":
                    struct_score += 12; struct_notes.append(f"CHoCH bullish — trend reversal at {last_choch['break_level']}")
                else:
                    struct_score -= 12; struct_notes.append(f"CHoCH bearish — trend reversal at {last_choch['break_level']}")
    signals["structure"] = {"score": max(0, min(100, struct_score)), "notes": struct_notes}

    # PATTERNS
    pat_score = 50
    pat_notes = []
    for p in chart_patterns:
        if p.get("bias") == "bullish": pat_score += 12; pat_notes.append(f"{p['name']} — {p.get('note', '')}")
        elif p.get("bias") == "bearish": pat_score -= 12; pat_notes.append(f"{p['name']} — {p.get('note', '')}")
    recent = [p for p in candlestick_patterns if p["bars_ago"] <= 2]
    for p in recent:
        lv = f" at {p['near_level']['type']} {p['near_level']['level']:.2f}" if p.get("near_level") else ""
        if p["type"] == "bullish": pat_score += 8; pat_notes.append(f"{p['name']}{lv}")
        elif p["type"] == "bearish": pat_score -= 8; pat_notes.append(f"{p['name']}{lv}")
    signals["patterns"] = {"score": max(0, min(100, pat_score)), "notes": pat_notes}

    # OI (only if data available)
    if oi_data:
        oi_score = 50
        oi_notes = []
        pcr = oi_data.get("pcr")
        if pcr is not None:
            if pcr > 1.0:
                oi_score += 15
                oi_notes.append(f"PCR {pcr:.2f} — high put writing, bullish")
            elif pcr < 0.7:
                oi_score -= 15
                oi_notes.append(f"PCR {pcr:.2f} — low put writing, bearish")
            else:
                oi_notes.append(f"PCR {pcr:.2f} — neutral")
        buildup = oi_data.get("oi_buildup")
        if buildup == "long_buildup":
            oi_score += 20; oi_notes.append("Long buildup — fresh longs being added")
        elif buildup == "short_buildup":
            oi_score -= 20; oi_notes.append("Short buildup — fresh shorts being added")
        elif buildup == "short_covering":
            oi_score += 10; oi_notes.append("Short covering — shorts exiting")
        elif buildup == "long_unwinding":
            oi_score -= 10; oi_notes.append("Long unwinding — longs exiting")
        cw = oi_data.get("call_wall")
        pw = oi_data.get("put_wall")
        if cw and abs(last - cw) / last <= 0.02:
            oi_score -= 5; oi_notes.append(f"Near call wall {cw} — resistance")
        if pw and abs(last - pw) / last <= 0.02:
            oi_score += 5; oi_notes.append(f"Near put wall {pw} — support")
        mp = oi_data.get("max_pain")
        if mp:
            oi_notes.append(f"Max pain: {mp}")
        signals["oi"] = {"score": max(0, min(100, oi_score)), "notes": oi_notes}

    return signals


def _position_guidance(
    last: float,
    buy_level: dict | None,
    sell_level: dict | None,
    buy_sl: float,
    sell_sl: float,
    buy_targets: list[float],
    sell_targets: list[float],
    sma20: float | None,
    d_rsi: float | None,
    macd_data: dict | None,
    chart_patterns: list[dict],
    entry_price: float | None = None,
    sr_zones: list[dict] | None = None,
    atr14: float = 0,
    sl_mult: float = 1.5,
) -> dict:

    if entry_price and entry_price > 0:
        return _personalized_guidance(
            last, entry_price, buy_level, sell_level, buy_sl, sell_sl,
            sma20, d_rsi, macd_data, chart_patterns, sr_zones or [], atr14, sl_mult,
        )

    hold_conditions = []
    book_levels = []
    exit_signals = []
    tighten_triggers = []

    if buy_level:
        hold_conditions.append(f"Hold longs while price stays above {buy_level['price']:.2f} ({buy_level['source']})")
        exit_signals.append(f"Exit longs if price closes below {buy_sl:.2f} (SL)")
    if sell_level and sell_level.get("price"):
        book_levels.append({"level": round(sell_level["price"], 2), "action": f"Book profits near {sell_level['price']:.2f} ({sell_level['source']})"})
    for i, t in enumerate(buy_targets[:2]):
        book_levels.append({"level": round(t, 2), "action": f"Target {i+1} for longs: {t:.2f}"})

    if sma20:
        hold_conditions.append(f"SMA20 at {sma20:.2f} — key trend reference")
    if d_rsi and d_rsi > 65:
        tighten_triggers.append(f"RSI elevated at {d_rsi:.1f} — consider tightening stop")
    if macd_data and macd_data["crossover"] == "bearish":
        tighten_triggers.append("MACD bearish crossover — momentum shifting down")

    bearish_pats = [p for p in chart_patterns if p.get("bias") == "bearish"]
    if bearish_pats:
        tighten_triggers.append(f"Bearish pattern: {bearish_pats[0]['name']} — reduce exposure or hedge")

    return {
        "has_position": False,
        "hold_conditions": hold_conditions,
        "profit_booking": book_levels,
        "exit_signals": exit_signals,
        "tighten_triggers": tighten_triggers,
    }


def _personalized_guidance(
    last: float,
    entry_price: float,
    buy_level: dict | None,
    sell_level: dict | None,
    buy_sl: float,
    sell_sl: float,
    sma20: float | None,
    d_rsi: float | None,
    macd_data: dict | None,
    chart_patterns: list[dict],
    sr_zones: list[dict],
    atr14: float,
    sl_mult: float,
) -> dict:
    pnl = last - entry_price
    pnl_pct = (pnl / entry_price) * 100 if entry_price else 0

    if abs(pnl_pct) < 0.5:
        pnl_status = "At Breakeven"
    elif pnl_pct > 0:
        pnl_status = "In Profit"
    else:
        pnl_status = "In Loss"

    supports_below = sorted(
        [z for z in sr_zones if z["type"] == "support" and z["level"] < last],
        key=lambda z: z["level"], reverse=True,
    )
    resists_above = sorted(
        [z for z in sr_zones if z["type"] == "resistance" and z["level"] > last],
        key=lambda z: z["level"],
    )

    hold_conditions = []
    book_levels = []
    exit_signals = []
    tighten_triggers = []

    # --- Score-based verdict ---
    # Positive = bullish (hold/add), negative = bearish (sell/exit)
    score = 0
    reasons = []

    # Trend alignment
    if sma20:
        if last > sma20:
            score += 2
            reasons.append(f"Above SMA20 ({sma20:.2f}) — trend supportive")
        else:
            score -= 2
            reasons.append(f"Below SMA20 ({sma20:.2f}) — trend weakening")

    # MACD
    if macd_data:
        if macd_data["crossover"] == "bullish":
            score += 2
            reasons.append("MACD bullish crossover — momentum building")
        elif macd_data["crossover"] == "bearish":
            score -= 2
            reasons.append("MACD bearish crossover — momentum fading")
        elif macd_data["above_zero"]:
            score += 1
        else:
            score -= 1

    # RSI
    if d_rsi:
        if d_rsi > 70:
            score -= 1
            reasons.append(f"RSI overbought at {d_rsi:.1f} — pullback risk")
        elif d_rsi < 30:
            score += 2
            reasons.append(f"RSI oversold at {d_rsi:.1f} — bounce likely")
        elif d_rsi < 40:
            score += 1

    # Patterns
    bearish_pats = [p for p in chart_patterns if p.get("bias") == "bearish"]
    bullish_pats = [p for p in chart_patterns if p.get("bias") == "bullish"]
    if bearish_pats:
        score -= 2
        reasons.append(f"Bearish pattern: {bearish_pats[0]['name']}")
    if bullish_pats:
        score += 1
        reasons.append(f"Bullish pattern: {bullish_pats[0]['name']}")

    # Support proximity
    if supports_below:
        dist_to_support = (last - supports_below[0]["level"]) / last * 100
        if dist_to_support < 1.5:
            score += 1
            reasons.append(f"Near support at {supports_below[0]['level']:.2f} ({supports_below[0]['touches']} touches)")
    if resists_above:
        dist_to_resist = (resists_above[0]["level"] - last) / last * 100
        if dist_to_resist < 1.5:
            score -= 1
            reasons.append(f"Near resistance at {resists_above[0]['level']:.2f} ({resists_above[0]['touches']} touches)")

    # P&L context adjustments
    if pnl_pct < -10:
        score -= 1
        reasons.append(f"Deep loss ({pnl_pct:.1f}%) — thesis needs re-evaluation")
    if pnl_pct > 15:
        score -= 1
        reasons.append(f"Large profit ({pnl_pct:.1f}%) — consider partial booking")

    # --- Determine verdict ---
    if score >= 3:
        verdict = "ADD MORE"
        verdict_note = "Multiple signals support the position — consider adding on dips"
    elif score >= 1:
        verdict = "HOLD"
        verdict_note = "Position is supported by current data — continue holding"
    elif score >= -1:
        verdict = "HOLD WITH CAUTION"
        verdict_note = "Mixed signals — hold but tighten stops and watch closely"
    elif score >= -3:
        verdict = "EXIT PARTIAL"
        verdict_note = "Weakening signals — book partial profits or reduce exposure"
    else:
        verdict = "EXIT"
        verdict_note = "Multiple signals against the position — consider exiting"

    # --- Build guidance using all technical levels relative to entry ---
    # Gather all levels: S/R + order blocks + SMAs
    all_levels = []
    for z in sr_zones:
        all_levels.append({"price": z["level"], "source": f"S/R ({z['touches']} touches)", "strength": z["touches"]})
    if buy_level:
        all_levels.append({"price": buy_level["price"], "source": buy_level["source"], "strength": 5})
    if sell_level:
        all_levels.append({"price": sell_level["price"], "source": sell_level["source"], "strength": 5})
    if sma20:
        all_levels.append({"price": sma20, "source": "SMA 20", "strength": 3})
    all_levels.sort(key=lambda l: l["price"])

    # Levels above CMP (targets for long, resistance)
    targets_up = [l for l in all_levels
                  if l["price"] > last * 1.005
                  and abs(l["price"] - entry_price) / entry_price > 0.005]
    # Levels below CMP (supports, SL zone)
    supports_down = [l for l in reversed(all_levels)
                     if l["price"] < last * 0.995
                     and abs(l["price"] - entry_price) / entry_price > 0.005]

    # SL: below nearest support with ATR buffer
    if supports_down:
        sl_level = supports_down[0]["price"] - sl_mult * atr14 * 0.5
        sl_source = supports_down[0]["source"]
    else:
        sl_level = last - sl_mult * atr14
        sl_source = "ATR-based"

    if pnl_status == "In Profit":
        trail_stop = max(entry_price, sl_level + sl_mult * atr14 * 0.5)
        hold_conditions.append(f"Trail stop: {trail_stop:.2f} — below {supports_down[0]['source'] if supports_down else 'entry'}")
        for t in targets_up[:2]:
            book_levels.append({"level": round(t["price"], 2), "action": f"Book at {t['price']:.2f} ({t['source']})"})
        if not targets_up:
            book_levels.append({"level": round(last + 2 * atr14, 2), "action": f"Target: {last + 2 * atr14:.2f} (2x ATR)"})
        exit_signals.append(f"Exit if closes below {trail_stop:.2f} (trail stop)")

    elif pnl_status == "In Loss":
        if supports_down and score >= 1:
            hold_conditions.append(f"Support at {supports_down[0]['price']:.2f} ({supports_down[0]['source']}) — averaging zone if thesis intact")
        hold_conditions.append(f"Hold if stays above {sl_level + sl_mult * atr14 * 0.5:.2f} ({sl_source})")
        # Targets: first the entry (breakeven), then levels above
        for t in targets_up[:2]:
            book_levels.append({"level": round(t["price"], 2), "action": f"Target: {t['price']:.2f} ({t['source']})"})
        if not book_levels:
            book_levels.append({"level": round(last + 2 * atr14, 2), "action": f"Target: {last + 2 * atr14:.2f} (2x ATR)"})
        exit_signals.append(f"Exit if closes below {sl_level:.2f} (SL — below {sl_source})")

    else:
        hold_conditions.append(f"Near breakeven — {verdict_note.lower()}")
        exit_signals.append(f"Exit if closes below {sl_level:.2f} ({sl_source})")

    return {
        "has_position": True,
        "entry_price": round(entry_price, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "pnl_status": pnl_status,
        "verdict": verdict,
        "verdict_note": verdict_note,
        "reasons": reasons,
        "hold_conditions": hold_conditions,
        "profit_booking": book_levels,
        "exit_signals": exit_signals,
        "tighten_triggers": tighten_triggers,
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def full_analysis(sym: str, candles: list[list[float]], timeframe: str = "positional", entry_price: float | None = None, oi_data: dict | None = None) -> dict:
    cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["positional"])

    sr_zones = find_support_resistance(candles, window=cfg["sr_window"])
    order_blocks = find_order_blocks(candles, lookback=cfg["ob_lookback"])
    fvgs = find_fair_value_gaps(candles, lookback=cfg["fvg_lookback"])
    chart_pats = detect_chart_patterns(candles)
    candle_pats = detect_candlestick_patterns(candles, sr_zones)
    swing = swing_analysis(sym, candles)

    market_struct = detect_market_structure(candles, window=cfg["structure_window"])
    vwap_data = compute_vwap(candles)
    pivots = pivot_points(candles, method=cfg["pivot_method"], timeframe=timeframe)
    fibs = fibonacci_retracement(candles, lookback=cfg["fib_lookback"])

    recommendation = generate_recommendation(
        candles=candles,
        sym=sym,
        order_blocks=order_blocks,
        sr_zones=sr_zones,
        chart_patterns=chart_pats,
        candlestick_patterns=candle_pats,
        swing=swing,
        timeframe=timeframe,
        entry_price=entry_price,
        oi_data=oi_data,
        market_structure=market_struct,
        vwap_data=vwap_data,
        pivot_data=pivots,
        fib_data=fibs,
    )

    return {
        "recommendation": recommendation,
        "order_blocks": order_blocks,
        "fair_value_gaps": fvgs,
        "support_resistance": sr_zones,
        "chart_patterns": chart_pats,
        "candlestick_patterns": candle_pats,
        "swing": swing,
        "oi": oi_data,
        "market_structure": market_struct,
        "vwap": vwap_data,
        "pivot_points": pivots,
        "fibonacci": fibs,
        "timeframe": timeframe,
        "timeframe_label": cfg["label"],
    }
