#!/usr/bin/env python3
"""Multi-strategy intraday backtest on selected F&O stocks.
Strategies: VWAP Bounce, PDH/PDL Breakout, RSI Reversal, EMA 9/21 Cross.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import yfinance as yf
from dotenv import load_dotenv

load_dotenv(override=True)

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = (9, 15)
EXIT_TIME = (15, 15)

STOCKS = [
    "KALYANKJIL", "MCX", "BHEL", "COFORGE", "PAYTM",
    "ADANIPOWER", "DIXON", "ETERNAL", "VEDL", "TATASTEEL",
    "HDFCBANK", "SAIL", "NATIONALUM", "IDEA", "KAYNES",
]


# ---------------------------------------------------------------------------
# Data fetching (reused from ORB backtest)
# ---------------------------------------------------------------------------

def fetch_5min_candles(symbol: str) -> list[list[float]]:
    app_id = os.environ.get("FYERS_APP_ID")
    token = os.environ.get("FYERS_ACCESS_TOKEN")
    if app_id and token:
        try:
            from fyers_apiv3 import fyersModel
            os.makedirs("logs", exist_ok=True)
            client = fyersModel.FyersModel(client_id=app_id, token=token, is_async=False, log_path="logs/")
            now = datetime.now(IST)
            resp = client.history(data={
                "symbol": f"NSE:{symbol}-EQ", "resolution": "5", "date_format": "1",
                "range_from": (now - timedelta(days=60)).strftime("%Y-%m-%d"),
                "range_to": now.strftime("%Y-%m-%d"), "cont_flag": "1",
            })
            time.sleep(0.15)
            candles = resp.get("candles", [])
            if candles:
                return [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in candles]
        except Exception:
            pass
    try:
        df = yf.Ticker(f"{symbol}.NS").history(period="60d", interval="5m", auto_adjust=False)
        if df is not None and not df.empty:
            rows = []
            for idx, row in df.iterrows():
                ts = int(idx.timestamp())
                rows.append([ts, float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), float(row["Volume"])])
            return sorted(rows, key=lambda r: r[0])
    except Exception:
        pass
    return []


def group_by_day(candles: list[list[float]]) -> dict[str, list[list[float]]]:
    days = defaultdict(list)
    for c in candles:
        dt = datetime.fromtimestamp(int(c[0]), tz=IST)
        if dt.weekday() < 5:
            days[dt.strftime("%Y-%m-%d")].append(c)
    return dict(sorted(days.items()))


def candle_time(c):
    return datetime.fromtimestamp(int(c[0]), tz=IST)


def minutes_from_open(c):
    dt = candle_time(c)
    return (dt.hour - MARKET_OPEN[0]) * 60 + (dt.minute - MARKET_OPEN[1])


def is_exit_time(c):
    dt = candle_time(c)
    return dt.hour > EXIT_TIME[0] or (dt.hour == EXIT_TIME[0] and dt.minute >= EXIT_TIME[1])


def resolve_trade(candles, entry_idx, entry_price, sl, target, direction):
    """Walk forward from entry to find SL/target/time exit."""
    for k in range(entry_idx, len(candles)):
        c = candles[k]
        _, o, h, l, cl, v = c
        hit_exit_time = is_exit_time(c)

        if direction == "LONG":
            if l <= sl:
                return {"exit": sl, "exit_reason": "SL", "exit_idx": k, "pnl": sl - entry_price}
            if h >= target:
                return {"exit": target, "exit_reason": "TARGET", "exit_idx": k, "pnl": target - entry_price}
            if hit_exit_time:
                return {"exit": cl, "exit_reason": "TIME_EXIT", "exit_idx": k, "pnl": cl - entry_price}
        else:
            if h >= sl:
                return {"exit": sl, "exit_reason": "SL", "exit_idx": k, "pnl": entry_price - sl}
            if l <= target:
                return {"exit": target, "exit_reason": "TARGET", "exit_idx": k, "pnl": entry_price - target}
            if hit_exit_time:
                return {"exit": cl, "exit_reason": "TIME_EXIT", "exit_idx": k, "pnl": entry_price - cl}

    last = candles[-1]
    pnl = (last[4] - entry_price) if direction == "LONG" else (entry_price - last[4])
    return {"exit": last[4], "exit_reason": "EOD", "exit_idx": len(candles) - 1, "pnl": pnl}


# ---------------------------------------------------------------------------
# Strategy 1: VWAP Bounce
# ---------------------------------------------------------------------------

def compute_vwap(candles_so_far):
    """Cumulative VWAP up to current point in the day."""
    cum_pv = 0.0
    cum_v = 0.0
    for c in candles_so_far:
        typical = (c[2] + c[3] + c[4]) / 3
        cum_pv += typical * c[5]
        cum_v += c[5]
    return cum_pv / cum_v if cum_v > 0 else None


def strategy_vwap_bounce(day_candles):
    """
    VWAP Bounce:
    - After first 30 min, establish trend (price above/below VWAP)
    - LONG: Price is above VWAP, pulls back to touch VWAP (candle low within 0.15% of VWAP),
            closes above VWAP → buy at close, SL = candle low, target = 1:1
    - SHORT: Price is below VWAP, pushes up to VWAP (candle high within 0.15% of VWAP),
             closes below VWAP → sell at close, SL = candle high, target = 1:1
    - Max 2 trades per day (1 long + 1 short)
    """
    trades = []
    long_taken = False
    short_taken = False

    for i in range(6, len(day_candles)):  # skip first 30 min (6 x 5-min candles)
        c = day_candles[i]
        if is_exit_time(c):
            break

        vwap = compute_vwap(day_candles[:i + 1])
        if vwap is None:
            continue

        _, o, h, l, cl, v = c
        proximity = abs(vwap) * 0.0015  # 0.15% band

        # Check trend: use 3-candle lookback above/below VWAP
        recent_above = sum(1 for j in range(max(0, i - 3), i) if day_candles[j][4] > compute_vwap(day_candles[:j + 1] if j > 0 else day_candles[:1]))
        recent_below = 3 - recent_above

        # LONG: uptrend, pullback to VWAP, bounce
        if not long_taken and recent_above >= 2 and l <= vwap + proximity and cl > vwap:
            risk = cl - l
            if risk > 0.001 * cl:
                target = cl + risk
                result = resolve_trade(day_candles, i + 1 if i + 1 < len(day_candles) else i, cl, l, target, "LONG")
                trades.append({
                    "direction": "LONG", "entry": cl, "sl": l, "target": target,
                    "entry_time": candle_time(c).strftime("%H:%M"),
                    "exit_time": candle_time(day_candles[result["exit_idx"]]).strftime("%H:%M"),
                    **result, "pnl_pct": round(result["pnl"] / cl * 100, 2),
                })
                long_taken = True

        # SHORT: downtrend, push up to VWAP, reject
        if not short_taken and recent_below >= 2 and h >= vwap - proximity and cl < vwap:
            risk = h - cl
            if risk > 0.001 * cl:
                target = cl - risk
                result = resolve_trade(day_candles, i + 1 if i + 1 < len(day_candles) else i, cl, h, target, "SHORT")
                trades.append({
                    "direction": "SHORT", "entry": cl, "sl": h, "target": target,
                    "entry_time": candle_time(c).strftime("%H:%M"),
                    "exit_time": candle_time(day_candles[result["exit_idx"]]).strftime("%H:%M"),
                    **result, "pnl_pct": round(result["pnl"] / cl * 100, 2),
                })
                short_taken = True

        if long_taken and short_taken:
            break

    return trades


# ---------------------------------------------------------------------------
# Strategy 2: Previous Day High/Low Breakout
# ---------------------------------------------------------------------------

def strategy_pdhl_breakout(day_candles, prev_day_candles):
    """
    PDH/PDL Breakout:
    - PDH = previous day's high, PDL = previous day's low
    - Wait first 15 min, then:
    - LONG: price breaks above PDH → entry at PDH, SL = PDH - (PDH-PDL)*0.3, target = 1:1
    - SHORT: price breaks below PDL → entry at PDL, SL = PDL + (PDH-PDL)*0.3, target = 1:1
    - Only first breakout in each direction
    """
    if not prev_day_candles:
        return []

    pdh = max(c[2] for c in prev_day_candles)
    pdl = min(c[3] for c in prev_day_candles)
    pd_range = pdh - pdl
    if pd_range <= 0:
        return []

    trades = []
    long_taken = False
    short_taken = False
    buffer = pdh * 0.001  # 0.1% buffer

    for i, c in enumerate(day_candles):
        if minutes_from_open(c) < 15:  # skip first 15 min
            continue
        if is_exit_time(c):
            break

        _, o, h, l, cl, v = c

        # LONG breakout above PDH
        if not long_taken and h >= pdh + buffer:
            entry = pdh + buffer
            sl = pdh - pd_range * 0.3
            risk = entry - sl
            target = entry + risk
            result = resolve_trade(day_candles, i, entry, sl, target, "LONG")
            trades.append({
                "direction": "LONG", "entry": round(entry, 2), "sl": round(sl, 2),
                "target": round(target, 2),
                "entry_time": candle_time(c).strftime("%H:%M"),
                "exit_time": candle_time(day_candles[result["exit_idx"]]).strftime("%H:%M"),
                **result, "pnl_pct": round(result["pnl"] / entry * 100, 2),
            })
            long_taken = True

        # SHORT breakout below PDL
        if not short_taken and l <= pdl - buffer:
            entry = pdl - buffer
            sl = pdl + pd_range * 0.3
            risk = sl - entry
            target = entry - risk
            result = resolve_trade(day_candles, i, entry, sl, target, "SHORT")
            trades.append({
                "direction": "SHORT", "entry": round(entry, 2), "sl": round(sl, 2),
                "target": round(target, 2),
                "entry_time": candle_time(c).strftime("%H:%M"),
                "exit_time": candle_time(day_candles[result["exit_idx"]]).strftime("%H:%M"),
                **result, "pnl_pct": round(result["pnl"] / entry * 100, 2),
            })
            short_taken = True

        if long_taken and short_taken:
            break

    return trades


# ---------------------------------------------------------------------------
# Strategy 3: RSI Extreme Reversal (14-period RSI on 5-min)
# ---------------------------------------------------------------------------

def compute_rsi_series(candles, period=14):
    """Return RSI value for each candle (None for first `period` candles)."""
    closes = [c[4] for c in candles]
    if len(closes) < period + 1:
        return [None] * len(closes)

    rsi_vals = [None] * period
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        rsi_vals.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_vals.append(100 - 100 / (1 + rs))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_vals.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_vals.append(100 - 100 / (1 + rs))

    return rsi_vals


def strategy_rsi_reversal(day_candles):
    """
    RSI Extreme Reversal:
    - Compute 14-period RSI on 5-min candles
    - LONG: RSI was below 30, crosses back above 30 → buy at candle close,
            SL = low of the lowest RSI candle in the oversold zone, target = 1:1
    - SHORT: RSI was above 70, crosses back below 70 → sell at candle close,
             SL = high of the highest RSI candle in the overbought zone, target = 1:1
    - Skip first 15 min, max 2 trades per direction per day
    """
    rsi_vals = compute_rsi_series(day_candles)
    trades = []
    long_count = 0
    short_count = 0

    in_oversold = False
    oversold_low = float("inf")
    in_overbought = False
    overbought_high = 0.0

    for i in range(1, len(day_candles)):
        if minutes_from_open(day_candles[i]) < 15:
            continue
        if is_exit_time(day_candles[i]):
            break

        rsi = rsi_vals[i]
        prev_rsi = rsi_vals[i - 1]
        if rsi is None or prev_rsi is None:
            continue

        c = day_candles[i]
        _, o, h, l, cl, v = c

        # Track oversold zone
        if rsi < 30:
            in_oversold = True
            oversold_low = min(oversold_low, l)
        elif in_oversold and rsi >= 30 and prev_rsi < 30 and long_count < 2:
            # RSI crossed back above 30 — reversal signal
            sl = oversold_low
            risk = cl - sl
            if risk > 0.001 * cl:
                target = cl + risk
                result = resolve_trade(day_candles, i + 1 if i + 1 < len(day_candles) else i, cl, sl, target, "LONG")
                trades.append({
                    "direction": "LONG", "entry": cl, "sl": sl, "target": target,
                    "entry_time": candle_time(c).strftime("%H:%M"),
                    "exit_time": candle_time(day_candles[result["exit_idx"]]).strftime("%H:%M"),
                    "rsi_at_entry": round(rsi, 1),
                    **result, "pnl_pct": round(result["pnl"] / cl * 100, 2),
                })
                long_count += 1
            in_oversold = False
            oversold_low = float("inf")

        if rsi >= 30:
            in_oversold = False
            oversold_low = float("inf")

        # Track overbought zone
        if rsi > 70:
            in_overbought = True
            overbought_high = max(overbought_high, h)
        elif in_overbought and rsi <= 70 and prev_rsi > 70 and short_count < 2:
            sl = overbought_high
            risk = sl - cl
            if risk > 0.001 * cl:
                target = cl - risk
                result = resolve_trade(day_candles, i + 1 if i + 1 < len(day_candles) else i, cl, sl, target, "SHORT")
                trades.append({
                    "direction": "SHORT", "entry": cl, "sl": sl, "target": target,
                    "entry_time": candle_time(c).strftime("%H:%M"),
                    "exit_time": candle_time(day_candles[result["exit_idx"]]).strftime("%H:%M"),
                    "rsi_at_entry": round(rsi, 1),
                    **result, "pnl_pct": round(result["pnl"] / cl * 100, 2),
                })
                short_count += 1
            in_overbought = False
            overbought_high = 0.0

        if rsi <= 70:
            in_overbought = False
            overbought_high = 0.0

    return trades


# ---------------------------------------------------------------------------
# Strategy 4: EMA 9/21 Crossover
# ---------------------------------------------------------------------------

def compute_ema_series(values, period):
    if len(values) < period:
        return [None] * len(values)
    sma = sum(values[:period]) / period
    mult = 2 / (period + 1)
    result = [None] * (period - 1) + [sma]
    for v in values[period:]:
        result.append(v * mult + result[-1] * (1 - mult))
    return result


def strategy_ema_cross(day_candles):
    """
    EMA 9/21 Crossover:
    - Compute 9 and 21 EMA on 5-min closes
    - LONG: 9 EMA crosses above 21 EMA → buy at candle close,
            SL = low of last 3 candles, target = 1:1
    - SHORT: 9 EMA crosses below 21 EMA → sell at candle close,
             SL = high of last 3 candles, target = 1:1
    - Skip first 30 min, max 2 trades per direction
    """
    closes = [c[4] for c in day_candles]
    ema9 = compute_ema_series(closes, 9)
    ema21 = compute_ema_series(closes, 21)

    trades = []
    long_count = 0
    short_count = 0

    for i in range(1, len(day_candles)):
        if minutes_from_open(day_candles[i]) < 30:
            continue
        if is_exit_time(day_candles[i]):
            break
        if ema9[i] is None or ema21[i] is None or ema9[i - 1] is None or ema21[i - 1] is None:
            continue

        c = day_candles[i]
        _, o, h, l, cl, v = c

        # Bullish cross: 9 EMA was below 21, now above
        if ema9[i - 1] <= ema21[i - 1] and ema9[i] > ema21[i] and long_count < 2:
            recent_low = min(day_candles[j][3] for j in range(max(0, i - 3), i + 1))
            sl = recent_low
            risk = cl - sl
            if risk > 0.001 * cl:
                target = cl + risk
                result = resolve_trade(day_candles, i + 1 if i + 1 < len(day_candles) else i, cl, sl, target, "LONG")
                trades.append({
                    "direction": "LONG", "entry": cl, "sl": sl, "target": target,
                    "entry_time": candle_time(c).strftime("%H:%M"),
                    "exit_time": candle_time(day_candles[result["exit_idx"]]).strftime("%H:%M"),
                    **result, "pnl_pct": round(result["pnl"] / cl * 100, 2),
                })
                long_count += 1

        # Bearish cross: 9 EMA was above 21, now below
        if ema9[i - 1] >= ema21[i - 1] and ema9[i] < ema21[i] and short_count < 2:
            recent_high = max(day_candles[j][2] for j in range(max(0, i - 3), i + 1))
            sl = recent_high
            risk = sl - cl
            if risk > 0.001 * cl:
                target = cl - risk
                result = resolve_trade(day_candles, i + 1 if i + 1 < len(day_candles) else i, cl, sl, target, "SHORT")
                trades.append({
                    "direction": "SHORT", "entry": cl, "sl": sl, "target": target,
                    "entry_time": candle_time(c).strftime("%H:%M"),
                    "exit_time": candle_time(day_candles[result["exit_idx"]]).strftime("%H:%M"),
                    **result, "pnl_pct": round(result["pnl"] / cl * 100, 2),
                })
                short_count += 1

    return trades


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

STRATEGIES = {
    "VWAP Bounce": strategy_vwap_bounce,
    "PDH/PDL Breakout": strategy_pdhl_breakout,
    "RSI Reversal": strategy_rsi_reversal,
    "EMA 9/21 Cross": strategy_ema_cross,
}


def backtest_stock(symbol: str, all_day_candles: dict[str, list]) -> dict:
    dates = sorted(all_day_candles.keys())
    results = {}

    for strat_name, strat_fn in STRATEGIES.items():
        all_trades = []
        days_traded = 0

        for di, date_str in enumerate(dates):
            dc = all_day_candles[date_str]
            if len(dc) < 20:
                continue

            if strat_name == "PDH/PDL Breakout":
                if di == 0:
                    continue
                prev_dc = all_day_candles[dates[di - 1]]
                trades = strat_fn(dc, prev_dc)
            else:
                trades = strat_fn(dc)

            if trades:
                for t in trades:
                    t["date"] = date_str
                all_trades.extend(trades)
                days_traded += 1

        if not all_trades:
            results[strat_name] = {"trades": 0, "days_traded": 0}
            continue

        winners = [t for t in all_trades if t["pnl_pct"] > 0]
        losers = [t for t in all_trades if t["pnl_pct"] < 0]
        longs = [t for t in all_trades if t["direction"] == "LONG"]
        shorts = [t for t in all_trades if t["direction"] == "SHORT"]
        long_wins = [t for t in longs if t["pnl_pct"] > 0]
        short_wins = [t for t in shorts if t["pnl_pct"] > 0]

        total_pnl = sum(t["pnl_pct"] for t in all_trades)
        by_reason = defaultdict(int)
        for t in all_trades:
            by_reason[t["exit_reason"]] += 1

        results[strat_name] = {
            "trades": len(all_trades),
            "days_traded": days_traded,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / len(all_trades) * 100, 1),
            "total_pnl_pct": round(total_pnl, 2),
            "avg_pnl_pct": round(total_pnl / len(all_trades), 2),
            "avg_win_pct": round(sum(t["pnl_pct"] for t in winners) / len(winners), 2) if winners else 0,
            "avg_loss_pct": round(sum(t["pnl_pct"] for t in losers) / len(losers), 2) if losers else 0,
            "best_pct": round(max(t["pnl_pct"] for t in all_trades), 2),
            "worst_pct": round(min(t["pnl_pct"] for t in all_trades), 2),
            "long_trades": len(longs),
            "long_wins": len(long_wins),
            "long_wr": round(len(long_wins) / len(longs) * 100, 1) if longs else 0,
            "long_pnl": round(sum(t["pnl_pct"] for t in longs), 2),
            "short_trades": len(shorts),
            "short_wins": len(short_wins),
            "short_wr": round(len(short_wins) / len(shorts) * 100, 1) if shorts else 0,
            "short_pnl": round(sum(t["pnl_pct"] for t in shorts), 2),
            "by_reason": dict(by_reason),
            "all_trades": all_trades,
        }

    return results


def main():
    print(f"Multi-Strategy Intraday Backtest — {len(STOCKS)} stocks")
    print(f"Strategies: {', '.join(STRATEGIES.keys())}\n")

    stock_data = {}
    for i, sym in enumerate(STOCKS, 1):
        print(f"[{i}/{len(STOCKS)}] Fetching {sym}...")
        candles = fetch_5min_candles(sym)
        if candles:
            stock_data[sym] = group_by_day(candles)
        else:
            print(f"  ⚠ No data for {sym}")

    print(f"\nRunning backtests...")
    all_results = {}
    for sym, days in stock_data.items():
        all_results[sym] = backtest_stock(sym, days)

    # ======================================================================
    # PRINT RESULTS — per strategy, all stocks
    # ======================================================================
    for strat_name in STRATEGIES:
        print(f"\n{'━'*130}")
        print(f"  STRATEGY: {strat_name.upper()}")
        print(f"{'━'*130}")
        print(f"{'Symbol':<14} {'Days':>5} {'Trades':>7} {'Win%':>6} {'L-Win%':>7} {'S-Win%':>7} {'Tgt':>5} {'SL':>5} {'Time':>5} {'TotPnL%':>9} {'AvgPnL%':>8} {'AvgW%':>7} {'AvgL%':>7} {'Best%':>7} {'Worst%':>7}")
        print(f"{'─'*14} {'─'*5} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*5} {'─'*9} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

        totals = {"trades": 0, "wins": 0, "pnl": 0.0}

        sorted_syms = sorted(all_results.keys(),
                             key=lambda s: all_results[s].get(strat_name, {}).get("total_pnl_pct", -999),
                             reverse=True)

        for sym in sorted_syms:
            d = all_results[sym].get(strat_name, {})
            if d.get("trades", 0) == 0:
                print(f"{sym:<14} {'':>5} {'No trades':>7}")
                continue

            tgt = d["by_reason"].get("TARGET", 0)
            sl = d["by_reason"].get("SL", 0)
            te = d["by_reason"].get("TIME_EXIT", 0) + d["by_reason"].get("EOD", 0)

            print(f"{sym:<14} {d['days_traded']:>5} {d['trades']:>7} {d['win_rate']:>5.1f}% {d['long_wr']:>6.1f}% {d['short_wr']:>6.1f}% {tgt:>5} {sl:>5} {te:>5} {d['total_pnl_pct']:>+9.2f} {d['avg_pnl_pct']:>+8.2f} {d['avg_win_pct']:>+7.2f} {d['avg_loss_pct']:>+7.2f} {d['best_pct']:>+7.2f} {d['worst_pct']:>+7.2f}")
            totals["trades"] += d["trades"]
            totals["wins"] += d["winners"]
            totals["pnl"] += d["total_pnl_pct"]

        if totals["trades"] > 0:
            wr = round(totals["wins"] / totals["trades"] * 100, 1)
            avg = round(totals["pnl"] / totals["trades"], 2)
            print(f"{'─'*14} {'─'*5} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*5} {'─'*9} {'─'*8}")
            print(f"{'TOTAL':<14} {'':>5} {totals['trades']:>7} {wr:>5.1f}% {'':>7} {'':>7} {'':>5} {'':>5} {'':>5} {totals['pnl']:>+9.2f} {avg:>+8.2f}")

    # ======================================================================
    # GRAND COMPARISON TABLE — all strategies side by side
    # ======================================================================
    print(f"\n{'='*130}")
    print("GRAND COMPARISON — ALL STRATEGIES (per stock, Total PnL %)")
    print(f"{'='*130}")

    strat_names = list(STRATEGIES.keys())
    header = f"{'Symbol':<14}"
    for s in strat_names:
        header += f" {s:>18}"
    print(header)
    print(f"{'─'*14}" + f" {'─'*18}" * len(strat_names))

    for sym in sorted(all_results.keys()):
        row = f"{sym:<14}"
        for s in strat_names:
            d = all_results[sym].get(s, {})
            if d.get("trades", 0) == 0:
                row += f" {'N/A':>18}"
            else:
                val = f"{d['total_pnl_pct']:+.2f}% ({d['win_rate']:.0f}%W)"
                row += f" {val:>18}"
        print(row)

    # Strategy totals
    print(f"{'─'*14}" + f" {'─'*18}" * len(strat_names))
    row = f"{'TOTAL':<14}"
    for s in strat_names:
        tot = sum(all_results[sym].get(s, {}).get("total_pnl_pct", 0) for sym in all_results)
        cnt = sum(all_results[sym].get(s, {}).get("trades", 0) for sym in all_results)
        wins = sum(all_results[sym].get(s, {}).get("winners", 0) for sym in all_results)
        wr = round(wins / cnt * 100, 0) if cnt else 0
        val = f"{tot:+.1f}% ({wr:.0f}%W)"
        row += f" {val:>18}"
    print(row)

    row = f"{'TRADES':<14}"
    for s in strat_names:
        cnt = sum(all_results[sym].get(s, {}).get("trades", 0) for sym in all_results)
        row += f" {cnt:>18}"
    print(row)

    row = f"{'AVG/TRADE':<14}"
    for s in strat_names:
        tot = sum(all_results[sym].get(s, {}).get("total_pnl_pct", 0) for sym in all_results)
        cnt = sum(all_results[sym].get(s, {}).get("trades", 0) for sym in all_results)
        avg = round(tot / cnt, 3) if cnt else 0
        row += f" {avg:>+18.3f}"
    print(row)

    # Save
    out = "multi_strategy_results.json"
    export = {}
    for sym in all_results:
        export[sym] = {}
        for s in strat_names:
            d = all_results[sym].get(s, {})
            export[sym][s] = {k: v for k, v in d.items() if k != "all_trades"}
    with open(out, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
