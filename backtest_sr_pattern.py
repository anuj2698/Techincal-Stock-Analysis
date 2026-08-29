#!/usr/bin/env python3
"""S/R Pattern Breakout backtest on daily candles across ALL F&O stocks.

Sliding-window approach:
  - For each stock, get ~2 years of daily candles
  - At each day N (stepping every 5 days), run pattern detection on candles[:N]
  - If a pattern has a breakout level near an S/R zone (within 1.5%), record a setup
  - Track forward up to 20 trading days to see if entry triggered, then target or SL hit
  - Aggregate results by pattern type to find the most accurate breakout pattern
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

sys.path.insert(0, '.')
from app import get_fno_stocks, resolve_yahoo_ticker, fetch_candles
from analyzer import detect_chart_patterns, find_support_resistance, detect_ema_crossovers, atr as calc_atr, sma, rsi as calc_rsi

IST = timezone(timedelta(hours=5, minutes=30))

CONFLUENCE_THRESHOLD_PCT = 1.5
MIN_CANDLES = 80
STEP_DAYS = 5
FORWARD_DAYS = 20
MIN_RR = 1.0


def backtest_stock(sym):
    try:
        canonical, yahoo_sym = resolve_yahoo_ticker(sym)
        candles = fetch_candles(yahoo_sym, period="2y", interval="1d", canonical=canonical)
    except Exception:
        return None

    if not candles or len(candles) < MIN_CANDLES + FORWARD_DAYS + 10:
        return None

    total = len(candles)
    trades = []
    seen_setups = set()

    for end_idx in range(MIN_CANDLES, total - FORWARD_DAYS, STEP_DAYS):
        window = candles[:end_idx]
        cmp = float(window[-1][4])

        patterns = detect_chart_patterns(window)
        if not patterns:
            continue

        sr_zones = find_support_resistance(window, window=5)

        all_levels = []
        for z in sr_zones:
            all_levels.append({
                "price": z["level"],
                "strength": z["touches"] * 2 + z["recency_score"],
            })
        sma50 = sma([float(c[4]) for c in window], 50)
        sma200 = sma([float(c[4]) for c in window], 200)
        if sma50:
            all_levels.append({"price": sma50, "strength": 3})
        if sma200:
            all_levels.append({"price": sma200, "strength": 4})

        atr_val = calc_atr(window, 14)

        for p in patterns:
            if p.get("confirmed"):
                continue

            bl_up = p.get("breakout_level_up")
            bl_down = p.get("breakout_level_down")

            for bl, direction in [(bl_up, "bullish"), (bl_down, "bearish")]:
                if bl is None:
                    continue

                setup_key = f"{sym}_{p['name']}_{direction}_{round(bl, 1)}_{end_idx // 10}"
                if setup_key in seen_setups:
                    continue

                nearby = [lv for lv in all_levels if abs(lv["price"] - bl) / bl * 100 <= CONFLUENCE_THRESHOLD_PCT]
                if not nearby:
                    continue

                confluence_count = len(nearby)
                confluence_strength = sum(lv["strength"] for lv in nearby)

                entry = bl
                if direction == "bullish":
                    sl = p.get("sl_up", round(bl - atr_val, 2))
                    target = p.get("breakout_target_up", round(bl + atr_val * 2, 2))
                else:
                    sl = p.get("sl_down", round(bl + atr_val, 2))
                    target = p.get("breakout_target_down", round(bl - atr_val * 2, 2))

                risk = abs(entry - sl) if sl else 0
                reward = abs(target - entry) if target else 0
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                if rr_ratio < MIN_RR:
                    continue

                seen_setups.add(setup_key)

                result = simulate_forward(
                    candles[end_idx:end_idx + FORWARD_DAYS],
                    entry, sl, target, direction,
                )

                if result is None:
                    continue

                exit_price, outcome = result
                if direction == "bullish":
                    pnl_pct = (exit_price - entry) / entry * 100
                else:
                    pnl_pct = (entry - exit_price) / entry * 100

                entry_date = datetime.fromtimestamp(int(candles[end_idx][0]), tz=IST).strftime("%Y-%m-%d")

                trades.append({
                    "pattern": p["name"],
                    "bias": p.get("bias", "neutral"),
                    "direction": direction,
                    "confluence_count": confluence_count,
                    "confluence_strength": confluence_strength,
                    "rr_ratio": rr_ratio,
                    "outcome": outcome,
                    "pnl_pct": round(pnl_pct, 2),
                    "date": entry_date,
                })

    if not trades:
        return None

    return {"symbol": sym, "trades": trades}


def simulate_forward(candles, entry, sl, target, direction):
    triggered = False

    for c in candles:
        h, l, cl = float(c[2]), float(c[3]), float(c[4])

        if not triggered:
            if direction == "bullish" and h >= entry:
                triggered = True
            elif direction == "bearish" and l <= entry:
                triggered = True
            if not triggered:
                continue

        if direction == "bullish":
            if l <= sl:
                return sl, "SL"
            if h >= target:
                return target, "TARGET"
        else:
            if h >= sl:
                return sl, "SL"
            if l <= target:
                return target, "TARGET"

    if triggered:
        return float(candles[-1][4]), "TIME"

    return None


def main():
    print("Fetching F&O stock list...")
    stocks = get_fno_stocks()
    print(f"Found {len(stocks)} F&O stocks. Running S/R Pattern Breakout backtest...\n")

    all_results = []
    done = 0

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(backtest_stock, s["symbol"]): s["symbol"] for s in stocks}
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r:
                all_results.append(r)
            if done % 25 == 0 or done == len(stocks):
                print(f"  [{done}/{len(stocks)}] processed, {len(all_results)} stocks with trades")

    all_trades = []
    for r in all_results:
        all_trades.extend(r["trades"])

    if not all_trades:
        print("\nNo trades found.")
        return

    # ----- Aggregate by pattern -----
    pattern_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "targets": 0, "sls": 0, "time_exits": 0, "total_pnl": 0.0, "rr_sum": 0.0})

    for t in all_trades:
        ps = pattern_stats[t["pattern"]]
        ps["trades"] += 1
        ps["total_pnl"] += t["pnl_pct"]
        ps["rr_sum"] += t["rr_ratio"]
        if t["pnl_pct"] > 0:
            ps["wins"] += 1
        if t["outcome"] == "TARGET":
            ps["targets"] += 1
        elif t["outcome"] == "SL":
            ps["sls"] += 1
        else:
            ps["time_exits"] += 1

    # ----- Aggregate by pattern + direction -----
    pattern_dir_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "targets": 0, "sls": 0, "time_exits": 0, "total_pnl": 0.0})

    for t in all_trades:
        key = f"{t['pattern']} ({t['direction'].upper()})"
        ps = pattern_dir_stats[key]
        ps["trades"] += 1
        ps["total_pnl"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            ps["wins"] += 1
        if t["outcome"] == "TARGET":
            ps["targets"] += 1
        elif t["outcome"] == "SL":
            ps["sls"] += 1
        else:
            ps["time_exits"] += 1

    # ----- Aggregate by confluence count -----
    conf_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "targets": 0, "total_pnl": 0.0})
    for t in all_trades:
        bucket = f"{t['confluence_count']}+ sources" if t["confluence_count"] >= 4 else f"{t['confluence_count']} sources"
        cs = conf_stats[bucket]
        cs["trades"] += 1
        cs["total_pnl"] += t["pnl_pct"]
        if t["pnl_pct"] > 0:
            cs["wins"] += 1
        if t["outcome"] == "TARGET":
            cs["targets"] += 1

    # ----- Print results -----
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t["pnl_pct"] > 0)
    total_targets = sum(1 for t in all_trades if t["outcome"] == "TARGET")
    total_sls = sum(1 for t in all_trades if t["outcome"] == "SL")
    total_pnl = sum(t["pnl_pct"] for t in all_trades)

    print(f"\n{'='*110}")
    print(f"S/R PATTERN BREAKOUT — ALL F&O STOCKS — DAILY CANDLE BACKTEST")
    print(f"Min RR: {MIN_RR}:1 | Confluence threshold: {CONFLUENCE_THRESHOLD_PCT}% | Forward window: {FORWARD_DAYS} days")
    print(f"{'='*110}")

    print(f"\n  OVERALL STATS")
    print(f"  {'─'*60}")
    print(f"  Stocks analyzed:    {len(all_results)}")
    print(f"  Total setups:       {total_trades}")
    print(f"  Winners:            {total_wins}")
    print(f"  Win Rate:           {round(total_wins/total_trades*100, 1)}%")
    print(f"  Target Hits:        {total_targets}")
    print(f"  SL Hits:            {total_sls}")
    print(f"  Time Exits:         {total_trades - total_targets - total_sls}")
    print(f"  Total PnL:          {total_pnl:+.2f}%")
    print(f"  Avg PnL/Trade:      {total_pnl/total_trades:+.3f}%")

    # Pattern breakdown
    sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]["wins"]/max(x[1]["trades"],1), reverse=True)

    print(f"\n{'─'*110}")
    print(f"PATTERN ACCURACY RANKING")
    print(f"{'─'*110}")
    print(f"{'#':<4} {'Pattern':<28} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'Target':>7} {'SL':>5} {'Time':>6} {'TotPnL%':>10} {'AvgPnL%':>10} {'AvgRR':>7}")
    print(f"{'─'*4} {'─'*28} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*5} {'─'*6} {'─'*10} {'─'*10} {'─'*7}")
    for i, (name, ps) in enumerate(sorted_patterns, 1):
        wr = round(ps["wins"] / ps["trades"] * 100, 1) if ps["trades"] else 0
        avg_pnl = ps["total_pnl"] / ps["trades"] if ps["trades"] else 0
        avg_rr = ps["rr_sum"] / ps["trades"] if ps["trades"] else 0
        print(f"{i:<4} {name:<28} {ps['trades']:>7} {ps['wins']:>6} {wr:>6.1f}% {ps['targets']:>7} {ps['sls']:>5} {ps['time_exits']:>6} {ps['total_pnl']:>+10.2f} {avg_pnl:>+10.3f} {avg_rr:>7.2f}")

    # Pattern + direction breakdown
    sorted_pd = sorted(pattern_dir_stats.items(), key=lambda x: x[1]["wins"]/max(x[1]["trades"],1), reverse=True)

    print(f"\n{'─'*110}")
    print(f"PATTERN + DIRECTION BREAKDOWN")
    print(f"{'─'*110}")
    print(f"{'#':<4} {'Pattern (Direction)':<38} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'Target':>7} {'SL':>5} {'Time':>6} {'TotPnL%':>10} {'AvgPnL%':>10}")
    print(f"{'─'*4} {'─'*38} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*5} {'─'*6} {'─'*10} {'─'*10}")
    for i, (name, ps) in enumerate(sorted_pd, 1):
        wr = round(ps["wins"] / ps["trades"] * 100, 1) if ps["trades"] else 0
        avg_pnl = ps["total_pnl"] / ps["trades"] if ps["trades"] else 0
        print(f"{i:<4} {name:<38} {ps['trades']:>7} {ps['wins']:>6} {wr:>6.1f}% {ps['targets']:>7} {ps['sls']:>5} {ps['time_exits']:>6} {ps['total_pnl']:>+10.2f} {avg_pnl:>+10.3f}")

    # Confluence impact
    sorted_conf = sorted(conf_stats.items())

    print(f"\n{'─'*110}")
    print(f"CONFLUENCE IMPACT ON WIN RATE")
    print(f"{'─'*110}")
    print(f"{'Confluence':<20} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'Target':>7} {'TotPnL%':>10} {'AvgPnL%':>10}")
    print(f"{'─'*20} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*10} {'─'*10}")
    for name, cs in sorted_conf:
        wr = round(cs["wins"] / cs["trades"] * 100, 1) if cs["trades"] else 0
        avg_pnl = cs["total_pnl"] / cs["trades"] if cs["trades"] else 0
        print(f"{name:<20} {cs['trades']:>7} {cs['wins']:>6} {wr:>6.1f}% {cs['targets']:>7} {cs['total_pnl']:>+10.2f} {avg_pnl:>+10.3f}")

    # Top stocks
    stock_stats = {}
    for r in all_results:
        sym = r["symbol"]
        trades = r["trades"]
        wins = sum(1 for t in trades if t["pnl_pct"] > 0)
        pnl = sum(t["pnl_pct"] for t in trades)
        stock_stats[sym] = {"trades": len(trades), "wins": wins, "win_rate": round(wins/len(trades)*100, 1), "total_pnl": round(pnl, 2)}

    sorted_stocks = sorted(stock_stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True)

    print(f"\n{'─'*110}")
    print(f"TOP 20 STOCKS BY PnL")
    print(f"{'─'*110}")
    print(f"{'#':<4} {'Symbol':<16} {'Trades':>7} {'Wins':>6} {'Win%':>7} {'TotPnL%':>10}")
    print(f"{'─'*4} {'─'*16} {'─'*7} {'─'*6} {'─'*7} {'─'*10}")
    for i, (sym, ss) in enumerate(sorted_stocks[:20], 1):
        print(f"{i:<4} {sym:<16} {ss['trades']:>7} {ss['wins']:>6} {ss['win_rate']:>6.1f}% {ss['total_pnl']:>+10.2f}")

    # Save
    output = {
        "overall": {
            "stocks": len(all_results),
            "total_trades": total_trades,
            "win_rate": round(total_wins / total_trades * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / total_trades, 3),
        },
        "pattern_ranking": [
            {
                "pattern": name,
                "trades": ps["trades"],
                "wins": ps["wins"],
                "win_rate": round(ps["wins"] / ps["trades"] * 100, 1),
                "targets": ps["targets"],
                "sls": ps["sls"],
                "total_pnl": round(ps["total_pnl"], 2),
                "avg_pnl": round(ps["total_pnl"] / ps["trades"], 3),
            }
            for name, ps in sorted_patterns
        ],
        "pattern_direction": [
            {
                "pattern": name,
                "trades": ps["trades"],
                "wins": ps["wins"],
                "win_rate": round(ps["wins"] / ps["trades"] * 100, 1),
                "total_pnl": round(ps["total_pnl"], 2),
            }
            for name, ps in sorted_pd
        ],
        "confluence_impact": [
            {
                "confluence": name,
                "trades": cs["trades"],
                "wins": cs["wins"],
                "win_rate": round(cs["wins"] / cs["trades"] * 100, 1),
                "total_pnl": round(cs["total_pnl"], 2),
            }
            for name, cs in sorted_conf
        ],
        "stock_results": sorted_stocks[:30],
    }

    out_file = "sr_pattern_backtest.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {out_file}")


if __name__ == "__main__":
    main()
