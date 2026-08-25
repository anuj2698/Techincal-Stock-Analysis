#!/usr/bin/env python3
"""Backtest Opening Range Breakout (ORB) strategy on 5-minute candles for selected F&O stocks."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

IST = timezone(timedelta(hours=5, minutes=30))

# Stocks selected from 6-month analysis: high volume + big intraday range
ORB_STOCKS = [
    "KALYANKJIL",
    "MCX",
    "BHEL",
    "COFORGE",
    "PAYTM",
    "ADANIPOWER",
    "DIXON",
    "ETERNAL",
    "VEDL",
    "TATASTEEL",
    "HDFCBANK",
    "SAIL",
    "NATIONALUM",
    "IDEA",
    "KAYNES",
]

# ORB parameters
ORB_MINUTES = 15          # opening range = first 15 min (9:15-9:30)
ORB_BUFFER_PCT = 0.1      # 0.1% buffer above/below OR to confirm breakout
EXIT_TIME = (15, 15)      # force exit at 3:15 PM
MARKET_OPEN = (9, 15)
RR_TARGETS = [1.0, 1.5, 2.0]  # risk-reward ratios to evaluate


def fetch_5min_candles(symbol: str) -> list[list[float]]:
    """Fetch 5-min candles — Fyers first, then Yahoo Finance fallback."""
    app_id = os.environ.get("FYERS_APP_ID")
    token = os.environ.get("FYERS_ACCESS_TOKEN")

    if app_id and token:
        try:
            from fyers_apiv3 import fyersModel
            os.makedirs("logs", exist_ok=True)
            client = fyersModel.FyersModel(client_id=app_id, token=token, is_async=False, log_path="logs/")
            now = datetime.now(IST)
            resp = client.history(data={
                "symbol": f"NSE:{symbol}-EQ",
                "resolution": "5",
                "date_format": "1",
                "range_from": (now - timedelta(days=60)).strftime("%Y-%m-%d"),
                "range_to": now.strftime("%Y-%m-%d"),
                "cont_flag": "1",
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


def get_opening_range(day_candles: list[list[float]]) -> dict | None:
    """Get the high/low of the first ORB_MINUTES minutes after market open."""
    or_candles = []
    for c in day_candles:
        dt = datetime.fromtimestamp(int(c[0]), tz=IST)
        candle_min = (dt.hour - MARKET_OPEN[0]) * 60 + (dt.minute - MARKET_OPEN[1])
        if 0 <= candle_min < ORB_MINUTES:
            or_candles.append(c)

    if not or_candles:
        return None

    or_high = max(c[2] for c in or_candles)
    or_low = min(c[3] for c in or_candles)
    or_open = or_candles[0][1]
    or_range = or_high - or_low
    or_range_pct = (or_range / or_open * 100) if or_open > 0 else 0

    return {
        "high": or_high,
        "low": or_low,
        "open": or_open,
        "range": or_range,
        "range_pct": round(or_range_pct, 2),
        "candle_count": len(or_candles),
    }


def simulate_orb_day(day_candles: list[list[float]], or_data: dict, rr: float) -> dict | None:
    """Simulate ORB trades for one day at a given R:R target."""
    or_high = or_data["high"]
    or_low = or_data["low"]
    or_range = or_data["range"]

    if or_range <= 0:
        return None

    buffer = or_high * ORB_BUFFER_PCT / 100
    long_entry = or_high + buffer
    short_entry = or_low - buffer
    long_sl = or_low
    short_sl = or_high
    long_target = long_entry + or_range * rr
    short_target = short_entry - or_range * rr

    long_trade = None
    short_trade = None

    post_or_candles = []
    for c in day_candles:
        dt = datetime.fromtimestamp(int(c[0]), tz=IST)
        candle_min = (dt.hour - MARKET_OPEN[0]) * 60 + (dt.minute - MARKET_OPEN[1])
        if candle_min >= ORB_MINUTES:
            post_or_candles.append(c)

    for c in post_or_candles:
        dt = datetime.fromtimestamp(int(c[0]), tz=IST)
        _, o, h, l, cl, v = c

        is_exit_time = (dt.hour > EXIT_TIME[0]) or (dt.hour == EXIT_TIME[0] and dt.minute >= EXIT_TIME[1])

        # --- LONG TRADE ---
        if long_trade is None and not is_exit_time:
            if h >= long_entry:
                long_trade = {
                    "direction": "LONG",
                    "entry": long_entry,
                    "sl": long_sl,
                    "target": long_target,
                    "entry_time": dt.strftime("%H:%M"),
                    "risk": long_entry - long_sl,
                }

        if long_trade and long_trade.get("result") is None:
            if l <= long_trade["sl"]:
                long_trade["result"] = "SL"
                long_trade["exit"] = long_trade["sl"]
                long_trade["exit_time"] = dt.strftime("%H:%M")
                long_trade["pnl"] = long_trade["sl"] - long_trade["entry"]
                long_trade["pnl_pct"] = round(long_trade["pnl"] / long_trade["entry"] * 100, 2)
            elif h >= long_trade["target"]:
                long_trade["result"] = "TARGET"
                long_trade["exit"] = long_trade["target"]
                long_trade["exit_time"] = dt.strftime("%H:%M")
                long_trade["pnl"] = long_trade["target"] - long_trade["entry"]
                long_trade["pnl_pct"] = round(long_trade["pnl"] / long_trade["entry"] * 100, 2)
            elif is_exit_time:
                long_trade["result"] = "TIME_EXIT"
                long_trade["exit"] = cl
                long_trade["exit_time"] = dt.strftime("%H:%M")
                long_trade["pnl"] = cl - long_trade["entry"]
                long_trade["pnl_pct"] = round(long_trade["pnl"] / long_trade["entry"] * 100, 2)

        # --- SHORT TRADE ---
        if short_trade is None and not is_exit_time:
            if l <= short_entry:
                short_trade = {
                    "direction": "SHORT",
                    "entry": short_entry,
                    "sl": short_sl,
                    "target": short_target,
                    "entry_time": dt.strftime("%H:%M"),
                    "risk": short_sl - short_entry,
                }

        if short_trade and short_trade.get("result") is None:
            if h >= short_trade["sl"]:
                short_trade["result"] = "SL"
                short_trade["exit"] = short_trade["sl"]
                short_trade["exit_time"] = dt.strftime("%H:%M")
                short_trade["pnl"] = short_trade["entry"] - short_trade["sl"]
                short_trade["pnl_pct"] = round(short_trade["pnl"] / short_trade["entry"] * 100, 2)
            elif l <= short_trade["target"]:
                short_trade["result"] = "TARGET"
                short_trade["exit"] = short_trade["target"]
                short_trade["exit_time"] = dt.strftime("%H:%M")
                short_trade["pnl"] = short_trade["entry"] - short_trade["target"]
                short_trade["pnl_pct"] = round(short_trade["pnl"] / short_trade["entry"] * 100, 2)
            elif is_exit_time:
                short_trade["result"] = "TIME_EXIT"
                short_trade["exit"] = cl
                short_trade["exit_time"] = dt.strftime("%H:%M")
                short_trade["pnl"] = short_trade["entry"] - cl
                short_trade["pnl_pct"] = round(short_trade["pnl"] / short_trade["entry"] * 100, 2)

    # Close any open trades at EOD
    if long_trade and long_trade.get("result") is None and post_or_candles:
        last = post_or_candles[-1]
        long_trade["result"] = "EOD_EXIT"
        long_trade["exit"] = last[4]
        long_trade["exit_time"] = datetime.fromtimestamp(int(last[0]), tz=IST).strftime("%H:%M")
        long_trade["pnl"] = last[4] - long_trade["entry"]
        long_trade["pnl_pct"] = round(long_trade["pnl"] / long_trade["entry"] * 100, 2)

    if short_trade and short_trade.get("result") is None and post_or_candles:
        last = post_or_candles[-1]
        short_trade["result"] = "EOD_EXIT"
        short_trade["exit"] = last[4]
        short_trade["exit_time"] = datetime.fromtimestamp(int(last[0]), tz=IST).strftime("%H:%M")
        short_trade["pnl"] = short_trade["entry"] - last[4]
        short_trade["pnl_pct"] = round(short_trade["pnl"] / short_trade["entry"] * 100, 2)

    trades = []
    if long_trade and long_trade.get("result"):
        trades.append(long_trade)
    if short_trade and short_trade.get("result"):
        trades.append(short_trade)

    return trades if trades else None


def backtest_stock(symbol: str) -> dict:
    """Run full ORB backtest for a single stock."""
    print(f"  Fetching 5m data for {symbol}...")
    candles = fetch_5min_candles(symbol)
    if not candles:
        return {"symbol": symbol, "error": "No data", "days": 0}

    days = group_by_day(candles)
    results_by_rr = {}

    for rr in RR_TARGETS:
        all_trades = []
        days_traded = 0
        days_no_trade = 0

        for date_str, day_candles in days.items():
            if len(day_candles) < 10:
                continue

            or_data = get_opening_range(day_candles)
            if not or_data or or_data["range"] <= 0:
                days_no_trade += 1
                continue

            trades = simulate_orb_day(day_candles, or_data, rr)
            if trades:
                for t in trades:
                    t["date"] = date_str
                    t["or_range_pct"] = or_data["range_pct"]
                all_trades.extend(trades)
                days_traded += 1
            else:
                days_no_trade += 1

        if not all_trades:
            results_by_rr[rr] = {"trades": 0}
            continue

        winners = [t for t in all_trades if t["pnl"] > 0]
        losers = [t for t in all_trades if t["pnl"] < 0]
        breakeven = [t for t in all_trades if t["pnl"] == 0]

        long_trades = [t for t in all_trades if t["direction"] == "LONG"]
        short_trades = [t for t in all_trades if t["direction"] == "SHORT"]
        long_wins = [t for t in long_trades if t["pnl"] > 0]
        short_wins = [t for t in short_trades if t["pnl"] > 0]

        target_hits = [t for t in all_trades if t["result"] == "TARGET"]
        sl_hits = [t for t in all_trades if t["result"] == "SL"]
        time_exits = [t for t in all_trades if t["result"] in ("TIME_EXIT", "EOD_EXIT")]

        total_pnl_pct = sum(t["pnl_pct"] for t in all_trades)
        avg_winner_pct = sum(t["pnl_pct"] for t in winners) / len(winners) if winners else 0
        avg_loser_pct = sum(t["pnl_pct"] for t in losers) / len(losers) if losers else 0

        results_by_rr[rr] = {
            "trades": len(all_trades),
            "days_traded": days_traded,
            "days_no_trade": days_no_trade,
            "winners": len(winners),
            "losers": len(losers),
            "breakeven": len(breakeven),
            "win_rate": round(len(winners) / len(all_trades) * 100, 1),
            "long_trades": len(long_trades),
            "long_wins": len(long_wins),
            "long_win_rate": round(len(long_wins) / len(long_trades) * 100, 1) if long_trades else 0,
            "short_trades": len(short_trades),
            "short_wins": len(short_wins),
            "short_win_rate": round(len(short_wins) / len(short_trades) * 100, 1) if short_trades else 0,
            "target_hits": len(target_hits),
            "sl_hits": len(sl_hits),
            "time_exits": len(time_exits),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "avg_pnl_pct": round(total_pnl_pct / len(all_trades), 2),
            "avg_winner_pct": round(avg_winner_pct, 2),
            "avg_loser_pct": round(avg_loser_pct, 2),
            "best_trade_pct": round(max(t["pnl_pct"] for t in all_trades), 2),
            "worst_trade_pct": round(min(t["pnl_pct"] for t in all_trades), 2),
            "all_trades": all_trades,
        }

    return {
        "symbol": symbol,
        "total_days": len(days),
        "data_from": min(days.keys()) if days else "N/A",
        "data_to": max(days.keys()) if days else "N/A",
        "results_by_rr": results_by_rr,
    }


def print_results(all_results: list[dict]):
    print(f"\n{'='*120}")
    print(f"OPENING RANGE BREAKOUT (ORB) BACKTEST — {ORB_MINUTES}-MIN OPENING RANGE")
    print(f"Buffer: {ORB_BUFFER_PCT}% | Exit by: {EXIT_TIME[0]}:{EXIT_TIME[1]:02d} | First trade after: 9:{15+ORB_MINUTES:02d}")
    print(f"{'='*120}")

    for rr in RR_TARGETS:
        print(f"\n{'━'*120}")
        print(f"  RISK-REWARD TARGET: 1:{rr}")
        print(f"{'━'*120}")
        print(f"{'Symbol':<14} {'Days':>5} {'Trades':>7} {'Win%':>6} {'L-Win%':>7} {'S-Win%':>7} {'Tgt':>5} {'SL':>5} {'Time':>5} {'TotPnL%':>9} {'AvgPnL%':>8} {'AvgW%':>7} {'AvgL%':>7} {'Best%':>7} {'Worst%':>7}")
        print(f"{'─'*14} {'─'*5} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*5} {'─'*9} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")

        totals = {"trades": 0, "winners": 0, "losers": 0, "pnl": 0.0}
        for r in sorted(all_results, key=lambda x: x.get("results_by_rr", {}).get(rr, {}).get("total_pnl_pct", 0), reverse=True):
            rr_data = r.get("results_by_rr", {}).get(rr, {})
            if not rr_data or rr_data.get("trades", 0) == 0:
                print(f"{r['symbol']:<14} {'—':>5} {'No trades':>7}")
                continue
            d = rr_data
            print(f"{r['symbol']:<14} {r['total_days']:>5} {d['trades']:>7} {d['win_rate']:>5.1f}% {d['long_win_rate']:>6.1f}% {d['short_win_rate']:>6.1f}% {d['target_hits']:>5} {d['sl_hits']:>5} {d['time_exits']:>5} {d['total_pnl_pct']:>+9.2f} {d['avg_pnl_pct']:>+8.2f} {d['avg_winner_pct']:>+7.2f} {d['avg_loser_pct']:>+7.2f} {d['best_trade_pct']:>+7.2f} {d['worst_trade_pct']:>+7.2f}")
            totals["trades"] += d["trades"]
            totals["winners"] += d["winners"]
            totals["losers"] += d["losers"]
            totals["pnl"] += d["total_pnl_pct"]

        if totals["trades"] > 0:
            print(f"{'─'*14} {'─'*5} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*5} {'─'*5} {'─'*5} {'─'*9} {'─'*8}")
            wr = round(totals["winners"] / totals["trades"] * 100, 1)
            avg = round(totals["pnl"] / totals["trades"], 2)
            print(f"{'TOTAL':<14} {'':>5} {totals['trades']:>7} {wr:>5.1f}% {'':>7} {'':>7} {'':>5} {'':>5} {'':>5} {totals['pnl']:>+9.2f} {avg:>+8.2f}")

    # --- Per-stock summary across all RR ---
    print(f"\n{'='*120}")
    print("PER-STOCK SUMMARY (best R:R for each stock)")
    print(f"{'='*120}")
    print(f"{'Symbol':<14} {'Data Range':<25} {'Best RR':>8} {'Trades':>7} {'Win%':>6} {'TotPnL%':>9} {'AvgPnL%':>8}")
    print(f"{'─'*14} {'─'*25} {'─'*8} {'─'*7} {'─'*6} {'─'*9} {'─'*8}")

    for r in all_results:
        best_rr = None
        best_pnl = float("-inf")
        for rr in RR_TARGETS:
            rr_data = r.get("results_by_rr", {}).get(rr, {})
            if rr_data.get("trades", 0) > 0 and rr_data["total_pnl_pct"] > best_pnl:
                best_pnl = rr_data["total_pnl_pct"]
                best_rr = rr

        if best_rr is None:
            print(f"{r['symbol']:<14} {r.get('data_from','?')} to {r.get('data_to','?'):<10} {'N/A':>8}")
            continue

        d = r["results_by_rr"][best_rr]
        date_range = f"{r['data_from']} to {r['data_to']}"
        print(f"{r['symbol']:<14} {date_range:<25} {f'1:{best_rr}':>8} {d['trades']:>7} {d['win_rate']:>5.1f}% {d['total_pnl_pct']:>+9.2f} {d['avg_pnl_pct']:>+8.2f}")

    # --- Long vs Short bias ---
    print(f"\n{'='*120}")
    print("LONG vs SHORT BIAS (at 1:1 R:R)")
    print(f"{'='*120}")
    print(f"{'Symbol':<14} {'L-Trades':>9} {'L-Win%':>7} {'L-PnL%':>9} {'S-Trades':>9} {'S-Win%':>7} {'S-PnL%':>9} {'Bias':<10}")
    print(f"{'─'*14} {'─'*9} {'─'*7} {'─'*9} {'─'*9} {'─'*7} {'─'*9} {'─'*10}")

    for r in sorted(all_results, key=lambda x: x.get("results_by_rr", {}).get(1.0, {}).get("total_pnl_pct", 0), reverse=True):
        rr_data = r.get("results_by_rr", {}).get(1.0, {})
        if not rr_data or rr_data.get("trades", 0) == 0:
            continue
        d = rr_data
        long_pnl = sum(t["pnl_pct"] for t in d["all_trades"] if t["direction"] == "LONG")
        short_pnl = sum(t["pnl_pct"] for t in d["all_trades"] if t["direction"] == "SHORT")
        bias = "LONG" if long_pnl > short_pnl else "SHORT" if short_pnl > long_pnl else "NEUTRAL"
        print(f"{r['symbol']:<14} {d['long_trades']:>9} {d['long_win_rate']:>6.1f}% {long_pnl:>+9.2f} {d['short_trades']:>9} {d['short_win_rate']:>6.1f}% {short_pnl:>+9.2f} {bias:<10}")


def main():
    print(f"ORB Backtest — {ORB_MINUTES}-min Opening Range | {len(ORB_STOCKS)} stocks")
    print(f"R:R Targets: {', '.join(f'1:{rr}' for rr in RR_TARGETS)}\n")

    all_results = []
    for i, sym in enumerate(ORB_STOCKS, 1):
        print(f"[{i}/{len(ORB_STOCKS)}] {sym}")
        result = backtest_stock(sym)
        all_results.append(result)

    print_results(all_results)

    out = "orb_backtest_results.json"
    export = []
    for r in all_results:
        e = {k: v for k, v in r.items() if k != "results_by_rr"}
        e["results_by_rr"] = {}
        for rr, data in r.get("results_by_rr", {}).items():
            e["results_by_rr"][str(rr)] = {k: v for k, v in data.items() if k != "all_trades"}
        export.append(e)
    with open(out, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
