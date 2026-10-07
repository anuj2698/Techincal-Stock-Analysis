#!/usr/bin/env python3
"""RSI Momentum backtest across ALL F&O stocks.

Strategy: 5m + 15m RSI > 65 (buy) or < 35 (short), 2H hold.
Saves per-stock results to rsi_momentum_backtest.json for use in live scanner.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

sys.path.insert(0, '.')
from analyzer import rsi_series
from atomic_json import write_json_atomic

IST = timezone(timedelta(hours=5, minutes=30))

RSI_BULL = 65
RSI_BEAR = 35
FORWARD_2H = 24
MIN_MOVE = 0.1
EXECUTION_COST_PCT = 0.15
METHODOLOGY_VERSION = 3

OUTPUT_FILE = Path("rsi_momentum_backtest.json")


def summarize_trades(trades):
    wins = sum(1 for trade in trades if trade["win"])
    test = trades[int(len(trades) * 0.8):]
    test_avg = sum(trade["pnl"] for trade in test) / len(test) if test else 0
    stats = {
        "trades": len(trades),
        "wins": wins,
        "win_rate": round(wins / len(trades) * 100, 1),
        "avg_pnl": round(sum(t["pnl"] for t in trades) / len(trades), 3),
        "avg_mfe": round(sum(t["mfe"] for t in trades) / len(trades), 3),
        "avg_mae": round(sum(t["mae"] for t in trades) / len(trades), 3),
        "total_pnl": round(sum(t["pnl"] for t in trades), 2),
        "methodology_version": METHODOLOGY_VERSION,
        "out_of_sample": {
            "split": "chronological_80_20",
            "trades": len(test),
            "avg_pnl": round(test_avg, 3),
            "win_rate": round(sum(1 for t in test if t["win"]) / len(test) * 100, 1) if test else 0,
            "eligible": len(trades) >= 30 and len(test) >= 8 and test_avg > 0,
        },
    }
    stats["rr"] = round(stats["avg_mfe"] / stats["avg_mae"], 1) if stats["avg_mae"] > 0.001 else 0
    return stats


def fetch_5min_candles(symbol):
    app_id = os.environ.get('FYERS_APP_ID')
    token = os.environ.get('FYERS_ACCESS_TOKEN')
    if app_id and token:
        try:
            from fyers_apiv3 import fyersModel
            os.makedirs('logs', exist_ok=True)
            client = fyersModel.FyersModel(client_id=app_id, token=token, is_async=False, log_path='logs/')
            now = datetime.now(IST)
            resp = client.history(data={
                'symbol': f'NSE:{symbol}-EQ', 'resolution': '5', 'date_format': '1',
                'range_from': (now - timedelta(days=60)).strftime('%Y-%m-%d'),
                'range_to': now.strftime('%Y-%m-%d'), 'cont_flag': '1',
            })
            time.sleep(0.12)
            candles = resp.get('candles', [])
            if candles:
                return [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in candles]
        except Exception:
            pass
    try:
        import yfinance as yf
        df = yf.Ticker(f'{symbol}.NS').history(period='60d', interval='5m', auto_adjust=False)
        if df is not None and not df.empty:
            rows = []
            for idx, row in df.iterrows():
                rows.append([int(idx.timestamp()), float(row['Open']), float(row['High']),
                             float(row['Low']), float(row['Close']), float(row['Volume'])])
            return sorted(rows, key=lambda r: r[0])
    except Exception:
        pass
    return []


def resample(candles_5m, factor):
    """Build exchange-open-aligned bars without combining different sessions."""
    buckets = {}
    for candle in candles_5m:
        dt = datetime.fromtimestamp(int(candle[0]), tz=IST)
        minute = dt.hour * 60 + dt.minute
        key = (dt.strftime("%Y-%m-%d"), (minute - (9 * 60 + 15)) // (factor * 5))
        buckets.setdefault(key, []).append(candle)
    out = []
    for chunk in buckets.values():
        out.append([chunk[0][0], chunk[0][1], max(c[2] for c in chunk),
                    min(c[3] for c in chunk), chunk[-1][4], sum(c[5] for c in chunk)])
    return sorted(out, key=lambda row: row[0])


def map_htf_rsi(candles_5m, candles_htf, rsi_htf):
    offset = len(candles_htf) - len(rsi_htf)
    ts_map = {
        int(candles_htf[i + offset][0]) + 15 * 60: value
        for i, value in enumerate(rsi_htf)
    }
    keys = sorted(ts_map.keys())
    result = [None] * len(candles_5m)
    for i, c5 in enumerate(candles_5m):
        best = None
        for k in keys:
            # A 15-minute RSI becomes available only after that bar closes.
            if k <= c5[0] + 5 * 60:
                best = k
            else:
                break
        if best is not None:
            result[i] = ts_map[best]
    return result


def is_market_hours(ts):
    dt = datetime.fromtimestamp(ts, tz=IST)
    if dt.weekday() >= 5:
        return False
    t = dt.hour * 60 + dt.minute
    return 9 * 60 + 20 <= t <= 15 * 60 + 10


def backtest_stock(sym):
    candles_5m = fetch_5min_candles(sym)
    if not candles_5m or len(candles_5m) < 200:
        return None

    n = len(candles_5m)
    rsi_5m_raw = rsi_series([c[4] for c in candles_5m], 14)
    rsi_5m = [None] * (n - len(rsi_5m_raw)) + rsi_5m_raw

    candles_15m = resample(candles_5m, 3)
    rsi_15m_raw = rsi_series([c[4] for c in candles_15m], 14)
    rsi_15m = map_htf_rsi(candles_5m, candles_15m, rsi_15m_raw)

    bull_trades = []
    bear_trades = []

    traded_days = set()
    for idx in range(50, n - 1):
        if not is_market_hours(candles_5m[idx][0]):
            continue
        signal_dt = datetime.fromtimestamp(candles_5m[idx][0], tz=IST)
        day_key = signal_dt.strftime("%Y-%m-%d")
        if day_key in traded_days or (signal_dt.hour, signal_dt.minute) > (14, 30):
            continue

        r5 = rsi_5m[idx]
        r15 = rsi_15m[idx]
        if r5 is None or r15 is None:
            continue

        # Check previous candle to avoid re-triggering
        r5p = rsi_5m[idx - 1]
        r15p = rsi_15m[idx - 1]

        is_bull = r5 > RSI_BULL and r15 > RSI_BULL
        was_bull = r5p is not None and r15p is not None and r5p > RSI_BULL and r15p > RSI_BULL

        is_bear = r5 < RSI_BEAR and r15 < RSI_BEAR
        was_bear = r5p is not None and r15p is not None and r5p < RSI_BEAR and r15p < RSI_BEAR

        direction = "bull" if is_bull and not was_bull else "bear" if is_bear and not was_bear else None
        if not direction:
            continue
        entry = float(candles_5m[idx][4])
        stop = entry * (0.995 if direction == "bull" else 1.005)
        target = entry * (1.0075 if direction == "bull" else 0.9925)
        future = []
        for candle in candles_5m[idx + 1:idx + 1 + FORWARD_2H]:
            dt = datetime.fromtimestamp(candle[0], tz=IST)
            if dt.strftime("%Y-%m-%d") != day_key or (dt.hour, dt.minute) > (15, 15):
                break
            future.append(candle)
        if not future:
            continue

        exit_price = float(future[-1][4])
        result = "eod"
        for candle in future:
            high, low = float(candle[2]), float(candle[3])
            if direction == "bull":
                if low <= stop:
                    exit_price, result = stop, "stop"
                    break
                if high >= target:
                    exit_price, result = target, "target"
                    break
            else:
                if high >= stop:
                    exit_price, result = stop, "stop"
                    break
                if low <= target:
                    exit_price, result = target, "target"
                    break

        max_high = max(float(c[2]) for c in future)
        min_low = min(float(c[3]) for c in future)
        gross = ((exit_price - entry) / entry * 100 if direction == "bull"
                 else (entry - exit_price) / entry * 100)
        pnl = gross - EXECUTION_COST_PCT
        mfe = ((max_high - entry) / entry * 100 if direction == "bull"
               else (entry - min_low) / entry * 100)
        mae = ((entry - min_low) / entry * 100 if direction == "bull"
               else (max_high - entry) / entry * 100)
        trade = {"date": day_key, "pnl": pnl, "mfe": mfe, "mae": mae,
                 "win": pnl > 0, "result": result}
        (bull_trades if direction == "bull" else bear_trades).append(trade)
        traded_days.add(day_key)

    result = {"symbol": sym}

    if bull_trades:
        result["bull"] = summarize_trades(bull_trades)

    if bear_trades:
        result["bear"] = summarize_trades(bear_trades)

    if not bull_trades and not bear_trades:
        return None

    return result


def main():
    # Import get_fno_stocks from app context
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("Fetching F&O stock list...")
    from app import get_fno_stocks, resolve_yahoo_ticker
    stocks = get_fno_stocks()
    print(f"Found {len(stocks)} F&O stocks. Running RSI Momentum backtest...\n")

    from concurrent.futures import ThreadPoolExecutor

    results = {}
    done = 0
    total = len(stocks)

    def process(stock):
        nonlocal done
        sym = stock["symbol"]
        r = backtest_stock(sym)
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  [{done}/{total}] processed")
        return sym, r

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = list(pool.map(process, stocks))

    all_results = {}
    for sym, r in futures:
        if r:
            all_results[sym] = r

    # Save to JSON
    write_json_atomic(OUTPUT_FILE, {
        "methodology_version": METHODOLOGY_VERSION,
        "execution_cost_pct": EXECUTION_COST_PCT,
        "generated_at": datetime.now(IST).isoformat(),
        "results": all_results,
    }, indent=2)
    print(f"\nSaved {len(all_results)} stock results to {OUTPUT_FILE}")

    # Print summary table
    print(f"\n{'=' * 110}")
    print(f"RSI MOMENTUM BACKTEST — ALL F&O STOCKS (5m+15m RSI > {RSI_BULL} buy / < {RSI_BEAR} short, 2H hold)")
    print(f"{'=' * 110}")

    # Aggregate
    all_bull = []
    all_bear = []

    rows = []
    for sym, r in sorted(all_results.items()):
        b = r.get("bull", {})
        be = r.get("bear", {})
        bull_str = f"{b.get('win_rate', '-'):>5}% ({b['trades']}t, {b['avg_pnl']:+.3f}%, RR {b['rr']}x)" if b else "—"
        bear_str = f"{be.get('win_rate', '-'):>5}% ({be['trades']}t, {be['avg_pnl']:+.3f}%, RR {be['rr']}x)" if be else "—"
        total_trades = b.get("trades", 0) + be.get("trades", 0)
        total_wins = b.get("wins", 0) + be.get("wins", 0)
        combined_wr = round(total_wins / total_trades * 100, 1) if total_trades else 0
        rows.append((sym, combined_wr, total_trades, bull_str, bear_str))

        if b:
            all_bull.extend([1] * b["wins"] + [0] * (b["trades"] - b["wins"]))
        if be:
            all_bear.extend([1] * be["wins"] + [0] * (be["trades"] - be["wins"]))

    rows.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Stock':<18} {'Combined':>9} {'Trades':>7}   {'Bullish (WR / trades / avgPnL / RR)':<45} {'Bearish (WR / trades / avgPnL / RR)'}")
    print(f"  {'─' * 18} {'─' * 9} {'─' * 7}   {'─' * 45} {'─' * 45}")
    for sym, cwr, tt, bs, bes in rows:
        print(f"  {sym:<18} {cwr:>8.1f}% {tt:>7}   {bs:<45} {bes}")

    print(f"\n{'─' * 110}")
    total_b = len(all_bull)
    total_be = len(all_bear)
    print(f"  AGGREGATE BULL: {sum(all_bull)}/{total_b} wins = {sum(all_bull)/total_b*100:.1f}% WR" if total_b else "  AGGREGATE BULL: no trades")
    print(f"  AGGREGATE BEAR: {sum(all_bear)}/{total_be} wins = {sum(all_bear)/total_be*100:.1f}% WR" if total_be else "  AGGREGATE BEAR: no trades")
    total_all = total_b + total_be
    total_wins = sum(all_bull) + sum(all_bear)
    print(f"  OVERALL: {total_wins}/{total_all} wins = {total_wins/total_all*100:.1f}% WR" if total_all else "")


if __name__ == "__main__":
    main()
