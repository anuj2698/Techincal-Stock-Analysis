#!/usr/bin/env python3
"""RSI Momentum backtest across ALL F&O stocks.

Strategy: 5m + 15m RSI > 65 (buy) or < 35 (short), 2H hold.
Saves per-stock results to rsi_momentum_backtest.json for use in live scanner.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

sys.path.insert(0, '.')
from analyzer import rsi_series

IST = timezone(timedelta(hours=5, minutes=30))

RSI_BULL = 65
RSI_BEAR = 35
FORWARD_2H = 24
MIN_MOVE = 0.1

OUTPUT_FILE = Path("rsi_momentum_backtest.json")


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
    out = []
    for i in range(0, len(candles_5m) - factor + 1, factor):
        chunk = candles_5m[i:i + factor]
        out.append([chunk[0][0], chunk[0][1], max(c[2] for c in chunk),
                     min(c[3] for c in chunk), chunk[-1][4],
                     sum(c[5] for c in chunk)])
    return out


def map_htf_rsi(candles_5m, candles_htf, rsi_htf):
    ts_map = {}
    for i, c in enumerate(candles_htf):
        if i < len(rsi_htf) and rsi_htf[i] is not None:
            ts_map[c[0]] = rsi_htf[i]
    keys = sorted(ts_map.keys())
    result = [None] * len(candles_5m)
    for i, c5 in enumerate(candles_5m):
        best = None
        for k in keys:
            if k <= c5[0]:
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
    rsi_5m = rsi_series([c[4] for c in candles_5m], 14)
    while len(rsi_5m) < n:
        rsi_5m.append(None)

    candles_15m = resample(candles_5m, 3)
    rsi_15m_raw = rsi_series([c[4] for c in candles_15m], 14)
    rsi_15m = map_htf_rsi(candles_5m, candles_15m, rsi_15m_raw)

    bull_trades = []
    bear_trades = []

    for idx in range(50, n - FORWARD_2H):
        if not is_market_hours(candles_5m[idx][0]):
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

        entry = candles_5m[idx][4]
        fwd_end = min(idx + FORWARD_2H, n - 1)
        exit_price = candles_5m[fwd_end][4]
        max_high = max(c[2] for c in candles_5m[idx + 1:fwd_end + 1])
        min_low = min(c[3] for c in candles_5m[idx + 1:fwd_end + 1])

        if is_bull and not was_bull:
            pnl = (exit_price - entry) / entry * 100
            mfe = (max_high - entry) / entry * 100
            mae = (entry - min_low) / entry * 100
            bull_trades.append({"pnl": pnl, "mfe": mfe, "mae": mae, "win": pnl > MIN_MOVE})

        if is_bear and not was_bear:
            pnl = (entry - exit_price) / entry * 100
            mfe = (entry - min_low) / entry * 100
            mae = (max_high - entry) / entry * 100
            bear_trades.append({"pnl": pnl, "mfe": mfe, "mae": mae, "win": pnl > MIN_MOVE})

    result = {"symbol": sym}

    if bull_trades:
        wins = sum(1 for t in bull_trades if t["win"])
        result["bull"] = {
            "trades": len(bull_trades),
            "wins": wins,
            "win_rate": round(wins / len(bull_trades) * 100, 1),
            "avg_pnl": round(sum(t["pnl"] for t in bull_trades) / len(bull_trades), 3),
            "avg_mfe": round(sum(t["mfe"] for t in bull_trades) / len(bull_trades), 3),
            "avg_mae": round(sum(t["mae"] for t in bull_trades) / len(bull_trades), 3),
            "total_pnl": round(sum(t["pnl"] for t in bull_trades), 2),
        }
        result["bull"]["rr"] = round(result["bull"]["avg_mfe"] / result["bull"]["avg_mae"], 1) if result["bull"]["avg_mae"] > 0.001 else 0

    if bear_trades:
        wins = sum(1 for t in bear_trades if t["win"])
        result["bear"] = {
            "trades": len(bear_trades),
            "wins": wins,
            "win_rate": round(wins / len(bear_trades) * 100, 1),
            "avg_pnl": round(sum(t["pnl"] for t in bear_trades) / len(bear_trades), 3),
            "avg_mfe": round(sum(t["mfe"] for t in bear_trades) / len(bear_trades), 3),
            "avg_mae": round(sum(t["mae"] for t in bear_trades) / len(bear_trades), 3),
            "total_pnl": round(sum(t["pnl"] for t in bear_trades), 2),
        }
        result["bear"]["rr"] = round(result["bear"]["avg_mfe"] / result["bear"]["avg_mae"], 1) if result["bear"]["avg_mae"] > 0.001 else 0

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
    OUTPUT_FILE.write_text(json.dumps(all_results, indent=2))
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
