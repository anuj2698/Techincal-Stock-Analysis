#!/usr/bin/env python3
"""PDH/PDL Breakout backtest on 5-min candles across ALL F&O stocks."""
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

import yfinance as yf

sys.path.insert(0, '.')
from app import get_fno_stocks, resolve_yahoo_ticker, fetch_candles

IST = timezone(timedelta(hours=5, minutes=30))
PDHL_BUFFER_PCT = 0.001
PDHL_SL_RATIO = 0.3


def fetch_5min(symbol):
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
                'range_from': (now - timedelta(days=90)).strftime('%Y-%m-%d'),
                'range_to': now.strftime('%Y-%m-%d'), 'cont_flag': '1',
            })
            time.sleep(0.12)
            candles = resp.get('candles', [])
            if candles:
                return [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in candles]
        except:
            pass
    try:
        df = yf.Ticker(f'{symbol}.NS').history(period='60d', interval='5m', auto_adjust=False)
        if df is not None and not df.empty:
            rows = []
            for idx, row in df.iterrows():
                rows.append([int(idx.timestamp()), float(row['Open']), float(row['High']),
                             float(row['Low']), float(row['Close']), float(row['Volume'])])
            return sorted(rows, key=lambda r: r[0])
    except:
        pass
    return []


def group_by_day(candles):
    days = defaultdict(list)
    for c in candles:
        dt = datetime.fromtimestamp(int(c[0]), tz=IST)
        if dt.weekday() < 5:
            days[dt.strftime('%Y-%m-%d')].append(c)
    return dict(sorted(days.items()))


def sim_breakout(day_candles, entry_price, sl, target, direction, start_min=15):
    for c in day_candles:
        dt = datetime.fromtimestamp(int(c[0]), tz=IST)
        mins = (dt.hour - 9) * 60 + (dt.minute - 15)
        if mins < start_min:
            continue
        is_exit = dt.hour > 15 or (dt.hour == 15 and dt.minute >= 15)
        h, l, cl = c[2], c[3], c[4]

        if not hasattr(sim_breakout, '_triggered'):
            sim_breakout._triggered = False

        if not sim_breakout._triggered:
            if direction == 'LONG' and h >= entry_price:
                sim_breakout._triggered = True
            elif direction == 'SHORT' and l <= entry_price:
                sim_breakout._triggered = True

        if sim_breakout._triggered:
            if direction == 'LONG':
                if l <= sl: return sl, 'SL'
                if h >= target: return target, 'TARGET'
                if is_exit: return cl, 'TIME'
            else:
                if h >= sl: return sl, 'SL'
                if l <= target: return target, 'TARGET'
                if is_exit: return cl, 'TIME'
    return None


def backtest_stock(sym):
    candles_5m = fetch_5min(sym)
    if not candles_5m or len(candles_5m) < 100:
        return None

    days = group_by_day(candles_5m)
    sorted_dates = sorted(days.keys())

    # Also get daily candles for PDH/PDL
    try:
        canonical, yahoo_sym = resolve_yahoo_ticker(sym)
        candles_daily = fetch_candles(yahoo_sym, period='6mo', interval='1d', canonical=canonical)
    except:
        return None

    if not candles_daily or len(candles_daily) < 10:
        return None

    daily_map = {}
    for c in candles_daily:
        d = datetime.fromtimestamp(int(c[0]), tz=IST).strftime('%Y-%m-%d')
        daily_map[d] = c

    sorted_daily = sorted(daily_map.keys())
    trades = []

    for day_str in sorted_dates:
        dc = days[day_str]
        if len(dc) < 15:
            continue

        # Find previous day
        prev_day = None
        for d in reversed(sorted_daily):
            if d < day_str:
                prev_day = d
                break
        if not prev_day or prev_day not in daily_map:
            continue

        pc = daily_map[prev_day]
        pdh = float(pc[2])
        pdl = float(pc[3])
        pd_range = pdh - pdl
        if pd_range <= 0:
            continue

        buf = pdh * PDHL_BUFFER_PCT

        for direction in ['LONG', 'SHORT']:
            if direction == 'LONG':
                ep = pdh + buf
                sl = pdh - pd_range * PDHL_SL_RATIO
                risk = ep - sl
                tgt = ep + risk
            else:
                ep = pdl - buf
                sl = pdl + pd_range * PDHL_SL_RATIO
                risk = sl - ep
                tgt = ep - risk

            # Reset trigger state
            sim_breakout._triggered = False
            result = sim_breakout(dc, ep, sl, tgt, direction, 15)

            if result:
                xp, reason = result
                pnl_pct = (xp - ep) / ep * 100 if direction == 'LONG' else (ep - xp) / ep * 100
                trades.append({
                    'dir': direction, 'result': reason,
                    'pnl_pct': round(pnl_pct, 2), 'date': day_str,
                })

    if not trades:
        return None

    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    targets = [t for t in trades if t['result'] == 'TARGET']
    sls = [t for t in trades if t['result'] == 'SL']
    total_pnl = sum(t['pnl_pct'] for t in trades)
    last_close = float(candles_5m[-1][4])
    avg_vol = sum(float(c[5]) for c in candles_5m[-75:]) / min(75, len(candles_5m))
    avg_value_cr = round(avg_vol * last_close / 1e7, 1)

    return {
        'symbol': sym,
        'days': len(set(t['date'] for t in trades)),
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'targets': len(targets),
        'sls': len(sls),
        'win_rate': round(len(wins) / len(trades) * 100, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(total_pnl / len(trades), 3),
        'avg_value_cr': avg_value_cr,
        'data_from': min(t['date'] for t in trades),
        'data_to': max(t['date'] for t in trades),
    }


def main():
    print('Fetching F&O stock list...')
    stocks = get_fno_stocks()
    print(f'Found {len(stocks)} F&O stocks. Fetching 5-min data & running PDH/PDL backtest...\n')

    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(backtest_stock, s['symbol']): s['symbol'] for s in stocks}
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r:
                results.append(r)
            if done % 25 == 0 or done == len(stocks):
                print(f'  [{done}/{len(stocks)}] processed, {len(results)} valid')

    results.sort(key=lambda x: x['total_pnl'], reverse=True)

    # Data range
    all_froms = [r['data_from'] for r in results]
    all_tos = [r['data_to'] for r in results]

    print(f'\n{"="*100}')
    print(f'PDH/PDL BREAKOUT — ALL F&O STOCKS — 5-MIN CANDLE BACKTEST')
    print(f'Data: {min(all_froms)} to {max(all_tos)}')
    print(f'{"="*100}')

    all_trades = sum(r['trades'] for r in results)
    all_wins = sum(r['wins'] for r in results)
    all_losses = sum(r['losses'] for r in results)
    all_targets = sum(r['targets'] for r in results)
    all_sls = sum(r['sls'] for r in results)
    all_pnl = sum(r['total_pnl'] for r in results)
    overall_wr = round(all_wins / all_trades * 100, 1) if all_trades else 0

    profitable = [r for r in results if r['total_pnl'] > 0]

    print(f'\n  OVERALL STATS')
    print(f'  {"─"*50}')
    print(f'  Stocks analyzed:    {len(results)}')
    print(f'  Total trades:       {all_trades}')
    print(f'  Winners:            {all_wins}')
    print(f'  Losers:             {all_losses}')
    print(f'  Win Rate:           {overall_wr}%')
    print(f'  Target Hits:        {all_targets}')
    print(f'  SL Hits:            {all_sls}')
    print(f'  Total PnL:          {all_pnl:+.2f}%')
    print(f'  Avg PnL/Trade:      {all_pnl/all_trades:+.3f}%')
    print(f'  Profitable stocks:  {len(profitable)} / {len(results)} ({round(len(profitable)/len(results)*100)}%)')

    # Top 30
    print(f'\n{"─"*100}')
    print(f'TOP 30 STOCKS BY PnL')
    print(f'{"─"*100}')
    print(f'{"#":<4} {"Symbol":<16} {"Days":>5} {"Trades":>7} {"W":>4} {"L":>4} {"Win%":>6} {"Tgt":>5} {"SL":>5} {"TotPnL%":>9} {"AvgPnL%":>9} {"Val Cr":>8}')
    print(f'{"─"*4} {"─"*16} {"─"*5} {"─"*7} {"─"*4} {"─"*4} {"─"*6} {"─"*5} {"─"*5} {"─"*9} {"─"*9} {"─"*8}')
    for i, r in enumerate(results[:30], 1):
        print(f'{i:<4} {r["symbol"]:<16} {r["days"]:>5} {r["trades"]:>7} {r["wins"]:>4} {r["losses"]:>4} {r["win_rate"]:>5.1f}% {r["targets"]:>5} {r["sls"]:>5} {r["total_pnl"]:>+9.2f} {r["avg_pnl"]:>+9.3f} {r["avg_value_cr"]:>8.1f}')

    # Bottom 15
    print(f'\n{"─"*100}')
    print(f'BOTTOM 15 STOCKS BY PnL')
    print(f'{"─"*100}')
    print(f'{"#":<4} {"Symbol":<16} {"Days":>5} {"Trades":>7} {"W":>4} {"L":>4} {"Win%":>6} {"Tgt":>5} {"SL":>5} {"TotPnL%":>9} {"AvgPnL%":>9}')
    print(f'{"─"*4} {"─"*16} {"─"*5} {"─"*7} {"─"*4} {"─"*4} {"─"*6} {"─"*5} {"─"*5} {"─"*9} {"─"*9}')
    for i, r in enumerate(results[-15:], len(results) - 14):
        print(f'{i:<4} {r["symbol"]:<16} {r["days"]:>5} {r["trades"]:>7} {r["wins"]:>4} {r["losses"]:>4} {r["win_rate"]:>5.1f}% {r["targets"]:>5} {r["sls"]:>5} {r["total_pnl"]:>+9.2f} {r["avg_pnl"]:>+9.3f}')

    # Win rate distribution
    print(f'\n{"─"*100}')
    print(f'WIN RATE DISTRIBUTION')
    print(f'{"─"*100}')
    brackets = [(70, 100), (60, 70), (50, 60), (40, 50), (0, 40)]
    for lo, hi in brackets:
        count = len([r for r in results if lo <= r['win_rate'] < hi])
        syms = ', '.join(r['symbol'] for r in results if lo <= r['win_rate'] < hi)[:80]
        print(f'  {lo}-{hi}% WR:  {count:>4} stocks  {syms}')

    # Save
    out = 'pdhl_all_fno_backtest.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nFull results saved to {out}')


if __name__ == '__main__':
    main()
