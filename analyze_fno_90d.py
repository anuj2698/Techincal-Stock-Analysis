#!/usr/bin/env python3
"""Analyze F&O stocks for the last 90 days — high volume + big intraday movers."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

sys.path.insert(0, ".")
from app import get_fno_stocks, resolve_yahoo_ticker, fetch_candles

IST = timezone(timedelta(hours=5, minutes=30))
LOOKBACK_DAYS = 180


def analyze_stock(stock: dict) -> dict | None:
    sym = stock["symbol"]
    name = stock["name"]
    try:
        canonical, yahoo_sym = resolve_yahoo_ticker(sym)
        candles = fetch_candles(yahoo_sym, period="6mo", interval="1d", canonical=canonical)
        if not candles or len(candles) < 20:
            return None

        cutoff_ts = (datetime.now(IST) - timedelta(days=LOOKBACK_DAYS)).timestamp()
        candles_90d = [c for c in candles if c[0] >= cutoff_ts]
        if len(candles_90d) < 15:
            return None

        volumes = [c[5] for c in candles_90d]
        avg_vol = sum(volumes) / len(volumes)
        max_vol = max(volumes)
        last_vol = volumes[-1]
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0

        intraday_ranges = []
        intraday_pcts = []
        for c in candles_90d:
            _, o, h, l, cl, v = c
            rng = h - l
            pct = (rng / cl * 100) if cl > 0 else 0
            intraday_ranges.append(rng)
            intraday_pcts.append(pct)

        avg_range_pct = sum(intraday_pcts) / len(intraday_pcts)
        max_range_pct = max(intraday_pcts)
        max_range_idx = intraday_pcts.index(max_range_pct)
        max_range_date = datetime.fromtimestamp(candles_90d[max_range_idx][0], tz=IST).strftime("%Y-%m-%d")

        last_close = candles_90d[-1][4]
        last_range_pct = intraday_pcts[-1]

        recent_5d_vol = sum(volumes[-5:]) / min(5, len(volumes[-5:])) if len(volumes) >= 5 else avg_vol
        recent_5d_range = sum(intraday_pcts[-5:]) / min(5, len(intraday_pcts[-5:])) if len(intraday_pcts) >= 5 else avg_range_pct

        total_value = avg_vol * last_close

        return {
            "symbol": sym,
            "name": name,
            "last_close": round(last_close, 2),
            "trading_days": len(candles_90d),
            "avg_volume": int(avg_vol),
            "avg_volume_cr": round(avg_vol * last_close / 1e7, 1),
            "max_volume": int(max_vol),
            "last_volume": int(last_vol),
            "vol_ratio_last": round(vol_ratio, 2),
            "recent_5d_avg_vol": int(recent_5d_vol),
            "avg_range_pct": round(avg_range_pct, 2),
            "max_range_pct": round(max_range_pct, 2),
            "max_range_date": max_range_date,
            "last_range_pct": round(last_range_pct, 2),
            "recent_5d_avg_range_pct": round(recent_5d_range, 2),
            "total_value_traded": round(total_value / 1e7, 1),
        }
    except Exception as e:
        return None


def main():
    print("Fetching F&O stock list...")
    stocks = get_fno_stocks()
    print(f"Found {len(stocks)} F&O stocks. Fetching 90-day data...\n")

    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(analyze_stock, s): s for s in stocks}
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r:
                results.append(r)
            if done % 25 == 0 or done == len(stocks):
                print(f"  [{done}/{len(stocks)}] processed, {len(results)} valid")

    print(f"\n{'='*100}")
    print(f"F&O STOCK ANALYSIS — LAST 90 DAYS ({len(results)} stocks analyzed)")
    print(f"{'='*100}")

    # --- TOP 30 BY AVERAGE DAILY VOLUME (SHARES) ---
    print(f"\n{'─'*100}")
    print("TOP 30 STOCKS BY AVERAGE DAILY VOLUME (Shares)")
    print(f"{'─'*100}")
    by_vol = sorted(results, key=lambda x: x["avg_volume"], reverse=True)[:30]
    print(f"{'#':<4} {'Symbol':<18} {'CMP':>10} {'Avg Vol':>14} {'Avg Val(Cr)':>12} {'5D Avg Vol':>14} {'Vol Ratio':>10}")
    print(f"{'─'*4} {'─'*18} {'─'*10} {'─'*14} {'─'*12} {'─'*14} {'─'*10}")
    for i, r in enumerate(by_vol, 1):
        print(f"{i:<4} {r['symbol']:<18} {r['last_close']:>10.2f} {r['avg_volume']:>14,} {r['avg_volume_cr']:>12.1f} {r['recent_5d_avg_vol']:>14,} {r['vol_ratio_last']:>10.2f}")

    # --- TOP 30 BY AVERAGE DAILY TRADED VALUE (CR) ---
    print(f"\n{'─'*100}")
    print("TOP 30 STOCKS BY AVERAGE DAILY TRADED VALUE (₹ Crores)")
    print(f"{'─'*100}")
    by_val = sorted(results, key=lambda x: x["avg_volume_cr"], reverse=True)[:30]
    print(f"{'#':<4} {'Symbol':<18} {'CMP':>10} {'Avg Val(Cr)':>12} {'Avg Vol':>14} {'5D Avg Vol':>14}")
    print(f"{'─'*4} {'─'*18} {'─'*10} {'─'*12} {'─'*14} {'─'*14}")
    for i, r in enumerate(by_val, 1):
        print(f"{i:<4} {r['symbol']:<18} {r['last_close']:>10.2f} {r['avg_volume_cr']:>12.1f} {r['avg_volume']:>14,} {r['recent_5d_avg_vol']:>14,}")

    # --- TOP 30 BIG INTRADAY MOVERS (AVG HIGH-LOW %) ---
    print(f"\n{'─'*100}")
    print("TOP 30 BIG INTRADAY MOVERS — Average Daily Range (High-Low as % of Close)")
    print(f"{'─'*100}")
    by_range = sorted(results, key=lambda x: x["avg_range_pct"], reverse=True)[:30]
    print(f"{'#':<4} {'Symbol':<18} {'CMP':>10} {'Avg Range%':>11} {'5D Range%':>10} {'Max Range%':>11} {'Max Date':>12} {'Avg Vol':>14}")
    print(f"{'─'*4} {'─'*18} {'─'*10} {'─'*11} {'─'*10} {'─'*11} {'─'*12} {'─'*14}")
    for i, r in enumerate(by_range, 1):
        print(f"{i:<4} {r['symbol']:<18} {r['last_close']:>10.2f} {r['avg_range_pct']:>11.2f} {r['recent_5d_avg_range_pct']:>10.2f} {r['max_range_pct']:>11.2f} {r['max_range_date']:>12} {r['avg_volume']:>14,}")

    # --- TOP 30 BIGGEST SINGLE-DAY INTRADAY MOVES ---
    print(f"\n{'─'*100}")
    print("TOP 30 BIGGEST SINGLE-DAY INTRADAY MOVES (Max High-Low %)")
    print(f"{'─'*100}")
    by_max_range = sorted(results, key=lambda x: x["max_range_pct"], reverse=True)[:30]
    print(f"{'#':<4} {'Symbol':<18} {'CMP':>10} {'Max Range%':>11} {'Date':>12} {'Avg Range%':>11} {'Avg Vol':>14}")
    print(f"{'─'*4} {'─'*18} {'─'*10} {'─'*11} {'─'*12} {'─'*11} {'─'*14}")
    for i, r in enumerate(by_max_range, 1):
        print(f"{i:<4} {r['symbol']:<18} {r['last_close']:>10.2f} {r['max_range_pct']:>11.2f} {r['max_range_date']:>12} {r['avg_range_pct']:>11.2f} {r['avg_volume']:>14,}")

    # --- STOCKS WITH RECENT VOLUME SPIKE (last day vol > 2x average) ---
    print(f"\n{'─'*100}")
    print("STOCKS WITH RECENT VOLUME SPIKE (Last Day Volume > 2x 90-Day Average)")
    print(f"{'─'*100}")
    vol_spikes = sorted([r for r in results if r["vol_ratio_last"] >= 2.0], key=lambda x: x["vol_ratio_last"], reverse=True)
    if vol_spikes:
        print(f"{'#':<4} {'Symbol':<18} {'CMP':>10} {'Last Vol':>14} {'Avg Vol':>14} {'Ratio':>8} {'Avg Range%':>11}")
        print(f"{'─'*4} {'─'*18} {'─'*10} {'─'*14} {'─'*14} {'─'*8} {'─'*11}")
        for i, r in enumerate(vol_spikes, 1):
            print(f"{i:<4} {r['symbol']:<18} {r['last_close']:>10.2f} {r['last_volume']:>14,} {r['avg_volume']:>14,} {r['vol_ratio_last']:>8.2f} {r['avg_range_pct']:>11.2f}")
    else:
        print("No stocks with volume spike > 2x on the last trading day.")

    # --- COMBINED: HIGH VOLUME + BIG MOVERS ---
    print(f"\n{'─'*100}")
    print("COMBINED: HIGH VOLUME + BIG INTRADAY RANGE (Top 50 by Volume ∩ Top 50 by Range)")
    print(f"{'─'*100}")
    top_vol_set = {r["symbol"] for r in sorted(results, key=lambda x: x["avg_volume_cr"], reverse=True)[:50]}
    top_range_set = {r["symbol"] for r in sorted(results, key=lambda x: x["avg_range_pct"], reverse=True)[:50]}
    overlap_syms = top_vol_set & top_range_set
    overlap = sorted([r for r in results if r["symbol"] in overlap_syms], key=lambda x: x["avg_range_pct"], reverse=True)
    if overlap:
        print(f"{'#':<4} {'Symbol':<18} {'CMP':>10} {'Avg Val(Cr)':>12} {'Avg Range%':>11} {'5D Range%':>10} {'Avg Vol':>14}")
        print(f"{'─'*4} {'─'*18} {'─'*10} {'─'*12} {'─'*11} {'─'*10} {'─'*14}")
        for i, r in enumerate(overlap, 1):
            print(f"{i:<4} {r['symbol']:<18} {r['last_close']:>10.2f} {r['avg_volume_cr']:>12.1f} {r['avg_range_pct']:>11.2f} {r['recent_5d_avg_range_pct']:>10.2f} {r['avg_volume']:>14,}")
    else:
        print("No stocks in both top-50 volume and top-50 range.")

    # Save raw results
    out_path = "fno_90d_analysis.json"
    with open(out_path, "w") as f:
        json.dump(sorted(results, key=lambda x: x["avg_volume_cr"], reverse=True), f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
