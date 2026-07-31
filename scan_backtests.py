#!/usr/bin/env python3
"""Weekly batch backtest for NSE 500 + BSE stocks across all timeframes.
Run every Sunday: python scan_backtests.py
"""

import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, timezone

import requests
from pathlib import Path
from datetime import datetime

from nselib import capital_market

from app import resolve_yahoo_ticker, fetch_candles
from backtester import run_backtest

IST = timezone(timedelta(hours=5, minutes=30))
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "scanner_cache"
CACHE_FILE = CACHE_DIR / "backtest_results.json"


def fetch_scan_universe() -> list[dict]:
    print("[1/3] Fetching stock universe (NSE 500 + BSE)...")

    # NSE 500
    frames = [
        capital_market.nifty50_equity_list(),
        capital_market.niftynext50_equity_list(),
        capital_market.niftymidcap150_equity_list(),
        capital_market.niftysmallcap250_equity_list(),
    ]
    seen = set()
    stocks = []
    for df in frames:
        for _, row in df.iterrows():
            sym = row["Symbol"]
            if sym in seen:
                continue
            seen.add(sym)
            stocks.append({
                "symbol": sym,
                "name": row.get("Company Name", sym),
            })
    nse_count = len(stocks)

    # BSE — active equities not already in NSE 500
    try:
        bse_url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Atea=&segment=Equity&status=Active"
        bse_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com/"}
        resp = requests.get(bse_url, headers=bse_headers, timeout=15)
        if resp.status_code == 200:
            nse_upper = {s.upper() for s in seen}
            bse_added = 0
            for s in resp.json():
                sym = str(s.get("scrip_id", "")).strip().upper()
                mktcap = float(s.get("Mktcap", 0) or 0)
                if sym and sym not in nse_upper and mktcap > 500:
                    stocks.append({
                        "symbol": f"BSE:{sym}",
                        "name": s.get("Scrip_Name", sym),
                    })
                    bse_added += 1
            print(f"  BSE: added {bse_added} stocks not in NSE 500")
    except Exception as e:
        print(f"  BSE fetch failed: {e} — continuing with NSE 500 only")

    print(f"  Total: {len(stocks)} stocks ({nse_count} NSE + {len(stocks) - nse_count} BSE)")
    return stocks


def fetch_all_candles(stocks: list[dict]) -> dict:
    print(f"\n[2/3] Fetching candles for {len(stocks)} stocks...")
    candles = {}

    def _fetch(stock):
        sym = stock["symbol"]
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)
            daily = fetch_candles(yahoo_sym, period="1y", interval="1d", canonical=canonical)
            hourly = fetch_candles(yahoo_sym, period="60d", interval="1h", canonical=canonical)
            return sym, daily, hourly
        except Exception:
            return sym, [], []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch, s): s for s in stocks}
        done = 0
        for future in as_completed(futures):
            done += 1
            sym, daily, hourly = future.result()
            candles[sym] = {"daily": daily, "hourly": hourly}
            if done % 20 == 0 or done == len(stocks):
                print(f"  Fetched {done}/{len(stocks)} stocks")

    return candles


def run_all_backtests(stocks: list[dict], candles: dict) -> dict:
    print(f"\n[3/3] Running backtests...")
    results = {}
    total = len(stocks) * 3
    count = 0

    for stock in stocks:
        sym = stock["symbol"]
        name = stock["name"]
        stock_candles = candles.get(sym, {})
        daily = stock_candles.get("daily", [])
        hourly = stock_candles.get("hourly", [])

        for tf, cndls in [("intraday", hourly), ("short_term", daily), ("positional", daily)]:
            count += 1
            if not cndls or len(cndls) < 100:
                continue
            try:
                bt = run_backtest(sym, cndls, timeframe=tf)
                if bt.get("error"):
                    continue
                summary = bt.get("summary", {})
                win_rate = summary.get("win_rate", 0)
                key = f"{sym}_{tf}"
                results[key] = {
                    "symbol": sym,
                    "name": name,
                    "timeframe": tf,
                    "summary": summary,
                }
                print(f"  [{count}/{total}] {sym} ({tf})... win_rate={win_rate:.1f}%")
            except Exception as e:
                print(f"  [{count}/{total}] {sym} ({tf})... ERROR: {e}")

    return results


def main():
    stocks = fetch_scan_universe()
    candles = fetch_all_candles(stocks)
    results = run_all_backtests(stocks, candles)

    CACHE_DIR.mkdir(exist_ok=True)
    data = {
        "timestamp": datetime.now(IST).isoformat(),
        "total_stocks": len(stocks),
        "results": results,
    }
    CACHE_FILE.write_text(json.dumps(data, indent=2, default=str))

    tf_counts = {}
    for entry in results.values():
        tf = entry["timeframe"]
        wr = entry.get("summary", {}).get("win_rate", 0)
        tf_counts.setdefault(tf, {"total": 0, "above_80": 0})
        tf_counts[tf]["total"] += 1
        if wr >= 75:
            tf_counts[tf]["above_80"] += 1

    print(f"\nDone! Results saved to {CACHE_FILE}")
    for tf, c in tf_counts.items():
        print(f"  {tf}: {c['total']} backtested, {c['above_80']} with win_rate >= 75%")


if __name__ == "__main__":
    main()
