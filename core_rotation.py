#!/usr/bin/env python3
"""Monthly auto-rotation of intraday core stocks.

Runs a 6-month volume+range analysis and a 60-day PDH/PDL backtest to
decide which stocks deserve core status. Stocks are demoted when their
trailing backtest deteriorates and promoted when they consistently rank
in the top tier.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
CONFIG_PATH = Path(__file__).resolve().parent / "core_stocks_config.json"
MAX_CORE = 15
LOOKBACK_DAYS = 180
PROMOTE_MONTHS_REQUIRED = 2
DEMOTE_WR_THRESHOLD = 50.0
DEMOTE_PNL_THRESHOLD = 0.0

PDHL_BUFFER_PCT = 0.1
PDHL_SL_RATIO = 0.3


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {
        "last_rotation": None,
        "rotation_interval_days": 30,
        "core_stocks": [],
        "backtest_stats": {},
        "promotion_history": {},
        "rotation_log": [],
    }


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def needs_rotation() -> bool:
    cfg = load_config()
    last = cfg.get("last_rotation")
    interval = cfg.get("rotation_interval_days", 30)
    if not last:
        return True
    last_dt = datetime.fromisoformat(last)
    return (datetime.now(IST) - last_dt).days >= interval


def _analyze_stock(stock: dict, fetch_candles, resolve_yahoo_ticker) -> dict | None:
    sym = stock["symbol"]
    name = stock["name"]
    try:
        canonical, yahoo_sym = resolve_yahoo_ticker(sym)
        candles = fetch_candles(yahoo_sym, period="6mo", interval="1d", canonical=canonical)
        if not candles or len(candles) < 25:
            return None

        cutoff_ts = (datetime.now(IST) - timedelta(days=LOOKBACK_DAYS)).timestamp()
        candles_period = [c for c in candles if c[0] >= cutoff_ts]
        if len(candles_period) < 15:
            return None

        volumes = [c[5] for c in candles_period]
        avg_vol = sum(volumes) / len(volumes)
        last_close = candles_period[-1][4]

        if last_close < 40:
            return None
        avg_vol_5 = sum(volumes[-5:]) / min(5, len(volumes[-5:])) if len(volumes) >= 5 else avg_vol
        if avg_vol_5 < 500_000:
            return None

        avg_value_cr = avg_vol * last_close / 1e7

        intraday_pcts = []
        for c in candles_period:
            _, o, h, l, cl, v = c
            pct = (h - l) / cl * 100 if cl > 0 else 0
            intraday_pcts.append(pct)

        avg_range_pct = sum(intraday_pcts) / len(intraday_pcts)
        avg_range_5 = sum(intraday_pcts[-5:]) / min(5, len(intraday_pcts[-5:])) if len(intraday_pcts) >= 5 else avg_range_pct

        return {
            "symbol": sym,
            "name": name,
            "last_close": round(last_close, 2),
            "avg_volume": int(avg_vol),
            "avg_volume_cr": round(avg_value_cr, 1),
            "avg_range_pct": round(avg_range_pct, 2),
            "avg_range_5d": round(avg_range_5, 2),
            "avg_vol_5d": int(avg_vol_5),
        }
    except Exception:
        return None


def _backtest_pdhl_stock(sym: str, fetch_candles, resolve_yahoo_ticker) -> dict | None:
    """Run a 60-day PDH/PDL backtest on 5-min candles for a single stock."""
    try:
        canonical, yahoo_sym = resolve_yahoo_ticker(sym)

        candles_5m = fetch_candles(yahoo_sym, period="2mo", interval="5m", canonical=canonical)
        if not candles_5m or len(candles_5m) < 100:
            return None

        day_groups = defaultdict(list)
        for c in candles_5m:
            dt = datetime.fromtimestamp(int(c[0]), tz=IST)
            if dt.weekday() < 5:
                day_groups[dt.strftime("%Y-%m-%d")].append(c)

        days_sorted = sorted(day_groups.keys())
        if len(days_sorted) < 3:
            return None

        all_trades = []
        for i in range(1, len(days_sorted)):
            prev_day = days_sorted[i - 1]
            today = days_sorted[i]

            prev_candles = day_groups[prev_day]
            today_candles = day_groups[today]
            if len(prev_candles) < 5 or len(today_candles) < 10:
                continue

            pdh = max(float(c[2]) for c in prev_candles)
            pdl = min(float(c[3]) for c in prev_candles)
            pd_range = pdh - pdl
            if pd_range <= 0:
                continue

            buffer = pdh * PDHL_BUFFER_PCT / 100
            long_entry = pdh + buffer
            short_entry = pdl - buffer
            long_sl = pdh - pd_range * PDHL_SL_RATIO
            short_sl = pdl + pd_range * PDHL_SL_RATIO
            long_risk = long_entry - long_sl
            short_risk = short_sl - short_entry
            long_target = long_entry + long_risk
            short_target = short_entry - short_risk

            post_15 = []
            for c in today_candles:
                dt = datetime.fromtimestamp(int(c[0]), tz=IST)
                mins_from_open = (dt.hour - 9) * 60 + (dt.minute - 15)
                if mins_from_open >= 15:
                    post_15.append(c)

            if not post_15:
                continue

            for direction, entry, sl, target in [
                ("LONG", long_entry, long_sl, long_target),
                ("SHORT", short_entry, short_sl, short_target),
            ]:
                triggered = False
                result = None
                pnl_pct = None

                for c in post_15:
                    h, l, cl = float(c[2]), float(c[3]), float(c[4])
                    dt = datetime.fromtimestamp(int(c[0]), tz=IST)
                    is_eod = dt.hour >= 15 and dt.minute >= 15

                    if not triggered:
                        if direction == "LONG" and h >= entry:
                            triggered = True
                        elif direction == "SHORT" and l <= entry:
                            triggered = True

                    if triggered and result is None:
                        if direction == "LONG":
                            if l <= sl:
                                result = "sl"
                                pnl_pct = round((sl - entry) / entry * 100, 2)
                            elif h >= target:
                                result = "target"
                                pnl_pct = round((target - entry) / entry * 100, 2)
                            elif is_eod:
                                result = "eod"
                                pnl_pct = round((cl - entry) / entry * 100, 2)
                        else:
                            if h >= sl:
                                result = "sl"
                                pnl_pct = round((entry - sl) / entry * 100, 2)
                            elif l <= target:
                                result = "target"
                                pnl_pct = round((entry - target) / entry * 100, 2)
                            elif is_eod:
                                result = "eod"
                                pnl_pct = round((entry - cl) / entry * 100, 2)

                    if result:
                        break

                if triggered and result:
                    all_trades.append({"direction": direction, "result": result, "pnl_pct": pnl_pct})

        if not all_trades:
            return None

        winners = [t for t in all_trades if t["pnl_pct"] and t["pnl_pct"] > 0]
        total_pnl = sum(t["pnl_pct"] for t in all_trades if t["pnl_pct"])
        wr = round(len(winners) / len(all_trades) * 100, 1) if all_trades else 0

        return {"wr": wr, "pnl": round(total_pnl, 2), "trades": len(all_trades)}

    except Exception:
        return None


def run_rotation(fetch_candles, resolve_yahoo_ticker, get_fno_stocks, progress_cb=None):
    """Run the full rotation: analyze all F&O stocks, backtest top candidates, update config.

    Args:
        fetch_candles: app.fetch_candles function
        resolve_yahoo_ticker: app.resolve_yahoo_ticker function
        get_fno_stocks: app.get_fno_stocks function
        progress_cb: optional callback(step, done, total) for progress updates
    """
    cfg = load_config()
    current_core = set(cfg.get("core_stocks", []))
    promotion_history = cfg.get("promotion_history", {})
    now = datetime.now(IST)
    month_key = now.strftime("%Y-%m")

    if progress_cb:
        progress_cb("Fetching F&O stock list", 0, 4)

    try:
        stocks = get_fno_stocks()
    except Exception:
        return {"error": "Failed to fetch F&O list"}

    # Step 1: 6-month volume + range analysis
    if progress_cb:
        progress_cb("Analyzing 6-month volume & range", 1, 4)

    def _analyze(stock):
        return _analyze_stock(stock, fetch_candles, resolve_yahoo_ticker)

    with ThreadPoolExecutor(max_workers=8) as pool:
        analyzed = list(pool.map(_analyze, stocks))

    scored = [r for r in analyzed if r is not None]

    # Rank by value traded and by range, intersect top 50
    by_value = sorted(scored, key=lambda x: x["avg_volume_cr"], reverse=True)
    by_range = sorted(scored, key=lambda x: x["avg_range_pct"], reverse=True)
    top_value_set = {r["symbol"] for r in by_value[:50]}
    top_range_set = {r["symbol"] for r in by_range[:50]}
    overlap = top_value_set & top_range_set
    candidates = sorted(
        [r for r in scored if r["symbol"] in overlap],
        key=lambda x: x["avg_range_pct"],
        reverse=True,
    )

    # Step 2: Backtest top candidates
    if progress_cb:
        progress_cb("Running 60-day PDH/PDL backtests", 2, 4)

    bt_symbols = [c["symbol"] for c in candidates[:30]]
    for sym in current_core:
        if sym not in bt_symbols:
            bt_symbols.append(sym)

    def _backtest(sym):
        return sym, _backtest_pdhl_stock(sym, fetch_candles, resolve_yahoo_ticker)

    with ThreadPoolExecutor(max_workers=6) as pool:
        bt_results_raw = list(pool.map(_backtest, bt_symbols))

    bt_results = {sym: bt for sym, bt in bt_results_raw if bt is not None}

    # Step 3: Apply demotion / promotion rules
    if progress_cb:
        progress_cb("Applying rotation rules", 3, 4)

    # Update promotion history — track which stocks appear in candidates each month
    for c in candidates[:MAX_CORE]:
        sym = c["symbol"]
        if sym not in promotion_history:
            promotion_history[sym] = []
        if month_key not in promotion_history[sym]:
            promotion_history[sym].append(month_key)
        # Keep only last 6 months
        promotion_history[sym] = promotion_history[sym][-6:]

    # Build lookup for volume/range data of analyzed stocks
    analyzed_lookup = {r["symbol"]: r for r in scored}

    demoted = []
    for sym in list(current_core):
        reasons = []

        # Check backtest deterioration
        bt = bt_results.get(sym)
        if bt is not None:
            if bt["wr"] < DEMOTE_WR_THRESHOLD:
                reasons.append(f"WR={bt['wr']}% (below {DEMOTE_WR_THRESHOLD}%)")
            if bt["pnl"] < DEMOTE_PNL_THRESHOLD:
                reasons.append(f"PnL={bt['pnl']}% (negative)")

        # Check if stock dropped out of the volume+range overlap
        if sym not in overlap:
            stock_data = analyzed_lookup.get(sym)
            if stock_data:
                in_top_vol = sym in top_value_set
                in_top_range = sym in top_range_set
                if not in_top_vol and not in_top_range:
                    reasons.append(f"Not in top-50 volume ({stock_data['avg_volume_cr']}Cr) or range ({stock_data['avg_range_pct']}%)")
                elif not in_top_vol:
                    reasons.append(f"Volume dropped: {stock_data['avg_volume_cr']}Cr (not in top-50 value)")
                elif not in_top_range:
                    reasons.append(f"Range compressed: {stock_data['avg_range_pct']}% (not in top-50 range)")
            else:
                reasons.append("No analysis data (delisted or insufficient history)")

        if reasons:
            demoted.append({"symbol": sym, "reason": "; ".join(reasons)})
            current_core.discard(sym)

    promoted = []
    for c in candidates:
        if len(current_core) >= MAX_CORE:
            break
        sym = c["symbol"]
        if sym in current_core:
            continue
        bt = bt_results.get(sym)
        if bt is None or bt["wr"] < DEMOTE_WR_THRESHOLD or bt["pnl"] < DEMOTE_PNL_THRESHOLD:
            continue
        months_seen = promotion_history.get(sym, [])
        recent_months = [m for m in months_seen if m >= (now - timedelta(days=90)).strftime("%Y-%m")]
        if len(recent_months) >= PROMOTE_MONTHS_REQUIRED or cfg.get("last_rotation") is None:
            promoted.append({"symbol": sym, "reason": f"WR={bt['wr']}%, PnL={bt['pnl']}%, months_in_top={len(recent_months)}"})
            current_core.add(sym)

    # Build new backtest stats for all current core stocks
    new_backtest_stats = {}
    for sym in current_core:
        bt = bt_results.get(sym)
        if bt:
            new_backtest_stats[sym] = {"wr": bt["wr"], "pnl": bt["pnl"]}
        else:
            old_bt = cfg.get("backtest_stats", {}).get(sym)
            if old_bt:
                new_backtest_stats[sym] = old_bt

    rotation_entry = {
        "date": now.isoformat(),
        "stocks_analyzed": len(scored),
        "candidates_in_overlap": len(candidates),
        "backtested": len(bt_results),
        "demoted": demoted,
        "promoted": promoted,
        "final_core": sorted(current_core),
    }

    rotation_log = cfg.get("rotation_log", [])
    rotation_log.append(rotation_entry)
    rotation_log = rotation_log[-12:]

    cfg["last_rotation"] = now.isoformat()
    cfg["core_stocks"] = sorted(current_core)
    cfg["backtest_stats"] = new_backtest_stats
    cfg["promotion_history"] = promotion_history
    cfg["rotation_log"] = rotation_log
    save_config(cfg)

    if progress_cb:
        progress_cb("Rotation complete", 4, 4)

    return rotation_entry
