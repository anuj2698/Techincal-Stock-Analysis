#!/usr/bin/env python3
"""Flask web app for Indian stock analysis — buy/sell zones, chart patterns, swing setups."""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from concurrent.futures import ThreadPoolExecutor

from analyzer import full_analysis, TIMEFRAME_CONFIG, detect_chart_patterns, detect_ema_crossovers, resample_weekly
from oi_fetcher import fetch_oi
from prediction_logger import log_prediction, get_predictions, get_all_predictions
from backtester import run_backtest, backtest_logged_predictions

load_dotenv()

import threading

# ---------------------------------------------------------------------------
# Background scanner
# ---------------------------------------------------------------------------
_scan_lock = threading.Lock()
_scan_running = False
_scan_progress = {"done": 0, "total": 0, "status": "idle"}


def _run_background_scan():
    global _scan_running, _scan_progress
    _scan_progress = {"done": 0, "total": 0, "status": "running"}
    try:
        import subprocess, sys
        proc = subprocess.Popen(
            [sys.executable, "scan_backtests.py"],
            cwd=str(Path(__file__).resolve().parent),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            if "/" in line and "]" in line:
                try:
                    part = line.split("]")[0].split("[")[1]
                    done, total = part.split("/")
                    _scan_progress["done"] = int(done)
                    _scan_progress["total"] = int(total)
                except (IndexError, ValueError):
                    pass
        proc.wait()
        _scan_progress["status"] = "done"
    except Exception as e:
        _scan_progress["status"] = f"error: {e}"
    finally:
        with _scan_lock:
            _scan_running = False


def _ensure_scan_cache():
    """Start a background scan if cache is missing or older than 24 hours. Returns True if cache is usable."""
    global _scan_running
    cache_path = Path(__file__).resolve().parent / "scanner_cache" / "backtest_results.json"
    if cache_path.exists():
        try:
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=IST)
            age_hours = (datetime.now(IST) - mtime).total_seconds() / 3600
            if age_hours < 24:
                return True
        except Exception:
            pass

    with _scan_lock:
        if _scan_running:
            return cache_path.exists()
        _scan_running = True

    t = threading.Thread(target=_run_background_scan, daemon=True)
    t.start()
    return cache_path.exists()


# ---------------------------------------------------------------------------
# Fyers API client (lazy init)
# ---------------------------------------------------------------------------
_fyers_client = None


def _get_fyers():
    global _fyers_client
    if _fyers_client is not None:
        return _fyers_client
    app_id = os.environ.get("FYERS_APP_ID")
    token = os.environ.get("FYERS_ACCESS_TOKEN")
    if not app_id or not token:
        return None
    try:
        from fyers_apiv3 import fyersModel
        _fyers_client = fyersModel.FyersModel(
            client_id=app_id, token=token, is_async=False, log_path="logs/",
        )
        os.makedirs("logs", exist_ok=True)
        return _fyers_client
    except Exception:
        return None


INTERVAL_TO_FYERS = {"1d": "D", "1h": "60", "15m": "15", "5m": "5", "1m": "1"}
PERIOD_TO_DAYS = {"1mo": 30, "2mo": 60, "60d": 60, "3mo": 90, "6mo": 180, "1y": 365}


def _fyers_symbol(canonical: str) -> str | None:
    """Convert canonical symbol to Fyers format. Returns None for unsupported symbols."""
    if canonical.startswith("NSE:") or canonical.startswith("BSE:"):
        return canonical
    if canonical.startswith("^") or canonical.endswith("=F"):
        return None
    return f"NSE:{canonical}-EQ"


def _fetch_fyers(canonical: str, period: str = "1y", interval: str = "1d") -> list[list[float]]:
    """Fetch candles from Fyers API. Returns [] on failure."""
    fyers = _get_fyers()
    if not fyers:
        return []
    fyers_sym = _fyers_symbol(canonical)
    if not fyers_sym:
        return []
    resolution = INTERVAL_TO_FYERS.get(interval)
    if not resolution:
        return []
    days = PERIOD_TO_DAYS.get(period, 365)
    now = datetime.now(IST)
    range_to = now.strftime("%Y-%m-%d")
    range_from = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        resp = fyers.history(data={
            "symbol": fyers_sym,
            "resolution": resolution,
            "date_format": "1",
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": "1",
        })
        time.sleep(0.12)
        candles = resp.get("candles", [])
        if not candles:
            return []
        return [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in candles]
    except Exception:
        return []
from performance import analyze_performance, analyze_attribution

app = Flask(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def market_status(last_candle_date: str) -> dict:
    now = datetime.now(IST)
    weekday = now.weekday()
    current_time = now.hour * 60 + now.minute
    open_time = MARKET_OPEN[0] * 60 + MARKET_OPEN[1]
    close_time = MARKET_CLOSE[0] * 60 + MARKET_CLOSE[1]

    if weekday >= 5:
        return {
            "is_open": False,
            "note": f"Weekend — showing last close from {last_candle_date}",
            "price_label": "Last Close",
        }

    if current_time < open_time:
        return {
            "is_open": False,
            "note": f"Pre-market — showing last close from {last_candle_date}",
            "price_label": "Prev Close",
        }

    if current_time > close_time:
        return {
            "is_open": False,
            "note": f"Market closed — showing close from {last_candle_date}",
            "price_label": "Close",
        }

    return {
        "is_open": True,
        "note": f"Market open — showing last completed close from {last_candle_date}",
        "price_label": "Last Close",
    }


INDEX_MAP = {
    "NIFTY": ("NIFTY 50", "^NSEI"),
    "NIFTY 50": ("NIFTY 50", "^NSEI"),
    "NIFTY50": ("NIFTY 50", "^NSEI"),
    "SENSEX": ("SENSEX", "^BSESN"),
    "BANKNIFTY": ("BANK NIFTY", "^NSEBANK"),
    "BANK NIFTY": ("BANK NIFTY", "^NSEBANK"),
    "NIFTYBANK": ("BANK NIFTY", "^NSEBANK"),
    "CRUDE": ("Crude Oil", "CL=F"),
    "CRUDE OIL": ("Crude Oil", "CL=F"),
    "CRUDEOIL": ("Crude Oil", "CL=F"),
    "GOLD": ("Gold", "GC=F"),
    "SILVER": ("Silver", "SI=F"),
    "NIFTYIT": ("NIFTY IT", "^CNXIT"),
    "NIFTY IT": ("NIFTY IT", "^CNXIT"),
}


def resolve_yahoo_ticker(symbol: str) -> tuple[str, str]:
    sym = symbol.strip().upper()

    idx = INDEX_MAP.get(sym)
    if idx:
        return idx[0], idx[1]

    if sym.startswith("^") or sym.endswith("=F"):
        return sym, sym

    if sym.endswith(".NS") or sym.endswith(".BO"):
        return sym, sym

    if sym.startswith("BSE:"):
        name = sym[4:]
        for suffix in ("-A", "-B", "-M", "-Z", "-X", "-XT", "-P", "-T"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return sym, name + ".BO"

    if sym.startswith("NSE:"):
        name = sym[4:]
    else:
        name = sym
        sym = f"NSE:{name}-EQ"

    for suffix in ("-EQ", "-BE", "-SM", "-ST"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    return sym, name + ".NS"


def _cache_path(yahoo_sym: str, interval: str = "1d") -> Path:
    safe = yahoo_sym.replace(".", "_").replace("^", "_")
    suffix = f"_{interval}" if interval != "1d" else ""
    return CACHE_DIR / f"{safe}{suffix}.json"


def _save_cache(yahoo_sym: str, candles: list[list[float]], interval: str = "1d") -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(yahoo_sym, interval)
    path.write_text(json.dumps({"ts": datetime.now(IST).isoformat(), "candles": candles}))


def _load_cache(yahoo_sym: str, interval: str = "1d") -> list[list[float]] | None:
    path = _cache_path(yahoo_sym, interval)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("candles")
    except Exception:
        return None


def fetch_candles(yahoo_sym: str, period: str = "1y", interval: str = "1d", canonical: str | None = None) -> list[list[float]]:
    # Try Fyers first (faster, no rate limits)
    if canonical:
        fyers_rows = _fetch_fyers(canonical, period=period, interval=interval)
        if fyers_rows:
            _save_cache(yahoo_sym, fyers_rows, interval)
            return fyers_rows

    # Fallback to Yahoo Finance
    tkr = yf.Ticker(yahoo_sym)
    df = tkr.history(period=period, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        cached = _load_cache(yahoo_sym, interval)
        return cached if cached else []
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    rows: list[list[float]] = []
    for idx, row in df.iterrows():
        ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else 0
        o = float(row["Open"])
        h = float(row["High"])
        l_val = float(row["Low"])
        c = float(row["Close"])
        v = float(row["Volume"]) if row["Volume"] == row["Volume"] else 0.0
        rows.append([ts, o, h, l_val, c, v])
    rows.sort(key=lambda r: r[0])

    if not rows:
        cached = _load_cache(yahoo_sym, interval)
        return cached if cached else []

    cached = _load_cache(yahoo_sym, interval)
    if cached and len(cached) > 0:
        cached_last_ts = cached[-1][0]
        fresh_last_ts = rows[-1][0]
        if cached_last_ts > fresh_last_ts:
            merged_map: dict[int, list[float]] = {}
            for r in cached:
                merged_map[int(r[0])] = r
            for r in rows:
                merged_map[int(r[0])] = r
            rows = sorted(merged_map.values(), key=lambda r: r[0])

    _save_cache(yahoo_sym, rows, interval)
    return rows


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


_stock_list_cache: list[dict] | None = None


@app.route("/stocks")
def stocks():
    global _stock_list_cache
    if _stock_list_cache is not None:
        return jsonify(_stock_list_cache)
    try:
        from nselib import capital_market
        df = capital_market.equity_list()
        nse_syms = set()
        result = []
        for _, row in df.iterrows():
            sym = str(row["SYMBOL"]).strip()
            name = str(row["NAME OF COMPANY"]).strip()
            if sym and name:
                result.append({"s": sym, "n": name})
                nse_syms.add(sym.upper())

        try:
            import requests as _req
            bse_url = "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Atea=&segment=Equity&status=Active"
            bse_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com/"}
            bse_resp = _req.get(bse_url, headers=bse_headers, timeout=10)
            if bse_resp.status_code == 200:
                for s in bse_resp.json():
                    sym = str(s.get("scrip_id", "")).strip().upper()
                    mktcap = float(s.get("Mktcap", 0) or 0)
                    if sym and sym not in nse_syms and mktcap > 500:
                        result.append({"s": f"BSE:{sym}", "n": s.get("Scrip_Name", sym)})
        except Exception:
            pass

        indices = [
            {"s": "NIFTY 50", "n": "Nifty 50 Index"},
            {"s": "SENSEX", "n": "BSE Sensex Index"},
            {"s": "BANKNIFTY", "n": "Bank Nifty Index"},
            {"s": "NIFTY IT", "n": "Nifty IT Index"},
            {"s": "CRUDE OIL", "n": "Crude Oil (WTI Futures)"},
            {"s": "GOLD", "n": "Gold (COMEX Futures)"},
            {"s": "SILVER", "n": "Silver (COMEX Futures)"},
        ]
        result.extend(indices)

        result.sort(key=lambda x: x["s"])
        _stock_list_cache = result
        return jsonify(result)
    except Exception:
        return jsonify([])


@app.route("/analyze")
def analyze():
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"error": "Please enter a stock symbol"}), 400

    timeframe = request.args.get("timeframe", "positional").strip()
    if timeframe not in ("short_term", "positional", "intraday"):
        timeframe = "positional"
    entry_price_str = request.args.get("entry_price", "").strip()
    entry_price = float(entry_price_str) if entry_price_str else None

    cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["positional"])
    interval = cfg.get("candle_interval", "1d")
    period = cfg.get("candle_period", "1y")

    try:
        canonical, yahoo_sym = resolve_yahoo_ticker(symbol)
        candles = fetch_candles(yahoo_sym, period=period, interval=interval, canonical=canonical)
        if not candles:
            return jsonify({"error": f"No data found for '{symbol}'. Try formats like: RELIANCE, TCS, INFY"}), 404

        daily_candles = None
        if timeframe == "intraday":
            daily_candles = fetch_candles(yahoo_sym, period="1y", interval="1d", canonical=canonical)

        last_ts = candles[-1][0]
        last_date = datetime.fromtimestamp(last_ts, tz=IST).strftime("%d %b %Y")

        oi_data = fetch_oi(canonical)
        result = full_analysis(canonical, candles, timeframe=timeframe, entry_price=entry_price, oi_data=oi_data, daily_candles=daily_candles)
        result["symbol"] = canonical
        result["yahoo_ticker"] = yahoo_sym
        result["candle_count"] = len(candles)
        result["last_price"] = round(candles[-1][4], 2) if candles else None
        result["last_date"] = last_date
        result["market"] = market_status(last_date)

        pred_id = log_prediction(result, canonical, timeframe, result["last_price"])
        result["prediction_id"] = pred_id
        result["prediction_history"] = get_predictions(canonical, limit=10)

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/predictions")
def predictions():
    symbol = request.args.get("symbol", "").strip()
    if symbol:
        canonical, _ = resolve_yahoo_ticker(symbol)
        return jsonify(get_predictions(canonical))
    return jsonify(get_all_predictions())


@app.route("/backtest")
def backtest():
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    timeframe = request.args.get("timeframe", "positional").strip()
    if timeframe not in ("short_term", "positional"):
        timeframe = "positional"
    period = "1y"

    try:
        canonical, yahoo_sym = resolve_yahoo_ticker(symbol)
        candles = fetch_candles(yahoo_sym, period=period, canonical=canonical)
        if not candles or len(candles) < 100:
            return jsonify({"error": "Insufficient data for backtest"}), 400

        result = run_backtest(canonical, candles, timeframe=timeframe)
        result["symbol"] = canonical
        result["timeframe"] = timeframe
        result["performance"] = analyze_performance(result.get("predictions", []))
        result["attribution"] = analyze_attribution(result.get("predictions", []))
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Backtest failed: {str(e)}"}), 500


@app.route("/backtest/update")
def backtest_update():
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400

    try:
        canonical, yahoo_sym = resolve_yahoo_ticker(symbol)
        candles = fetch_candles(yahoo_sym, period="1y", canonical=canonical)
        if not candles:
            return jsonify({"error": "No data"}), 404

        count = backtest_logged_predictions(canonical, candles)
        return jsonify({"updated": count, "symbol": canonical})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/scanner")
def scanner():
    return render_template("scanner.html")


@app.route("/scanner/picks")
def scanner_picks():
    timeframe = request.args.get("timeframe", "positional").strip()
    if timeframe not in ("intraday", "short_term", "positional"):
        timeframe = "positional"

    cache_path = Path(__file__).resolve().parent / "scanner_cache" / "backtest_results.json"
    has_cache = _ensure_scan_cache()
    if not has_cache:
        return jsonify({"scanning": True, "progress": _scan_progress, "picks": []})

    cache = json.loads(cache_path.read_text())
    timestamp = cache.get("timestamp", "")

    qualifying = []
    for key, entry in cache.get("results", {}).items():
        if entry["timeframe"] != timeframe:
            continue
        summary = entry.get("summary", {})
        win_rate = summary.get("win_rate", 0)
        profit_factor = summary.get("profit_factor", 0)
        if win_rate >= 75 and profit_factor >= 1.0:
            qualifying.append(entry)

    if not qualifying:
        return jsonify({"picks": [], "timestamp": timestamp, "qualifying_count": 0})

    # Fetch Nifty 50 returns for relative strength
    nifty_rs = 0.0
    try:
        nifty_candles = fetch_candles("^NSEI", period="3mo", interval="1d")
        if nifty_candles and len(nifty_candles) >= 2:
            nifty_rs = (nifty_candles[-1][4] - nifty_candles[0][4]) / nifty_candles[0][4] * 100
    except Exception:
        pass

    cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["positional"])
    interval = cfg.get("candle_interval", "1d")
    period = cfg.get("candle_period", "1y")

    def _analyze_stock(entry):
        sym = entry["symbol"]
        name = entry["name"]
        win_rate = entry["summary"]["win_rate"]
        profit_factor = entry["summary"].get("profit_factor", 0)
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)
            candles = fetch_candles(yahoo_sym, period=period, interval=interval, canonical=canonical)
            if not candles:
                return None

            daily_candles = None
            if timeframe == "intraday":
                daily_candles = fetch_candles(yahoo_sym, period="1y", interval="1d", canonical=canonical)

            result = full_analysis(canonical, candles, timeframe=timeframe, daily_candles=daily_candles)
            ap = result.get("recommendation", {}).get("action_plan", {})
            buy = ap.get("buy", {})
            chart_patterns = result.get("chart_patterns", [])

            conviction = buy.get("conviction", {})
            conv_score = conviction.get("score", 0)
            entry_conf = buy.get("entry_confirmation", {})
            conf_confirmed = entry_conf.get("confirmed", False)

            bullish_patterns = [p for p in chart_patterns if p.get("bias") in ("bullish", "neutral") and p.get("breakout_target_up")]
            has_breakout = len(bullish_patterns) > 0

            verdict = ap.get("action_summary", {}).get("verdict", "WAIT")
            if verdict == "WAIT":
                return None

            cmp = round(candles[-1][4], 2)
            buy_level = buy.get("level")
            if not buy_level:
                return None
            distance_pct = round((buy_level - cmp) / cmp * 100, 2)

            # Relative strength vs Nifty
            rs = 0.0
            if len(candles) >= 60:
                lookback = min(63, len(candles) - 1)
                stock_return = (candles[-1][4] - candles[-lookback][4]) / candles[-lookback][4] * 100
                rs = round(stock_return - nifty_rs, 2)

            # Composite ranking: conviction 30% + RS 20% + breakout 20% + confirmation 15% + PF 15%
            rs_score = max(0, min(100, 50 + rs * 2))
            breakout_score = 100 if has_breakout else 0
            conf_score = 100 if conf_confirmed else 0
            pf_score = max(0, min(100, profit_factor * 40))
            rank_score = round(conv_score * 0.30 + rs_score * 0.20 + breakout_score * 0.20 + conf_score * 0.15 + pf_score * 0.15, 1)

            pattern_names = [p["name"] for p in bullish_patterns]

            # EMA crossover info
            ema_data = ap.get("ema_crossovers", {})
            ema_alignment = ema_data.get("alignment", "neutral")
            ema_crosses = [c["type"] for c in ema_data.get("crossovers", []) if c["bias"] == "bullish"]

            # Position sizing (2% risk per trade on 10K capital)
            capital = 10000.0
            risk_per_trade = capital * 0.02
            buy_sl = buy.get("sl")
            risk_per_share = abs(buy_level - buy_sl) if buy_sl else 0
            suggested_qty = int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
            capital_required = round(buy_level * suggested_qty, 2) if suggested_qty > 0 else 0

            return {
                "symbol": canonical,
                "name": name,
                "cmp": cmp,
                "buy_level": buy_level,
                "buy_distance_pct": distance_pct,
                "sl": buy.get("sl"),
                "targets": buy.get("targets", []),
                "conviction": conviction,
                "rr": buy.get("rr"),
                "confluence_count": buy.get("confluence_count", 0),
                "trend_warning": buy.get("trend_warning"),
                "status": ap.get("status"),
                "volume_environment": ap.get("volume_environment"),
                "win_rate": win_rate,
                "profit_factor": round(profit_factor, 2) if profit_factor else None,
                "relative_strength": rs,
                "patterns": pattern_names,
                "rank_score": rank_score,
                "entry_confirmed": conf_confirmed,
                "confirmation_signals": entry_conf.get("signals", []),
                "ema_alignment": ema_alignment,
                "ema_crossovers": ema_crosses,
                "suggested_qty": suggested_qty,
                "capital_required": capital_required,
                "risk_per_trade": round(risk_per_trade, 2),
                "verdict": ap.get("action_summary", {}).get("verdict", "WAIT"),
                "verdict_confidence": ap.get("action_summary", {}).get("confidence", "—"),
                "verdict_reasons": ap.get("action_summary", {}).get("reasons", [])[:3],
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=5) as pool:
        picks_raw = list(pool.map(_analyze_stock, qualifying))

    picks = [p for p in picks_raw if p is not None]
    picks.sort(key=lambda p: p.get("rank_score", 0), reverse=True)
    picks = picks[:20]

    return jsonify({
        "picks": picks,
        "timestamp": timestamp,
        "qualifying_count": len(qualifying),
    })


@app.route("/scanner/status")
def scanner_status():
    cache_path = Path(__file__).resolve().parent / "scanner_cache" / "backtest_results.json"
    return jsonify({
        "scanning": _scan_running,
        "progress": _scan_progress,
        "has_cache": cache_path.exists(),
    })


@app.route("/patterns")
def patterns_page():
    return render_template("patterns.html")


@app.route("/patterns/scan")
def patterns_scan():
    tf = request.args.get("timeframe", "daily").strip()
    if tf not in ("hourly", "daily", "weekly"):
        tf = "daily"

    try:
        from nselib import capital_market
        df = capital_market.fno_equity_list()
        stocks = [{"symbol": row["symbol"], "name": row.get("underlying", row["symbol"])} for _, row in df.iterrows()]
    except Exception as e:
        return jsonify({"error": f"Failed to fetch F&O list: {e}", "results": []})

    interval_map = {"hourly": ("1h", "60d"), "daily": ("1d", "1y"), "weekly": ("1d", "1y")}
    interval, period = interval_map[tf]

    def _scan_stock(stock):
        sym = stock["symbol"]
        name = stock["name"]
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)
            candles = fetch_candles(yahoo_sym, period=period, interval=interval, canonical=canonical)
            if not candles or len(candles) < 30:
                return None

            if tf == "weekly":
                candles = resample_weekly(candles)
                if len(candles) < 20:
                    return None

            cmp = round(float(candles[-1][4]), 2)
            patterns = detect_chart_patterns(candles)
            if not patterns:
                return None

            closes = [float(c[4]) for c in candles]
            ema_data = detect_ema_crossovers(closes)

            from analyzer import rsi as _rsi
            d_rsi = _rsi(closes)

            bullish_pats = [p for p in patterns if p.get("bias") == "bullish"]
            bearish_pats = [p for p in patterns if p.get("bias") == "bearish"]
            ema_align = ema_data.get("alignment", "neutral")
            bull_crosses = [c for c in ema_data.get("crossovers", []) if c["bias"] == "bullish"]
            bear_crosses = [c for c in ema_data.get("crossovers", []) if c["bias"] == "bearish"]

            score_buy = len(bullish_pats) * 2
            score_sell = len(bearish_pats) * 2
            reasons = []

            if bullish_pats:
                reasons.append("Bullish pattern: " + ", ".join(p["name"] for p in bullish_pats))
            if bearish_pats:
                reasons.append("Bearish pattern: " + ", ".join(p["name"] for p in bearish_pats))

            if ema_align in ("bullish", "strong_bullish"):
                score_buy += 2
                reasons.append("EMAs aligned bullish")
            elif ema_align in ("bearish", "strong_bearish"):
                score_sell += 2
                reasons.append("EMAs aligned bearish")

            if bull_crosses:
                score_buy += 2
                reasons.append(bull_crosses[0]["type"])
            if bear_crosses:
                score_sell += 2
                reasons.append(bear_crosses[0]["type"])

            if d_rsi and d_rsi < 35:
                score_buy += 1; reasons.append(f"RSI oversold ({d_rsi:.0f})")
            elif d_rsi and d_rsi > 65:
                score_sell += 1; reasons.append(f"RSI overbought ({d_rsi:.0f})")

            if score_buy > score_sell and score_buy >= 3:
                verdict = "BUY"
                confidence = "High" if score_buy >= 6 else "Medium" if score_buy >= 4 else "Low"
            elif score_sell > score_buy and score_sell >= 3:
                verdict = "SELL"
                confidence = "High" if score_sell >= 6 else "Medium" if score_sell >= 4 else "Low"
            else:
                verdict = "WATCH"
                confidence = "—"
                reasons.append("Pattern forming — wait for breakout confirmation")

            return {
                "symbol": canonical,
                "display_symbol": sym,
                "name": name,
                "cmp": cmp,
                "patterns": patterns,
                "ema_alignment": ema_data.get("alignment", "neutral"),
                "ema_crossovers": [c["type"] for c in ema_data.get("crossovers", [])],
                "rsi": round(d_rsi, 1) if d_rsi else None,
                "verdict": verdict,
                "confidence": confidence,
                "reasons": reasons[:4],
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results_raw = list(pool.map(_scan_stock, stocks))

    results = [r for r in results_raw if r is not None]
    results.sort(key=lambda r: len(r["patterns"]), reverse=True)

    pattern_summary = {}
    for r in results:
        for p in r["patterns"]:
            name = p["name"]
            pattern_summary.setdefault(name, {"count": 0, "bullish": 0, "bearish": 0, "neutral": 0})
            pattern_summary[name]["count"] += 1
            bias = p.get("bias", "neutral")
            if bias in pattern_summary[name]:
                pattern_summary[name][bias] += 1

    return jsonify({
        "results": results,
        "total_scanned": len(stocks),
        "stocks_with_patterns": len(results),
        "pattern_summary": pattern_summary,
        "timeframe": tf,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
