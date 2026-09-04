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

from analyzer import full_analysis, TIMEFRAME_CONFIG, detect_chart_patterns, detect_ema_crossovers, resample_weekly, rsi_series, ema_series
from oi_fetcher import fetch_oi
from prediction_logger import log_prediction, get_predictions, get_all_predictions
from backtester import run_backtest, backtest_logged_predictions
from results_fetcher import get_cached_data as get_results_cached, start_background_fetch as start_results_fetch, get_fetch_status as get_results_status, is_cache_fresh as is_results_fresh

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

ADMIN_SECRET = os.environ.get("ADMIN_SECRET")


def _check_admin(req):
    """Check admin secret from header or query param. Returns error response or None if OK."""
    if not ADMIN_SECRET:
        return None
    token = req.headers.get("X-Admin-Secret") or req.args.get("admin_secret")
    if token != ADMIN_SECRET:
        return jsonify({"error": "Unauthorized"}), 401
    return None


IST = timezone(timedelta(hours=5, minutes=30))
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
FNO_CACHE_FILE = CACHE_DIR / "fno_list.json"
FNO_CACHE_MAX_AGE_DAYS = 10


def get_fno_stocks() -> list[dict]:
    """Return F&O stock list, cached to disk for 10 days."""
    if FNO_CACHE_FILE.exists():
        try:
            data = json.loads(FNO_CACHE_FILE.read_text())
            cached_at = datetime.fromisoformat(data["ts"])
            if (datetime.now(IST) - cached_at).days < FNO_CACHE_MAX_AGE_DAYS:
                return data["stocks"]
        except Exception:
            pass

    from nselib import capital_market
    df = capital_market.fno_equity_list()
    stocks = [{"symbol": row["symbol"], "name": row.get("underlying", row["symbol"])} for _, row in df.iterrows()]
    CACHE_DIR.mkdir(exist_ok=True)
    FNO_CACHE_FILE.write_text(json.dumps({"ts": datetime.now(IST).isoformat(), "stocks": stocks}))
    return stocks


RSI_SIGNALS_DIR = Path(__file__).resolve().parent / "rsi_signals"
RSI_SUMMARY_FILE = RSI_SIGNALS_DIR / "_summary.json"


def _rsi_signal_path(symbol: str) -> Path:
    safe = symbol.replace(":", "_").replace("/", "_") + ".json"
    return RSI_SIGNALS_DIR / safe


def _rsi_signal_key(ce: dict) -> str:
    return f"{ce['date_key']}|{ce['time']}|{ce['type']}"


def save_rsi_signals(results: list[dict]) -> None:
    """Persist RSI extreme signals from scan results, deduplicating by date+time+type."""
    RSI_SIGNALS_DIR.mkdir(exist_ok=True)
    now_iso = datetime.now(IST).isoformat()

    for r in results:
        symbol = r.get("symbol")
        if not symbol or not r.get("common"):
            continue
        path = _rsi_signal_path(symbol)
        existing = []
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                existing = []
        existing_keys = {s.get("key") for s in existing}

        added = False
        for ce in r["common"]:
            key = _rsi_signal_key(ce)
            if key in existing_keys:
                continue
            record = {
                "key": key,
                "date": ce.get("date"),
                "date_key": ce.get("date_key"),
                "time": ce.get("time"),
                "type": ce.get("type"),
                "price": ce.get("price"),
                "rsi_5m": ce.get("rsi_5m"),
                "rsi_15m": ce.get("rsi_15m"),
                "move": ce.get("move"),
                "next30": ce.get("next30"),
                "ema8_trade": ce.get("ema8_trade"),
                "recorded_at": now_iso,
            }
            existing.append(record)
            existing_keys.add(key)
            added = True

        if added:
            path.write_text(json.dumps(existing, indent=2, default=str))

    _rebuild_rsi_summary()


def _rebuild_rsi_summary() -> None:
    """Recompute per-stock summary from all signal files."""
    if not RSI_SIGNALS_DIR.exists():
        return
    summary = {}
    now_iso = datetime.now(IST).isoformat()

    for path in RSI_SIGNALS_DIR.glob("*.json"):
        if path.name.startswith("_"):
            continue
        try:
            signals = json.loads(path.read_text())
        except Exception:
            continue
        if not signals:
            continue

        symbol = None
        total = 0
        reversed_count = 0
        e8_wins = 0
        e8_losses = 0
        last_date = ""

        for s in signals:
            total += 1
            move = s.get("move")
            if move:
                if move.get("reversed"):
                    reversed_count += 1
            e8 = s.get("ema8_trade")
            if e8:
                if e8.get("result") == "target":
                    e8_wins += 1
                elif e8.get("result") == "sl":
                    e8_losses += 1
            dk = s.get("date_key", "")
            if dk > last_date:
                last_date = dk

        fname = path.stem.replace("_", ":", 1)
        for orig_char, safe_char in [(":", "_"), ("/", "_")]:
            pass
        sym_key = path.stem
        if sym_key.startswith("NSE_"):
            sym_key = "NSE:" + sym_key[4:]
        elif sym_key.startswith("BSE_"):
            sym_key = "BSE:" + sym_key[4:]

        e8_decided = e8_wins + e8_losses
        summary[sym_key] = {
            "total": total,
            "reversed": reversed_count,
            "continued": total - reversed_count,
            "reversal_pct": round(reversed_count / total * 100, 1) if total > 0 else 0,
            "ema8_wins": e8_wins,
            "ema8_losses": e8_losses,
            "ema8_win_pct": round(e8_wins / e8_decided * 100, 1) if e8_decided > 0 else None,
            "last_signal": last_date,
            "last_updated": now_iso,
        }

    RSI_SUMMARY_FILE.write_text(json.dumps(summary, indent=2))


def load_rsi_summary() -> dict:
    """Load the stored per-stock RSI signal performance summary."""
    if RSI_SUMMARY_FILE.exists():
        try:
            return json.loads(RSI_SUMMARY_FILE.read_text())
        except Exception:
            pass
    return {}


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

            rr = buy.get("rr") or 0
            if rr < 1.0:
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
    if tf not in ("daily", "weekly", "monthly"):
        tf = "daily"

    try:
        stocks = get_fno_stocks()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch F&O list: {e}", "results": []})

    interval_map = {"daily": ("1d", "1y"), "weekly": ("1d", "1y"), "monthly": ("1d", "1y")}
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
            elif tf == "monthly":
                from analyzer import resample_monthly as _resample_monthly
                candles = _resample_monthly(candles)
                if len(candles) < 12:
                    return None

            cmp = round(float(candles[-1][4]), 2)
            patterns = detect_chart_patterns(candles)
            if not patterns:
                return None

            closes = [float(c[4]) for c in candles]
            ema_data = detect_ema_crossovers(closes)

            from analyzer import rsi as _rsi
            d_rsi = _rsi(closes)

            ema_align = ema_data.get("alignment", "neutral")
            bull_crosses = [c for c in ema_data.get("crossovers", []) if c["bias"] == "bullish"]
            bear_crosses = [c for c in ema_data.get("crossovers", []) if c["bias"] == "bearish"]

            trend_bullish = ema_align in ("bullish", "strong_bullish")
            trend_bearish = ema_align in ("bearish", "strong_bearish")

            for p in patterns:
                bias = p.get("bias", "neutral")
                if bias == "bullish":
                    p["trend_aligned"] = trend_bullish
                elif bias == "bearish":
                    p["trend_aligned"] = trend_bearish
                else:
                    p["trend_aligned"] = True

            confirmed_bull = [p for p in patterns if p.get("bias") == "bullish" and p.get("confirmed")]
            confirmed_bear = [p for p in patterns if p.get("bias") == "bearish" and p.get("confirmed")]
            unconfirmed_bull = [p for p in patterns if p.get("bias") == "bullish" and not p.get("confirmed")]
            unconfirmed_bear = [p for p in patterns if p.get("bias") == "bearish" and not p.get("confirmed")]

            score_buy = len(confirmed_bull) * 3 + len(unconfirmed_bull) * 1
            score_sell = len(confirmed_bear) * 3 + len(unconfirmed_bear) * 1
            reasons = []

            if confirmed_bull:
                reasons.append("Confirmed bullish: " + ", ".join(p["name"] for p in confirmed_bull))
            if confirmed_bear:
                reasons.append("Confirmed bearish: " + ", ".join(p["name"] for p in confirmed_bear))
            if unconfirmed_bull:
                reasons.append("Forming bullish: " + ", ".join(p["name"] for p in unconfirmed_bull))
            if unconfirmed_bear:
                reasons.append("Forming bearish: " + ", ".join(p["name"] for p in unconfirmed_bear))

            if trend_bullish:
                score_buy += 2
                reasons.append("EMAs aligned bullish")
            elif trend_bearish:
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

            patterns.sort(key=lambda p: (p.get("confirmed", False), p.get("trend_aligned", False)), reverse=True)

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


@app.route("/results")
def results_page():
    return render_template("results.html")


@app.route("/results/data")
def results_data():
    cached = get_results_cached()
    if cached:
        cached["cache_fresh"] = is_results_fresh()
        return jsonify(cached)
    status = get_results_status()
    if status["running"]:
        return jsonify({"fetching": True, "progress": status["progress"]})
    start_results_fetch()
    return jsonify({"fetching": True, "progress": {"step": "Starting data fetch...", "done": 0, "total": 4}})


@app.route("/results/refresh", methods=["POST"])
def results_refresh():
    started = start_results_fetch()
    if started:
        return jsonify({"status": "started"})
    return jsonify({"status": "already_running"})


@app.route("/results/status")
def results_status():
    return jsonify(get_results_status())


@app.route("/today")
def today_page():
    return render_template("today.html")


@app.route("/today/picks")
def today_picks():
    timeframe = request.args.get("timeframe", "short_term").strip()
    if timeframe not in ("intraday", "short_term"):
        timeframe = "short_term"

    cache_path = Path(__file__).resolve().parent / "scanner_cache" / "backtest_results.json"
    has_cache = _ensure_scan_cache()
    if not has_cache:
        return jsonify({"scanning": True, "progress": _scan_progress, "picks": []})

    cache = json.loads(cache_path.read_text())
    timestamp = cache.get("timestamp", "")

    is_intraday = timeframe == "intraday"

    qualifying = []
    for key, entry in cache.get("results", {}).items():
        if entry["timeframe"] != timeframe:
            continue
        summary = entry.get("summary", {})
        win_rate = summary.get("win_rate", 0)
        profit_factor = summary.get("profit_factor", 0)
        if is_intraday:
            if win_rate >= 50 and profit_factor >= 1.5:
                qualifying.append(entry)
        else:
            if win_rate >= 70 and profit_factor >= 0.8:
                qualifying.append(entry)

    if not qualifying:
        return jsonify({"picks": [], "timestamp": timestamp, "qualifying_count": 0})

    cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["short_term"])
    interval = cfg.get("candle_interval", "1d")
    period = cfg.get("candle_period", "1y")
    max_distance_pct = 1.0

    nifty_intraday_return = 0.0
    if is_intraday:
        try:
            nifty_candles = fetch_candles("^NSEI", period="1mo", interval="1h")
            if nifty_candles and len(nifty_candles) >= 7:
                from collections import defaultdict as _dd
                _nifty_days = _dd(list)
                for c in nifty_candles:
                    _d = datetime.fromtimestamp(int(c[0]), tz=IST).strftime("%Y-%m-%d")
                    _nifty_days[_d].append(c)
                _ndays = sorted(_nifty_days.keys())
                if len(_ndays) >= 2:
                    _prev_close = _nifty_days[_ndays[-2]][-1][4]
                    _today_last = _nifty_days[_ndays[-1]][-1][4]
                    nifty_intraday_return = round((_today_last - _prev_close) / _prev_close * 100, 2)
        except Exception:
            pass

    def _intraday_context(candles):
        from collections import defaultdict as _dd
        daily_groups = _dd(list)
        for c in candles:
            dt = datetime.fromtimestamp(int(c[0]), tz=IST).strftime("%Y-%m-%d")
            daily_groups[dt].append(c)
        days = sorted(daily_groups.keys())
        if len(days) < 2:
            return None
        prev = daily_groups[days[-2]]
        today = daily_groups[days[-1]]

        pdh = max(c[2] for c in prev)
        pdl = min(c[3] for c in prev)
        pdc = prev[-1][4]
        today_open = today[0][1]
        today_high = max(c[2] for c in today)
        today_low = min(c[3] for c in today)
        today_last = today[-1][4]
        gap_pct = round((today_open - pdc) / pdc * 100, 2) if pdc else 0
        or_high = today[0][2]
        or_low = today[0][3]
        stock_return = round((today_last - pdc) / pdc * 100, 2) if pdc else 0
        rs_vs_nifty = round(stock_return - nifty_intraday_return, 2)

        closes = [c[4] for c in candles]
        atr_period = min(14, len(candles) - 1)
        trs = []
        for i in range(1, len(candles)):
            tr = max(candles[i][2] - candles[i][3], abs(candles[i][2] - candles[i-1][4]), abs(candles[i][3] - candles[i-1][4]))
            trs.append(tr)
        day_atr = sum(trs[-atr_period:]) / atr_period if trs else 0
        day_range = today_high - today_low
        range_used_pct = round(day_range / (day_atr * 7) * 100, 1) if day_atr > 0 else 0

        return {
            "pdh": round(pdh, 2), "pdl": round(pdl, 2), "pdc": round(pdc, 2),
            "today_open": round(today_open, 2),
            "gap_pct": gap_pct,
            "or_high": round(or_high, 2), "or_low": round(or_low, 2),
            "above_pdc": today_last > pdc,
            "above_pdh": today_last > pdh,
            "above_or_high": today_last > or_high,
            "stock_return": stock_return,
            "rs_vs_nifty": rs_vs_nifty,
            "range_used_pct": range_used_pct,
        }

    def _analyze_for_today(entry):
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
            if is_intraday:
                daily_candles = fetch_candles(yahoo_sym, period="1y", interval="1d", canonical=canonical)

            oi_data = fetch_oi(canonical)
            result = full_analysis(canonical, candles, timeframe=timeframe, oi_data=oi_data, daily_candles=daily_candles)
            ap = result.get("recommendation", {}).get("action_plan", {})
            buy = ap.get("buy", {})

            verdict = ap.get("action_summary", {}).get("verdict", "WAIT")
            if verdict == "WAIT":
                return None

            cmp = round(candles[-1][4], 2)
            buy_level = buy.get("level")
            if not buy_level:
                return None

            distance_pct = round((buy_level - cmp) / cmp * 100, 2)
            if distance_pct < -max_distance_pct:
                return None

            rr = buy.get("rr") or 0
            if rr < 1.0:
                return None

            conviction = buy.get("conviction", {})
            conv_score = conviction.get("score", 0)
            entry_conf = buy.get("entry_confirmation", {})
            conf_confirmed = entry_conf.get("confirmed", False)
            vol_env = ap.get("volume_environment", "normal")

            if vol_env == "low":
                return None
            if conv_score < 40:
                return None
            if buy.get("trend_warning"):
                return None

            indicators = result.get("indicators", {})
            rvol = indicators.get("rvol")

            vwap_data = indicators.get("vwap", {})
            vwap_price = vwap_data.get("vwap")
            vwap_position = vwap_data.get("position", "unknown")

            oi_result = result.get("oi", {})
            pcr = oi_result.get("pcr")
            oi_buildup = oi_result.get("oi_buildup")
            max_pain = oi_result.get("max_pain")

            intra_ctx = None
            if is_intraday:
                if rvol is not None and rvol < 0.8:
                    return None
                intra_ctx = _intraday_context(candles)
                if intra_ctx and intra_ctx["range_used_pct"] > 85:
                    return None

            ema_data = ap.get("ema_crossovers", {})
            chart_patterns = result.get("chart_patterns", [])
            bullish_patterns = [p for p in chart_patterns if p.get("bias") in ("bullish", "neutral") and p.get("breakout_target_up")]

            capital = 10000.0
            risk_per_trade = capital * 0.02
            buy_sl = buy.get("sl")
            risk_per_share = abs(buy_level - buy_sl) if buy_sl else 0
            suggested_qty = int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
            capital_required = round(buy_level * suggested_qty, 2) if suggested_qty > 0 else 0

            proximity_score = max(0, 100 - abs(distance_pct) * 100)
            conf_score_w = 100 if conf_confirmed else 0
            vol_score = 100 if vol_env == "confirmed" else 50 if vol_env == "normal" else 0
            rr_score = min(100, rr * 33)

            if is_intraday:
                vwap_score = 100 if vwap_position == "above" else 50 if vwap_position == "at" else 0
                oi_score = 0
                if oi_buildup in ("long_buildup", "short_covering"):
                    oi_score = 100
                elif pcr and pcr > 1.0:
                    oi_score = 60

                ctx_score = 0
                if intra_ctx:
                    if intra_ctx["above_pdc"]:
                        ctx_score += 40
                    if intra_ctx["rs_vs_nifty"] > 0:
                        ctx_score += 30
                    if intra_ctx["gap_pct"] > 0:
                        ctx_score += 15
                    if intra_ctx["above_or_high"]:
                        ctx_score += 15
                    ctx_score = min(100, ctx_score)

                actionability = round(
                    proximity_score * 0.15
                    + conf_score_w * 0.20
                    + vol_score * 0.15
                    + vwap_score * 0.10
                    + oi_score * 0.10
                    + ctx_score * 0.20
                    + rr_score * 0.10,
                    1,
                )
            else:
                actionability = round(
                    proximity_score * 0.30
                    + conf_score_w * 0.20
                    + vol_score * 0.15
                    + conv_score * 0.15
                    + rr_score * 0.20,
                    1,
                )

            signals = []
            if ap.get("status") == "BUY ZONE":
                signals.append("IN BUY ZONE")
            if conf_confirmed:
                signals.extend(entry_conf.get("signals", []))
            if vol_env == "confirmed":
                signals.append("Volume confirmed")
            if is_intraday:
                if intra_ctx and intra_ctx["above_pdc"]:
                    signals.append("Above prev close")
                if intra_ctx and intra_ctx["rs_vs_nifty"] > 0.5:
                    signals.append(f"RS vs Nifty +{intra_ctx['rs_vs_nifty']:.1f}%")
                if intra_ctx and intra_ctx["gap_pct"] > 0.3:
                    signals.append(f"Gap up {intra_ctx['gap_pct']:+.1f}%")
                if intra_ctx and intra_ctx["above_or_high"]:
                    signals.append("Above opening range")
                if vwap_position == "above":
                    signals.append("Above VWAP")
                if oi_buildup in ("long_buildup", "short_covering"):
                    signals.append(oi_buildup.replace("_", " ").title())
                elif pcr and pcr > 1.0:
                    signals.append(f"PCR {pcr:.2f}")
            if bullish_patterns:
                signals.extend(p["name"] for p in bullish_patterns[:2])
            ema_crosses = [c["type"] for c in ema_data.get("crossovers", []) if c["bias"] == "bullish"]
            if ema_crosses:
                signals.append(ema_crosses[0])

            pick = {
                "symbol": canonical,
                "name": name,
                "cmp": cmp,
                "buy_level": buy_level,
                "distance_pct": distance_pct,
                "sl": buy_sl,
                "targets": buy.get("targets", []),
                "rr": rr,
                "conviction": conviction,
                "entry_confirmed": conf_confirmed,
                "volume_environment": vol_env,
                "signals": signals[:6],
                "verdict": verdict,
                "verdict_confidence": ap.get("action_summary", {}).get("confidence", "—"),
                "verdict_reasons": ap.get("action_summary", {}).get("reasons", [])[:3],
                "win_rate": win_rate,
                "profit_factor": round(profit_factor, 2),
                "ema_alignment": ema_data.get("alignment", "neutral"),
                "suggested_qty": suggested_qty,
                "capital_required": capital_required,
                "actionability": actionability,
                "status": ap.get("status"),
            }

            if is_intraday:
                pick["rvol"] = round(rvol, 2) if rvol else None
                pick["vwap"] = round(vwap_price, 2) if vwap_price else None
                pick["vwap_position"] = vwap_position
                pick["pcr"] = round(pcr, 2) if pcr else None
                pick["oi_buildup"] = oi_buildup
                pick["max_pain"] = max_pain
                if intra_ctx:
                    pick["pdh"] = intra_ctx["pdh"]
                    pick["pdl"] = intra_ctx["pdl"]
                    pick["pdc"] = intra_ctx["pdc"]
                    pick["gap_pct"] = intra_ctx["gap_pct"]
                    pick["stock_return"] = intra_ctx["stock_return"]
                    pick["rs_vs_nifty"] = intra_ctx["rs_vs_nifty"]
                    pick["range_used_pct"] = intra_ctx["range_used_pct"]
                    pick["or_high"] = intra_ctx["or_high"]
                    pick["or_low"] = intra_ctx["or_low"]

            return pick
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=5) as pool:
        picks_raw = list(pool.map(_analyze_for_today, qualifying))

    picks = [p for p in picks_raw if p is not None]
    picks.sort(key=lambda p: p.get("actionability", 0), reverse=True)

    from news_fetcher import fetch_news_batch
    if picks:
        news_map = fetch_news_batch([p["symbol"] for p in picks[:10]], limit=3)
        for p in picks:
            p["news"] = news_map.get(p["symbol"], [])

    return jsonify({
        "picks": picks,
        "timestamp": timestamp,
        "qualifying_count": len(qualifying),
        "max_distance_pct": max_distance_pct,
    })


@app.route("/today/backtest")
def today_backtest():
    timeframe = request.args.get("timeframe", "short_term").strip()
    if timeframe not in ("intraday", "short_term"):
        timeframe = "short_term"

    days_back = 7
    hold_candles = 3 if timeframe == "short_term" else 7

    cache_path = Path(__file__).resolve().parent / "scanner_cache" / "backtest_results.json"
    if not cache_path.exists():
        return jsonify({"error": "No scanner cache. Run the scanner first."})

    cache = json.loads(cache_path.read_text())

    is_intraday_bt = timeframe == "intraday"
    qualifying = []
    for key, entry in cache.get("results", {}).items():
        if entry["timeframe"] != timeframe:
            continue
        summary = entry.get("summary", {})
        wr = summary.get("win_rate", 0)
        pf = summary.get("profit_factor", 0)
        if is_intraday_bt:
            if wr >= 50 and pf >= 1.5:
                qualifying.append(entry)
        else:
            if wr >= 70 and pf >= 0.8:
                qualifying.append(entry)

    if not qualifying:
        return jsonify({"error": "No qualifying stocks", "daily_results": []})

    cfg = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG["short_term"])
    interval = cfg.get("candle_interval", "1d")
    period = cfg.get("candle_period", "1y")

    def _fetch_candles(entry):
        sym = entry["symbol"]
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)
            candles = fetch_candles(yahoo_sym, period=period, interval=interval, canonical=canonical)
            if candles and len(candles) > 60:
                return sym, entry["name"], canonical, candles
        except Exception:
            pass
        return sym, entry["name"], None, None

    with ThreadPoolExecutor(max_workers=5) as pool:
        fetched = list(pool.map(_fetch_candles, qualifying))

    stock_data = [(s, n, c, cndls) for s, n, c, cndls in fetched if cndls]
    if not stock_data:
        return jsonify({"error": "No candle data", "daily_results": []})

    ref_timestamps = sorted(set(c[0] for c in stock_data[0][3]))

    if timeframe == "intraday":
        from collections import defaultdict
        daily_groups = defaultdict(list)
        for ts in ref_timestamps:
            d = datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d")
            daily_groups[d].append(ts)
        day_keys = sorted(daily_groups.keys())
        last_ts_per_day = [max(daily_groups[d]) for d in day_keys]
        sim_dates = last_ts_per_day[-(days_back + 1):-1]
    else:
        sim_dates = ref_timestamps[-(days_back + 1):-1]

    all_day_results = {ts: [] for ts in sim_dates}

    def _simulate_stock(args):
        sym, name, canonical, candles, = args
        results = {}
        for sim_ts in sim_dates:
            truncated = [c for c in candles if c[0] <= sim_ts]
            if len(truncated) < 60:
                continue
            future = [c for c in candles if c[0] > sim_ts][:hold_candles]
            if not future:
                continue

            try:
                daily_candles = None
                if timeframe == "intraday":
                    daily_candles = candles
                result = full_analysis(canonical, truncated, timeframe=timeframe, daily_candles=daily_candles)
                ap = result.get("recommendation", {}).get("action_plan", {})
                buy = ap.get("buy", {})

                verdict = ap.get("action_summary", {}).get("verdict", "WAIT")
                if verdict == "WAIT":
                    continue

                cmp = round(truncated[-1][4], 2)
                buy_level = buy.get("level")
                if not buy_level:
                    continue

                distance_pct = round((buy_level - cmp) / cmp * 100, 2)
                if distance_pct < -1.0:
                    continue

                rr = buy.get("rr") or 0
                if rr < 1.0:
                    continue

                bt_conv = buy.get("conviction", {}).get("score", 0)
                if bt_conv < 40:
                    continue
                bt_vol_env = ap.get("volume_environment", "normal")
                if bt_vol_env == "low":
                    continue
                if buy.get("trend_warning"):
                    continue

                if timeframe == "intraday":
                    indicators = result.get("indicators", {})
                    bt_rvol = indicators.get("rvol")
                    if bt_rvol is not None and bt_rvol < 0.8:
                        continue

                sl = buy.get("sl")
                targets = buy.get("targets", [])
                t1 = targets[0] if targets else None

                outcome = {"entered": False, "result": "no_trigger", "pnl_pct": None, "exit_price": None, "days_held": 0}

                entry_idx = None
                for i, fc in enumerate(future):
                    if fc[3] <= buy_level:
                        outcome["entered"] = True
                        entry_idx = i
                        break

                if outcome["entered"] and sl and t1:
                    for j in range(entry_idx, len(future)):
                        fc = future[j]
                        if fc[3] <= sl:
                            outcome["result"] = "sl_hit"
                            outcome["exit_price"] = round(sl, 2)
                            outcome["pnl_pct"] = round((sl - buy_level) / buy_level * 100, 2)
                            outcome["days_held"] = j - entry_idx + 1
                            break
                        if fc[2] >= t1:
                            outcome["result"] = "t1_hit"
                            outcome["exit_price"] = round(t1, 2)
                            outcome["pnl_pct"] = round((t1 - buy_level) / buy_level * 100, 2)
                            outcome["days_held"] = j - entry_idx + 1
                            break
                    else:
                        last_close = round(future[-1][4], 2)
                        outcome["result"] = "open"
                        outcome["exit_price"] = last_close
                        outcome["pnl_pct"] = round((last_close - buy_level) / buy_level * 100, 2)
                        outcome["days_held"] = len(future) - entry_idx
                elif outcome["entered"] and future:
                    last_close = round(future[-1][4], 2)
                    outcome["result"] = "open"
                    outcome["exit_price"] = last_close
                    outcome["pnl_pct"] = round((last_close - buy_level) / buy_level * 100, 2)

                conv = buy.get("conviction", {})
                results[sim_ts] = {
                    "symbol": canonical,
                    "name": name,
                    "cmp_at_pick": cmp,
                    "buy_level": round(buy_level, 2),
                    "distance_pct": distance_pct,
                    "sl": round(sl, 2) if sl else None,
                    "t1": round(t1, 2) if t1 else None,
                    "rr": round(rr, 1),
                    "conviction": conv.get("score", 0),
                    "outcome": outcome,
                }
            except Exception:
                continue
        return results

    with ThreadPoolExecutor(max_workers=5) as pool:
        all_stock_results = list(pool.map(_simulate_stock, stock_data))

    for stock_results in all_stock_results:
        for ts, pick in stock_results.items():
            if ts in all_day_results:
                all_day_results[ts].append(pick)

    daily_results = []
    for ts in sim_dates:
        picks = sorted(all_day_results[ts], key=lambda p: abs(p["distance_pct"]))
        date_str = datetime.fromtimestamp(ts, tz=IST).strftime("%d %b %Y (%a)")
        daily_results.append({"date": date_str, "picks": picks})

    all_picks = [p for d in daily_results for p in d["picks"]]
    entered = [p for p in all_picks if p["outcome"]["entered"]]
    wins = [p for p in entered if p["outcome"]["result"] == "t1_hit"]
    losses = [p for p in entered if p["outcome"]["result"] == "sl_hit"]
    still_open = [p for p in entered if p["outcome"]["result"] == "open"]
    total_pnl = sum(p["outcome"]["pnl_pct"] for p in entered if p["outcome"]["pnl_pct"] is not None)

    return jsonify({
        "daily_results": daily_results,
        "summary": {
            "total_picks": len(all_picks),
            "entries_triggered": len(entered),
            "wins": len(wins),
            "losses": len(losses),
            "open": len(still_open),
            "win_rate": round(len(wins) / max(len(wins) + len(losses), 1) * 100, 1),
            "total_pnl_pct": round(total_pnl, 2),
            "avg_pnl_pct": round(total_pnl / len(entered), 2) if entered else 0,
        },
        "timeframe": timeframe,
        "hold_period": "3 days" if timeframe == "short_term" else "1 day (hourly)",
    })


# ---------------------------------------------------------------------------
# Background RSI scan cache
# ---------------------------------------------------------------------------
RSI_CACHE_DIR = CACHE_DIR / "rsi_scans"
RSI_CACHE_MAX_AGE_SEC = 300  # 5 minutes during market hours

_rsi_scan_locks = {
    "today": threading.Lock(),
    "historical": threading.Lock(),
    "momentum": threading.Lock(),
}
_rsi_scan_running = {"today": False, "historical": False, "momentum": False}


def _rsi_cache_path(scan_type: str, extra: str = "") -> Path:
    return RSI_CACHE_DIR / f"{scan_type}{extra}.json"


def _rsi_cache_fresh(scan_type: str, extra: str = "") -> dict | None:
    path = _rsi_cache_path(scan_type, extra)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(data.get("_cached_at", ""))
        age = (datetime.now(IST) - cached_at).total_seconds()
        if age < RSI_CACHE_MAX_AGE_SEC:
            return data
    except Exception:
        pass
    return None


def _save_rsi_cache(scan_type: str, data: dict, extra: str = "") -> None:
    RSI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data["_cached_at"] = datetime.now(IST).isoformat()
    _rsi_cache_path(scan_type, extra).write_text(json.dumps(data))


@app.route("/rsi-extremes")
def rsi_extremes_page():
    return render_template("rsi_extremes.html")


def _do_rsi_extremes_today():
    """Run RSI extremes live scan in background, save to cache."""
    try:
        stocks = get_fno_stocks()
    except Exception:
        return

    def _scan_current(stock):
        sym = stock["symbol"]
        name = stock["name"]
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)
            cmp = None
            cur_rsi = {}
            ema_ctx = {}

            for interval in ["5m", "15m", "1h"]:
                candles = fetch_candles(yahoo_sym, period="1mo", interval=interval, canonical=canonical)
                if not candles or len(candles) < 20:
                    if interval in ("5m", "15m"):
                        return None
                    continue

                closes = [float(c[4]) for c in candles]
                rsi_vals = rsi_series(closes, 14)
                if not rsi_vals:
                    if interval in ("5m", "15m"):
                        return None
                    continue
                cur_rsi[interval] = round(rsi_vals[-1], 1)

                if interval == "5m":
                    cmp = round(float(candles[-1][4]), 2)
                    e9 = ema_series(closes, 9)
                    e20 = ema_series(closes, 20)
                    ema_ctx = {
                        "ema9": round(e9[-1], 2) if e9 else None,
                        "ema20": round(e20[-1], 2) if e20 else None,
                    }

            r5 = cur_rsi.get("5m")
            r15 = cur_rsi.get("15m")
            if r5 is None or r15 is None:
                return None

            both_ob = r5 > 80 and r15 > 80
            both_os = r5 < 20 and r15 < 20
            if not (both_ob or both_os):
                return None

            signal_type = "overbought" if both_ob else "oversold"

            return {
                "symbol": canonical,
                "display_symbol": sym,
                "name": name,
                "cmp": cmp,
                "rsi_5m": r5,
                "rsi_15m": r15,
                "rsi_1h": cur_rsi.get("1h"),
                "type": signal_type,
                "emas": ema_ctx,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        results_raw = list(pool.map(_scan_current, stocks))

    results = [r for r in results_raw if r is not None]

    def _sort_key(r):
        r5, r15 = r["rsi_5m"], r["rsi_15m"]
        if r["type"] == "overbought":
            return r15
        else:
            return -r15

    results.sort(key=_sort_key, reverse=True)

    summary = load_rsi_summary()
    for r in results:
        sp = summary.get(r["symbol"])
        if sp:
            r["stored_perf"] = sp

    _save_rsi_cache("today", {
        "results": results,
        "total_scanned": len(stocks),
        "count": len(results),
        "time": datetime.now(IST).strftime("%H:%M:%S"),
    })
    _rsi_scan_running["today"] = False


@app.route("/rsi-extremes/today")
def rsi_extremes_today():
    cached = _rsi_cache_fresh("today")
    if cached:
        cached.pop("_cached_at", None)
        return jsonify(cached)

    if not _rsi_scan_running["today"]:
        with _rsi_scan_locks["today"]:
            if not _rsi_scan_running["today"]:
                _rsi_scan_running["today"] = True
                threading.Thread(target=_do_rsi_extremes_today, daemon=True).start()

    stale = None
    path = _rsi_cache_path("today")
    if path.exists():
        try:
            stale = json.loads(path.read_text())
            stale.pop("_cached_at", None)
            stale["_stale"] = True
            return jsonify(stale)
        except Exception:
            pass

    return jsonify({"scanning": True, "results": [], "count": 0, "time": datetime.now(IST).strftime("%H:%M:%S")})


def _do_rsi_extremes_scan(scan_days: int):
    """Run RSI extremes historical scan in background, save to cache."""
    candle_period = "2mo" if scan_days > 15 else "1mo"

    try:
        stocks = get_fno_stocks()
    except Exception:
        _rsi_scan_running["historical"] = False
        return

    def _find_daily_extremes(candles, last_10_days):
        closes = [float(c[4]) for c in candles]
        rsi_vals = rsi_series(closes, 14)
        if not rsi_vals:
            return [], None

        offset = len(closes) - len(rsi_vals)
        day_overbought = {}
        day_oversold = {}

        for j, rv in enumerate(rsi_vals):
            ci = j + offset
            c = candles[ci]
            dt = datetime.fromtimestamp(int(c[0]), tz=IST)
            day_str = dt.strftime("%Y-%m-%d")

            if day_str not in last_10_days:
                continue

            if rv > 80:
                if day_str not in day_overbought or rv > day_overbought[day_str]["rsi"]:
                    day_overbought[day_str] = {
                        "date": dt.strftime("%d %b"),
                        "date_key": day_str,
                        "time": dt.strftime("%H:%M"),
                        "rsi": round(rv, 1),
                        "price": round(float(c[4]), 2),
                        "type": "overbought",
                    }
            elif rv < 20:
                if day_str not in day_oversold or rv < day_oversold[day_str]["rsi"]:
                    day_oversold[day_str] = {
                        "date": dt.strftime("%d %b"),
                        "date_key": day_str,
                        "time": dt.strftime("%H:%M"),
                        "rsi": round(rv, 1),
                        "price": round(float(c[4]), 2),
                        "type": "oversold",
                    }

        extremes = list(day_overbought.values()) + list(day_oversold.values())
        extremes.sort(key=lambda e: e["date_key"], reverse=True)
        return extremes, round(rsi_vals[-1], 1)

    def _build_rsi_lookup(candles, last_10_days):
        """Return {timestamp: (rsi_val, price, datetime_obj)} for candles in last_10_days."""
        closes = [float(c[4]) for c in candles]
        rsi_vals = rsi_series(closes, 14)
        if not rsi_vals:
            return {}, None
        offset = len(closes) - len(rsi_vals)
        lookup = {}
        for j, rv in enumerate(rsi_vals):
            ci = j + offset
            c = candles[ci]
            ts = int(c[0])
            dt = datetime.fromtimestamp(ts, tz=IST)
            if dt.strftime("%Y-%m-%d") in last_10_days:
                lookup[ts] = (rv, round(float(c[4]), 2), dt)
        return lookup, round(rsi_vals[-1], 1)

    def _scan_stock(stock):
        sym = stock["symbol"]
        name = stock["name"]
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)
            display_sym = sym

            result = {"symbol": canonical, "display_symbol": display_sym, "name": name}
            has_any = False
            cmp = None
            rsi_lookups = {}
            candles_by_tf = {}

            for interval in ["5m", "15m"]:
                candles = fetch_candles(yahoo_sym, period=candle_period, interval=interval, canonical=canonical)
                if not candles or len(candles) < 20:
                    result[interval] = {"extremes": [], "current_rsi": None, "overbought_count": 0, "oversold_count": 0}
                    continue

                candles_by_tf[interval] = candles

                if cmp is None:
                    cmp = round(float(candles[-1][4]), 2)

                all_dates = sorted(set(
                    datetime.fromtimestamp(int(c[0]), tz=IST).strftime("%Y-%m-%d")
                    for c in candles
                ))
                last_n = set(all_dates[-scan_days:]) if len(all_dates) >= scan_days else set(all_dates)

                extremes, current_rsi = _find_daily_extremes(candles, last_n)
                lookup, _ = _build_rsi_lookup(candles, last_n)
                rsi_lookups[interval] = lookup

                result[interval] = {
                    "extremes": extremes,
                    "current_rsi": current_rsi,
                    "overbought_count": sum(1 for e in extremes if e["type"] == "overbought"),
                    "oversold_count": sum(1 for e in extremes if e["type"] == "oversold"),
                }

                if extremes:
                    has_any = True

            if not has_any:
                return None

            # Find common extremes: both 5m and 15m extreme at overlapping times
            common_extremes = []
            rsi_5m = rsi_lookups.get("5m", {})
            rsi_15m = rsi_lookups.get("15m", {})

            if rsi_5m and rsi_15m:
                for ts_15m, (rv_15m, price_15m, dt_15m) in sorted(rsi_15m.items()):
                    if 20 <= rv_15m <= 80:
                        continue
                    hhmm = dt_15m.hour * 60 + dt_15m.minute
                    if hhmm < 9 * 60 + 20 or hhmm >= 15 * 60:
                        continue
                    extreme_type = "overbought" if rv_15m > 80 else "oversold"

                    for offset_sec in range(0, 15 * 60, 5 * 60):
                        ts_5m = ts_15m + offset_sec
                        if ts_5m not in rsi_5m:
                            continue
                        rv_5m, price_5m, dt_5m = rsi_5m[ts_5m]
                        match = (extreme_type == "overbought" and rv_5m > 80) or \
                                (extreme_type == "oversold" and rv_5m < 20)
                        if match:
                            common_extremes.append({
                                "date": dt_15m.strftime("%d %b"),
                                "date_key": dt_15m.strftime("%Y-%m-%d"),
                                "time": dt_5m.strftime("%H:%M"),
                                "rsi_5m": round(rv_5m, 1),
                                "rsi_15m": round(rv_15m, 1),
                                "price": price_5m,
                                "type": extreme_type,
                                "_ts_5m": ts_5m,
                            })
                            break

                # Deduplicate to peak per day per direction
                common_by_day = {}
                for ce in common_extremes:
                    key = (ce["date_key"], ce["type"])
                    if key not in common_by_day:
                        common_by_day[key] = ce
                    elif ce["type"] == "overbought" and ce["rsi_15m"] > common_by_day[key]["rsi_15m"]:
                        common_by_day[key] = ce
                    elif ce["type"] == "oversold" and ce["rsi_15m"] < common_by_day[key]["rsi_15m"]:
                        common_by_day[key] = ce
                common_extremes = sorted(common_by_day.values(), key=lambda e: e["date_key"], reverse=True)

            # Forward movement: scan 5m candles until price moves 1% in either direction
            candles_5m = candles_by_tf.get("5m", [])
            if candles_5m and common_extremes:
                ts_to_idx = {int(c[0]): i for i, c in enumerate(candles_5m)}
                for ce in common_extremes:
                    ts = ce.pop("_ts_5m", None)
                    if ts is None:
                        continue
                    idx = ts_to_idx.get(ts)
                    if idx is None:
                        continue
                    entry_price = float(candles_5m[idx][4])
                    target_up = entry_price * 1.01
                    target_down = entry_price * 0.99

                    hit_dir = None
                    hit_price = None
                    hit_time = None
                    hit_date = None
                    candles_scanned = 0
                    max_high = entry_price
                    max_low = entry_price

                    for k in range(1, len(candles_5m) - idx):
                        fc = candles_5m[idx + k]
                        candles_scanned += 1
                        fh = float(fc[2])
                        fl = float(fc[3])
                        if fh > max_high:
                            max_high = fh
                        if fl < max_low:
                            max_low = fl

                        up_hit = fh >= target_up
                        down_hit = fl <= target_down
                        fc_dt = datetime.fromtimestamp(int(fc[0]), tz=IST)
                        if up_hit and down_hit:
                            fc_o = float(fc[1])
                            fc_c = float(fc[4])
                            hit_dir = "down" if fc_c < fc_o else "up"
                            hit_price = round(target_down if hit_dir == "down" else target_up, 2)
                            hit_time = fc_dt.strftime("%H:%M")
                            hit_date = fc_dt.strftime("%d %b")
                            break
                        elif up_hit:
                            hit_dir = "up"
                            hit_price = round(target_up, 2)
                            hit_time = fc_dt.strftime("%H:%M")
                            hit_date = fc_dt.strftime("%d %b")
                            break
                        elif down_hit:
                            hit_dir = "down"
                            hit_price = round(target_down, 2)
                            hit_time = fc_dt.strftime("%H:%M")
                            hit_date = fc_dt.strftime("%d %b")
                            break

                    # OB expects down move → down hit = reversed
                    # OS expects up move → up hit = reversed
                    if hit_dir:
                        if ce["type"] == "overbought":
                            reversed_ok = hit_dir == "down"
                        else:
                            reversed_ok = hit_dir == "up"
                    else:
                        reversed_ok = False

                    ce["move"] = {
                        "entry": round(entry_price, 2),
                        "max_high": round(max_high, 2),
                        "max_low": round(max_low, 2),
                        "hit_1pct": hit_dir is not None,
                        "hit_dir": hit_dir,
                        "hit_price": hit_price,
                        "hit_date": hit_date,
                        "hit_time": hit_time,
                        "candles": candles_scanned,
                        "minutes": candles_scanned * 5,
                        "reversed": reversed_ok,
                    }

                    # Next 30 minutes high/low (6 five-minute candles)
                    next30_candles = candles_5m[idx + 1: idx + 7]
                    if next30_candles:
                        n30_high = max(float(fc[2]) for fc in next30_candles)
                        n30_low = min(float(fc[3]) for fc in next30_candles)
                        ce["next30"] = {
                            "high": round(n30_high, 2),
                            "low": round(n30_low, 2),
                            "high_pct": round((n30_high - entry_price) / entry_price * 100, 2),
                            "low_pct": round((n30_low - entry_price) / entry_price * 100, 2),
                            "candles": len(next30_candles),
                        }

                    # EMA observation — how far does the reversal go relative to EMAs?
                    closes_up_to = [float(c[4]) for c in candles_5m[:idx + 1]]
                    if len(closes_up_to) >= 20:
                        e8 = ema_series(closes_up_to, 8)
                        e9 = ema_series(closes_up_to, 9)
                        e20 = ema_series(closes_up_to, 20)
                        ema8_val = round(e8[-1], 2) if e8 else None
                        ema9_val = round(e9[-1], 2) if e9 else None
                        ema20_val = round(e20[-1], 2) if e20 else None

                        is_short = ce["type"] == "overbought"
                        adverse_limit = entry_price * 1.01 if is_short else entry_price * 0.99
                        max_fav = entry_price

                        for k in range(1, len(candles_5m) - idx):
                            fc = candles_5m[idx + k]
                            fh, fl = float(fc[2]), float(fc[3])
                            if is_short:
                                if fl < max_fav:
                                    max_fav = fl
                                if fh >= adverse_limit:
                                    break
                            else:
                                if fh > max_fav:
                                    max_fav = fh
                                if fl <= adverse_limit:
                                    break

                        max_fav_pct = round(abs(max_fav - entry_price) / entry_price * 100, 2)

                        def _reached(ema_v):
                            if ema_v is None:
                                return False
                            return max_fav <= ema_v if is_short else max_fav >= ema_v

                        ce["emas"] = {
                            "ema8": ema8_val, "ema9": ema9_val, "ema20": ema20_val,
                            "ema8_dist": round(abs(entry_price - ema8_val) / entry_price * 100, 2) if ema8_val else None,
                            "ema9_dist": round(abs(entry_price - ema9_val) / entry_price * 100, 2) if ema9_val else None,
                            "ema20_dist": round(abs(entry_price - ema20_val) / entry_price * 100, 2) if ema20_val else None,
                            "max_fav": round(max_fav, 2),
                            "max_fav_pct": max_fav_pct,
                            "reached_8": _reached(ema8_val),
                            "reached_9": _reached(ema9_val),
                            "reached_20": _reached(ema20_val),
                        }

                        # 8 EMA trade with 1:1 R:R
                        if ema8_val is not None:
                            on_target_side = (is_short and ema8_val < entry_price) or \
                                             (not is_short and ema8_val > entry_price)
                            if on_target_side:
                                dist = abs(entry_price - ema8_val)
                                sl_8 = round(entry_price + dist, 2) if is_short else round(entry_price - dist, 2)
                                e8_result = "open"
                                e8_at = None
                                for k in range(1, len(candles_5m) - idx):
                                    fc = candles_5m[idx + k]
                                    fh, fl = float(fc[2]), float(fc[3])
                                    fc_dt = datetime.fromtimestamp(int(fc[0]), tz=IST)
                                    if is_short:
                                        if fh >= sl_8:
                                            e8_result = "sl"
                                            e8_at = fc_dt.strftime("%d %b %H:%M")
                                            break
                                        if fl <= ema8_val:
                                            e8_result = "target"
                                            e8_at = fc_dt.strftime("%d %b %H:%M")
                                            break
                                    else:
                                        if fl <= sl_8:
                                            e8_result = "sl"
                                            e8_at = fc_dt.strftime("%d %b %H:%M")
                                            break
                                        if fh >= ema8_val:
                                            e8_result = "target"
                                            e8_at = fc_dt.strftime("%d %b %H:%M")
                                            break
                                ce["ema8_trade"] = {
                                    "target": ema8_val,
                                    "sl": sl_8,
                                    "dist_pct": round(dist / entry_price * 100, 2),
                                    "result": e8_result,
                                    "at": e8_at,
                                }
            else:
                for ce in common_extremes:
                    ce.pop("_ts_5m", None)

            result["common"] = common_extremes
            result["common_count"] = len(common_extremes)
            result["common_overbought"] = sum(1 for e in common_extremes if e["type"] == "overbought")
            result["common_oversold"] = sum(1 for e in common_extremes if e["type"] == "oversold")

            # Per-stock performance
            with_move = [ce for ce in common_extremes if ce.get("move")]
            if with_move:
                rev_count = sum(1 for ce in with_move if ce["move"]["reversed"])
                result["stock_perf"] = {
                    "total": len(with_move),
                    "reversed": rev_count,
                    "continued": len(with_move) - rev_count,
                    "reversal_pct": round(rev_count / len(with_move) * 100, 1),
                }

            result["cmp"] = cmp
            total = sum(
                result.get(tf, {}).get("overbought_count", 0) + result.get(tf, {}).get("oversold_count", 0)
                for tf in ["5m", "15m"]
            )
            result["total_extreme_days"] = total
            result["both_timeframes"] = bool(
                result.get("5m", {}).get("extremes") and result.get("15m", {}).get("extremes")
            )
            days_set = set()
            for tf in ["5m", "15m"]:
                for e in result.get(tf, {}).get("extremes", []):
                    days_set.add(e["date_key"])
            result["unique_extreme_days"] = len(days_set)

            return result
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        results_raw = list(pool.map(_scan_stock, stocks))

    results = [r for r in results_raw if r is not None]
    results.sort(key=lambda r: (r["common_count"], r["both_timeframes"], r["total_extreme_days"]), reverse=True)

    # Aggregate movement stats across all common extremes
    all_ob_moves = []
    all_os_moves = []
    for r in results:
        for ce in r.get("common", []):
            m = ce.get("move")
            if not m:
                continue
            if ce["type"] == "overbought":
                all_ob_moves.append(m)
            else:
                all_os_moves.append(m)

    def _move_stats(moves):
        if not moves:
            return None
        with_hit = [m for m in moves if m["hit_1pct"]]
        reversals = sum(1 for m in moves if m["reversed"])
        hit_minutes = [m["minutes"] for m in with_hit]
        return {
            "count": len(moves),
            "hit_1pct_count": len(with_hit),
            "hit_rate": round(len(with_hit) / len(moves) * 100, 1),
            "reversal_count": reversals,
            "reversal_rate": round(reversals / len(moves) * 100, 1),
            "avg_minutes_to_hit": round(sum(hit_minutes) / len(hit_minutes), 0) if hit_minutes else None,
            "no_hit_count": len(moves) - len(with_hit),
        }

    # Aggregate EMA reach patterns
    ema_patterns = {"short": [], "long": []}
    for r in results:
        for ce in r.get("common", []):
            ea = ce.get("emas")
            if ea:
                key = "short" if ce["type"] == "overbought" else "long"
                ema_patterns[key].append(ea)

    def _ema_stats(pats):
        if not pats:
            return None
        n = len(pats)
        r8 = sum(1 for p in pats if p["reached_8"])
        r9 = sum(1 for p in pats if p["reached_9"])
        r20 = sum(1 for p in pats if p["reached_20"])
        favs = [p["max_fav_pct"] for p in pats]
        d8 = [p["ema8_dist"] for p in pats if p["ema8_dist"] is not None]
        d9 = [p["ema9_dist"] for p in pats if p["ema9_dist"] is not None]
        d20 = [p["ema20_dist"] for p in pats if p["ema20_dist"] is not None]
        stopped_before_8 = n - r8
        stopped_8_to_9 = r8 - r9
        stopped_9_to_20 = r9 - r20
        past_20 = r20
        return {
            "count": n,
            "reached_8_pct": round(r8 / n * 100, 1),
            "reached_9_pct": round(r9 / n * 100, 1),
            "reached_20_pct": round(r20 / n * 100, 1),
            "avg_max_fav_pct": round(sum(favs) / len(favs), 2) if favs else 0,
            "avg_ema8_dist": round(sum(d8) / len(d8), 2) if d8 else None,
            "avg_ema9_dist": round(sum(d9) / len(d9), 2) if d9 else None,
            "avg_ema20_dist": round(sum(d20) / len(d20), 2) if d20 else None,
            "stopped_before_8": stopped_before_8,
            "stopped_8_to_9": stopped_8_to_9,
            "stopped_9_to_20": stopped_9_to_20,
            "past_20": past_20,
            "reached_8_count": r8,
            "ema8_bounce_rate": round(stopped_8_to_9 / r8 * 100, 1) if r8 > 0 else None,
            "ema8_continue_rate": round((r8 - stopped_8_to_9) / r8 * 100, 1) if r8 > 0 else None,
        }

    # Aggregate 8 EMA 1:1 R:R trade stats
    e8_trades = {"short": [], "long": []}
    for r in results:
        for ce in r.get("common", []):
            t = ce.get("ema8_trade")
            if t:
                key = "short" if ce["type"] == "overbought" else "long"
                e8_trades[key].append(t)

    def _e8_stats(trades):
        if not trades:
            return None
        wins = sum(1 for t in trades if t["result"] == "target")
        losses = sum(1 for t in trades if t["result"] == "sl")
        opens = sum(1 for t in trades if t["result"] == "open")
        decided = wins + losses
        dists = [t["dist_pct"] for t in trades]
        return {
            "count": len(trades),
            "wins": wins, "losses": losses, "opens": opens,
            "win_rate": round(wins / decided * 100, 1) if decided else None,
            "avg_dist_pct": round(sum(dists) / len(dists), 2) if dists else None,
        }

    # Persist signals and rebuild summary
    save_rsi_signals(results)

    # Attach stored all-time performance to each result
    summary = load_rsi_summary()
    for r in results:
        sp = summary.get(r["symbol"])
        if sp:
            r["stored_perf"] = sp

    _save_rsi_cache("historical", {
        "results": results,
        "total_scanned": len(stocks),
        "stocks_with_extremes": len(results),
        "scan_days": scan_days,
        "move_stats": {
            "overbought": _move_stats(all_ob_moves),
            "oversold": _move_stats(all_os_moves),
        },
        "ema_stats": {
            "short": _ema_stats(ema_patterns["short"]),
            "long": _ema_stats(ema_patterns["long"]),
        },
        "ema8_trade_stats": {
            "short": _e8_stats(e8_trades["short"]),
            "long": _e8_stats(e8_trades["long"]),
        },
    }, extra=f"_{scan_days}d")
    _rsi_scan_running["historical"] = False


@app.route("/rsi-extremes/scan")
def rsi_extremes_scan():
    scan_days = min(int(request.args.get("days", "10")), 30) if request.args.get("days", "").isdigit() else 10

    cached = _rsi_cache_fresh("historical", extra=f"_{scan_days}d")
    if cached:
        cached.pop("_cached_at", None)
        return jsonify(cached)

    if not _rsi_scan_running["historical"]:
        with _rsi_scan_locks["historical"]:
            if not _rsi_scan_running["historical"]:
                _rsi_scan_running["historical"] = True
                threading.Thread(target=_do_rsi_extremes_scan, args=(scan_days,), daemon=True).start()

    stale = None
    path = _rsi_cache_path("historical", extra=f"_{scan_days}d")
    if path.exists():
        try:
            stale = json.loads(path.read_text())
            stale.pop("_cached_at", None)
            stale["_stale"] = True
            return jsonify(stale)
        except Exception:
            pass

    return jsonify({"scanning": True, "results": [], "stocks_with_extremes": 0, "scan_days": scan_days})


# ---------------------------------------------------------------------------
# Intraday Strategies — ORB & PDH/PDL Breakout
# ---------------------------------------------------------------------------

from core_rotation import load_config as _load_core_config, needs_rotation as _needs_core_rotation, run_rotation as _run_core_rotation

_core_rotation_running = False
_core_rotation_progress = {"step": "idle", "done": 0, "total": 0}


def _get_core_stocks() -> list[str]:
    return _load_core_config().get("core_stocks", [])


def _get_backtest_stats() -> dict:
    return _load_core_config().get("backtest_stats", {})


def _rotation_progress_cb(step, done, total):
    global _core_rotation_progress
    _core_rotation_progress = {"step": step, "done": done, "total": total}


def _run_background_rotation():
    global _core_rotation_running
    try:
        _run_core_rotation(fetch_candles, resolve_yahoo_ticker, get_fno_stocks, progress_cb=_rotation_progress_cb)
    except Exception as e:
        _core_rotation_progress["step"] = f"error: {e}"
    finally:
        _core_rotation_running = False


def _ensure_core_rotation():
    """Trigger background core stock rotation if due (monthly). Non-blocking."""
    global _core_rotation_running
    if _core_rotation_running:
        return
    if not _needs_core_rotation():
        return
    with _scan_lock:
        if _core_rotation_running:
            return
        _core_rotation_running = True
    t = threading.Thread(target=_run_background_rotation, daemon=True)
    t.start()

PDHL_BUFFER_PCT = 0.1
PDHL_SL_RATIO = 0.3

INTRADAY_PICK_CACHE_FILE = CACHE_DIR / "intraday_picks.json"
INTRADAY_PICK_MAX_STOCKS = 20
INTRADAY_MIN_AVG_VOL_5D = 500_000     # min 5-day avg volume in shares
INTRADAY_MIN_VOL_RATIO_5D_20D = 0.40  # 5-day avg must be >= 40% of 20-day avg
INTRADAY_MIN_PREV_DAY_VOL_RATIO = 0.20  # yesterday's vol must be >= 20% of 20-day avg
INTRADAY_MIN_PRICE = 40               # min stock price in rupees


def _score_stock_for_intraday(sym, candles_daily):
    """Score a stock for intraday suitability based on daily candles."""
    if not candles_daily or len(candles_daily) < 25:
        return None

    recent_20 = candles_daily[-20:]
    recent_5 = candles_daily[-5:]
    prev_day = candles_daily[-1]

    pd_high = float(prev_day[2])
    pd_low = float(prev_day[3])
    pd_close = float(prev_day[4])
    pd_vol = float(prev_day[5])
    pd_range_pct = (pd_high - pd_low) / pd_close * 100 if pd_close > 0 else 0

    avg_vol_20 = sum(float(c[5]) for c in recent_20) / len(recent_20)
    avg_range_20 = sum((float(c[2]) - float(c[3])) / float(c[4]) * 100 for c in recent_20 if float(c[4]) > 0) / len(recent_20)
    avg_value_cr = avg_vol_20 * pd_close / 1e7

    avg_range_5 = sum((float(c[2]) - float(c[3])) / float(c[4]) * 100 for c in recent_5 if float(c[4]) > 0) / len(recent_5)
    avg_vol_5 = sum(float(c[5]) for c in recent_5) / len(recent_5)

    # Price and volume filters
    if pd_close < INTRADAY_MIN_PRICE:
        return None
    if avg_vol_5 < INTRADAY_MIN_AVG_VOL_5D:
        return None
    if avg_vol_20 > 0 and avg_vol_5 / avg_vol_20 < INTRADAY_MIN_VOL_RATIO_5D_20D:
        return None
    if avg_vol_20 > 0 and pd_vol / avg_vol_20 < INTRADAY_MIN_PREV_DAY_VOL_RATIO:
        return None

    # Scoring components (each 0-100)
    vol_spike = min(pd_vol / avg_vol_20, 3.0) / 3.0 * 100 if avg_vol_20 > 0 else 0
    range_expansion = min(avg_range_5 / avg_range_20, 2.0) / 2.0 * 100 if avg_range_20 > 0 else 0
    abs_range = min(avg_range_5 / 4.0, 1.0) * 100
    vol_trend = min(avg_vol_5 / avg_vol_20, 2.0) / 2.0 * 100 if avg_vol_20 > 0 else 0
    liquidity = min(avg_vol_5 / 5_000_000, 1.0) * 100  # 50L shares/day = max score

    # Weighted score
    score = (
        vol_spike * 0.20 +
        range_expansion * 0.25 +
        abs_range * 0.25 +
        vol_trend * 0.15 +
        liquidity * 0.15
    )

    is_core = sym in _get_core_stocks()
    backtest = _get_backtest_stats().get(sym)

    return {
        "symbol": sym,
        "score": round(score, 1),
        "is_core": is_core,
        "cmp": round(pd_close, 2),
        "pd_range_pct": round(pd_range_pct, 2),
        "avg_range_5d": round(avg_range_5, 2),
        "avg_range_20d": round(avg_range_20, 2),
        "vol_spike": round(pd_vol / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        "vol_trend_5d": round(avg_vol_5 / avg_vol_20, 2) if avg_vol_20 > 0 else 0,
        "avg_vol_5d": int(avg_vol_5),
        "avg_vol_20d": int(avg_vol_20),
        "avg_value_cr": round(avg_value_cr, 1),
        "backtest": backtest,
    }


def pick_intraday_stocks() -> list[dict]:
    """Scan all F&O stocks and return top candidates ranked by intraday score."""
    # Check cache (valid for same day)
    if INTRADAY_PICK_CACHE_FILE.exists():
        try:
            cached = json.loads(INTRADAY_PICK_CACHE_FILE.read_text())
            if cached.get("date") == datetime.now(IST).strftime("%Y-%m-%d"):
                return cached["picks"]
        except Exception:
            pass

    try:
        stocks = get_fno_stocks()
    except Exception:
        _bt = _get_backtest_stats()
        return [{"symbol": s, "score": 0, "is_core": True, "backtest": _bt.get(s)} for s in _get_core_stocks()]

    scored = []

    def _score_one(stock):
        sym = stock["symbol"]
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)
            candles = fetch_candles(yahoo_sym, period="2mo", interval="1d", canonical=canonical)
            return _score_stock_for_intraday(sym, candles)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_score_one, stocks))

    scored = [r for r in results if r is not None]
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Always include core stocks if they pass liquidity filter, fill rest from top scorers
    core_picks = [s for s in scored if s["is_core"]]
    non_core_picks = [s for s in scored if not s["is_core"]]

    core_syms = {s["symbol"] for s in core_picks}
    final = list(core_picks)
    for s in non_core_picks:
        if len(final) >= INTRADAY_PICK_MAX_STOCKS:
            break
        if s["symbol"] not in core_syms:
            final.append(s)

    final.sort(key=lambda x: x["score"], reverse=True)

    # Cache for today
    CACHE_DIR.mkdir(exist_ok=True)
    INTRADAY_PICK_CACHE_FILE.write_text(json.dumps({
        "date": datetime.now(IST).strftime("%Y-%m-%d"),
        "picks": final,
        "total_scanned": len(scored),
    }))

    return final


@app.route("/intraday")
def intraday_page():
    return render_template("intraday.html")


FYERS_TOKEN_REFRESH_FILE = CACHE_DIR / "fyers_token_refresh.json"


@app.route("/api/fyers-status")
def fyers_status():
    """Check Fyers token status and last refresh time."""
    app_id = os.environ.get("FYERS_APP_ID")
    token = os.environ.get("FYERS_ACCESS_TOKEN")
    status = "not_configured"
    name = None
    if app_id and token:
        try:
            from fyers_apiv3 import fyersModel
            client = fyersModel.FyersModel(client_id=app_id, token=token, is_async=False, log_path="logs/")
            resp = client.get_profile()
            if resp.get("s") == "ok":
                status = "active"
                name = resp["data"]["name"]
            else:
                status = "expired"
        except Exception:
            status = "error"

    today = datetime.now(IST).strftime("%Y-%m-%d")
    refreshed_today = False
    if FYERS_TOKEN_REFRESH_FILE.exists():
        try:
            data = json.loads(FYERS_TOKEN_REFRESH_FILE.read_text())
            refreshed_today = data.get("date") == today
        except Exception:
            pass

    return jsonify({"status": status, "name": name, "refreshed_today": refreshed_today})


@app.route("/api/refresh-fyers-token", methods=["POST"])
def refresh_fyers_token():
    """Initiate Fyers token refresh. Returns auth URL for the browser to redirect to."""
    auth_err = _check_admin(request)
    if auth_err:
        return auth_err
    global _fyers_client
    today = datetime.now(IST).strftime("%Y-%m-%d")

    if FYERS_TOKEN_REFRESH_FILE.exists():
        try:
            data = json.loads(FYERS_TOKEN_REFRESH_FILE.read_text())
            if data.get("date") == today:
                return jsonify({"success": False, "status": "already_refreshed",
                                "message": f"Token already refreshed today at {data.get('time', '?')}"})
        except Exception:
            pass

    try:
        from fyers_auth import get_auth_url, exchange_auth_code

        callback_uri = request.host_url.rstrip("/") + "/fyers/callback"
        auth_code, auth_url = get_auth_url(redirect_uri_override=callback_uri)

        if auth_code:
            # Auth code captured directly — no browser redirect needed
            access_token = exchange_auth_code(auth_code)
            _fyers_client = None

            name = "Unknown"
            try:
                from fyers_apiv3 import fyersModel
                client = fyersModel.FyersModel(client_id=os.environ.get("FYERS_APP_ID"),
                                                token=access_token, is_async=False, log_path="logs/")
                resp = client.get_profile()
                if resp.get("s") == "ok":
                    name = resp["data"]["name"]
            except Exception:
                pass

            CACHE_DIR.mkdir(exist_ok=True)
            FYERS_TOKEN_REFRESH_FILE.write_text(json.dumps({
                "date": today, "time": datetime.now(IST).strftime("%H:%M:%S"), "name": name,
            }))

            return jsonify({"success": True, "status": "active", "name": name,
                            "message": f"Token active — {name}"})

        return jsonify({"success": True, "status": "redirect", "auth_url": auth_url})
    except Exception as e:
        return jsonify({"success": False, "status": "error", "message": str(e)})


@app.route("/fyers/callback")
def fyers_callback():
    """OAuth callback — Fyers redirects here with auth_code after user authorizes."""
    global _fyers_client
    auth_code = request.args.get("auth_code")
    if not auth_code:
        return """<html><body style="background:#0f1117;color:#f85149;font-family:sans-serif;
            display:flex;justify-content:center;align-items:center;height:100vh;font-size:1.3rem;">
            Error: No auth_code received from Fyers. Please try again.
            </body></html>""", 400

    try:
        from fyers_auth import exchange_auth_code
        access_token = exchange_auth_code(auth_code)
        _fyers_client = None

        # Verify
        app_id = os.environ.get("FYERS_APP_ID")
        name = "Unknown"
        try:
            from fyers_apiv3 import fyersModel
            client = fyersModel.FyersModel(client_id=app_id, token=access_token, is_async=False, log_path="logs/")
            resp = client.get_profile()
            if resp.get("s") == "ok":
                name = resp["data"]["name"]
        except Exception:
            pass

        # Mark as refreshed today
        today = datetime.now(IST).strftime("%Y-%m-%d")
        CACHE_DIR.mkdir(exist_ok=True)
        FYERS_TOKEN_REFRESH_FILE.write_text(json.dumps({
            "date": today,
            "time": datetime.now(IST).strftime("%H:%M:%S"),
            "name": name,
        }))

        return f"""<html><body style="background:#0f1117;color:#3fb950;font-family:sans-serif;
            display:flex;flex-direction:column;justify-content:center;align-items:center;height:100vh;">
            <div style="font-size:1.5rem;font-weight:700;margin-bottom:1rem;">Fyers Token Generated</div>
            <div style="font-size:1rem;color:#8b949e;">Authenticated as: {name}</div>
            <a href="/" style="margin-top:2rem;padding:0.7rem 1.5rem;background:#1f6feb;color:#fff;
               border-radius:8px;text-decoration:none;font-weight:600;">Go to Dashboard</a>
            </body></html>"""
    except Exception as e:
        return f"""<html><body style="background:#0f1117;color:#f85149;font-family:sans-serif;
            display:flex;flex-direction:column;justify-content:center;align-items:center;height:100vh;">
            <div style="font-size:1.3rem;font-weight:700;">Token generation failed</div>
            <div style="font-size:0.9rem;color:#8b949e;margin-top:0.5rem;">{e}</div>
            <a href="/" style="margin-top:2rem;padding:0.7rem 1.5rem;background:#1f6feb;color:#fff;
               border-radius:8px;text-decoration:none;font-weight:600;">Back to Dashboard</a>
            </body></html>""", 500


@app.route("/intraday/pick")
def intraday_pick():
    """Scan all F&O stocks and return today's top intraday candidates."""
    force = request.args.get("force", "0") == "1"
    if force and INTRADAY_PICK_CACHE_FILE.exists():
        INTRADAY_PICK_CACHE_FILE.unlink()

    picks = pick_intraday_stocks()
    return jsonify({
        "picks": picks,
        "count": len(picks),
        "date": datetime.now(IST).strftime("%Y-%m-%d"),
        "time": datetime.now(IST).strftime("%H:%M:%S"),
    })


INTRADAY_EOD_CACHE_FILE = CACHE_DIR / "intraday_eod.json"


@app.route("/intraday/rotation-status")
def intraday_rotation_status():
    cfg = _load_core_config()
    return jsonify({
        "running": _core_rotation_running,
        "progress": _core_rotation_progress,
        "last_rotation": cfg.get("last_rotation"),
        "core_stocks": cfg.get("core_stocks", []),
        "core_count": len(cfg.get("core_stocks", [])),
        "last_log": cfg.get("rotation_log", [])[-1] if cfg.get("rotation_log") else None,
    })


@app.route("/intraday/rotate", methods=["POST"])
def intraday_force_rotate():
    """Force a core stock rotation now (ignores the monthly interval)."""
    global _core_rotation_running
    if _core_rotation_running:
        return jsonify({"status": "already_running"})
    from core_rotation import save_config as _save_core_config
    cfg = _load_core_config()
    cfg["last_rotation"] = None
    _save_core_config(cfg)
    _ensure_core_rotation()
    return jsonify({"status": "started"})


@app.route("/intraday/levels")
def intraday_levels():
    _ensure_core_rotation()

    now = datetime.now(IST)
    mkt_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    mkt_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)

    if now < mkt_open_time:
        market_status = "pre_open"
    elif now <= mkt_close_time:
        market_status = "open"
    else:
        market_status = "closed"

    today_str = now.strftime("%Y-%m-%d")

    # After market close, serve cached EOD data instead of re-fetching
    if market_status == "closed" and INTRADAY_EOD_CACHE_FILE.exists():
        try:
            cached = json.loads(INTRADAY_EOD_CACHE_FILE.read_text())
            if cached.get("date") == today_str:
                return jsonify(cached)
        except Exception:
            pass

    # Get dynamically picked stocks
    picks = pick_intraday_stocks()
    pick_symbols = [p["symbol"] for p in picks]
    pick_data = {p["symbol"]: p for p in picks}

    def _process_stock(sym):
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)

            # Fetch 5-min candles for intraday (recent data)
            candles_5m = fetch_candles(yahoo_sym, period="1mo", interval="5m", canonical=canonical)
            if not candles_5m or len(candles_5m) < 20:
                return {"symbol": sym, "error": "No intraday data"}

            # Fetch daily candles for PDH/PDL
            candles_daily = fetch_candles(yahoo_sym, period="1mo", interval="1d", canonical=canonical)

            # Group 5m candles by day
            from collections import defaultdict as _dd
            day_groups = _dd(list)
            for c in candles_5m:
                dt = datetime.fromtimestamp(int(c[0]), tz=IST)
                day_groups[dt.strftime("%Y-%m-%d")].append(c)

            days_sorted = sorted(day_groups.keys())
            today_candles = day_groups.get(today_str, [])

            # Current price
            if today_candles:
                cmp = round(float(today_candles[-1][4]), 2)
                day_open = float(today_candles[0][1])
                change_pct = round((cmp - day_open) / day_open * 100, 2) if day_open > 0 else 0
            elif candles_5m:
                cmp = round(float(candles_5m[-1][4]), 2)
                change_pct = None
            else:
                return {"symbol": sym, "error": "No price data"}

            result = {"symbol": sym, "cmp": cmp, "change_pct": change_pct}

            # ── PDH/PDL levels ──
            pdhl_data = None
            if candles_daily and len(candles_daily) >= 2:
                # Find previous completed day
                daily_days = {}
                for c in candles_daily:
                    d = datetime.fromtimestamp(int(c[0]), tz=IST).strftime("%Y-%m-%d")
                    daily_days[d] = c

                sorted_daily = sorted(daily_days.keys())
                prev_day = None
                for d in reversed(sorted_daily):
                    if d < today_str:
                        prev_day = d
                        break

                if prev_day and prev_day in daily_days:
                    pc = daily_days[prev_day]
                    pdh = round(float(pc[2]), 2)
                    pdl = round(float(pc[3]), 2)
                    pd_range = pdh - pdl

                    if pd_range > 0:
                        buffer = pdh * PDHL_BUFFER_PCT / 100
                        long_entry = round(pdh + buffer, 2)
                        short_entry = round(pdl - buffer, 2)
                        long_sl = round(pdh - pd_range * PDHL_SL_RATIO, 2)
                        short_sl = round(pdl + pd_range * PDHL_SL_RATIO, 2)
                        long_risk = long_entry - long_sl
                        short_risk = short_sl - short_entry
                        long_target = round(long_entry + long_risk, 2)
                        short_target = round(short_entry - short_risk, 2)

                        # Check signal status from today's 5m candles
                        long_status = "waiting"
                        short_status = "waiting"
                        long_pnl_pct = None
                        short_pnl_pct = None

                        post_15 = [c for c in today_candles
                                   if (datetime.fromtimestamp(int(c[0]), tz=IST).hour - 9) * 60 +
                                      (datetime.fromtimestamp(int(c[0]), tz=IST).minute - 15) >= 0]

                        long_triggered = False
                        short_triggered = False
                        long_trigger_time = None
                        short_trigger_time = None
                        long_exit_time = None
                        short_exit_time = None
                        for c in post_15:
                            h, l, cl = c[2], c[3], c[4]
                            c_dt = datetime.fromtimestamp(int(c[0]), tz=IST)
                            c_time = c_dt.strftime("%H:%M")
                            if not long_triggered and h >= long_entry:
                                long_triggered = True
                                long_status = "triggered"
                                long_trigger_time = c_time
                            if long_triggered and long_status == "triggered":
                                if l <= long_sl:
                                    long_status = "sl"
                                    long_pnl_pct = round((long_sl - long_entry) / long_entry * 100, 2)
                                    long_exit_time = c_time
                                elif h >= long_target:
                                    long_status = "target"
                                    long_pnl_pct = round((long_target - long_entry) / long_entry * 100, 2)
                                    long_exit_time = c_time

                            if not short_triggered and l <= short_entry:
                                short_triggered = True
                                short_status = "triggered"
                                short_trigger_time = c_time
                            if short_triggered and short_status == "triggered":
                                if h >= short_sl:
                                    short_status = "sl"
                                    short_pnl_pct = round((short_entry - short_sl) / short_entry * 100, 2)
                                    short_exit_time = c_time
                                elif l <= short_target:
                                    short_status = "target"
                                    short_pnl_pct = round((short_entry - short_target) / short_entry * 100, 2)
                                    short_exit_time = c_time

                        eod_closed = market_status == "closed"
                        if long_status == "triggered":
                            long_pnl_pct = round((cmp - long_entry) / long_entry * 100, 2)
                            if eod_closed:
                                long_status = "eod"
                                long_exit_time = "15:30"
                        if short_status == "triggered":
                            short_pnl_pct = round((short_entry - cmp) / short_entry * 100, 2)
                            if eod_closed:
                                short_status = "eod"
                                short_exit_time = "15:30"

                        pdhl_data = {
                            "pdh": pdh, "pdl": pdl, "prev_date": prev_day,
                            "long_entry": long_entry, "short_entry": short_entry,
                            "long_sl": long_sl, "short_sl": short_sl,
                            "long_target": long_target, "short_target": short_target,
                            "range_abs": round(pd_range, 2),
                            "range_pct": round(pd_range / pdl * 100, 2) if pdl > 0 else 0,
                            "long_status": long_status, "short_status": short_status,
                            "long_pnl_pct": long_pnl_pct, "short_pnl_pct": short_pnl_pct,
                            "long_trigger_time": long_trigger_time, "short_trigger_time": short_trigger_time,
                            "long_exit_time": long_exit_time, "short_exit_time": short_exit_time,
                        }

            result["pdhl"] = pdhl_data

            # RSI levels (5-min uses last 100 candles for continuity, daily uses all)
            from analyzer import rsi as _rsi
            rsi_5m = None
            rsi_daily = None
            if candles_5m and len(candles_5m) >= 20:
                closes_5m = [float(c[4]) for c in candles_5m[-100:]]
                rsi_5m = _rsi(closes_5m, 14)
            if candles_daily and len(candles_daily) >= 20:
                closes_d = [float(c[4]) for c in candles_daily]
                rsi_daily = _rsi(closes_d, 14)
            result["rsi_5m"] = round(rsi_5m, 1) if rsi_5m is not None else None
            result["rsi_daily"] = round(rsi_daily, 1) if rsi_daily is not None else None

            # Attach pick metadata
            pd = pick_data.get(sym, {})
            result["score"] = pd.get("score", 0)
            result["is_core"] = pd.get("is_core", False)
            result["avg_range_5d"] = pd.get("avg_range_5d")
            result["vol_spike"] = pd.get("vol_spike")
            result["avg_value_cr"] = pd.get("avg_value_cr")
            result["backtest"] = pd.get("backtest") or _get_backtest_stats().get(sym)
            return result

        except Exception as e:
            return {"symbol": sym, "error": str(e)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_process_stock, pick_symbols))

    # Sort: active signals first, then by score
    def _sort_key(r):
        has_signal = 0
        strat = r.get("pdhl")
        if strat:
            if strat.get("long_status") == "triggered" or strat.get("short_status") == "triggered":
                has_signal = 2
            elif strat.get("long_status") in ("target", "sl") or strat.get("short_status") in ("target", "sl"):
                has_signal = 1
        return (-has_signal, -r.get("score", 0))

    results.sort(key=_sort_key)

    # Check Fyers token status
    fyers_status = "not_configured"
    fyers = _get_fyers()
    if fyers:
        try:
            resp = fyers.get_profile()
            fyers_status = "active" if resp.get("s") == "ok" else "expired"
        except Exception:
            fyers_status = "error"

    response_data = {
        "stocks": results,
        "market_status": market_status,
        "time": now.strftime("%H:%M:%S"),
        "date": today_str,
        "total_picked": len(pick_symbols),
        "total_scanned": len(picks),
        "data_source": "Fyers" if fyers_status == "active" else "Yahoo Finance",
        "fyers_status": fyers_status,
    }

    # Cache EOD results after market close so we don't re-fetch
    if market_status == "closed":
        CACHE_DIR.mkdir(exist_ok=True)
        INTRADAY_EOD_CACHE_FILE.write_text(json.dumps(response_data))

    return jsonify(response_data)


# ---------------------------------------------------------------------------
# S/R Pattern Breakout Scanner
# ---------------------------------------------------------------------------

@app.route("/sr-breakout")
def sr_breakout_page():
    return render_template("sr_breakout.html")


@app.route("/sr-breakout/scan")
def sr_breakout_scan():
    from analyzer import (
        rsi as _rsi,
        find_support_resistance,
        pivot_points,
        fibonacci_retracement,
        atr as _atr,
        sma as _sma,
    )

    try:
        stocks = get_fno_stocks()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch F&O list: {e}", "results": []})

    CONFLUENCE_THRESHOLD_PCT = 1.5
    TRUSTED_PATTERNS = {"Cup & Handle", "Ascending Triangle", "Descending Triangle"}

    def _scan_stock(stock):
        sym = stock["symbol"]
        name = stock["name"]
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)
            candles = fetch_candles(yahoo_sym, period="1y", interval="1d", canonical=canonical)
            if not candles or len(candles) < 60:
                return None

            cmp = round(float(candles[-1][4]), 2)
            closes = [float(c[4]) for c in candles]
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]

            patterns = detect_chart_patterns(candles)
            if not patterns:
                return None

            patterns = [p for p in patterns if p["name"] in TRUSTED_PATTERNS]
            if not patterns:
                return None

            has_breakout_level = any(
                p.get("breakout_level_up") or p.get("breakout_level_down")
                for p in patterns
            )
            if not has_breakout_level:
                return None

            sr_zones = find_support_resistance(candles, window=5)
            ema_data = detect_ema_crossovers(closes)
            d_rsi = _rsi(closes)
            atr_val = _atr(candles, 14)
            pivot_data = pivot_points(candles, method="standard", timeframe="positional")
            fib_data = fibonacci_retracement(candles)

            all_levels = []
            for z in sr_zones:
                all_levels.append({
                    "price": z["level"],
                    "source": f"S/R ({z['touches']} touches)",
                    "strength": z["touches"] * 2 + z["recency_score"],
                    "touches": z["touches"],
                })
            if pivot_data:
                for lk in ("r1", "r2", "r3", "s1", "s2", "s3"):
                    val = pivot_data.get(lk)
                    if val:
                        all_levels.append({"price": val, "source": f"Pivot {lk.upper()}", "strength": 4})
                all_levels.append({"price": pivot_data["pp"], "source": "Pivot PP", "strength": 5})
            if fib_data:
                for fl in fib_data.get("levels", []):
                    if fl["ratio"] in (0.382, 0.5, 0.618):
                        all_levels.append({"price": fl["price"], "source": f"Fib {fl['label']}", "strength": 5 if fl["ratio"] == 0.618 else 4})

            sma50 = _sma(closes, 50)
            sma200 = _sma(closes, 200)
            if sma50:
                all_levels.append({"price": sma50, "source": "SMA 50", "strength": 3})
            if sma200:
                all_levels.append({"price": sma200, "source": "SMA 200", "strength": 4})

            matched = []
            for p in patterns:
                bl_up = p.get("breakout_level_up")
                bl_down = p.get("breakout_level_down")

                for bl, direction in [(bl_up, "bullish"), (bl_down, "bearish")]:
                    if bl is None:
                        continue
                    nearby = [
                        lv for lv in all_levels
                        if abs(lv["price"] - bl) / bl * 100 <= CONFLUENCE_THRESHOLD_PCT
                    ]
                    if not nearby:
                        continue

                    confluence_strength = sum(lv["strength"] for lv in nearby)
                    confluence_sources = [lv["source"] for lv in nearby]
                    confluence_count = len(nearby)

                    if p.get("confirmed"):
                        status = "Breakout Confirmed"
                    else:
                        status = "Approaching"

                    distance_pct = round(abs(cmp - bl) / bl * 100, 2)

                    score = confluence_strength * 2
                    if p.get("confirmed"):
                        score += 10
                    if p.get("volume_confirmed"):
                        score += 5
                    if confluence_count >= 3:
                        score += 5
                    elif confluence_count >= 2:
                        score += 2

                    ema_align = ema_data.get("alignment", "neutral")
                    if direction == "bullish" and ema_align in ("bullish", "strong_bullish"):
                        score += 4
                    elif direction == "bearish" and ema_align in ("bearish", "strong_bearish"):
                        score += 4

                    if d_rsi:
                        if direction == "bullish" and d_rsi < 40:
                            score += 2
                        elif direction == "bearish" and d_rsi > 60:
                            score += 2

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

                    if rr_ratio < 1.0:
                        continue

                    matched.append({
                        "pattern": p["name"],
                        "bias": p.get("bias", "neutral"),
                        "direction": direction,
                        "breakout_level": round(bl, 2),
                        "status": status,
                        "confirmed": p.get("confirmed", False),
                        "distance_pct": distance_pct,
                        "confluence_count": confluence_count,
                        "confluence_strength": confluence_strength,
                        "confluence_sources": confluence_sources[:5],
                        "entry": round(entry, 2),
                        "sl": round(sl, 2) if sl else None,
                        "target": round(target, 2) if target else None,
                        "rr_ratio": rr_ratio,
                        "volume_confirmed": p.get("volume_confirmed", False),
                        "components": p.get("components", []),
                    })

            if not matched:
                return None

            matched.sort(key=lambda m: m["confluence_strength"], reverse=True)

            best = matched[0]
            ema_align = ema_data.get("alignment", "neutral")
            bull_crosses = [c for c in ema_data.get("crossovers", []) if c["bias"] == "bullish"]
            bear_crosses = [c for c in ema_data.get("crossovers", []) if c["bias"] == "bearish"]

            reasons = []
            reasons.append(f"{best['pattern']} at ₹{best['breakout_level']} with {best['confluence_count']}-way confluence")
            reasons.append(f"S/R sources: {', '.join(best['confluence_sources'][:3])}")
            if best["confirmed"]:
                reasons.append("Breakout confirmed — price beyond level")
            else:
                reasons.append(f"Price {best['distance_pct']}% from breakout level")
            if ema_align in ("bullish", "strong_bullish"):
                reasons.append("EMAs aligned bullish")
            elif ema_align in ("bearish", "strong_bearish"):
                reasons.append("EMAs aligned bearish")
            if bull_crosses:
                reasons.append(bull_crosses[0]["type"])
            if bear_crosses:
                reasons.append(bear_crosses[0]["type"])

            total_score = sum(m["confluence_strength"] for m in matched)

            return {
                "symbol": canonical,
                "display_symbol": sym,
                "name": name,
                "cmp": cmp,
                "rsi": round(d_rsi, 1) if d_rsi else None,
                "ema_alignment": ema_align,
                "ema_crossovers": [c["type"] for c in ema_data.get("crossovers", [])],
                "setups": matched,
                "top_status": best["status"],
                "top_direction": best["direction"],
                "score": total_score,
                "reasons": reasons[:5],
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results_raw = list(pool.map(_scan_stock, stocks))

    results = [r for r in results_raw if r is not None]
    results.sort(key=lambda r: r["score"], reverse=True)

    summary = {
        "total_approaching": sum(1 for r in results if r["top_status"] == "Approaching"),
        "total_confirmed": sum(1 for r in results if r["top_status"] == "Breakout Confirmed"),
        "total_bullish": sum(1 for r in results if r["top_direction"] == "bullish"),
        "total_bearish": sum(1 for r in results if r["top_direction"] == "bearish"),
    }

    return jsonify({
        "results": results,
        "total_scanned": len(stocks),
        "total_with_setups": len(results),
        "summary": summary,
    })


# ---------------------------------------------------------------------------
# RSI Momentum Scanner (5m + 15m RSI alignment)
# ---------------------------------------------------------------------------

@app.route("/rsi-momentum")
def rsi_momentum_page():
    return render_template("rsi_momentum.html")


def _do_rsi_momentum_scan():
    """Run RSI momentum scan in background, save to cache."""
    RSI_BULL_THRESH = 65
    RSI_BEAR_THRESH = 35

    try:
        stocks = get_fno_stocks()
    except Exception:
        _rsi_scan_running["momentum"] = False
        return

    now = datetime.now(IST)
    market_hour = now.hour
    market_min = now.minute
    market_time_min = market_hour * 60 + market_min
    close_min = 15 * 60 + 30

    def _time_zone(h, m):
        t = h * 60 + m
        if t < 10 * 60:
            return "open"
        elif t < 12 * 60:
            return "morning"
        elif t < 13 * 60:
            return "lunch"
        elif t < 15 * 60:
            return "afternoon"
        else:
            return "close"

    def _scan_stock(stock):
        sym = stock["symbol"]
        name = stock["name"]
        try:
            canonical, yahoo_sym = resolve_yahoo_ticker(sym)
            candles_5m = fetch_candles(yahoo_sym, period="1mo", interval="5m", canonical=canonical)
            if not candles_5m or len(candles_5m) < 50:
                return None

            closes_5m = [float(c[4]) for c in candles_5m]
            cmp = round(closes_5m[-1], 2)

            rsi_5m_vals = rsi_series(closes_5m, 14)
            if not rsi_5m_vals or len(rsi_5m_vals) < 2:
                return None
            rsi_5m = round(rsi_5m_vals[-1], 1)

            # Resample to 15min
            candles_15m = []
            for i in range(0, len(candles_5m) - 2, 3):
                chunk = candles_5m[i:i + 3]
                candles_15m.append([
                    chunk[0][0], chunk[0][1],
                    max(float(c[2]) for c in chunk),
                    min(float(c[3]) for c in chunk),
                    chunk[-1][4],
                    sum(float(c[5]) if len(c) > 5 else 0 for c in chunk),
                ])
            if len(candles_15m) < 20:
                return None

            closes_15m = [float(c[4]) for c in candles_15m]
            rsi_15m_vals = rsi_series(closes_15m, 14)
            if not rsi_15m_vals or len(rsi_15m_vals) < 2:
                return None
            rsi_15m = round(rsi_15m_vals[-1], 1)

            # Resample to 1H for display context
            candles_1h = []
            for i in range(0, len(candles_5m) - 11, 12):
                chunk = candles_5m[i:i + 12]
                candles_1h.append([
                    chunk[0][0], chunk[0][1],
                    max(float(c[2]) for c in chunk),
                    min(float(c[3]) for c in chunk),
                    chunk[-1][4],
                    sum(float(c[5]) if len(c) > 5 else 0 for c in chunk),
                ])
            rsi_1h = None
            if len(candles_1h) >= 20:
                closes_1h = [float(c[4]) for c in candles_1h]
                rsi_1h_vals = rsi_series(closes_1h, 14)
                if rsi_1h_vals:
                    rsi_1h = round(rsi_1h_vals[-1], 1)

            # EMAs for context
            e9 = ema_series(closes_5m, 9)
            e20 = ema_series(closes_5m, 20)
            ema_ctx = {
                "ema9": round(e9[-1], 2) if e9 else None,
                "ema20": round(e20[-1], 2) if e20 else None,
            }

            # Check signal
            is_bull = rsi_5m > RSI_BULL_THRESH and rsi_15m > RSI_BULL_THRESH
            is_bear = rsi_5m < RSI_BEAR_THRESH and rsi_15m < RSI_BEAR_THRESH
            if not is_bull and not is_bear:
                return None

            signal_type = "bullish" if is_bull else "bearish"

            # RSI trend — is it strengthening?
            rsi_5m_prev = round(rsi_5m_vals[-2], 1) if len(rsi_5m_vals) >= 2 else None
            rsi_15m_prev = round(rsi_15m_vals[-2], 1) if len(rsi_15m_vals) >= 2 else None
            if is_bull:
                strengthening = (rsi_5m_prev and rsi_5m > rsi_5m_prev) or (rsi_15m_prev and rsi_15m > rsi_15m_prev)
            else:
                strengthening = (rsi_5m_prev and rsi_5m < rsi_5m_prev) or (rsi_15m_prev and rsi_15m < rsi_15m_prev)

            # Time context
            can_hold_2h = (close_min - market_time_min) >= 120
            signal_time = now.strftime("%H:%M")
            time_zone = _time_zone(market_hour, market_min)

            # Signal strength tier based on RSI brackets
            if is_bull:
                min_rsi = min(rsi_5m, rsi_15m)
                if min_rsi >= 75:
                    signal_strength = "perfect"
                    expected_wr = 100.0
                    expected_rr = 10.5
                    expected_pnl = 2.64
                elif min_rsi >= 70:
                    signal_strength = "very_strong"
                    expected_wr = 98.0
                    expected_rr = 8.4
                    expected_pnl = 1.67
                else:
                    signal_strength = "strong"
                    expected_wr = 91.4
                    expected_rr = 5.5
                    expected_pnl = 1.09
            else:
                max_rsi = max(rsi_5m, rsi_15m)
                if max_rsi <= 25:
                    signal_strength = "perfect"
                    expected_wr = 98.3
                    expected_rr = 10.8
                    expected_pnl = 2.08
                elif max_rsi <= 30:
                    signal_strength = "very_strong"
                    expected_wr = 97.2
                    expected_rr = 7.1
                    expected_pnl = 1.34
                else:
                    signal_strength = "strong"
                    expected_wr = 92.1
                    expected_rr = 4.8
                    expected_pnl = 0.86

            # SL and target levels
            sl_pct = 0.5
            if is_bull:
                sl_price = round(cmp * (1 - sl_pct / 100), 2)
                target_price = round(cmp * (1 + expected_pnl / 100), 2)
            else:
                sl_price = round(cmp * (1 + sl_pct / 100), 2)
                target_price = round(cmp * (1 - expected_pnl / 100), 2)

            # Score: higher RSI alignment = stronger signal
            if is_bull:
                score = (rsi_5m - RSI_BULL_THRESH) + (rsi_15m - RSI_BULL_THRESH)
                if rsi_1h and rsi_1h > 60:
                    score += 5
            else:
                score = (RSI_BEAR_THRESH - rsi_5m) + (RSI_BEAR_THRESH - rsi_15m)
                if rsi_1h and rsi_1h < 40:
                    score += 5
            if strengthening:
                score += 3
            if time_zone == "afternoon":
                score += 5
            if signal_strength == "perfect":
                score += 10
            elif signal_strength == "very_strong":
                score += 5

            hold_note = "Hold 2H from entry" if can_hold_2h else "Hold till close — carry if needed"

            return {
                "symbol": canonical,
                "display_symbol": sym,
                "name": name,
                "cmp": cmp,
                "rsi_5m": rsi_5m,
                "rsi_15m": rsi_15m,
                "rsi_1h": rsi_1h,
                "rsi_5m_prev": rsi_5m_prev,
                "rsi_15m_prev": rsi_15m_prev,
                "signal_type": signal_type,
                "signal_strength": signal_strength,
                "expected_wr": expected_wr,
                "expected_rr": expected_rr,
                "expected_pnl": expected_pnl,
                "sl_pct": sl_pct,
                "sl_price": sl_price,
                "target_price": target_price,
                "strengthening": strengthening,
                "signal_time": signal_time,
                "time_zone": time_zone,
                "can_hold_2h": can_hold_2h,
                "hold_note": hold_note,
                "emas": ema_ctx,
                "score": round(score, 1),
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        results_raw = list(pool.map(_scan_stock, stocks))

    results = [r for r in results_raw if r is not None]
    results.sort(key=lambda r: r["score"], reverse=True)

    bt_file = Path("rsi_momentum_backtest.json")
    bt_data = {}
    if bt_file.exists():
        try:
            bt_data = json.loads(bt_file.read_text())
        except Exception:
            pass

    for r in results:
        sym = r["display_symbol"]
        bt = bt_data.get(sym, {})
        direction = r["signal_type"]
        bt_dir = bt.get("bull" if direction == "bullish" else "bear", {})
        r["backtest"] = {
            "trades": bt_dir.get("trades", 0),
            "win_rate": bt_dir.get("win_rate", 0),
            "avg_pnl": bt_dir.get("avg_pnl", 0),
            "rr": bt_dir.get("rr", 0),
        } if bt_dir else None

    bulls = [r for r in results if r["signal_type"] == "bullish"]
    bears = [r for r in results if r["signal_type"] == "bearish"]

    _save_rsi_cache("momentum", {
        "results": results,
        "total_scanned": len(stocks),
        "count": len(results),
        "bulls": len(bulls),
        "bears": len(bears),
        "time": now.strftime("%H:%M:%S"),
        "thresholds": {"bull": RSI_BULL_THRESH, "bear": RSI_BEAR_THRESH},
    })
    _rsi_scan_running["momentum"] = False


@app.route("/rsi-momentum/scan")
def rsi_momentum_scan():
    cached = _rsi_cache_fresh("momentum")
    if cached:
        cached.pop("_cached_at", None)
        return jsonify(cached)

    if not _rsi_scan_running["momentum"]:
        with _rsi_scan_locks["momentum"]:
            if not _rsi_scan_running["momentum"]:
                _rsi_scan_running["momentum"] = True
                threading.Thread(target=_do_rsi_momentum_scan, daemon=True).start()

    stale = None
    path = _rsi_cache_path("momentum")
    if path.exists():
        try:
            stale = json.loads(path.read_text())
            stale.pop("_cached_at", None)
            stale["_stale"] = True
            return jsonify(stale)
        except Exception:
            pass

    return jsonify({"scanning": True, "results": [], "count": 0, "bulls": 0, "bears": 0,
                    "time": datetime.now(IST).strftime("%H:%M:%S"), "thresholds": {"bull": 65, "bear": 35}})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
