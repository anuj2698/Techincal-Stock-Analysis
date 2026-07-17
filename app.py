#!/usr/bin/env python3
"""Flask web app for Indian stock analysis — buy/sell zones, chart patterns, swing setups."""
from __future__ import annotations

import json
import math
import os
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf
from flask import Flask, jsonify, render_template, request

from analyzer import full_analysis
from oi_fetcher import fetch_oi
from prediction_logger import log_prediction, get_predictions, get_all_predictions
from backtester import run_backtest, backtest_logged_predictions
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


def resolve_yahoo_ticker(symbol: str) -> tuple[str, str]:
    sym = symbol.strip().upper()

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


def _cache_path(yahoo_sym: str) -> Path:
    safe = yahoo_sym.replace(".", "_").replace("^", "_")
    return CACHE_DIR / f"{safe}.json"


def _save_cache(yahoo_sym: str, candles: list[list[float]]) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(yahoo_sym)
    path.write_text(json.dumps({"ts": datetime.now(IST).isoformat(), "candles": candles}))


def _load_cache(yahoo_sym: str) -> list[list[float]] | None:
    path = _cache_path(yahoo_sym)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("candles")
    except Exception:
        return None


def fetch_candles(yahoo_sym: str, period: str = "1y") -> list[list[float]]:
    tkr = yf.Ticker(yahoo_sym)
    df = tkr.history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        cached = _load_cache(yahoo_sym)
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
        cached = _load_cache(yahoo_sym)
        return cached if cached else []

    # If Yahoo returned fewer recent candles than cache, merge the latest from cache
    cached = _load_cache(yahoo_sym)
    if cached and len(cached) > 0:
        cached_last_ts = cached[-1][0]
        fresh_last_ts = rows[-1][0]
        if cached_last_ts > fresh_last_ts:
            # Cache has more recent data (Yahoo returned stale/NaN data that got dropped)
            # Merge: use fresh rows up to their last timestamp, then append cached rows after
            merged_map: dict[int, list[float]] = {}
            for r in cached:
                merged_map[int(r[0])] = r
            for r in rows:
                merged_map[int(r[0])] = r
            rows = sorted(merged_map.values(), key=lambda r: r[0])

    _save_cache(yahoo_sym, rows)
    return rows


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze")
def analyze():
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"error": "Please enter a stock symbol"}), 400

    timeframe = request.args.get("timeframe", "positional").strip()
    if timeframe not in ("short_term", "positional"):
        timeframe = "positional"
    period = "3mo" if timeframe == "short_term" else "1y"
    entry_price_str = request.args.get("entry_price", "").strip()
    entry_price = float(entry_price_str) if entry_price_str else None

    try:
        canonical, yahoo_sym = resolve_yahoo_ticker(symbol)
        candles = fetch_candles(yahoo_sym, period=period)
        if not candles:
            return jsonify({"error": f"No data found for '{symbol}'. Try formats like: RELIANCE, TCS, INFY"}), 404

        last_ts = candles[-1][0]
        last_date = datetime.fromtimestamp(last_ts, tz=IST).strftime("%d %b %Y")

        oi_data = fetch_oi(canonical)
        result = full_analysis(canonical, candles, timeframe=timeframe, entry_price=entry_price, oi_data=oi_data)
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
        candles = fetch_candles(yahoo_sym, period=period)
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
        candles = fetch_candles(yahoo_sym, period="1y")
        if not candles:
            return jsonify({"error": "No data"}), 404

        count = backtest_logged_predictions(canonical, candles)
        return jsonify({"updated": count, "symbol": canonical})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
