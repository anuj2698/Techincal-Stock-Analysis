#!/usr/bin/env python3
"""Log predictions from the analysis engine for backtesting and performance tracking."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

IST = timezone(timedelta(hours=5, minutes=30))
PREDICTIONS_DIR = Path(__file__).resolve().parent / "predictions"


def _symbol_filename(symbol: str) -> str:
    return symbol.replace(":", "_").replace("/", "_") + ".json"


def _load_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return []


def _save_file(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def log_prediction(analysis_result: dict, symbol: str, timeframe: str, spot_price: float) -> str | None:
    rec = analysis_result.get("recommendation", {})
    ap = rec.get("action_plan", {})
    signals = rec.get("signals", {})
    indicators = rec.get("indicators", {})
    swing = analysis_result.get("swing")
    oi = analysis_result.get("oi")

    now = datetime.now(IST)
    today_str = now.strftime("%Y-%m-%d")

    file_path = PREDICTIONS_DIR / _symbol_filename(symbol)
    existing = _load_file(file_path)

    for p in existing:
        if p.get("date") == today_str and p.get("timeframe") == timeframe:
            return None

    chart_pats = [p.get("name", "") for p in analysis_result.get("chart_patterns", [])]
    candle_pats = [p.get("name", "") for p in analysis_result.get("candlestick_patterns", [])]

    prediction = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": now.isoformat(),
        "date": today_str,
        "symbol": symbol,
        "timeframe": timeframe,
        "spot_price": spot_price,
        "buy": ap.get("buy"),
        "sell": ap.get("sell"),
        "status": ap.get("status"),
        "signals": {k: v.get("score") for k, v in signals.items()},
        "signal_notes": {k: v.get("notes", []) for k, v in signals.items()},
        "indicators": {
            "rsi": indicators.get("rsi"),
            "sma20": indicators.get("sma20"),
            "sma50": indicators.get("sma50"),
            "sma200": indicators.get("sma200"),
            "ema8": indicators.get("ema8"),
            "atr14": indicators.get("atr14"),
            "rvol": indicators.get("rvol"),
            "macd_crossover": indicators.get("macd", {}).get("crossover") if indicators.get("macd") else None,
            "bb_squeeze": indicators.get("bollinger", {}).get("squeeze") if indicators.get("bollinger") else None,
        },
        "oi_summary": {
            "pcr": oi.get("pcr"),
            "max_pain": oi.get("max_pain"),
            "buildup": oi.get("oi_buildup"),
            "call_wall": oi.get("call_wall"),
            "put_wall": oi.get("put_wall"),
        } if oi else None,
        "patterns_detected": chart_pats + candle_pats,
        "swing": {
            "conviction": swing.get("conviction"),
            "bias": swing.get("bias"),
            "score": swing.get("score"),
        } if swing else None,
        "outcome": None,
    }

    existing.append(prediction)
    _save_file(file_path, existing)
    return prediction["id"]


def get_predictions(symbol: str, limit: int = 20) -> list[dict]:
    file_path = PREDICTIONS_DIR / _symbol_filename(symbol)
    data = _load_file(file_path)
    data.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
    return data[:limit]


def get_all_predictions(limit: int = 100) -> list[dict]:
    if not PREDICTIONS_DIR.exists():
        return []
    all_preds: list[dict] = []
    for f in PREDICTIONS_DIR.glob("*.json"):
        all_preds.extend(_load_file(f))
    all_preds.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
    return all_preds[:limit]


def update_outcome(symbol: str, prediction_id: str, outcome: dict) -> bool:
    file_path = PREDICTIONS_DIR / _symbol_filename(symbol)
    data = _load_file(file_path)
    for p in data:
        if p.get("id") == prediction_id:
            p["outcome"] = outcome
            _save_file(file_path, data)
            return True
    return False
