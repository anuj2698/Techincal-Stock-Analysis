#!/usr/bin/env python3
"""Backtest simulator — replay candles to validate predictions without look-ahead bias."""
from __future__ import annotations

import os
import math
from datetime import datetime, timedelta, timezone

from analyzer import full_analysis
from prediction_logger import get_predictions, update_outcome

IST = timezone(timedelta(hours=5, minutes=30))
HORIZON = {"intraday": 40, "short_term": 10, "positional": 40}
WARMUP = 65
BACKTEST_VERSION = 3
EXECUTION_COST_PCT = float(os.environ.get("BACKTEST_EXECUTION_COST_PCT", "0.15"))
MIN_SAMPLE_SIZE = int(os.environ.get("BACKTEST_MIN_SAMPLE_SIZE", "12"))


def _net_pnl(gross_pnl: float, execution_cost_pct: float) -> float:
    return gross_pnl - execution_cost_pct


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return a 95% Wilson score interval as percentages."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return round(max(0, centre - margin) * 100, 1), round(min(1, centre + margin) * 100, 1)


def _check_side(
    side: dict,
    future_candles: list[list[float]],
    is_buy: bool,
    horizon: int,
    execution_cost_pct: float = EXECUTION_COST_PCT,
) -> dict | None:
    level = side.get("level")
    sl = side.get("sl")
    targets = side.get("targets", [])
    if not level or not sl or not targets:
        return None

    entry_triggered = False
    entry_day = 0

    for i, c in enumerate(future_candles):
        if i >= horizon:
            break
        high, low = float(c[2]), float(c[3])

        if not entry_triggered:
            if is_buy and low <= level:
                entry_triggered = True
                entry_day = i
            elif not is_buy and high >= level:
                entry_triggered = True
                entry_day = i

            if not entry_triggered:
                continue

        days_since_entry = i - entry_day
        if days_since_entry > horizon:
            break

        if is_buy:
            if low <= sl:
                pnl = _net_pnl((sl - level) / level * 100, execution_cost_pct)
                return _outcome("sl_hit", "buy", level, sl, entry_day, i, pnl, None, future_candles)
            for ti, t in enumerate(targets):
                if high >= t:
                    pnl = _net_pnl((t - level) / level * 100, execution_cost_pct)
                    return _outcome("target_hit", "buy", level, t, entry_day, i, pnl, ti + 1, future_candles)
        else:
            if high >= sl:
                pnl = _net_pnl((level - sl) / level * 100, execution_cost_pct)
                return _outcome("sl_hit", "sell", level, sl, entry_day, i, pnl, None, future_candles)
            for ti, t in enumerate(targets):
                if low <= t:
                    pnl = _net_pnl((level - t) / level * 100, execution_cost_pct)
                    return _outcome("target_hit", "sell", level, t, entry_day, i, pnl, ti + 1, future_candles)

    if not entry_triggered:
        return {"result": "not_triggered", "side": "buy" if is_buy else "sell"}

    exit_idx = min(len(future_candles) - 1, horizon - 1)
    exit_price = float(future_candles[exit_idx][4])
    gross_pnl = ((exit_price - level) / level * 100) if is_buy else ((level - exit_price) / level * 100)
    return _outcome(
        "expired", "buy" if is_buy else "sell", level, exit_price, entry_day,
        exit_idx, _net_pnl(gross_pnl, execution_cost_pct), None, future_candles,
    )


def _outcome(result, side, entry, exit_price, entry_idx, exit_idx, pnl, tgt_num, candles):
    entry_date = datetime.fromtimestamp(int(candles[entry_idx][0]), tz=IST).strftime("%Y-%m-%d") if entry_idx < len(candles) else None
    exit_date = datetime.fromtimestamp(int(candles[exit_idx][0]), tz=IST).strftime("%Y-%m-%d") if exit_idx < len(candles) else None
    return {
        "result": result,
        "side": side,
        "entry_price": round(entry, 2) if entry else None,
        "exit_price": round(exit_price, 2) if exit_price else None,
        "entry_triggered_on": entry_date,
        "exit_date": exit_date,
        "pnl_pct": round(pnl, 2) if pnl else 0,
        "days_to_entry": entry_idx,
        "days_to_exit": exit_idx,
        "target_hit": tgt_num,
    }


def backtest_prediction(
    prediction: dict,
    future_candles: list[list[float]],
    horizon: int,
    execution_cost_pct: float = EXECUTION_COST_PCT,
) -> dict | None:
    buy = prediction.get("buy")
    sell = prediction.get("sell")
    status = prediction.get("status", "NO TRADE")

    action_summary = prediction.get("action_summary") or {}
    verdict = str(action_summary.get("verdict", "")).upper()

    # New predictions carry an explicit verdict. Older logged predictions only
    # carry status, so retain that as a backwards-compatible fallback.
    if verdict == "BUY" or (not verdict and status == "BUY ZONE"):
        return _check_side(buy, future_candles, True, horizon, execution_cost_pct) if buy else None
    if verdict == "SELL" or (not verdict and status == "SELL ZONE"):
        return _check_side(sell, future_candles, False, horizon, execution_cost_pct) if sell else None

    return {"result": "skipped", "side": None, "reason": "no_actionable_verdict"}


def run_backtest(symbol: str, candles: list[list[float]], timeframe: str = "positional") -> dict:
    horizon = HORIZON.get(timeframe, 40)
    total = len(candles)
    if total < WARMUP + horizon:
        return {"version": BACKTEST_VERSION, "error": "Insufficient data", "predictions": [], "summary": {}}

    predictions: list[dict] = []
    # Do not open overlapping test positions for the same stock.
    step = max(5, horizon)

    for i in range(WARMUP, total - horizon, step):
        past = candles[:i + 1]
        future = candles[i + 1: i + 1 + horizon]
        if len(future) < 3:
            continue

        pred_date = datetime.fromtimestamp(int(candles[i][0]), tz=IST).strftime("%Y-%m-%d")
        spot = float(candles[i][4])

        result = full_analysis(symbol, past, timeframe=timeframe)
        ap = result.get("recommendation", {}).get("action_plan", {})
        signals = result.get("recommendation", {}).get("signals", {})

        outcome = backtest_prediction(ap, future, horizon)

        predictions.append({
            "date": pred_date,
            "spot": round(spot, 2),
            "buy": ap.get("buy"),
            "sell": ap.get("sell"),
            "status": ap.get("status"),
            "action_summary": ap.get("action_summary", {}),
            "signals": {k: v.get("score") for k, v in signals.items()},
            "signal_notes": {k: v.get("notes", []) for k, v in signals.items()},
            "outcome": outcome,
        })

    summary = _compute_summary(predictions)
    train_end = int(len(predictions) * 0.60)
    validation_end = int(len(predictions) * 0.80)
    segments = {
        "train": predictions[:train_end],
        "validation": predictions[train_end:validation_end],
        "test": predictions[validation_end:],
    }
    segment_summaries = {name: _compute_summary(rows) for name, rows in segments.items()}
    test_summary = segment_summaries["test"]
    summary["out_of_sample"] = {
        "split": "chronological_60_20_20",
        "segments": segment_summaries,
        "eligible": (
            summary["triggered"] >= MIN_SAMPLE_SIZE
            and test_summary["triggered"] >= 8
            and test_summary["expectancy_pct"] > 0
            and test_summary["profit_factor"] > 1
        ),
    }
    return {"version": BACKTEST_VERSION, "predictions": predictions, "summary": summary}


def _compute_summary(predictions: list[dict]) -> dict:
    triggered = [p for p in predictions if p["outcome"] and p["outcome"]["result"] in ("target_hit", "sl_hit", "expired")]
    not_triggered = [p for p in predictions if not p["outcome"] or p["outcome"]["result"] == "not_triggered"]
    expired = [p for p in predictions if p["outcome"] and p["outcome"]["result"] == "expired"]
    skipped = [p for p in predictions if p["outcome"] and p["outcome"]["result"] == "skipped"]

    wins = [p for p in triggered if p["outcome"]["pnl_pct"] > 0]
    losses = [p for p in triggered if p["outcome"]["pnl_pct"] <= 0]

    total = len(predictions)
    total_triggered = len(triggered)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_triggered * 100) if total_triggered > 0 else 0

    avg_win = sum(p["outcome"]["pnl_pct"] for p in wins) / win_count if wins else 0
    avg_loss = sum(p["outcome"]["pnl_pct"] for p in losses) / loss_count if losses else 0
    total_pnl = sum(p["outcome"]["pnl_pct"] for p in triggered)

    avg_days = sum(p["outcome"]["days_to_exit"] - p["outcome"]["days_to_entry"] for p in triggered) / total_triggered if triggered else 0
    avg_days_win = sum(p["outcome"]["days_to_exit"] - p["outcome"]["days_to_entry"] for p in wins) / win_count if wins else 0
    avg_days_loss = sum(p["outcome"]["days_to_exit"] - p["outcome"]["days_to_entry"] for p in losses) / loss_count if losses else 0

    gross_wins = sum(p["outcome"]["pnl_pct"] for p in wins) if wins else 0
    gross_losses = abs(sum(p["outcome"]["pnl_pct"] for p in losses)) if losses else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 99.0 if gross_wins > 0 else 0

    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss) if total_triggered > 0 else 0

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for prediction in triggered:
        equity += prediction["outcome"]["pnl_pct"]
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    confidence = "sufficient" if total_triggered >= MIN_SAMPLE_SIZE else "insufficient"
    ci_low, ci_high = _wilson_interval(win_count, total_triggered)
    return {
        "methodology_version": BACKTEST_VERSION,
        "execution_cost_pct": EXECUTION_COST_PCT,
        "minimum_sample_size": MIN_SAMPLE_SIZE,
        "sample_confidence": confidence,
        "total_predictions": total,
        "triggered": total_triggered,
        "not_triggered": len(not_triggered),
        "expired": len(expired),
        "skipped": len(skipped),
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(win_rate, 1),
        "win_rate_ci_95": [ci_low, ci_high],
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "total_pnl_pct": round(total_pnl, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy_pct": round(expectancy, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "avg_days_held": round(avg_days, 1),
        "avg_days_win": round(avg_days_win, 1),
        "avg_days_loss": round(avg_days_loss, 1),
    }


def backtest_logged_predictions(symbol: str, candles: list[list[float]]) -> int:
    preds = get_predictions(symbol, limit=100)
    updated = 0

    for p in preds:
        if p.get("outcome"):
            continue

        pred_date = p.get("date")
        if not pred_date:
            continue

        timeframe = p.get("timeframe", "positional")
        horizon = HORIZON.get(timeframe, 40)

        pred_ts = None
        for i, c in enumerate(candles):
            d = datetime.fromtimestamp(int(c[0]), tz=IST).strftime("%Y-%m-%d")
            if d == pred_date:
                pred_ts = i
                break

        if pred_ts is None:
            continue

        future = candles[pred_ts + 1:]
        if len(future) < 3:
            continue

        outcome = backtest_prediction(p, future, horizon)
        if outcome and outcome["result"] != "not_triggered":
            if update_outcome(symbol, p["id"], outcome):
                updated += 1

    return updated
