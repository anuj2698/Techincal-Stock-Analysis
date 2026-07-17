#!/usr/bin/env python3
"""Backtest simulator — replay candles to validate predictions without look-ahead bias."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from analyzer import full_analysis, TIMEFRAME_CONFIG
from prediction_logger import get_predictions, update_outcome

IST = timezone(timedelta(hours=5, minutes=30))
HORIZON = {"short_term": 10, "positional": 40}
WARMUP = 65


def _check_side(side: dict, future_candles: list[list[float]], is_buy: bool, horizon: int) -> dict | None:
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
        low, high = float(c[3]), float(c[2])

        if not entry_triggered:
            if is_buy and low <= level:
                entry_triggered = True
                entry_day = i
            elif not is_buy and high >= level:
                entry_triggered = True
                entry_day = i
            continue

        days_since_entry = i - entry_day
        if days_since_entry > horizon:
            break

        if is_buy:
            if low <= sl:
                pnl = (sl - level) / level * 100
                return _outcome("sl_hit", "buy", level, sl, entry_day, i, pnl, None, future_candles)
            for ti, t in enumerate(targets):
                if high >= t:
                    pnl = (t - level) / level * 100
                    return _outcome("target_hit", "buy", level, t, entry_day, i, pnl, ti + 1, future_candles)
        else:
            if high >= sl:
                pnl = (level - sl) / level * 100
                return _outcome("sl_hit", "sell", level, sl, entry_day, i, pnl, None, future_candles)
            for ti, t in enumerate(targets):
                if low <= t:
                    pnl = (level - t) / level * 100
                    return _outcome("target_hit", "sell", level, t, entry_day, i, pnl, ti + 1, future_candles)

    if not entry_triggered:
        return {"result": "not_triggered", "side": "buy" if is_buy else "sell"}

    return _outcome("expired", "buy" if is_buy else "sell", level, None, entry_day, min(len(future_candles) - 1, horizon - 1), 0, None, future_candles)


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


def backtest_prediction(prediction: dict, future_candles: list[list[float]], horizon: int) -> dict | None:
    buy = prediction.get("buy")
    sell = prediction.get("sell")
    status = prediction.get("status", "NO TRADE")

    results = []
    if status == "BUY ZONE" and buy:
        r = _check_side(buy, future_candles, is_buy=True, horizon=horizon)
        if r:
            results.append(r)
    elif status == "SELL ZONE" and sell:
        r = _check_side(sell, future_candles, is_buy=False, horizon=horizon)
        if r:
            results.append(r)
    else:
        if buy:
            r = _check_side(buy, future_candles, is_buy=True, horizon=horizon)
            if r:
                results.append(r)
        if sell:
            r = _check_side(sell, future_candles, is_buy=False, horizon=horizon)
            if r:
                results.append(r)

    if not results:
        return None

    triggered = [r for r in results if r["result"] != "not_triggered"]
    if triggered:
        wins = [r for r in triggered if r["result"] == "target_hit"]
        if wins:
            return wins[0]
        return triggered[0]
    return results[0]


def run_backtest(symbol: str, candles: list[list[float]], timeframe: str = "positional") -> dict:
    horizon = HORIZON.get(timeframe, 40)
    total = len(candles)
    if total < WARMUP + horizon:
        return {"error": "Insufficient data", "predictions": [], "summary": {}}

    predictions: list[dict] = []
    step = 5

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
            "signals": {k: v.get("score") for k, v in signals.items()},
            "signal_notes": {k: v.get("notes", []) for k, v in signals.items()},
            "outcome": outcome,
        })

    summary = _compute_summary(predictions)
    return {"predictions": predictions, "summary": summary}


def _compute_summary(predictions: list[dict]) -> dict:
    triggered = [p for p in predictions if p["outcome"] and p["outcome"]["result"] in ("target_hit", "sl_hit")]
    not_triggered = [p for p in predictions if not p["outcome"] or p["outcome"]["result"] == "not_triggered"]
    expired = [p for p in predictions if p["outcome"] and p["outcome"]["result"] == "expired"]

    wins = [p for p in triggered if p["outcome"]["result"] == "target_hit"]
    losses = [p for p in triggered if p["outcome"]["result"] == "sl_hit"]

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
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0

    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss) if total_triggered > 0 else 0

    return {
        "total_predictions": total,
        "triggered": total_triggered,
        "not_triggered": len(not_triggered),
        "expired": len(expired),
        "wins": win_count,
        "losses": loss_count,
        "win_rate": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "total_pnl_pct": round(total_pnl, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy_pct": round(expectancy, 2),
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
