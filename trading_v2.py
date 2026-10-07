"""Version 2 trading primitives: session-correct backtests and account risk gates."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, time, timedelta, timezone


IST = timezone(timedelta(hours=5, minutes=30))


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [round(max(0, centre - margin) * 100, 1), round(min(1, centre + margin) * 100, 1)]


def group_intraday_sessions(candles: list[list[float]]) -> dict[str, list[list[float]]]:
    """Group and sort exchange candles without allowing indicators/trades across days."""
    sessions: dict[str, list[list[float]]] = defaultdict(list)
    for candle in candles:
        dt = datetime.fromtimestamp(int(candle[0]), tz=IST)
        sessions[dt.strftime("%Y-%m-%d")].append(candle)
    return {day: sorted(rows, key=lambda row: row[0]) for day, rows in sorted(sessions.items())}


def _trade_metrics(trades: list[dict], execution_cost_pct: float, evaluated_sessions: int) -> dict:
    decided = [trade for trade in trades if trade["result"] in {"target", "stop", "eod"}]
    wins = [trade for trade in decided if trade["net_pnl_pct"] > 0]
    losses = [trade for trade in decided if trade["net_pnl_pct"] <= 0]
    gross_profit = sum(trade["net_pnl_pct"] for trade in wins)
    gross_loss = abs(sum(trade["net_pnl_pct"] for trade in losses))

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in decided:
        equity += trade["net_pnl_pct"]
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    count = len(decided)
    win_rate = len(wins) / count * 100 if count else 0.0
    expectancy = sum(trade["net_pnl_pct"] for trade in decided) / count if count else 0.0
    return {
        "methodology_version": 3,
        "strategy": "PDH/PDL 5m session breakout",
        "execution_cost_pct": execution_cost_pct,
        "total_sessions": evaluated_sessions,
        "ambiguous_sessions": sum(1 for trade in trades if trade.get("result") == "ambiguous"),
        "triggered": count,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "win_rate_ci_95": _wilson_interval(len(wins), count),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else (99.0 if gross_profit else 0.0),
        "expectancy_pct": round(expectancy, 3),
        "total_pnl_pct": round(sum(trade["net_pnl_pct"] for trade in decided), 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sample_confidence": "sufficient" if count >= 30 else "insufficient",
        "overnight_positions": 0,
    }


def backtest_pdhl_intraday(
    candles: list[list[float]],
    *,
    buffer_pct: float = 0.1,
    stop_range_ratio: float = 0.3,
    reward_risk: float = 1.0,
    execution_cost_pct: float = 0.15,
    entry_not_before: time = time(9, 30),
    last_entry_time: time = time(14, 30),
    forced_exit_time: time = time(15, 20),
) -> dict:
    """Backtest one PDH/PDL trade per day with no overnight positions.

    Fills that gap through a trigger use the opening price, same-bar stop/target
    ambiguity is resolved stop-first, and open positions are closed at 15:20.
    """
    sessions = group_intraday_sessions(candles)
    days = list(sessions)
    trades: list[dict] = []

    for index in range(1, len(days)):
        day = days[index]
        previous = sessions[days[index - 1]]
        current = sessions[day]
        if not previous or not current:
            continue

        pdh = max(float(candle[2]) for candle in previous)
        pdl = min(float(candle[3]) for candle in previous)
        previous_range = pdh - pdl
        if previous_range <= 0:
            continue

        long_trigger = pdh * (1 + buffer_pct / 100)
        short_trigger = pdl * (1 - buffer_pct / 100)
        planned_long_stop = pdh - previous_range * stop_range_ratio
        planned_short_stop = pdl + previous_range * stop_range_ratio
        eligible = [
            candle for candle in current
            if entry_not_before <= datetime.fromtimestamp(int(candle[0]), tz=IST).time() <= forced_exit_time
        ]
        if not eligible:
            continue

        trade = None
        for candle in eligible:
            dt = datetime.fromtimestamp(int(candle[0]), tz=IST)
            if dt.time() > last_entry_time:
                break
            open_, high, low = map(float, candle[1:4])
            long_touched = high >= long_trigger
            short_touched = low <= short_trigger
            if long_touched and short_touched:
                trade = {"date": day, "result": "ambiguous", "reason": "both_triggers_same_candle"}
                break
            if long_touched:
                fill = max(open_, long_trigger)
                risk = fill - planned_long_stop
                if risk <= 0:
                    break
                trade = {
                    "date": day, "side": "long", "entry_time": dt.strftime("%H:%M"),
                    "entry": fill, "stop": planned_long_stop, "target": fill + risk * reward_risk,
                }
                break
            if short_touched:
                fill = min(open_, short_trigger)
                risk = planned_short_stop - fill
                if risk <= 0:
                    break
                trade = {
                    "date": day, "side": "short", "entry_time": dt.strftime("%H:%M"),
                    "entry": fill, "stop": planned_short_stop, "target": fill - risk * reward_risk,
                }
                break

        if not trade:
            continue
        if trade.get("result") == "ambiguous":
            trades.append(trade)
            continue

        entry_index = next(
            i for i, candle in enumerate(eligible)
            if datetime.fromtimestamp(int(candle[0]), tz=IST).strftime("%H:%M") == trade["entry_time"]
        )
        exit_price = None
        result = "eod"
        exit_time = None
        for candle in eligible[entry_index:]:
            dt = datetime.fromtimestamp(int(candle[0]), tz=IST)
            high, low = float(candle[2]), float(candle[3])
            if trade["side"] == "long":
                if low <= trade["stop"]:
                    result, exit_price = "stop", trade["stop"]
                elif high >= trade["target"]:
                    result, exit_price = "target", trade["target"]
            else:
                if high >= trade["stop"]:
                    result, exit_price = "stop", trade["stop"]
                elif low <= trade["target"]:
                    result, exit_price = "target", trade["target"]
            if exit_price is not None:
                exit_time = dt.strftime("%H:%M")
                break

        if exit_price is None:
            eod_candle = eligible[-1]
            exit_price = float(eod_candle[4])
            exit_time = datetime.fromtimestamp(int(eod_candle[0]), tz=IST).strftime("%H:%M")

        if trade["side"] == "long":
            gross_pnl = (exit_price - trade["entry"]) / trade["entry"] * 100
        else:
            gross_pnl = (trade["entry"] - exit_price) / trade["entry"] * 100
        trade.update({
            "result": result,
            "exit": round(exit_price, 2),
            "exit_time": exit_time,
            "gross_pnl_pct": round(gross_pnl, 3),
            "net_pnl_pct": round(gross_pnl - execution_cost_pct, 3),
        })
        for key in ("entry", "stop", "target"):
            trade[key] = round(trade[key], 2)
        trades.append(trade)

    summary = _trade_metrics(trades, execution_cost_pct, max(0, len(days) - 1))
    decided = [trade for trade in trades if trade.get("result") in {"target", "stop", "eod"}]
    train_end = int(len(decided) * 0.60)
    validation_end = int(len(decided) * 0.80)
    segments = {
        "train": decided[:train_end],
        "validation": decided[train_end:validation_end],
        "test": decided[validation_end:],
    }
    segment_metrics = {
        name: _trade_metrics(rows, execution_cost_pct, len(rows))
        for name, rows in segments.items()
    }
    test_metrics = segment_metrics["test"]
    summary["out_of_sample"] = {
        "split": "chronological_60_20_20",
        "segments": segment_metrics,
        "eligible": (
            summary["triggered"] >= 30
            and test_metrics["triggered"] >= 8
            and test_metrics["expectancy_pct"] > 0
            and test_metrics["profit_factor"] > 1
        ),
    }
    return {"summary": summary, "trades": trades}


def account_risk_status(
    journal: list[dict],
    *,
    capital: float,
    risk_per_trade_pct: float = 0.5,
    max_daily_loss_pct: float = 1.0,
    max_trades_per_day: int = 3,
    max_open_positions: int = 2,
) -> dict:
    """Return a deterministic trade gate from today's journal state."""
    completed = [row for row in journal if row.get("status") == "closed"]
    open_positions = [row for row in journal if row.get("status") == "open"]
    realized_pnl = sum(float(row.get("pnl", 0) or 0) for row in completed)
    daily_loss_limit = capital * max_daily_loss_pct / 100
    reasons = []
    if realized_pnl <= -daily_loss_limit:
        reasons.append("Daily loss limit reached")
    if len(journal) >= max_trades_per_day:
        reasons.append("Maximum trades reached")
    if len(open_positions) >= max_open_positions:
        reasons.append("Maximum open positions reached")
    return {
        "can_open_trade": not reasons,
        "reasons": reasons,
        "capital": round(capital, 2),
        "risk_per_trade": round(capital * risk_per_trade_pct / 100, 2),
        "daily_loss_limit": round(daily_loss_limit, 2),
        "realized_pnl": round(realized_pnl, 2),
        "trades_taken": len(journal),
        "open_positions": len(open_positions),
        "limits": {
            "risk_per_trade_pct": risk_per_trade_pct,
            "max_daily_loss_pct": max_daily_loss_pct,
            "max_trades_per_day": max_trades_per_day,
            "max_open_positions": max_open_positions,
        },
    }
