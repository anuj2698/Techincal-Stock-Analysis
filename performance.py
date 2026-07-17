#!/usr/bin/env python3
"""Performance analysis and indicator attribution for backtest results."""
from __future__ import annotations

import math
import statistics


def analyze_performance(predictions: list[dict]) -> dict:
    triggered = [p for p in predictions if p.get("outcome") and p["outcome"]["result"] in ("target_hit", "sl_hit")]
    if not triggered:
        return {"error": "No triggered trades to analyze"}

    returns = [p["outcome"]["pnl_pct"] for p in triggered]

    # Equity curve
    equity = []
    cumulative = 0
    for p in triggered:
        cumulative += p["outcome"]["pnl_pct"]
        equity.append({
            "date": p.get("date", ""),
            "pnl": round(p["outcome"]["pnl_pct"], 2),
            "cumulative": round(cumulative, 2),
        })

    # Max drawdown
    peak = 0
    max_dd = 0
    for e in equity:
        if e["cumulative"] > peak:
            peak = e["cumulative"]
        dd = peak - e["cumulative"]
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (annualized, assuming ~250 trading days, trades every 5 days)
    if len(returns) >= 2:
        avg_ret = statistics.fmean(returns)
        std_ret = statistics.stdev(returns)
        trades_per_year = 50
        sharpe = (avg_ret / std_ret) * math.sqrt(trades_per_year) if std_ret > 0 else 0
    else:
        sharpe = 0

    # Streaks
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for p in triggered:
        if p["outcome"]["result"] == "target_hit":
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # Best / worst
    best = max(triggered, key=lambda p: p["outcome"]["pnl_pct"])
    worst = min(triggered, key=lambda p: p["outcome"]["pnl_pct"])

    return {
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "best_trade": {
            "date": best.get("date"),
            "pnl_pct": round(best["outcome"]["pnl_pct"], 2),
            "side": best["outcome"]["side"],
            "entry": best["outcome"]["entry_price"],
        },
        "worst_trade": {
            "date": worst.get("date"),
            "pnl_pct": round(worst["outcome"]["pnl_pct"], 2),
            "side": worst["outcome"]["side"],
            "entry": worst["outcome"]["entry_price"],
        },
        "equity_curve": equity,
    }


def analyze_attribution(predictions: list[dict]) -> dict:
    triggered = [p for p in predictions if p.get("outcome") and p["outcome"]["result"] in ("target_hit", "sl_hit") and p.get("signals")]
    if not triggered:
        return {"signal_attribution": {}, "contribution_scores": {}}

    signal_keys = set()
    for p in triggered:
        signal_keys.update(p["signals"].keys())

    # Signal-level attribution
    signal_attr: dict[str, dict] = {}
    for key in sorted(signal_keys):
        bullish = [p for p in triggered if p["signals"].get(key, 50) > 55]
        neutral = [p for p in triggered if 45 <= p["signals"].get(key, 50) <= 55]
        bearish = [p for p in triggered if p["signals"].get(key, 50) < 45]

        def wr(trades):
            if not trades:
                return {"count": 0, "win_rate": 0, "wins": 0}
            wins = sum(1 for t in trades if t["outcome"]["result"] == "target_hit")
            return {"count": len(trades), "win_rate": round(wins / len(trades) * 100, 1), "wins": wins}

        bull_stats = wr(bullish)
        neut_stats = wr(neutral)
        bear_stats = wr(bearish)

        contribution = bull_stats["win_rate"] - bear_stats["win_rate"] if bull_stats["count"] > 0 and bear_stats["count"] > 0 else 0

        signal_attr[key] = {
            "bullish": bull_stats,
            "neutral": neut_stats,
            "bearish": bear_stats,
            "contribution": round(contribution, 1),
        }

    # Rank by contribution
    contribution_scores = {k: v["contribution"] for k, v in signal_attr.items()}

    # Indicator-level attribution
    indicator_attr: dict[str, list[dict]] = {}

    # RSI buckets
    rsi_data = _indicator_buckets(triggered, lambda p: _get_indicator_note(p, "momentum", "RSI"),
                                   {"Oversold (<40)": lambda n: "oversold" in n.lower() or "approaching oversold" in n.lower(),
                                    "Neutral (40-60)": lambda n: "neutral" in n.lower(),
                                    "Overbought (>60)": lambda n: "overbought" in n.lower() or "elevated" in n.lower()})
    if rsi_data:
        indicator_attr["RSI"] = rsi_data

    # MACD
    macd_data = _indicator_buckets(triggered, lambda p: _get_indicator_note(p, "trend", "MACD"),
                                    {"Bullish Crossover": lambda n: "bullish crossover" in n.lower(),
                                     "Bearish Crossover": lambda n: "bearish crossover" in n.lower(),
                                     "No Crossover": lambda n: "crossover" not in n.lower()})
    if macd_data:
        indicator_attr["MACD"] = macd_data

    # OI Buildup
    oi_data = _indicator_buckets(triggered, lambda p: _get_indicator_note(p, "oi", "buildup"),
                                  {"Long Buildup": lambda n: "long buildup" in n.lower(),
                                   "Short Buildup": lambda n: "short buildup" in n.lower(),
                                   "Short Covering": lambda n: "short covering" in n.lower(),
                                   "Long Unwinding": lambda n: "long unwinding" in n.lower()})
    if oi_data:
        indicator_attr["OI Buildup"] = oi_data

    # Trend (SMA alignment)
    trend_data = _indicator_buckets(triggered, lambda p: _get_indicator_note(p, "trend", "SMA"),
                                     {"Uptrend": lambda n: "uptrend" in n.lower() or "price > sma" in n.lower(),
                                      "Downtrend": lambda n: "downtrend" in n.lower() or "price < sma" in n.lower(),
                                      "Pullback/Recovery": lambda n: "pullback" in n.lower() or "recovery" in n.lower()})
    if trend_data:
        indicator_attr["Trend (SMA)"] = trend_data

    return {
        "signal_attribution": signal_attr,
        "contribution_scores": contribution_scores,
        "indicator_attribution": indicator_attr,
    }


def _get_indicator_note(prediction: dict, signal_key: str, keyword: str) -> str:
    notes = prediction.get("signal_notes", {})
    if not notes:
        outcome = prediction.get("outcome", {})
        if not outcome:
            return ""
    sig_notes = notes.get(signal_key, []) if isinstance(notes, dict) else []
    for note in sig_notes:
        if keyword.lower() in note.lower():
            return note
    return ""


def _indicator_buckets(predictions: list[dict], note_fn, buckets: dict) -> list[dict]:
    results = []
    for label, match_fn in buckets.items():
        matching = [p for p in predictions if match_fn(note_fn(p))]
        if not matching:
            results.append({"label": label, "count": 0, "win_rate": 0})
            continue
        wins = sum(1 for p in matching if p["outcome"]["result"] == "target_hit")
        results.append({
            "label": label,
            "count": len(matching),
            "win_rate": round(wins / len(matching) * 100, 1),
        })
    has_data = any(r["count"] > 0 for r in results)
    return results if has_data else []
