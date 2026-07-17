#!/usr/bin/env python3
"""Fetch Open Interest data from niftytrader.in for F&O stocks."""
from __future__ import annotations

import json
import re

import requests
from bs4 import BeautifulSoup


def _symbol_to_slug(symbol: str) -> str:
    sym = symbol.strip().upper()
    for prefix in ("NSE:", "BSE:"):
        if sym.startswith(prefix):
            sym = sym[len(prefix):]
    for suffix in ("-EQ", "-BE", "-SM", "-ST", "-B", "-A", "-M", "-Z", "-X", "-XT", "-P", "-T"):
        if sym.endswith(suffix):
            sym = sym[: -len(suffix)]
            break
    return sym.lower().replace("&", "")


def _compute_max_pain(strikes: list[dict]) -> float | None:
    if not strikes:
        return None
    strike_prices = [s["strike"] for s in strikes]
    min_pain = float("inf")
    max_pain_strike = strike_prices[0]

    for settle in strike_prices:
        total_pain = 0
        for s in strikes:
            call_itm = max(0, settle - s["strike"]) * s["call_oi"]
            put_itm = max(0, s["strike"] - settle) * s["put_oi"]
            total_pain += call_itm + put_itm
        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = settle

    return max_pain_strike


def _determine_buildup(spot: float, prev_close: float, total_oi_change: float) -> str:
    price_up = spot >= prev_close
    oi_up = total_oi_change > 0

    if price_up and oi_up:
        return "long_buildup"
    if not price_up and oi_up:
        return "short_buildup"
    if price_up and not oi_up:
        return "short_covering"
    return "long_unwinding"


def fetch_oi(symbol: str) -> dict | None:
    slug = _symbol_to_slug(symbol)
    url = f"https://www.niftytrader.in/stock-options-chart/{slug}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        }
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        if not script or not script.string:
            return None

        data = json.loads(script.string)
        props = data.get("props", {}).get("pageProps", {})

        oi_data = props.get("initialOiData")
        if not oi_data or not isinstance(oi_data, list) or len(oi_data) == 0:
            return None

        spot_data = props.get("initialSpot", {})
        spot = spot_data.get("last_trade_price", 0)
        prev_close = spot_data.get("close", spot)

        pcr = props.get("pcrSymbol")
        pcr_all = props.get("pcrAll")
        if isinstance(pcr, str):
            pcr = float(pcr) if pcr else None
        if isinstance(pcr_all, str):
            pcr_all = float(pcr_all) if pcr_all else None

        strikes = []
        for row in oi_data:
            strikes.append({
                "strike": row["strike_price"],
                "call_oi": row.get("calls_oi", 0),
                "put_oi": row.get("puts_oi", 0),
                "call_chg": row.get("calls_change_oi", 0),
                "put_chg": row.get("puts_change_oi", 0),
                "call_vol": row.get("calls_volume", 0),
                "put_vol": row.get("puts_volume", 0),
            })

        if not strikes:
            return None

        call_wall_row = max(strikes, key=lambda s: s["call_oi"])
        put_wall_row = max(strikes, key=lambda s: s["put_oi"])

        max_pain = _compute_max_pain(strikes)

        total_call_chg = sum(s["call_chg"] for s in strikes)
        total_put_chg = sum(s["put_chg"] for s in strikes)
        total_oi_change = total_call_chg + total_put_chg
        buildup = _determine_buildup(spot, prev_close, total_oi_change)

        atm_strikes = sorted(strikes, key=lambda s: abs(s["strike"] - spot))[:7]
        atm_strikes.sort(key=lambda s: s["strike"])

        return {
            "pcr": round(pcr, 3) if pcr else None,
            "pcr_all": round(pcr_all, 4) if pcr_all else None,
            "max_pain": max_pain,
            "call_wall": call_wall_row["strike"],
            "call_wall_oi": call_wall_row["call_oi"],
            "put_wall": put_wall_row["strike"],
            "put_wall_oi": put_wall_row["put_oi"],
            "oi_buildup": buildup,
            "total_call_chg_oi": total_call_chg,
            "total_put_chg_oi": total_put_chg,
            "spot": spot,
            "prev_close": prev_close,
            "expiry": oi_data[0].get("expiry_date", ""),
            "strikes_near_atm": atm_strikes,
        }

    except Exception:
        return None
