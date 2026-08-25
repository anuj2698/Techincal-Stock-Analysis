#!/usr/bin/env python3
"""Fetch F&O stock result dates, expected moves, and post-result analytics."""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import requests as _requests
from nselib import capital_market

warnings.filterwarnings("ignore")

CACHE_DIR = Path(__file__).resolve().parent / "results_cache"
CACHE_FILE = CACHE_DIR / "results_data.json"
HISTORY_FILE = CACHE_DIR / "results_history.json"

_fetch_lock = threading.Lock()
_fetch_running = False
_fetch_progress = {"step": "", "done": 0, "total": 0, "status": "idle"}


def _load_history() -> dict:
    """Load history of finalized announced stocks, keyed by quarter label."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return {}


def _save_history(history: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2, default=str))


def _is_finalized(stock: dict) -> bool:
    """A stock is finalized when it's announced with complete post-result data."""
    return (
        stock.get("result_status") == "announced"
        and stock.get("post_result_change_pct") is not None
        and stock.get("revenue_cr") is not None
        and stock.get("cmp") is not None
        and stock.get("rsi") is not None
        and stock.get("expected_move_pct") is not None
    )


def get_results_quarter() -> tuple[str, str, str]:
    """Return (label, from_date, to_date) for the current results season."""
    today = date.today()
    m, y = today.month, today.year

    if 7 <= m <= 9:
        fy = y - 2000 + 1
        return f"Q1 FY{fy}", f"01-07-{y}", f"30-09-{y}"
    elif 10 <= m <= 12:
        fy = y - 2000 + 1
        return f"Q2 FY{fy}", f"01-10-{y}", f"31-12-{y}"
    elif 1 <= m <= 3:
        fy = y - 2000
        return f"Q3 FY{fy}", f"01-01-{y}", f"31-03-{y}"
    else:
        fy = y - 2000 + 1
        return f"Q4 FY{fy - 1}", f"01-04-{y}", f"30-06-{y}"


def _mc_week_starts(from_date: str, to_date: str) -> list[str]:
    """Generate weekly start dates for MoneyControl scraping."""
    d = datetime.strptime(from_date, "%d-%m-%Y").date()
    end = datetime.strptime(to_date, "%d-%m-%Y").date()
    starts = []
    while d <= end:
        starts.append(d.isoformat())
        d += timedelta(days=9)
    return starts


def fetch_fno_stocks() -> dict:
    df = capital_market.fno_equity_list()
    stocks = {}
    for _, row in df.iterrows():
        symbol = row["symbol"]
        stocks[symbol] = {
            "symbol": symbol,
            "name": row.get("underlying", symbol),
            "sector": "N/A",
            "cmp": None,
            "prev_close": None,
            "change_pct": None,
            "result_date": None,
            "result_status": "not_announced",
            "expected_move_pct": None,
            "direction": "neutral",
            "rsi": None,
            "dma_50": None,
            "above_dma": None,
            "atm_call_premium": None,
            "atm_put_premium": None,
            "atm_strike": None,
            "expiry_used": None,
            "revenue_cr": None,
            "net_profit_cr": None,
            "eps": None,
            "revenue_growth_yoy": None,
            "profit_growth_yoy": None,
            "pre_result_close": None,
            "next_day_close": None,
            "next_day_date": None,
            "post_result_change_pct": None,
            "beat_expected_move": None,
        }
    return stocks


def fetch_result_dates(from_date: str, to_date: str) -> dict:
    df = capital_market.event_calendar_for_equity(
        from_date=from_date, to_date=to_date, fno_only=True
    )
    meetings = {}
    for _, row in df.iterrows():
        symbol = row["symbol"]
        purpose = str(row.get("purpose", "")).lower()
        date_str = row["date"]

        is_results = any(
            kw in purpose
            for kw in ["financial results", "quarterly results", "un-audited", "unaudited"]
        )
        if not is_results:
            continue

        try:
            parsed = datetime.strptime(date_str.strip(), "%d-%b-%Y").date()
        except ValueError:
            continue

        if symbol not in meetings or parsed > meetings[symbol]["date"]:
            meetings[symbol] = {"date": parsed, "purpose": row.get("purpose", "")}

    return meetings


def fetch_moneycontrol_fallback(fno_symbols: set, already_have: dict, from_date: str, to_date: str) -> dict:
    missing = fno_symbols - set(already_have.keys())
    if not missing:
        return {}

    mc_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    found = {}
    seen_dates = set()
    week_starts = _mc_week_starts(from_date, to_date)

    for ws in week_starts:
        try:
            url = f"https://www.moneycontrol.com/markets/earnings/results-calendar/?activeDate={ws}"
            r = _requests.get(url, headers=mc_headers, timeout=15)
            if r.status_code != 200:
                continue

            html = r.text
            idx = html.find("__NEXT_DATA__")
            if idx < 0:
                continue
            start = html.find("{", idx)
            end_idx = html.find("</script>", start)
            data = json.loads(html[start:end_idx])
            rc = data["props"]["pageProps"]["resultCalendarData"]

            dates_in_strip = rc.get("dateStripData", {}).get("list", [])
            dates_to_fetch = []
            for d_info in dates_in_strip:
                dt = d_info[2]
                cnt = int(d_info[1]) if d_info[1] else 0
                if cnt > 0 and dt not in seen_dates:
                    dates_to_fetch.append(dt)
                    seen_dates.add(dt)

            for fetch_date in dates_to_fetch:
                time.sleep(0.3)
                try:
                    url2 = f"https://www.moneycontrol.com/markets/earnings/results-calendar/?activeDate={fetch_date}"
                    r2 = _requests.get(url2, headers=mc_headers, timeout=15)
                    if r2.status_code != 200:
                        continue
                    html2 = r2.text
                    idx2 = html2.find("__NEXT_DATA__")
                    if idx2 < 0:
                        continue
                    start2 = html2.find("{", idx2)
                    end_idx2 = html2.find("</script>", start2)
                    data2 = json.loads(html2[start2:end_idx2])
                    rc2 = data2["props"]["pageProps"]["resultCalendarData"]
                    items = rc2.get("tableData", {}).get("list", [])

                    for item in items:
                        stock_url = item.get("stockUrl", "")
                        slug = stock_url.rstrip("/").split("/")[-2] if "/" in stock_url else ""
                        slug_clean = slug.lower().replace("-", "").replace("limited", "").replace("ltd", "")

                        matched_symbol = None
                        for sym in missing:
                            sym_clean = sym.lower().replace("&", "").replace("-", "")
                            if sym_clean == slug_clean or slug_clean.startswith(sym_clean) or sym_clean.startswith(slug_clean):
                                matched_symbol = sym
                                break

                        if matched_symbol and matched_symbol not in found:
                            result_date = datetime.strptime(fetch_date, "%Y-%m-%d").date()
                            found[matched_symbol] = {"date": result_date, "purpose": item.get("resultType", "")}
                except Exception:
                    continue
        except Exception:
            continue
        time.sleep(0.3)

    return found


def _fetch_sector(symbol: str) -> tuple[str, str | None]:
    mc_headers = {"User-Agent": "Mozilla/5.0"}
    try:
        encoded = quote(symbol)
        url = (
            f"https://www.moneycontrol.com/mccode/common/autosuggestion_solr.php"
            f"?classic=true&query={encoded}&type=1&format=json"
        )
        r = _requests.get(url, headers=mc_headers, timeout=10)
        data = r.json()
        for item in data:
            dis = item.get("pdt_dis_nm", "")
            m = re.search(r"<span>[^,]+,\s*([^,]+),", dis)
            if m and m.group(1).strip() == symbol:
                return symbol, item.get("sc_sector", "N/A")
    except Exception:
        pass
    return symbol, None


def fetch_sectors(symbols: list[str]) -> dict:
    sectors = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        for sym, sector in pool.map(_fetch_sector, symbols):
            if sector:
                sectors[sym] = sector
    return sectors


def fetch_option_chain(symbol: str) -> dict | None:
    try:
        resp = _requests.get(
            f"https://www.niftytrader.in/nse-option-chain/{symbol.lower()}",
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
                "Accept": "text/html",
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text)
        if not m:
            return None
        data = json.loads(m.group(1))
    except Exception:
        return None

    pp = data.get("props", {}).get("pageProps", {})
    spot = pp.get("initialSpot")
    chain = pp.get("initialOptionChainData")

    if not spot or not chain:
        return None

    underlying = spot.get("last_trade_price") or spot.get("close")
    if not underlying:
        return None

    strikes = sorted(set(item["strike_price"] for item in chain))
    atm_strike = min(strikes, key=lambda s: abs(s - underlying))

    atm_row = next((r for r in chain if r["strike_price"] == atm_strike), None)
    if not atm_row:
        return None

    call_ltp = atm_row.get("calls_ltp", 0) or 0
    put_ltp = atm_row.get("puts_ltp", 0) or 0

    expected_move_pct = (
        round(((call_ltp + put_ltp) / underlying) * 100, 2) if underlying else 0
    )

    expiry = atm_row.get("expiry_date", "")
    if expiry:
        expiry = expiry.split("T")[0]

    prev_close = spot.get("close")
    change_pct = spot.get("change_per")

    return {
        "cmp": round(underlying, 2),
        "prev_close": round(prev_close, 2) if prev_close else None,
        "change_pct": round(change_pct, 2) if change_pct else None,
        "expected_move_pct": expected_move_pct,
        "atm_call_premium": round(call_ltp, 2),
        "atm_put_premium": round(put_ltp, 2),
        "atm_strike": atm_strike,
        "expiry_used": expiry,
    }


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[:period]]
    losses = [-d if d < 0 else 0 for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for d in deltas[period:]:
        gain = d if d > 0 else 0
        loss = -d if d < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _parse_nse_price(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    return float(str(val).replace(",", ""))


def fetch_technicals(symbol: str) -> dict | None:
    today = datetime.now()
    from_date = (today - timedelta(days=150)).strftime("%d-%m-%Y")
    to_date = today.strftime("%d-%m-%Y")

    closes = []
    try:
        df = capital_market.price_volume_data(symbol, from_date=from_date, to_date=to_date)
        if df is not None and len(df) >= 50:
            prices = df["ClosePrice"].apply(_parse_nse_price).tolist()
            closes = list(reversed(prices))
    except Exception:
        pass

    if len(closes) < 50:
        try:
            import yfinance as yf
            hist = yf.Ticker(f"{symbol}.NS").history(period="120d")
            if len(hist) >= 50:
                closes = hist["Close"].tolist()
        except Exception:
            pass

    if len(closes) < 50:
        return None

    cmp = round(closes[-1], 2)
    prev_close = round(closes[-2], 2) if len(closes) >= 2 else None
    change_pct = round((cmp - prev_close) / prev_close * 100, 2) if prev_close and prev_close != 0 else None
    rsi_val = _compute_rsi(closes, 14)
    dma_50 = round(sum(closes[-50:]) / 50, 2)
    above_dma = cmp > dma_50

    return {
        "cmp": cmp,
        "prev_close": prev_close,
        "change_pct": change_pct,
        "rsi": rsi_val,
        "dma_50": dma_50,
        "above_dma": above_dma,
    }


def determine_direction(rsi: float | None, above_dma: bool | None) -> str:
    if rsi is None or above_dma is None:
        return "neutral"
    if rsi > 80:
        return "overbought"
    elif rsi < 20:
        return "oversold"
    elif 60 <= rsi <= 80:
        return "bullish" if above_dma else "neutral"
    elif 20 <= rsi <= 40:
        return "bearish" if not above_dma else "neutral"
    else:
        return "mild-bullish" if above_dma else "mild-bearish"


def _safe_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def fetch_earnings(symbol: str) -> dict | None:
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        qs = ticker.quarterly_income_stmt
        if qs is None or qs.empty:
            return None

        latest_date = qs.columns[0]
        latest = qs.iloc[:, 0]

        rev = _safe_float(latest.get("Total Revenue"))
        ni = _safe_float(latest.get("Net Income"))
        eps_val = _safe_float(latest.get("Basic EPS"))

        if rev is None and ni is None:
            return None

        yoy_col = None
        for i, col in enumerate(qs.columns):
            if col.year == latest_date.year - 1 and col.quarter == latest_date.quarter:
                yoy_col = i
                break

        rev_growth = None
        ni_growth = None
        if yoy_col is not None:
            yoy = qs.iloc[:, yoy_col]
            yoy_rev = _safe_float(yoy.get("Total Revenue"))
            yoy_ni = _safe_float(yoy.get("Net Income"))
            if rev and yoy_rev and yoy_rev != 0:
                rev_growth = round((rev - yoy_rev) / abs(yoy_rev) * 100, 1)
            if ni and yoy_ni and yoy_ni != 0:
                ni_growth = round((ni - yoy_ni) / abs(yoy_ni) * 100, 1)

        return {
            "revenue_cr": round(rev / 1e7, 0) if rev else None,
            "net_profit_cr": round(ni / 1e7, 0) if ni else None,
            "eps": round(eps_val, 2) if eps_val else None,
            "revenue_growth_yoy": rev_growth,
            "profit_growth_yoy": ni_growth,
        }
    except Exception:
        return None


def fetch_post_result_price(symbol: str, result_date_str: str) -> dict | None:
    """Fetch pre-result close and post-result close to measure full reaction.

    Returns prev_day_close (last trading day before result) and next_day_close
    (first trading day after result) so the change captures the reaction
    regardless of whether results were announced before, during, or after hours.
    """
    try:
        result_dt = datetime.strptime(result_date_str, "%Y-%m-%d").date()
        from_d = (result_dt - timedelta(days=7)).strftime("%d-%m-%Y")
        to_d = (result_dt + timedelta(days=7)).strftime("%d-%m-%Y")

        df = capital_market.price_volume_data(symbol, from_date=from_d, to_date=to_d)
        if df is not None and len(df) >= 2:
            rows = []
            for _, row in df.iterrows():
                d = datetime.strptime(row["Date"].strip(), "%d-%b-%Y").date()
                close = _parse_nse_price(row["ClosePrice"])
                rows.append((d, close))
            rows.sort(key=lambda x: x[0])

            prev_day_close = None
            next_day_close = None
            next_day_date = None
            for d, close in rows:
                if d < result_dt:
                    prev_day_close = close
                elif d > result_dt and next_day_close is None:
                    next_day_close = close
                    next_day_date = d.isoformat()

            if prev_day_close and next_day_close:
                return {
                    "pre_result_close": round(prev_day_close, 2),
                    "next_day_close": round(next_day_close, 2),
                    "next_day_date": next_day_date,
                }
    except Exception:
        pass

    try:
        import yfinance as yf
        result_dt = datetime.strptime(result_date_str, "%Y-%m-%d").date()
        start = (result_dt - timedelta(days=7)).isoformat()
        end = (result_dt + timedelta(days=7)).isoformat()
        hist = yf.Ticker(f"{symbol}.NS").history(start=start, end=end)
        if len(hist) >= 3:
            closes = [(hist.index[i].date(), float(hist["Close"].iloc[i])) for i in range(len(hist))]
            prev_day_close = None
            next_day_close = None
            next_day_date = None
            for d, close in closes:
                if d < result_dt:
                    prev_day_close = close
                elif d > result_dt and next_day_close is None:
                    next_day_close = close
                    next_day_date = d.isoformat()
            if prev_day_close and next_day_close:
                return {
                    "pre_result_close": round(prev_day_close, 2),
                    "next_day_close": round(next_day_close, 2),
                    "next_day_date": next_day_date,
                }
    except Exception:
        pass

    return None


def run_full_fetch() -> dict:
    """Run the data fetch pipeline, reusing finalized history for announced stocks."""
    global _fetch_progress

    q_label, from_date, to_date = get_results_quarter()

    _fetch_progress = {"step": "Fetching F&O stock list", "done": 0, "total": 4, "status": "running"}
    stocks = fetch_fno_stocks()
    if not stocks:
        _fetch_progress["status"] = "error"
        _fetch_progress["step"] = "No F&O stocks found"
        return {}

    _fetch_progress = {"step": "Fetching result dates from NSE", "done": 1, "total": 4, "status": "running"}
    time.sleep(1)
    meetings = fetch_result_dates(from_date, to_date)

    today = date.today()
    matched = 0
    for symbol, meeting in meetings.items():
        if symbol in stocks:
            result_date = meeting["date"]
            stocks[symbol]["result_date"] = result_date.isoformat()
            if result_date < today:
                stocks[symbol]["result_status"] = "announced"
            elif result_date == today:
                stocks[symbol]["result_status"] = "today"
            else:
                stocks[symbol]["result_status"] = "upcoming"
            matched += 1

    mc_dates = fetch_moneycontrol_fallback(set(stocks.keys()), meetings, from_date, to_date)
    for symbol, meeting in mc_dates.items():
        if symbol in stocks and not stocks[symbol]["result_date"]:
            result_date = meeting["date"]
            stocks[symbol]["result_date"] = result_date.isoformat()
            if result_date < today:
                stocks[symbol]["result_status"] = "announced"
            elif result_date == today:
                stocks[symbol]["result_status"] = "today"
            else:
                stocks[symbol]["result_status"] = "upcoming"
            matched += 1

    symbols_with_dates = [s for s, d in stocks.items() if d["result_date"]]
    announced_symbols = {s for s in symbols_with_dates if stocks[s]["result_status"] == "announced"}

    history = _load_history()
    q_history = history.get(q_label, {})

    finalized_symbols = set()
    for symbol in symbols_with_dates:
        if symbol in q_history and _is_finalized(q_history[symbol]):
            stocks[symbol].update(q_history[symbol])
            finalized_symbols.add(symbol)

    to_process = [s for s in symbols_with_dates if s not in finalized_symbols]

    if not to_process:
        _fetch_progress = {"step": "All stocks loaded from history", "done": 4, "total": 4, "status": "done"}
    else:
        needs_sectors = [s for s in to_process if stocks[s].get("sector", "N/A") == "N/A"]
        if needs_sectors:
            _fetch_progress = {"step": f"Fetching sectors ({len(needs_sectors)} new)", "done": 2, "total": 4, "status": "running"}
            sectors = fetch_sectors(needs_sectors)
            for symbol, sector in sectors.items():
                if symbol in stocks:
                    stocks[symbol]["sector"] = sector

        _fetch_progress = {"step": f"Processing {len(to_process)} stocks (0/{len(to_process)})", "done": 3, "total": 4, "status": "running"}
        processed = {"count": 0}

        def _process_stock(symbol: str) -> None:
            stock = stocks[symbol]
            is_announced = symbol in announced_symbols

            opts = fetch_option_chain(symbol)
            if opts:
                stock.update(opts)

            tech = fetch_technicals(symbol)
            if tech:
                if not stock["cmp"]:
                    stock["cmp"] = tech["cmp"]
                if not stock["prev_close"]:
                    stock["prev_close"] = tech["prev_close"]
                if stock["change_pct"] is None:
                    stock["change_pct"] = tech["change_pct"]
                stock["rsi"] = tech["rsi"]
                stock["dma_50"] = tech["dma_50"]
                stock["above_dma"] = tech["above_dma"]
                stock["direction"] = determine_direction(tech["rsi"], tech["above_dma"])

            if is_announced and stock.get("result_date"):
                earnings = fetch_earnings(symbol)
                if earnings:
                    stock.update(earnings)

                price_data = fetch_post_result_price(symbol, stock["result_date"])
                if price_data:
                    stock.update(price_data)
                    if price_data["pre_result_close"] and price_data["next_day_close"]:
                        change = round(
                            (price_data["next_day_close"] - price_data["pre_result_close"])
                            / price_data["pre_result_close"] * 100, 2,
                        )
                        stock["post_result_change_pct"] = change
                        if stock["expected_move_pct"]:
                            stock["beat_expected_move"] = abs(change) > stock["expected_move_pct"]

            processed["count"] += 1
            _fetch_progress["step"] = f"Processing stocks ({processed['count']}/{len(to_process)})"

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_process_stock, sym) for sym in to_process]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    pass

    for symbol in symbols_with_dates:
        if _is_finalized(stocks[symbol]):
            q_history[symbol] = stocks[symbol]
    history[q_label] = q_history
    _save_history(history)

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "quarter_label": q_label,
        "total_fno_stocks": len(stocks),
        "stocks_with_dates": matched,
        "from_history": len(finalized_symbols),
        "freshly_fetched": len(to_process),
        "stocks": sorted(stocks.values(), key=lambda x: x.get("result_date") or "9999-12-31"),
    }

    CACHE_DIR.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(output, indent=2, default=str))

    _fetch_progress = {"step": "Done", "done": 4, "total": 4, "status": "done"}
    return output


IST = timezone(timedelta(hours=5, minutes=30))
MARKET_CLOSE_HOUR = 16  # 4 PM IST


def _is_market_closed() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return True
    return now.hour >= MARKET_CLOSE_HOUR


def _cache_has_post_close_fetch() -> bool:
    """Check if cache was generated after 4 PM IST today (i.e. has final closing prices)."""
    if not CACHE_FILE.exists():
        return False
    try:
        data = json.loads(CACHE_FILE.read_text())
        generated = data.get("generated_at")
        if not generated:
            return False
        gen_dt = datetime.fromisoformat(generated)
        if gen_dt.tzinfo is None:
            gen_dt = gen_dt.replace(tzinfo=IST)
        now = datetime.now(IST)
        same_day = gen_dt.date() == now.date()
        after_close = gen_dt.hour >= MARKET_CLOSE_HOUR
        return same_day and after_close
    except Exception:
        return False


def is_cache_fresh() -> bool:
    """Cache is fresh if we have a post-market-close fetch for today."""
    if not CACHE_FILE.exists():
        return False
    return _cache_has_post_close_fetch()


def get_cached_data() -> dict | None:
    if not CACHE_FILE.exists():
        return None
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return None


def get_cache_age_hours() -> float | None:
    if not CACHE_FILE.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        return (datetime.now() - mtime).total_seconds() / 3600
    except Exception:
        return None


def start_background_fetch() -> bool:
    """Start a background fetch. Returns True if started, False if already running."""
    global _fetch_running, _fetch_progress
    with _fetch_lock:
        if _fetch_running:
            return False
        _fetch_running = True

    def _run():
        global _fetch_running
        try:
            run_full_fetch()
        except Exception as e:
            _fetch_progress["status"] = f"error: {e}"
        finally:
            with _fetch_lock:
                _fetch_running = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return True


def get_fetch_status() -> dict:
    return {
        "running": _fetch_running,
        "progress": _fetch_progress.copy(),
        "has_cache": CACHE_FILE.exists(),
        "cache_age_hours": get_cache_age_hours(),
    }
