#!/usr/bin/env python3
"""Fetch recent news headlines for stocks from Google News RSS."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_news(symbol: str, limit: int = 3) -> list[dict]:
    """Fetch recent news headlines for a stock symbol from Google News RSS."""
    clean = symbol.replace("NSE:", "").replace("-EQ", "").replace("BSE:", "")
    try:
        url = (
            f"https://news.google.com/rss/search"
            f"?q={clean}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
        )
        resp = requests.get(url, timeout=8, headers=_HEADERS)
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.text)
        items = root.findall(".//item")[:limit]

        news = []
        for item in items:
            title = item.findtext("title", "")
            source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1]

            pub_date = item.findtext("pubDate", "")
            link = item.findtext("link", "")

            news.append({
                "title": title,
                "source": source,
                "date": pub_date,
                "url": link,
            })

        return news
    except Exception:
        return []


def fetch_news_batch(symbols: list[str], limit: int = 3) -> dict[str, list[dict]]:
    """Fetch news for multiple symbols in parallel."""
    results = {}

    def _fetch(sym):
        return sym, fetch_news(sym, limit)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for sym, news in pool.map(_fetch, symbols):
            results[sym] = news

    return results
