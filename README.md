# Stock Watchlist Scanner

A Flask application for Indian-market technical analysis, scanners, results tracking and strategy backtesting.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
flask --app app run --port 5001
```

Configure `ADMIN_SECRET` before using operational actions such as result refreshes, broker-token refreshes, prediction updates or stock-universe rotation. Production deliberately fails these actions closed when the secret is absent.

## Scanner data cadence

- S/R Breakout, Chart Patterns, Today's Picks and Top Picks use completed daily candles only. Any candle dated today in IST is excluded, and responses identify the `previous_close` price basis.
- Intraday Breakout, RSI Momentum and the live section of RSI Extremes use 5-minute market data. The server keeps a live snapshot for at most 60 seconds during market hours; 15-minute and 1-hour RSI views are derived locally from the same feed to reduce broker/API calls.
- Live scans use Fyers when its access token is valid and fall back to the configured market-data source otherwise. Refresh the Fyers connection when low latency matters.

## Trading Platform V2

Open `/v2` for the evidence-gated workflow. V2 keeps swing and intraday decisions separate, applies configurable account-level limits, and stores a local trade journal in the ignored `.cache` directory.

The V2 PDH/PDL backtester uses 5-minute candles within individual exchange sessions, permits at most one trade per stock per day, blocks entries before 09:30 and after 14:30, forces an exit at 15:20, applies execution costs, and reports chronological 60/20/20 train/validation/test results. A setup remains research-only until its overall sample and unseen test segment qualify.

## Backtest methodology

Backtests execute only explicit BUY or SELL verdicts. WAIT signals are skipped. If stop and target are both touched in one candle, the stop is assumed to occur first. Open trades are marked to market at the end of the horizon, and a configurable round-trip execution-cost allowance is deducted from every trade.

Scanner qualification requires at least `BACKTEST_MIN_SAMPLE_SIZE` completed trades and a 95% Wilson win-rate confidence interval whose lower bound is at least 50%. Old caches carry a methodology version and are regenerated after methodology changes.

Backtests remain simulations, not investment advice. Before using a strategy with real capital, validate it out of sample and account for liquidity, gaps, taxes, brokerage and position-level risk.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Production notes

The included Gunicorn configuration uses one threaded worker so in-process scan status remains coherent. JSON persistence now uses file locks and atomic replacement, but PostgreSQL plus a dedicated Redis-backed job worker is still recommended when deploying multiple application instances.
