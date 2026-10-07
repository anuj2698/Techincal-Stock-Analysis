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
