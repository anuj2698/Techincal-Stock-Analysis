import unittest
from datetime import datetime

import app as application


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = application.app.test_client()
        self.original_admin_secret = application.ADMIN_SECRET
        self.original_is_production = application.IS_PRODUCTION

    def tearDown(self):
        application.ADMIN_SECRET = self.original_admin_secret
        application.IS_PRODUCTION = self.original_is_production

    def test_health_includes_security_headers(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")

    def test_readiness_fails_when_admin_secret_is_missing(self):
        application.ADMIN_SECRET = None
        application.IS_PRODUCTION = True
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["checks"]["admin_auth_configured"])

    def test_analyze_rejects_invalid_entry_before_fetching_market_data(self):
        response = self.client.get("/analyze?symbol=RELIANCE&entry_price=not-a-number")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Entry price must be a number")

    def test_analyze_rejects_invalid_symbol(self):
        response = self.client.get("/analyze?symbol=%3Cscript%3E")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Invalid stock symbol")

    def test_admin_actions_fail_closed_without_secret(self):
        application.ADMIN_SECRET = None
        application.IS_PRODUCTION = True
        self.assertEqual(self.client.post("/results/refresh").status_code, 503)

    def test_admin_check_allows_local_development_without_secret(self):
        application.ADMIN_SECRET = None
        application.IS_PRODUCTION = False
        with application.app.test_request_context("/results/refresh", method="POST"):
            self.assertIsNone(application._check_admin(application.request))

    def test_admin_actions_reject_wrong_secret(self):
        application.ADMIN_SECRET = "correct-secret"
        response = self.client.post("/results/refresh", headers={"X-Admin-Secret": "wrong-secret"})
        self.assertEqual(response.status_code, 401)

    def test_backtest_update_is_not_available_via_get(self):
        self.assertEqual(self.client.get("/backtest/update?symbol=RELIANCE").status_code, 405)

    def test_forced_intraday_rescan_requires_admin(self):
        application.ADMIN_SECRET = "correct-secret"
        response = self.client.post("/intraday/pick?force=1")
        self.assertEqual(response.status_code, 401)

    def test_scanner_rejects_out_of_range_risk_before_starting_scan(self):
        response = self.client.get("/scanner/picks?timeframe=positional&capital=10000&risk_pct=25")
        self.assertEqual(response.status_code, 400)
        self.assertIn("between 0.1% and 10%", response.get_json()["error"])

    def test_previous_close_candles_excludes_current_session(self):
        prior = datetime(2026, 10, 6, 15, 30, tzinfo=application.IST)
        current = datetime(2026, 10, 7, 10, 0, tzinfo=application.IST)
        candles = [
            [int(prior.timestamp()), 100, 105, 99, 104, 1000],
            [int(current.timestamp()), 104, 107, 103, 106, 500],
        ]

        result = application.previous_close_candles(candles, now=current)

        self.assertEqual(result, candles[:1])

    def test_intraday_resampling_aligns_to_market_open_and_keeps_days_separate(self):
        def candle(day, hour, minute, open_, high, low, close, volume):
            ts = datetime(2026, 10, day, hour, minute, tzinfo=application.IST)
            return [int(ts.timestamp()), open_, high, low, close, volume]

        candles = [
            candle(6, 9, 15, 100, 102, 99, 101, 10),
            candle(6, 9, 20, 101, 104, 100, 103, 20),
            candle(6, 10, 15, 103, 106, 102, 105, 30),
            candle(7, 9, 15, 110, 112, 109, 111, 40),
        ]

        result = application.resample_intraday_minutes(candles, 60)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][1:], [100, 104.0, 99.0, 103, 30.0])
        self.assertEqual(result[1][4], 105)
        self.assertEqual(result[2][1], 110)


if __name__ == "__main__":
    unittest.main()
