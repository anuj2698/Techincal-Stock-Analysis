import unittest

import app as application


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = application.app.test_client()
        self.original_admin_secret = application.ADMIN_SECRET

    def tearDown(self):
        application.ADMIN_SECRET = self.original_admin_secret

    def test_health_includes_security_headers(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")

    def test_readiness_fails_when_admin_secret_is_missing(self):
        application.ADMIN_SECRET = None
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["checks"]["admin_secret_configured"])

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
        self.assertEqual(self.client.post("/results/refresh").status_code, 503)

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


if __name__ == "__main__":
    unittest.main()
