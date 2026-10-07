import unittest

from backtester import _compute_summary, backtest_prediction


def candle(day, open_, high, low, close):
    return [1_700_000_000 + day * 86_400, open_, high, low, close, 1_000]


BUY = {"level": 100, "sl": 95, "targets": [110]}
SELL = {"level": 100, "sl": 105, "targets": [90]}


class BacktesterTests(unittest.TestCase):
    def test_wait_verdict_does_not_cherry_pick_a_winning_side(self):
        prediction = {"status": "NO TRADE", "buy": BUY, "sell": SELL, "action_summary": {"verdict": "WAIT"}}
        result = backtest_prediction(prediction, [candle(0, 100, 111, 94, 100)], 10, 0)
        self.assertEqual(result["result"], "skipped")

    def test_buy_checks_stop_on_entry_candle_conservatively(self):
        prediction = {"status": "BUY ZONE", "buy": BUY, "sell": SELL, "action_summary": {"verdict": "BUY"}}
        result = backtest_prediction(prediction, [candle(0, 100, 112, 94, 101)], 10, 0)
        self.assertEqual(result["result"], "sl_hit")
        self.assertEqual(result["pnl_pct"], -5)

    def test_expired_trade_is_marked_to_market_and_costed(self):
        prediction = {"status": "BUY ZONE", "buy": BUY, "action_summary": {"verdict": "BUY"}}
        future = [candle(0, 101, 102, 99, 101), candle(1, 101, 104, 100, 103)]
        result = backtest_prediction(prediction, future, 10, 0.2)
        self.assertEqual(result["result"], "expired")
        self.assertEqual(result["exit_price"], 103)
        self.assertEqual(result["pnl_pct"], 2.8)

    def test_summary_counts_expired_results_and_skipped_predictions(self):
        predictions = [
            {"outcome": {"result": "target_hit", "pnl_pct": 4, "days_to_exit": 2, "days_to_entry": 0}},
            {"outcome": {"result": "expired", "pnl_pct": -1, "days_to_exit": 3, "days_to_entry": 0}},
            {"outcome": {"result": "skipped", "pnl_pct": 0}},
        ]
        summary = _compute_summary(predictions)
        self.assertEqual(summary["triggered"], 2)
        self.assertEqual(summary["expired"], 1)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["win_rate"], 50)
        self.assertEqual(len(summary["win_rate_ci_95"]), 2)
        self.assertLess(summary["win_rate_ci_95"][0], 50)
        self.assertGreater(summary["win_rate_ci_95"][1], 50)
        self.assertEqual(summary["total_pnl_pct"], 3)


if __name__ == "__main__":
    unittest.main()
