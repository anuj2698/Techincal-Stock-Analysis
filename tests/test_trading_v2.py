import unittest
from datetime import datetime

from trading_v2 import IST, account_risk_status, backtest_pdhl_intraday
from backtest_rsi_momentum import map_htf_rsi, resample


class TradingV2Tests(unittest.TestCase):
    @staticmethod
    def candle(day, hour, minute, open_, high, low, close, volume=1000):
        dt = datetime(2026, 10, day, hour, minute, tzinfo=IST)
        return [int(dt.timestamp()), open_, high, low, close, volume]

    def test_pdhl_backtest_uses_one_session_and_forces_no_overnight_position(self):
        candles = [
            self.candle(5, 9, 15, 95, 100, 90, 96),
            self.candle(5, 15, 20, 96, 98, 94, 97),
            self.candle(6, 9, 30, 99, 101, 98, 100),
            self.candle(6, 10, 0, 100, 102, 99, 101),
            self.candle(6, 15, 20, 101, 102, 100, 101.5),
            self.candle(7, 9, 30, 101, 102, 100, 101),
        ]

        result = backtest_pdhl_intraday(candles)

        self.assertEqual(result["summary"]["overnight_positions"], 0)
        self.assertEqual(result["summary"]["out_of_sample"]["split"], "chronological_60_20_20")
        self.assertFalse(result["summary"]["out_of_sample"]["eligible"])
        self.assertEqual(result["summary"]["triggered"], 1)
        self.assertEqual(result["trades"][0]["result"], "eod")
        self.assertEqual(result["trades"][0]["exit_time"], "15:20")

    def test_pdhl_backtest_counts_target_after_conservative_entry_bar(self):
        candles = [
            self.candle(5, 9, 15, 95, 100, 90, 96),
            self.candle(5, 15, 20, 96, 98, 94, 97),
            self.candle(6, 9, 30, 99, 101, 99, 100.5),
            self.candle(6, 10, 0, 101, 104, 100, 103.5),
            self.candle(6, 15, 20, 103, 104, 102, 103),
        ]

        result = backtest_pdhl_intraday(candles)

        self.assertEqual(result["trades"][0]["side"], "long")
        self.assertEqual(result["trades"][0]["result"], "target")
        self.assertGreater(result["trades"][0]["net_pnl_pct"], 0)

    def test_account_risk_gate_blocks_daily_loss_and_trade_count(self):
        journal = [
            {"status": "closed", "pnl": -600},
            {"status": "closed", "pnl": -500},
            {"status": "closed", "pnl": 100},
        ]

        status = account_risk_status(journal, capital=100_000)

        self.assertFalse(status["can_open_trade"])
        self.assertIn("Daily loss limit reached", status["reasons"])
        self.assertIn("Maximum trades reached", status["reasons"])
        self.assertEqual(status["risk_per_trade"], 500)

    def test_rsi_resampling_does_not_cross_sessions_or_leak_unclosed_15m_rsi(self):
        candles = [
            self.candle(5, 9, 15, 100, 101, 99, 100),
            self.candle(5, 9, 20, 100, 102, 99, 101),
            self.candle(5, 9, 25, 101, 103, 100, 102),
            self.candle(6, 9, 15, 110, 111, 109, 110),
        ]
        candles_15m = resample(candles, 3)
        mapped = map_htf_rsi(candles, candles_15m[:1], [70.0])

        self.assertEqual(len(candles_15m), 2)
        self.assertIsNone(mapped[1])
        self.assertEqual(mapped[2], 70.0)


if __name__ == "__main__":
    unittest.main()
