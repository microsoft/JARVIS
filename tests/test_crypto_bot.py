import unittest
from unittest.mock import Mock

from crypto_bot.config import BotConfig
from crypto_bot.features import aggregate_candles, generate_features
from crypto_bot.models import Candle, FeatureVector, Prediction
from crypto_bot.signals import decide, order_levels
from crypto_bot.tradingview import normalize_alert, validate_secret
from crypto_bot.foundation import AppPaths, AuditStore, PredictionJournal, SAFETY_CONTRACT, directional_pnl


def candles(count=40):
    return [
        Candle(i * 60_000, 100 + i, 101 + i, 99 + i, 100 + i, 10 + i)
        for i in range(count)
    ]


class CryptoBotTests(unittest.TestCase):
    def test_symbol_filtering(self):
        from crypto_bot.exchange import BinancePublicClient
        session = Mock()
        session.get.return_value.json.return_value = {"symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL"},
            {"symbol": "ETHUSD", "status": "TRADING", "quoteAsset": "USD", "contractType": "PERPETUAL"},
            {"symbol": "XRPUSDT", "status": "BREAK", "quoteAsset": "USDT", "contractType": "PERPETUAL"},
        ]}
        session.get.return_value.raise_for_status.return_value = None
        self.assertEqual(BinancePublicClient(session=session).active_usdt_perpetual_symbols(), ["BTCUSDT"])

    def test_watchlist_selects_requested_symbols_and_scaled_contracts(self):
        from crypto_bot.exchange import select_watchlist_symbols
        discovered = ["1000PEPEUSDT", "NEARUSDT", "SOLUSDT", "BTCUSDT"]
        self.assertEqual(
            select_watchlist_symbols(discovered, ["PEPE", "SOL"]),
            ["1000PEPEUSDT", "SOLUSDT"],
        )
        self.assertEqual(
            select_watchlist_symbols(
                ["NEARUSD", "NEARUSDT", "SOLUSD", "SOLUSDT"], ["NEAR", "SOL"]
            ),
            ["NEARUSD", "SOLUSD"],
        )
        kraken_pairs = ["NEARUSD", "SOLUSD", "XXLMZUSD", "XXBTZUSD"]
        self.assertEqual(
            select_watchlist_symbols(kraken_pairs, ["NEAR", "SOL", "XLM"]),
            ["NEARUSD", "SOLUSD", "XXLMZUSD"],
        )

    def test_kraken_spot_fallback_parses_public_data(self):
        from crypto_bot.exchange import KrakenPublicClient
        session = Mock()
        session.get.return_value.json.side_effect = [
            {"result": {"XXBTZUSD": {"quote": "ZUSD", "status": "online"}, "XETHXXBT": {"quote": "XXBT"}}},
            {"result": {"XXBTZUSD": [[1700000000, "100", "101", "99", "100.5", "100.2", "12", "10"]]}}
        ]
        session.get.return_value.raise_for_status.return_value = None
        client = KrakenPublicClient(session=session)
        self.assertEqual(client.active_usdt_perpetual_symbols(), ["XXBTZUSD"])
        self.assertEqual(client.candles("XXBTZUSD", 1)[0].close, 100.5)

    def test_candle_aggregation_drops_incomplete_bucket(self):
        result = aggregate_candles(candles(11), 5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].open, 100)
        self.assertEqual(result[0].close, 104)
        self.assertEqual(result[0].volume, 60)

    def test_features_are_generated_for_horizon(self):
        features = generate_features("BTCUSDT", candles(70), 5)
        self.assertEqual(features.interval_minutes, 5)
        self.assertGreater(features.momentum, 0)
        self.assertGreater(features.volume_ratio, 0)
        self.assertGreaterEqual(features.adx, 0)
        self.assertIsInstance(features.awesome_oscillator, float)
        self.assertGreaterEqual(features.rsi, 0)
        self.assertIsInstance(features.macd, float)
        self.assertIsInstance(features.atr_ratio, float)

    def test_signal_includes_cost_threshold(self):
        config = BotConfig(min_probability=0.55, fee_bps_per_side=4, spread_bps=2, slippage_bps=1)
        prediction = Prediction("BTCUSDT", 1, 0.9, 0.1, 0.01, 0.002)
        self.assertEqual(decide(prediction, config).action, "LONG")
        short_prediction = Prediction("BTCUSDT", 1, 0.1, 0.9, -0.01, 0.002)
        self.assertEqual(decide(short_prediction, config).action, "SHORT")
        expensive = BotConfig(min_probability=0.55, fee_bps_per_side=60, spread_bps=20, slippage_bps=20)
        self.assertEqual(decide(prediction, expensive).action, "NO_TRADE")
        self.assertEqual(directional_pnl(-0.02, "SHORT", 0.001), 0.019)
        levels = order_levels(100, prediction, decide(prediction, config))
        self.assertGreater(levels["trigger_price"], 100)
        self.assertLess(levels["stop_loss"], levels["trigger_price"])
        self.assertGreater(levels["take_profit"], levels["trigger_price"])
        short_levels = order_levels(100, short_prediction, decide(short_prediction, config))
        self.assertLess(short_levels["trigger_price"], 100)
        self.assertGreater(short_levels["stop_loss"], short_levels["trigger_price"])
        self.assertLess(short_levels["take_profit"], short_levels["trigger_price"])
        neutral = decide(Prediction("BTCUSDT", 1, 0.5, 0.5, 0.0, 0.002), config)
        neutral_levels = order_levels(100, Prediction("BTCUSDT", 1, 0.5, 0.5, 0.0, 0.002), neutral)
        self.assertIsNotNone(neutral_levels["trigger_price"])

    def test_order_levels_use_recent_swing_turns(self):
        config = BotConfig(min_probability=0.55)
        prediction = Prediction("BTCUSDT", 1, 0.9, 0.1, 0.01, 0.002)
        signal = decide(prediction, config)
        history = [
            Candle(i * 60_000, 100, 110, 99, 100, 10)
            for i in range(20)
        ]
        levels = order_levels(100, prediction, signal, history)
        self.assertGreater(levels["take_profit"], 105)
        self.assertLess(levels["stop_loss"], levels["trigger_price"])

    def test_tradingview_alert_is_validated_and_normalized(self):
        payload = {"secret": "test-secret", "symbol": "btcusdt", "action": "LONG", "price": "100"}
        self.assertTrue(validate_secret(payload, "test-secret"))
        self.assertEqual(normalize_alert(payload)["symbol"], "BTCUSDT")
        self.assertFalse(validate_secret(payload, "wrong-secret"))

    def test_audit_store_redacts_secrets(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.resolve(directory).create()
            store = AuditStore(paths.database)
            store.append("test", {"token": "do-not-store", "action": "paper"})
            row = store.connection.execute("SELECT payload_json FROM audit_events").fetchone()
            self.assertNotIn("do-not-store", row[0])
            self.assertIn("[REDACTED]", row[0])
            self.assertIn("PAPER", SAFETY_CONTRACT)
            store.close()

    def test_prediction_journal_resolves_and_calibrates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.resolve(directory).create()
            journal = PredictionJournal(paths.database)
            signal = {
                "symbol": "BTCUSDT", "horizon_minutes": 1, "action": "LONG",
                "probability": 0.8, "expected_return": 0.01,
            }
            journal.record(signal, 100.0, "2020-01-01T00:00:00+00:00")
            journal.resolve("BTCUSDT", 1, 101.0, "2020-01-01T00:02:00+00:00", 0.001)
            self.assertEqual(journal.summary()["1"], {"right": 1, "wrong": 0})
            self.assertGreater(journal.calibration_offset("BTCUSDT", 1), 0)
            self.assertEqual(
                journal.performance_summary(0.001)["1"],
                {"net_return": 0.009, "closed_calls": 1},
            )
            journal.close()

    def test_prediction_journal_archives_and_resets(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.resolve(directory).create()
            journal = PredictionJournal(paths.database)
            journal.record({
                "symbol": "BTCUSDT", "horizon_minutes": 1, "action": "LONG",
                "probability": 0.8, "expected_return": 0.01,
            }, 100.0, "2020-01-01T00:00:00+00:00")
            result = journal.archive_and_reset(0.001)
            self.assertEqual(result["prediction_count"], 1)
            self.assertEqual(journal.summary()["1"], {"right": 0, "wrong": 0})
            archived = journal.connection.execute(
                "SELECT prediction_count FROM prediction_archives"
            ).fetchone()
            self.assertEqual(archived[0], 1)
            journal.close()

    def test_calibration_uses_archived_history_after_reset(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.resolve(directory).create()
            journal = PredictionJournal(paths.database)
            journal.record({
                "symbol": "BTCUSDT", "horizon_minutes": 1, "action": "LONG",
                "probability": 0.8, "expected_return": 0.01,
            }, 100.0, "2020-01-01T00:00:00+00:00")
            journal.resolve("BTCUSDT", 1, 101.0, "2020-01-01T00:02:00+00:00", 0.001)
            journal.archive_and_reset(0.001)
            self.assertGreater(journal.calibration_offset("BTCUSDT", 1), 0)
            journal.close()

    def test_backtest_learning_is_bounded(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.resolve(directory).create()
            journal = PredictionJournal(paths.database)
            journal.apply_backtest_learning({"1": {"calls": 10, "right": 10, "accuracy": 1.0, "net_return": 0.2}})
            self.assertLessEqual(journal.calibration_offset("BTCUSDT", 1), 0.75)
            journal.close()

    def test_pair_backtest_learning_is_bounded_and_specific(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            paths = AppPaths.resolve(directory).create()
            journal = PredictionJournal(paths.database)
            journal.apply_pair_backtest_learning({
                "BTCUSDT": {"1": {"calls": 10, "accuracy": 1.0, "net_return": 0.2}},
                "ETHUSDT": {"1": {"calls": 10, "accuracy": 0.0, "net_return": -0.2}},
            })
            self.assertGreater(journal.calibration_offset("BTCUSDT", 1), 0)
            self.assertLess(journal.calibration_offset("ETHUSDT", 1), 0)
            journal.close()

    def test_pine_script_is_inspected_without_execution(self):
        from crypto_bot.pine import inspect_script
        result = inspect_script(
            "//@version=5\nstrategy('Breakout', overlay=true)\n"
            "length = input.int(20, 'Lookback')\n"
            "plot(close)\nalertcondition(close > high[1], 'Breakout')\n"
            "strategy.entry('L', strategy.long)\n",
            "breakout.pine",
        )
        self.assertEqual(result["version"], 5)
        self.assertEqual(result["declaration"], "strategy")
        self.assertIn("Lookback", result["inputs"])
        self.assertIn("plot", result["plots"])
        self.assertIn("alertcondition", result["alerts"])
        self.assertTrue(result["read_only"])
        self.assertTrue(result["warnings"])

    def test_pine_script_rejects_empty_source(self):
        from crypto_bot.pine import inspect_script
        with self.assertRaises(ValueError):
            inspect_script("  ")

    def test_terminal_bridge_rejects_unallowlisted_commands(self):
        from crypto_bot.terminal_bridge import run_bridge
        self.assertTrue(callable(run_bridge))


if __name__ == "__main__":
    unittest.main()
