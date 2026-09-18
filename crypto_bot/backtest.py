"""Small public-data walk-forward backtest for the signal-only baseline."""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable

from .features import aggregate_candles, generate_features
from .signals import decide, predict


def run_backtest(client, symbols: Iterable[str], config, journal) -> Dict:
    symbols = list(symbols)
    totals = {str(h): {"calls": 0, "right": 0, "net_return": 0.0} for h in (1, 5, 10)}
    per_symbol = {}

    def evaluate(symbol):
        candles = client.candles(symbol, config.candle_limit)
        result = {str(h): {"calls": 0, "right": 0, "net_return": 0.0} for h in (1, 5, 10)}
        for horizon in (1, 5, 10):
            bars = candles if horizon == 1 else aggregate_candles(candles, horizon)
            if len(bars) < 20:
                continue
            for end in range(12, len(bars) - 1):
                history = bars[:end]
                feature = generate_features(symbol, history, horizon)
                prediction = predict(feature, journal.calibration_offset(symbol, horizon))
                signal = decide(prediction, config)
                move = (bars[end + 1].close - bars[end].close) / bars[end].close
                pnl = move if signal.action == "LONG" else -move if signal.action == "SHORT" else 0.0
                if signal.action in ("LONG", "SHORT"):
                    pnl -= config.estimated_round_trip_cost
                    result[str(horizon)]["calls"] += 1
                    result[str(horizon)]["right"] += int(pnl > 0)
                    result[str(horizon)]["net_return"] += pnl
        per_symbol[symbol] = result
        return result

    with ThreadPoolExecutor(max_workers=min(config.scan_workers, max(1, len(list(symbols))))) as pool:
        for result in pool.map(evaluate, symbols):
            for horizon, values in result.items():
                for key in ("calls", "right"):
                    totals[horizon][key] += values[key]
                totals[horizon]["net_return"] += values["net_return"]
    for horizon, values in totals.items():
        values["net_return"] = round(values["net_return"], 6)
        values["accuracy"] = round(values["right"] / values["calls"], 4) if values["calls"] else 0.0
    journal.apply_backtest_learning(totals)
    for horizons in per_symbol.values():
        for values in horizons.values():
            values["net_return"] = round(values["net_return"], 6)
            values["accuracy"] = round(values["right"] / values["calls"], 4) if values["calls"] else 0.0
    journal.apply_pair_backtest_learning(per_symbol)
    totals["per_symbol"] = per_symbol
    return totals
