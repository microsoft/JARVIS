import math

from .config import BotConfig
from .models import Candle, FeatureVector, Prediction, Signal


def order_levels(price: float, prediction: Prediction, signal: Signal,
                 candles: list[Candle] | None = None):
    """Return paper levels using recent swings when candle history is available."""
    if price <= 0:
        return {"trigger_price": None, "stop_loss": None, "take_profit": None}
    cost = signal.estimated_cost
    risk = max(prediction.volatility * 2.0, cost * 2.0, 0.0005)
    reward = max(abs(prediction.expected_return), risk * 1.5)
    swing_high = max((bar.high for bar in (candles or [])[-20:]), default=0.0)
    swing_low = min((bar.low for bar in (candles or [])[-20:]), default=0.0)
    # A small buffer avoids placing a trigger directly on a recent extreme.
    buffer = max(risk * price * 0.15, cost * price)
    # NO_TRADE receives reference levels using the model's stronger direction;
    # its card remains non-actionable and these levels must not be treated as an order.
    direction = signal.action
    if direction == "NO_TRADE":
        direction = "LONG" if prediction.probability_up >= prediction.probability_down else "SHORT"
    if direction == "LONG":
        trigger = price * (1 + risk / 2)
        support = swing_low if 0 < swing_low < trigger else trigger * (1 - risk)
        resistance = swing_high if swing_high > trigger + buffer else trigger * (1 + reward)
        stop = max(support, trigger * (1 - risk * 1.5))
        target = max(trigger * (1 + reward), resistance)
    else:
        trigger = price * (1 - risk / 2)
        resistance = swing_high if swing_high > trigger else trigger * (1 + risk)
        support = swing_low if 0 < swing_low < trigger - buffer else trigger * (1 - reward)
        stop = min(resistance, trigger * (1 + risk * 1.5))
        target = min(trigger * (1 - reward), support)
    return {
        "trigger_price": round(trigger, 12),
        "stop_loss": round(stop, 12),
        "take_profit": round(target, 12),
    }


def predict(features: FeatureVector, calibration_offset: float = 0.0) -> Prediction:
    """Produce a conservative, deterministic probability from recent public data.

    This is a baseline model, not a claim of predictive certainty. The volatility
    term widens uncertainty and the output is bounded to avoid overconfidence.
    """
    scale = max(features.volatility * 3.0, 0.0005)
    direction = 1.0 if features.awesome_oscillator >= 0 else -1.0
    trend_strength = min(features.adx / 100.0, 1.0)
    indicator_bias = (
        0.12 * max(-1.0, min(1.0, (features.rsi - 50.0) / 20.0))
        + 0.10 * max(-1.0, min(1.0, features.macd / scale))
        + 0.08 * max(-1.0, min(1.0, features.vwap_distance / scale))
        + 0.06 * max(-1.0, min(1.0, (features.bollinger_position - 0.5) * 2))
    )
    score = (features.momentum + 0.5 * features.trend
             + 0.15 * direction * trend_strength + indicator_bias * scale) / scale
    score += calibration_offset
    probability_up = 1 / (1 + math.exp(-max(-6.0, min(6.0, score))))
    probability_up = min(0.95, max(0.05, probability_up))
    expected_return = features.momentum + 0.25 * features.trend + 0.1 * features.macd + 0.05 * features.vwap_distance
    return Prediction(
        features.symbol,
        features.interval_minutes,
        probability_up,
        1 - probability_up,
        expected_return,
        features.volatility,
    )


def decide(prediction: Prediction, config: BotConfig) -> Signal:
    cost = config.estimated_round_trip_cost
    if prediction.probability_up >= config.min_probability and prediction.expected_return > cost:
        return Signal(prediction.symbol, prediction.horizon_minutes, "LONG",
                      prediction.probability_up, prediction.expected_return, cost,
                      "up probability and expected return clear estimated costs")
    if prediction.probability_down >= config.min_probability and -prediction.expected_return > cost:
        return Signal(prediction.symbol, prediction.horizon_minutes, "SHORT",
                      prediction.probability_down, prediction.expected_return, cost,
                      "down probability and expected return clear estimated costs")
    return Signal(prediction.symbol, prediction.horizon_minutes, "NO_TRADE",
                  max(prediction.probability_up, prediction.probability_down),
                  prediction.expected_return, cost,
                  "probability or expected return does not clear the cost threshold")
