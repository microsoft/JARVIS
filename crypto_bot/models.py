from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int = 0

    @property
    def timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.open_time_ms / 1000, tz=timezone.utc)


@dataclass(frozen=True)
class FeatureVector:
    symbol: str
    interval_minutes: int
    momentum: float
    volatility: float
    volume_ratio: float
    trend: float
    adx: float
    awesome_oscillator: float
    rsi: float = 50.0
    ema_distance: float = 0.0
    macd: float = 0.0
    bollinger_position: float = 0.5
    atr_ratio: float = 0.0
    vwap_distance: float = 0.0


@dataclass(frozen=True)
class Prediction:
    symbol: str
    horizon_minutes: int
    probability_up: float
    probability_down: float
    expected_return: float
    volatility: float


@dataclass(frozen=True)
class Signal:
    symbol: str
    horizon_minutes: int
    action: str
    probability: float
    expected_return: float
    estimated_cost: float
    reason: str

    def as_dict(self):
        return {
            "symbol": self.symbol,
            "horizon_minutes": self.horizon_minutes,
            "action": self.action,
            "probability": round(self.probability, 6),
            "expected_return": round(self.expected_return, 6),
            "estimated_cost": round(self.estimated_cost, 6),
            "reason": self.reason,
        }
