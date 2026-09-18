import math
from typing import List

from .models import Candle, FeatureVector


def aggregate_candles(candles: List[Candle], interval_minutes: int) -> List[Candle]:
    """Aggregate aligned 1-minute candles, dropping incomplete buckets."""
    if interval_minutes < 1:
        raise ValueError("interval_minutes must be positive")
    grouped = {}
    for candle in sorted(candles, key=lambda item: item.open_time_ms):
        bucket = candle.open_time_ms // (interval_minutes * 60_000)
        grouped.setdefault(bucket, []).append(candle)
    result = []
    for bucket, rows in sorted(grouped.items()):
        if len(rows) != interval_minutes:
            continue
        result.append(
            Candle(
                open_time_ms=rows[0].open_time_ms,
                open=rows[0].open,
                high=max(row.high for row in rows),
                low=min(row.low for row in rows),
                close=rows[-1].close,
                volume=sum(row.volume for row in rows),
                close_time_ms=rows[-1].close_time_ms,
            )
        )
    return result


def _returns(candles: List[Candle]) -> List[float]:
    return [
        math.log(current.close / previous.close)
        for previous, current in zip(candles, candles[1:])
        if previous.close > 0 and current.close > 0
    ]


def _adx(candles: List[Candle], period: int = 7) -> float:
    if len(candles) <= period:
        return 0.0
    true_ranges = []
    plus_dm = []
    minus_dm = []
    for previous, current in zip(candles, candles[1:]):
        true_ranges.append(max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        ))
        up = current.high - previous.high
        down = previous.low - current.low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    tr = sum(true_ranges[-period:])
    plus = sum(plus_dm[-period:])
    minus = sum(minus_dm[-period:])
    if tr == 0:
        return 0.0
    plus_di = 100 * plus / tr
    minus_di = 100 * minus / tr
    return 100 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-12)


def _awesome_oscillator(candles: List[Candle], fast: int = 3, slow: int = 8) -> float:
    if len(candles) < slow:
        return 0.0
    midpoint = [(bar.high + bar.low) / 2 for bar in candles]
    return (
        sum(midpoint[-fast:]) / fast
        - sum(midpoint[-slow:]) / slow
    ) / max(abs(midpoint[-1]), 1e-12)


def _ema(values: List[float], period: int) -> float:
    if not values:
        return 0.0
    alpha = 2 / (period + 1)
    result = values[0]
    for value in values[1:]:
        result = alpha * value + (1 - alpha) * result
    return result


def _rsi(closes: List[float], period: int = 14) -> float:
    changes = [current - previous for previous, current in zip(closes, closes[1:])][-period:]
    gains = sum(max(change, 0) for change in changes)
    losses = sum(max(-change, 0) for change in changes)
    if losses == 0:
        return 100.0 if gains else 50.0
    return 100 - (100 / (1 + gains / losses))


def generate_features(symbol: str, candles: List[Candle], interval_minutes: int) -> FeatureVector:
    bars = candles if interval_minutes == 1 else aggregate_candles(candles, interval_minutes)
    if len(bars) < 12:
        raise ValueError("at least 12 complete bars are required")
    returns = _returns(bars)
    lookback = min(10, len(returns))
    momentum = sum(returns[-lookback:])
    mean = sum(returns[-lookback:]) / lookback
    volatility = math.sqrt(sum((item - mean) ** 2 for item in returns[-lookback:]) / lookback)
    volumes = [bar.volume for bar in bars[-min(20, len(bars)):]]
    average_volume = sum(volumes[:-1]) / max(1, len(volumes) - 1)
    volume_ratio = volumes[-1] / average_volume if average_volume else 1.0
    trend = (bars[-1].close - bars[-min(6, len(bars))].close) / bars[-min(6, len(bars))].close
    closes = [bar.close for bar in bars]
    ema_fast = _ema(closes[-50:], 9)
    ema_slow = _ema(closes[-100:], 21)
    window = closes[-20:]
    middle = sum(window) / len(window)
    deviation = math.sqrt(sum((value - middle) ** 2 for value in window) / len(window))
    typical_volume = sum(bar.volume for bar in bars[-20:])
    vwap = (
        sum(((bar.high + bar.low + bar.close) / 3) * bar.volume for bar in bars[-20:])
        / typical_volume if typical_volume else bars[-1].close
    )
    atr = sum(max(
        bar.high - bar.low,
        abs(bar.high - previous.close),
        abs(bar.low - previous.close),
    ) for previous, bar in zip(bars[-15:-1], bars[-14:])) / max(1, min(14, len(bars) - 1))
    return FeatureVector(
        symbol, interval_minutes, momentum, volatility, volume_ratio, trend,
        _adx(bars), _awesome_oscillator(bars),
        _rsi(closes), (ema_fast - ema_slow) / max(closes[-1], 1e-12),
        (ema_fast - ema_slow) / max(closes[-1], 1e-12),
        (closes[-1] - (middle - 2 * deviation)) / max(4 * deviation, 1e-12),
        atr / max(closes[-1], 1e-12),
        (closes[-1] - vwap) / max(closes[-1], 1e-12),
    )
