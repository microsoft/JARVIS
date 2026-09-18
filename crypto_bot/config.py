import os
from dataclasses import dataclass


DEFAULT_WATCHLIST = (
    "MOODENG", "NEAR", "ONDO", "OP", "PENDLE", "PENGU", "PEPE", "PNUT",
    "POL", "POPCAT", "PUMP", "RENDER", "S&P500", "S", "SHIB", "SILVER",
    "SOL", "STX", "SUI", "TAO", "TIA", "TRUMP", "TRX", "UNI", "VIRTUAL",
    "WIF", "WLD", "XLM", "XPL", "XRP", "XYZ100", "ZEC", "ZRO",
)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@dataclass(frozen=True)
class BotConfig:
    exchange: str = "auto"
    base_url: str = "https://fapi.binance.com"
    bybit_base_url: str = "https://api.bybit.com"
    kraken_base_url: str = "https://api.kraken.com"
    request_timeout_seconds: float = 7.0
    request_interval_seconds: float = 0.08
    candle_limit: int = 180
    max_symbols: int = 0
    watchlist: tuple = DEFAULT_WATCHLIST
    min_probability: float = 0.55
    fee_bps_per_side: float = 4.0
    spread_bps: float = 2.0
    slippage_bps: float = 1.0
    poll_seconds: int = 60
    scan_workers: int = 8
    tradingview_host: str = "127.0.0.1"
    tradingview_port: int = 8787
    tradingview_secret: str = ""

    @classmethod
    def from_env(cls) -> "BotConfig":
        watchlist_value = os.getenv("CRYPTO_BOT_WATCHLIST")
        watchlist = (
            tuple(item.strip().upper() for item in watchlist_value.split(",") if item.strip())
            if watchlist_value is not None
            else cls.watchlist
        )
        return cls(
            exchange=os.getenv("CRYPTO_BOT_EXCHANGE", cls.exchange).lower(),
            base_url=os.getenv("CRYPTO_BOT_BASE_URL", cls.base_url),
            bybit_base_url=os.getenv("CRYPTO_BOT_BYBIT_BASE_URL", cls.bybit_base_url),
            kraken_base_url=os.getenv("CRYPTO_BOT_KRAKEN_BASE_URL", cls.kraken_base_url),
            request_timeout_seconds=_float_env("CRYPTO_BOT_TIMEOUT", cls.request_timeout_seconds),
            request_interval_seconds=_float_env("CRYPTO_BOT_REQUEST_INTERVAL", cls.request_interval_seconds),
            candle_limit=_int_env("CRYPTO_BOT_CANDLE_LIMIT", cls.candle_limit),
            max_symbols=_int_env("CRYPTO_BOT_MAX_SYMBOLS", cls.max_symbols),
            watchlist=watchlist,
            min_probability=_float_env("CRYPTO_BOT_MIN_PROBABILITY", cls.min_probability),
            fee_bps_per_side=_float_env("CRYPTO_BOT_FEE_BPS", cls.fee_bps_per_side),
            spread_bps=_float_env("CRYPTO_BOT_SPREAD_BPS", cls.spread_bps),
            slippage_bps=_float_env("CRYPTO_BOT_SLIPPAGE_BPS", cls.slippage_bps),
            poll_seconds=_int_env("CRYPTO_BOT_POLL_SECONDS", cls.poll_seconds),
            scan_workers=max(1, min(16, _int_env("CRYPTO_BOT_SCAN_WORKERS", cls.scan_workers))),
            tradingview_host=os.getenv("CRYPTO_BOT_TRADINGVIEW_HOST", cls.tradingview_host),
            tradingview_port=_int_env("CRYPTO_BOT_TRADINGVIEW_PORT", cls.tradingview_port),
            tradingview_secret=os.getenv("CRYPTO_BOT_TRADINGVIEW_SECRET", cls.tradingview_secret),
        )

    @property
    def estimated_round_trip_cost(self) -> float:
        return (2 * self.fee_bps_per_side + self.spread_bps + self.slippage_bps) / 10_000
