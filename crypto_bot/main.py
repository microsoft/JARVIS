import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from requests import RequestException

from .config import BotConfig
from .exchange import (
    BinancePublicClient,
    BybitPublicClient,
    KrakenPublicClient,
    select_watchlist_symbols,
)
from .features import generate_features
from .signals import decide, order_levels, predict
from .tradingview import serve
from .dashboard import run_dashboard
from .foundation import PredictionJournal
from .foundation import directional_pnl


def scan(config: BotConfig, journal=None):
    client, symbols = create_client_and_symbols(config)
    output = []
    return _scan_symbols(client, symbols, config, journal)


def create_client_and_symbols(config):
    clients = []
    if config.exchange in ("auto", "binance"):
        clients.append(BinancePublicClient(
            config.base_url, config.request_timeout_seconds, config.request_interval_seconds
        ))
    if config.exchange in ("auto", "bybit"):
        clients.append(BybitPublicClient(
            config.bybit_base_url, config.request_timeout_seconds, config.request_interval_seconds
        ))
    if config.exchange in ("auto", "kraken"):
        clients.append(KrakenPublicClient(
            config.kraken_base_url, config.request_timeout_seconds, config.request_interval_seconds
        ))
    last_error = None
    for client in clients:
        try:
            symbols = client.active_usdt_perpetual_symbols()
            break
        except Exception as error:
            last_error = error
    else:
        raise RuntimeError("all configured public exchanges failed") from last_error
    symbols = select_watchlist_symbols(symbols, list(config.watchlist))
    if config.max_symbols and not config.watchlist:
        symbols = symbols[:config.max_symbols]
    return client, symbols


def _scan_symbols(client, symbols, config, journal):
    def collect(symbol):
        try:
            candles = client.candles(symbol, config.candle_limit)
            if not candles:
                return []
            current_price = candles[-1].close
            reference_price = candles[0].open
            price_change = current_price - reference_price
            percent_change = price_change / reference_price if reference_price else 0.0
            volume = sum(candle.volume for candle in candles)
            symbol_output = []
            for horizon in (1, 5, 10):
                features = generate_features(symbol, candles, horizon)
                if journal is not None:
                    journal.resolve(symbol, horizon, current_price, _now(), config.estimated_round_trip_cost)
                prediction = predict(features, journal.calibration_offset(symbol, horizon) if journal else 0.0)
                decided = decide(prediction, config)
                signal = decided.as_dict()
                signal.update(order_levels(current_price, prediction, decided, candles))
                signal.update({
                    "adx": round(features.adx, 4),
                    "awesome_oscillator": round(features.awesome_oscillator, 8),
                    "rsi": round(features.rsi, 4),
                    "ema_distance": round(features.ema_distance, 8),
                    "macd": round(features.macd, 8),
                    "bollinger_position": round(features.bollinger_position, 8),
                    "atr_ratio": round(features.atr_ratio, 8),
                    "vwap_distance": round(features.vwap_distance, 8),
                    "expected_pnl": round(
                        directional_pnl(signal["expected_return"], signal["action"]),
                        6,
                    ),
                })
                signal.update({
                    "price": round(current_price, 12),
                    "price_change": round(price_change, 12),
                    "percent_change": round(percent_change, 8),
                    "volume": round(volume, 8),
                    "provider": client.__class__.__name__.replace("PublicClient", "").lower(),
                })
                if journal is not None:
                    journal.record(signal, current_price, _now())
                symbol_output.append(signal)
            return symbol_output
        except (RequestException, RuntimeError, ValueError, TypeError, KeyError):
            # One incomplete symbol must not remove all other watchlist results.
            return []

    # Candle retrieval is the expensive public-data step. Bounded concurrency
    # keeps scans responsive while the client still paces request starts.
    with ThreadPoolExecutor(max_workers=min(config.scan_workers, max(1, len(symbols)))) as pool:
        batches = pool.map(collect, symbols)
    return [signal for batch in batches for signal in batch]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description="Signal-only Binance perpetual futures scanner")
    parser.add_argument("--once", action="store_true", help="scan once instead of polling")
    parser.add_argument("--tradingview-webhook", action="store_true",
                        help="listen for TradingView POST alerts at /tradingview")
    parser.add_argument("--desktop", action="store_true",
                        help="open the local INTRA-style desktop dashboard")
    parser.add_argument("--bridge", action="store_true",
                        help="run the local allowlisted terminal bridge")
    args = parser.parse_args()
    config = BotConfig.from_env()
    if args.desktop:
        if args.bridge:
            from .terminal_bridge import run_bridge
            import threading
            threading.Thread(
                target=run_bridge,
                args=(lambda: {"paper_mode": True}, lambda: scan(config)),
                kwargs={"open_browser": False},
                daemon=True,
            ).start()
        run_dashboard(config)
        return
    if args.bridge:
        from .terminal_bridge import run_bridge
        run_bridge(lambda: {"paper_mode": True}, lambda: scan(config))
        return
    if args.tradingview_webhook:
        serve(config.tradingview_host, config.tradingview_port, config.tradingview_secret)
        return
    while True:
        print(json.dumps(scan(config), separators=(",", ":")))
        if args.once:
            return
        time.sleep(config.poll_seconds)


if __name__ == "__main__":
    main()
