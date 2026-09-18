import time
import re
import threading
from typing import Any, Dict, List, Optional

import requests

from .models import Candle


def select_watchlist_symbols(discovered: List[str], watchlist: List[str]) -> List[str]:
    """Select one deterministic exchange symbol for each requested base."""
    if not watchlist:
        return sorted(discovered)
    selected = []
    normalized_watchlist = [
        re.sub(r"[^A-Z0-9]", "", name.upper()) for name in watchlist
    ]
    for requested in normalized_watchlist:
        matches = []
        for symbol in discovered:
            base = re.sub(r"[^A-Z0-9]", "", symbol.upper())
            candidates = {base}
            for quote in ("USDT", "USDC", "USD", "ZUSD"):
                if base.endswith(quote):
                    candidates.add(base[:-len(quote)])
            # Kraken's legacy pair IDs wrap assets as X...ZUSD.
            for candidate in list(candidates):
                kraken_candidate = candidate
                if kraken_candidate.startswith("X"):
                    kraken_candidate = kraken_candidate[1:]
                if kraken_candidate.endswith("Z"):
                    kraken_candidate = kraken_candidate[:-1]
                candidates.add(kraken_candidate)
            if any(
                candidate == requested or re.sub(r"^\d+", "", candidate) == requested
                for candidate in candidates
            ):
                # Prefer an unscaled exact pair, then USD/USDT over aliases.
                scale_penalty = 1 if re.match(r"^\d+", base) else 0
                alias_penalty = 0 if base.startswith(requested) else 1
                quote_penalty = 0 if base.endswith("USD") else 1
                matches.append((scale_penalty, alias_penalty, quote_penalty, symbol))
        if matches:
            selected.append(min(matches)[-1])
    return selected


class BinancePublicClient:
    """Small, rate-limit-aware client for Binance USDⓈ-M public endpoints."""

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        timeout: float = 10.0,
        min_interval: float = 0.15,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = max(0.0, min_interval)
        self.session = session or requests.Session()
        self._last_request = 0.0
        self._rate_lock = threading.Lock()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        for attempt in range(3):
            with self._rate_lock:
                wait = self.min_interval - (time.monotonic() - self._last_request)
                if wait > 0:
                    time.sleep(wait)
                self._last_request = time.monotonic()
            response = self.session.get(self.base_url + path, params=params, timeout=self.timeout)
            if response.status_code not in (418, 429, 500, 502, 503, 504):
                response.raise_for_status()
                return response.json()
            if attempt == 2:
                response.raise_for_status()
            retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
            time.sleep(min(30.0, max(0.1, retry_after)))
        raise RuntimeError("unreachable request retry state")

    def active_usdt_perpetual_symbols(self) -> List[str]:
        payload = self._get("/fapi/v1/exchangeInfo")
        return sorted(
            item["symbol"]
            for item in payload.get("symbols", [])
            if item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("contractType") == "PERPETUAL"
        )

    def candles(self, symbol: str, limit: int = 180) -> List[Candle]:
        rows = self._get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": "1m", "limit": min(max(1, limit), 1500)},
        )
        return [
            Candle(
                open_time_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time_ms=int(row[6]),
            )
            for row in rows
        ]


class BybitPublicClient(BinancePublicClient):
    """Public Bybit linear perpetual adapter with the same scanner interface."""

    def active_usdt_perpetual_symbols(self) -> List[str]:
        payload = self._get(
            "/v5/market/instruments-info",
            {"category": "linear", "limit": 1000},
        )
        return sorted(
            item["symbol"]
            for item in payload.get("result", {}).get("list", [])
            if item.get("status") == "Trading"
            and item.get("quoteCoin") == "USDT"
            and item.get("contractType") == "LinearPerpetual"
        )

    def candles(self, symbol: str, limit: int = 180) -> List[Candle]:
        payload = self._get(
            "/v5/market/kline",
            {"category": "linear", "symbol": symbol, "interval": "1", "limit": min(max(1, limit), 1000)},
        )
        rows = payload.get("result", {}).get("list", [])
        return [
            Candle(
                open_time_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time_ms=int(row[0]) + 59_999,
            )
            for row in reversed(rows)
        ]


class KrakenPublicClient(BinancePublicClient):
    """Credential-free Kraken spot fallback for networks blocking derivatives APIs."""

    def active_usdt_perpetual_symbols(self) -> List[str]:
        payload = self._get("/0/public/AssetPairs")
        pairs = []
        for name, item in payload.get("result", {}).items():
            quote = item.get("quote", "")
            if quote in {"ZUSD", "USD", "USDT"} and item.get("status", "online") == "online":
                pairs.append(name)
        return sorted(pairs)

    def candles(self, symbol: str, limit: int = 180) -> List[Candle]:
        payload = self._get("/0/public/OHLC", {"pair": symbol, "interval": 1})
        rows = next(
            (value for key, value in payload.get("result", {}).items() if key != "last"),
            [],
        )[-min(max(1, limit), 720):]
        return [
            Candle(
                open_time_ms=int(row[0]) * 1000,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[6]),
                close_time_ms=int(row[0]) * 1000 + 59_999,
            )
            for row in rows
        ]
