"""TradingView webhook receiver for signal-only alerts."""

import hmac
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict


def validate_secret(payload: Dict[str, Any], secret: str) -> bool:
    if not secret:
        return False
    supplied = str(payload.get("secret", ""))
    return hmac.compare_digest(supplied, secret)


def normalize_alert(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = str(payload.get("action", payload.get("signal", "NO_TRADE"))).upper()
    if action not in {"LONG", "SHORT", "NO_TRADE"}:
        raise ValueError("action must be LONG, SHORT, or NO_TRADE")
    symbol = str(payload.get("symbol", "")).upper()
    if not symbol:
        raise ValueError("symbol is required")
    return {
        "source": "tradingview",
        "symbol": symbol,
        "action": action,
        "timeframe": str(payload.get("timeframe", "")),
        "price": float(payload["price"]) if payload.get("price") is not None else None,
        "message": str(payload.get("message", "")),
    }


def serve(host: str, port: int, secret: str) -> None:
    if not secret:
        raise ValueError("CRYPTO_BOT_TRADINGVIEW_SECRET must be configured")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/tradingview":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                if not validate_secret(payload, secret):
                    self.send_error(401, "invalid webhook secret")
                    return
                result = normalize_alert(payload)
                body = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                print(json.dumps(result, separators=(",", ":")), flush=True)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.send_error(400, str(error))

        def log_message(self, format, *args):
            return

    HTTPServer((host, port), Handler).serve_forever()
