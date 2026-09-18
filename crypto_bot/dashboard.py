"""Local browser dashboard for the signal-only bot."""

import json
import os
import secrets
import threading
import webbrowser
import urllib.request
import urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .config import BotConfig
from .foundation import SAFETY_CONTRACT, AppPaths, AuditStore, PredictionJournal


WEB_DIR = Path(__file__).with_name("web")
CORE_AGENTS = [
    ("Volatility-Regime Agent", "RESEARCH", "Classify trending, ranging, high-volatility, and illiquid regimes from ATR ratio and Bollinger width."),
    ("Correlation/Leadership Agent", "RESEARCH", "Track BTC/ETH leadership and altcoin beta/correlation to identify market-wide beta."),
    ("Order-Flow Agent", "FEATURES", "Research order-book imbalance and taker buy/sell volume from public market data."),
    ("Funding Agent", "FEATURES", "Track public funding rate level and rate-of-change by symbol."),
    ("Basis Agent", "FEATURES", "Track mark-price versus index-price divergence."),
    ("Liquidation Agent", "FEATURES", "Research liquidation-cluster detection from public liquidation feeds."),
    ("Structure Agent", "FEATURES", "Detect Break of Structure, Change of Character, and Fair Value Gaps mechanically."),
    ("Risk Agent", "RISK", "Enforce actionability gates, cost-aware levels, and NO_TRADE conditions."),
    ("News Agent", "NEWS", "Summarize public news and event context as supplemental signal gating evidence."),
    ("Backtest Agent", "BACKTEST", "Own walk-forward validation, baselines, drawdown, and multiple-testing controls."),
    ("Calibration Agent", "BACKTEST", "Own bounded per-symbol/per-horizon calibration with minimum-sample safeguards."),
]


def run_dashboard(config: BotConfig, host: str = "127.0.0.1", port: int = 8788,
                  open_browser: bool = True) -> None:
    session_token = secrets.token_urlsafe(32)
    paths = AppPaths.resolve().create()
    audit = AuditStore(paths.database)
    journal = PredictionJournal(paths.database)
    scan_lock = threading.Lock()
    learning_path = paths.data / "learning_lab.json"
    learning_lock = threading.RLock()

    def load_learning() -> Dict[str, Any]:
        with learning_lock:
            if not learning_path.exists():
                data = {"entries": [], "agents": [], "upgrades": [], "pine_scripts": [], "agent_reports": []}
                data["agents"] = [
                    {"id": secrets.token_hex(8), "name": name, "mission": mission,
                     "scope": scope, "status": "ACTIVE // PAPER RESEARCH", "knowledge_count": 0}
                    for name, scope, mission in CORE_AGENTS
                ]
                save_learning(data)
                return data
            try:
                data = json.loads(learning_path.read_text(encoding="utf-8"))
                result = {key: data.get(key, []) if isinstance(data.get(key, []), list) else []
                          for key in ("entries", "agents", "upgrades", "pine_scripts", "agent_reports")}
                existing_names = {item.get("name") for item in result["agents"]}
                for name, scope, mission in CORE_AGENTS:
                    if name not in existing_names:
                        result["agents"].append({
                            "id": secrets.token_hex(8), "name": name, "mission": mission,
                            "scope": scope, "status": "ACTIVE // PAPER RESEARCH", "knowledge_count": 0,
                        })
                if len(result["agents"]) != len(data.get("agents", [])):
                    save_learning(result)
                return result
            except (OSError, ValueError):
                return {"entries": [], "agents": [], "upgrades": [], "pine_scripts": [], "agent_reports": []}

    def save_learning(data: Dict[str, Any]) -> None:
        with learning_lock:
            learning_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    audit.append("dashboard_started", {"mode": "paper", "safety_contract": SAFETY_CONTRACT})

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            return self.headers.get("Authorization", "") == f"Bearer {session_token}"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):
            path = urlparse(self.path).path
            if path.startswith("/api/") and not self._authorized():
                self._send(401, b'{"error":"unauthorized"}', "application/json")
                return
            if path == "/api/status":
                summary = journal.summary()
                self._send(200, json.dumps({
                    "exchange": config.exchange,
                    "watchlist": list(config.watchlist),
                    "watchlist_count": len(config.watchlist),
                    "paper_mode": True,
                    "tradingview_webhook": f"http://{config.tradingview_host}:{config.tradingview_port}/tradingview",
                    "cost_bps": round(config.estimated_round_trip_cost * 10_000, 2),
                    "safety_contract": SAFETY_CONTRACT,
                    "app_data": str(paths.data),
                    "prediction_outcomes": summary,
                    "paper_performance": journal.performance_summary(
                        config.estimated_round_trip_cost
                    ),
                }).encode(), "application/json")
                return
            if path == "/api/news":
                try:
                    from .news import scan_headlines
                    self._send(200, json.dumps({"news": scan_headlines(), "error": None}).encode(),
                               "application/json")
                except Exception as error:
                    self._send(503, json.dumps({"news": {}, "error": str(error)}).encode(),
                               "application/json")
                return
            if path == "/api/learning":
                self._send(200, json.dumps(load_learning()).encode(), "application/json")
                return
            if path == "/api/ollama/models":
                try:
                    request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
                    with urllib.request.urlopen(request, timeout=2) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    self._send(200, json.dumps({"models": payload.get("models", []), "error": None}).encode(),
                               "application/json")
                except (OSError, ValueError, urllib.error.URLError) as error:
                    self._send(503, json.dumps({"models": [], "error": str(error)}).encode(),
                               "application/json")
                return
            if path == "/api/scan":
                try:
                    from .main import scan
                    with scan_lock:
                        result = {"signals": scan(config, journal), "error": None}
                    self._send(200, json.dumps(result).encode(), "application/json")
                except Exception as error:
                    self._send(503, json.dumps({"signals": [], "error": str(error)}).encode(),
                               "application/json")
                return
            file_name = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (WEB_DIR / file_name).resolve()
            if WEB_DIR.resolve() not in target.parents or not target.is_file():
                self._send(404, b"Not found", "text/plain")
                return
            content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "text/css; charset=utf-8"
            if target.suffix == ".js":
                content_type = "application/javascript"
            body = target.read_bytes()
            if target.suffix == ".html":
                body = body.replace(b"__INTRA_SESSION_TOKEN__", session_token.encode("ascii"))
            self._send(200, body, content_type)

        def do_POST(self):
            path = urlparse(self.path).path
            if path.startswith("/api/") and not self._authorized():
                self._send(401, b'{"error":"unauthorized"}', "application/json")
                return
            if path == "/api/backtest":
                try:
                    self._json_body()
                    from .backtest import run_backtest
                    from .main import create_client_and_symbols
                    client, symbols = create_client_and_symbols(config)
                    with scan_lock:
                        result = run_backtest(client, symbols, config, journal)
                    audit.append("backtest_completed", {"pairs": len(symbols), "results": result})
                    self._send(200, json.dumps({"results": result, "error": None}).encode(), "application/json")
                except Exception as error:
                    self._send(503, json.dumps({"results": {}, "error": str(error)}).encode(), "application/json")
                return
            if path == "/api/learning/entry":
                body = self._json_body()
                title = str(body.get("title", "")).strip()
                content = str(body.get("content", "")).strip()
                if not title or not content:
                    self._send(400, b'{"error":"title and content are required"}', "application/json")
                    return
                data = load_learning()
                data["entries"].append({
                    "id": secrets.token_hex(8), "title": title, "content": content,
                    "category": str(body.get("category", "TECHNIQUE")).strip() or "TECHNIQUE",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                save_learning(data)
                self._send(200, json.dumps(data).encode(), "application/json")
                return
            if path == "/api/learning/agent":
                body = self._json_body()
                name = str(body.get("name", "")).strip()
                mission = str(body.get("mission", "")).strip()
                if not name or not mission:
                    self._send(400, b'{"error":"name and mission are required"}', "application/json")
                    return
                data = load_learning()
                data["agents"].append({
                    "id": secrets.token_hex(8), "name": name, "mission": mission,
                    "scope": str(body.get("scope", "RESEARCH")).strip() or "RESEARCH",
                    "status": "ACTIVE // PAPER RESEARCH",
                    "knowledge_count": 0,
                })
                save_learning(data)
                self._send(200, json.dumps(data).encode(), "application/json")
                return
            if path == "/api/learning/agent-report":
                body = self._json_body()
                agent_id = str(body.get("agent_id", "")).strip()
                report = str(body.get("report", "")).strip()
                if not agent_id or not report:
                    self._send(400, b'{"error":"agent_id and report are required"}', "application/json")
                    return
                data = load_learning()
                agent = next((item for item in data["agents"] if item.get("id") == agent_id), None)
                if agent is None:
                    self._send(404, b'{"error":"agent not found"}', "application/json")
                    return
                report_item = {
                    "id": secrets.token_hex(8), "agent_id": agent_id,
                    "agent": agent.get("name", "Unknown agent"), "report": report,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                data["agent_reports"].append(report_item)
                agent["knowledge_count"] = int(agent.get("knowledge_count", 0)) + 1
                data["entries"].append({
                    "id": secrets.token_hex(8),
                    "title": f"{agent.get('name', 'Agent')} report",
                    "content": report,
                    "category": "AGENT REPORT",
                    "source_agent_id": agent_id,
                    "created_at": report_item["created_at"],
                })
                save_learning(data)
                audit.append("agent_report_saved", {"agent_id": agent_id})
                self._send(200, json.dumps(data).encode(), "application/json")
                return
            if path == "/api/learning/pine":
                body = self._json_body()
                source = str(body.get("source", ""))
                name = str(body.get("name", "untitled.pine")).strip() or "untitled.pine"
                try:
                    from .pine import inspect_script
                    analysis = inspect_script(source, name)
                except ValueError as error:
                    self._send(400, json.dumps({"error": str(error)}).encode(), "application/json")
                    return
                data = load_learning()
                data["pine_scripts"].append({
                    "id": secrets.token_hex(8),
                    "analysis": analysis,
                    "source": source,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
                save_learning(data)
                audit.append("pine_script_added", {
                    "name": analysis["name"], "version": analysis["version"],
                    "declaration": analysis["declaration"],
                })
                self._send(200, json.dumps(data).encode(), "application/json")
                return
            if path == "/api/learning/upgrade":
                body = self._json_body()
                proposal = str(body.get("proposal", "")).strip()
                if not proposal:
                    self._send(400, b'{"error":"proposal is required"}', "application/json")
                    return
                data = load_learning()
                data["upgrades"].append({
                    "id": secrets.token_hex(8), "proposal": proposal,
                    "status": "PROPOSED // MANUAL REVIEW REQUIRED",
                })
                save_learning(data)
                audit.append("learning_upgrade_proposed", {"proposal": proposal})
                self._send(200, json.dumps(data).encode(), "application/json")
                return
            if path == "/api/ollama/chat":
                body = self._json_body()
                model = str(body.get("model", "")).strip()
                message = str(body.get("message", "")).strip()
                if not model or not message:
                    self._send(400, b'{"error":"model and message are required"}', "application/json")
                    return
                request_body = json.dumps({
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a paper-trading research assistant. Never claim certainty, never place orders, and label ideas as hypotheses."},
                        {"role": "user", "content": message},
                    ],
                    "stream": False,
                }).encode("utf-8")
                try:
                    request = urllib.request.Request(
                        "http://127.0.0.1:11434/api/chat", data=request_body,
                        headers={"Content-Type": "application/json"},
                    )
                    timeout = max(10, int(os.environ.get("INTRA_OLLAMA_TIMEOUT", "300")))
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    self._send(200, json.dumps({"reply": payload.get("message", {}).get("content", ""), "error": None}).encode(),
                               "application/json")
                except (OSError, ValueError, urllib.error.URLError) as error:
                    self._send(503, json.dumps({"reply": "", "error": str(error)}).encode(),
                               "application/json")
                return
            if path != "/api/archive-reset":
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            try:
                self._json_body()
                from .main import scan
                archive = journal.archive_and_reset(config.estimated_round_trip_cost)
                fresh_signals = scan(config, journal)
                audit.append("prediction_archive_reset", {
                    "archive_id": archive["archive_id"],
                    "prediction_count": archive["prediction_count"],
                })
                self._send(200, json.dumps({
                    "archive": archive,
                    "signals": fresh_signals,
                    "error": None,
                }).encode(), "application/json")
            except Exception as error:
                self._send(503, json.dumps({"signals": [], "error": str(error)}).encode(),
                           "application/json")

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"Dashboard running at {url} (signal-only / paper mode)", flush=True)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    server.serve_forever()
