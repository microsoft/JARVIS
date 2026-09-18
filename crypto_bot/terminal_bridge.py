"""Local, allowlisted terminal bridge for the INTRA desktop application."""

import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict


def run_bridge(status_provider: Callable[[], Dict[str, Any]],
               scan_provider: Callable[[], Any],
               host: str = "127.0.0.1", port: int = 8790,
               open_browser: bool = True) -> str:
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def _authorized(self) -> bool:
            return secrets.compare_digest(
                self.headers.get("X-INTRA-BRIDGE-TOKEN", ""), token
            )

        def _send(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "application": "INTRA", "paper_only": True})
                return
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            if self.path == "/capabilities":
                self._send(200, {"commands": [
                    "status", "scan", "open_dashboard", "open_learning_lab",
                ], "paper_only": True, "shell_access": False})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if not self._authorized():
                self._send(401, {"error": "unauthorized"})
                return
            if self.path != "/command":
                self._send(404, {"error": "not found"})
                return
            try:
                command = str(self._body().get("command", "")).strip().lower()
                if command == "status":
                    self._send(200, {"ok": True, "result": status_provider()})
                elif command == "scan":
                    self._send(200, {"ok": True, "result": scan_provider()})
                elif command == "open_dashboard":
                    webbrowser.open("http://127.0.0.1:8788/")
                    self._send(200, {"ok": True, "result": "dashboard opened"})
                elif command == "open_learning_lab":
                    webbrowser.open("http://127.0.0.1:8788/learning.html")
                    self._send(200, {"ok": True, "result": "learning lab opened"})
                else:
                    self._send(400, {"error": "command is not allowlisted"})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"INTRA terminal bridge listening at http://{host}:{port}", flush=True)
    print(f"INTRA bridge token: {token}", flush=True)
    if open_browser:
        threading.Timer(0.3, webbrowser.open,
                        args=(f"http://{host}:{port}/health",)).start()
    server.serve_forever()
    return token
