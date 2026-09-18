"""Local application foundation: paths, paper-only settings, and audit storage."""

import json
import os
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


SAFETY_CONTRACT = "NO BROKER CREDENTIALS - NO LIVE ORDERS - PAPER MODE ONLY"


def directional_pnl(move: float, action: str, cost: float = 0.0) -> float:
    """Convert market movement into strategy PNL; falling price benefits SHORT."""
    gross = move if action == "LONG" else -move if action == "SHORT" else 0.0
    return gross - cost if action in ("LONG", "SHORT") else gross


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(value: Any) -> Any:
    """Remove common secret fields before persistence or logging."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if re.search(r"(secret|token|password|api.?key|cookie)", key, re.I)
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


@dataclass(frozen=True)
class AppPaths:
    config: Path
    data: Path
    cache: Path
    logs: Path
    exports: Path

    @classmethod
    def resolve(cls, root: Optional[Path] = None) -> "AppPaths":
        if root is not None:
            base = Path(root)
            return cls(base / "config", base / "data", base / "cache", base / "logs", base / "exports")
        appdata = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
        local = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return cls(
            appdata / "INTRA Trading Research" / "config",
            local / "INTRA Trading Research" / "data",
            local / "INTRA Trading Research" / "cache",
            local / "INTRA Trading Research" / "logs",
            Path.home() / "Documents" / "INTRA Exports",
        )

    def create(self) -> "AppPaths":
        for path in (self.config, self.data, self.cache, self.logs, self.exports):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def database(self) -> Path:
        return self.data / "intra.sqlite3"


class AuditStore:
    """Small SQLite audit store with append-only events."""

    def __init__(self, database: Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.database), check_same_thread=False)
        self.lock = threading.RLock()
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )"""
        )
        self.connection.commit()

    def append(self, event_type: str, payload: Dict[str, Any]) -> int:
        with self.lock:
            cursor = self.connection.execute(
                "INSERT INTO audit_events(created_at,event_type,payload_json) VALUES(?,?,?)",
                (utc_now(), event_type, json.dumps(redact(payload), sort_keys=True)),
            )
            self.connection.commit()
            return int(cursor.lastrowid)

    def close(self) -> None:
        self.connection.close()


class PredictionJournal:
    """Durable paper prediction ledger with delayed outcome resolution."""

    def __init__(self, database: Path):
        self.connection = sqlite3.connect(str(database), check_same_thread=False)
        self.lock = threading.RLock()
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL, horizon_minutes INTEGER NOT NULL,
                created_at TEXT NOT NULL, entry_price REAL NOT NULL,
                action TEXT NOT NULL, probability REAL NOT NULL,
                expected_return REAL NOT NULL, resolved_at TEXT,
                exit_price REAL, correct INTEGER
            )"""
        )
        self.connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_key "
            "ON predictions(symbol, horizon_minutes, created_at)"
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS prediction_archives (
                archive_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                prediction_count INTEGER NOT NULL,
                analysis_json TEXT NOT NULL,
                predictions_json TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS strategy_learning (
                horizon_minutes INTEGER PRIMARY KEY,
                calibration_offset REAL NOT NULL,
                samples INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                net_return REAL NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS pair_strategy_learning (
                symbol TEXT NOT NULL, horizon_minutes INTEGER NOT NULL,
                calibration_offset REAL NOT NULL, samples INTEGER NOT NULL,
                accuracy REAL NOT NULL, net_return REAL NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY(symbol, horizon_minutes)
            )"""
        )
        self.connection.commit()

    def calibration_offset(self, symbol: str, horizon: int) -> float:
        with self.lock:
            rows = self.connection.execute(
                "SELECT correct FROM predictions WHERE symbol=? AND horizon_minutes=? "
                "AND correct IS NOT NULL",
                (symbol, horizon),
            ).fetchall()
            # Archives survive a fresh-start reset; include their resolved
            # outcomes so each refresh benefits from the complete paper record.
            archived_rows = self.connection.execute(
                "SELECT predictions_json FROM prediction_archives"
            ).fetchall()
            for (payload,) in archived_rows:
                for record in json.loads(payload):
                    if (record.get("symbol") == symbol
                            and record.get("horizon_minutes") == horizon
                            and record.get("correct") is not None):
                        rows.append((record["correct"],))
        accuracy = sum(row[0] for row in rows) / len(rows) if rows else 0.5
        learned = self.connection.execute(
            "SELECT calibration_offset FROM strategy_learning WHERE horizon_minutes=?",
            (horizon,),
        ).fetchone()
        learned_offset = learned[0] if learned else 0.0
        pair = self.connection.execute(
            "SELECT calibration_offset FROM pair_strategy_learning "
            "WHERE symbol=? AND horizon_minutes=?", (symbol, horizon)
        ).fetchone()
        if pair:
            learned_offset = pair[0]
        return max(-1.0, min(1.0, (accuracy - 0.5) * 2.0 + learned_offset))

    def apply_backtest_learning(self, results: Dict[str, Dict[str, Any]]) -> None:
        with self.lock:
            for horizon, values in results.items():
                accuracy = float(values["accuracy"])
                offset = max(-0.75, min(0.75, (accuracy - 0.5) * 1.5))
                self.connection.execute(
                    """INSERT INTO strategy_learning
                    (horizon_minutes,calibration_offset,samples,accuracy,net_return,updated_at)
                    VALUES(?,?,?,?,?,?)
                    ON CONFLICT(horizon_minutes) DO UPDATE SET
                    calibration_offset=excluded.calibration_offset,
                    samples=excluded.samples, accuracy=excluded.accuracy,
                    net_return=excluded.net_return, updated_at=excluded.updated_at""",
                    (int(horizon), offset, values["calls"], accuracy,
                     values["net_return"], utc_now()),
                )
            self.connection.commit()

    def apply_pair_backtest_learning(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        with self.lock:
            for symbol, horizons in results.items():
                for horizon, values in horizons.items():
                    calls = int(values["calls"])
                    accuracy = float(values["accuracy"])
                    offset = max(-0.75, min(0.75, (accuracy - 0.5) * 1.5))
                    self.connection.execute(
                        """INSERT INTO pair_strategy_learning
                        (symbol,horizon_minutes,calibration_offset,samples,accuracy,net_return,updated_at)
                        VALUES(?,?,?,?,?,?,?)
                        ON CONFLICT(symbol,horizon_minutes) DO UPDATE SET
                        calibration_offset=excluded.calibration_offset,
                        samples=excluded.samples, accuracy=excluded.accuracy,
                        net_return=excluded.net_return, updated_at=excluded.updated_at""",
                        (symbol, int(horizon), offset, calls, accuracy,
                         float(values["net_return"]), utc_now()),
                    )
            self.connection.commit()

    def record(self, signal: Dict[str, Any], entry_price: float, created_at: str) -> None:
        with self.lock:
            self.connection.execute(
                """INSERT OR IGNORE INTO predictions
                (symbol,horizon_minutes,created_at,entry_price,action,probability,expected_return)
                VALUES(?,?,?,?,?,?,?)""",
                (signal["symbol"], signal["horizon_minutes"], created_at, entry_price,
                 signal["action"], signal["probability"], signal["expected_return"]),
            )
            self.connection.commit()

    def resolve(self, symbol: str, horizon: int, current_price: float,
                now: str, cost: float) -> None:
        with self.lock:
            rows = self.connection.execute(
                "SELECT id,entry_price,action,created_at FROM predictions "
                "WHERE symbol=? AND horizon_minutes=? AND correct IS NULL",
                (symbol, horizon),
            ).fetchall()
            for row_id, entry, action, created_at in rows:
                try:
                    age_seconds = (datetime.fromisoformat(now) -
                                   datetime.fromisoformat(created_at)).total_seconds()
                except ValueError:
                    continue
                if age_seconds < horizon * 60:
                    continue
                move = (current_price - entry) / entry if entry else 0.0
                pnl = directional_pnl(move, action, cost)
                correct = (pnl > 0) if action in ("LONG", "SHORT") else abs(move) <= cost
                self.connection.execute(
                    "UPDATE predictions SET resolved_at=?,exit_price=?,correct=? WHERE id=?",
                    (now, current_price, int(correct), row_id),
                )
            self.connection.commit()

    def summary(self) -> Dict[str, Dict[str, int]]:
        with self.lock:
            result = {}
            for horizon in (1, 5, 10):
                rows = self.connection.execute(
                    "SELECT correct, COUNT(*) FROM predictions "
                    "WHERE horizon_minutes=? AND correct IS NOT NULL GROUP BY correct",
                    (horizon,),
                ).fetchall()
                counts = {0: 0, 1: 0}
                counts.update({int(value): count for value, count in rows})
                result[str(horizon)] = {"right": counts[1], "wrong": counts[0]}
            return result

    def performance_summary(self, cost: float) -> Dict[str, Dict[str, float]]:
        """Return hypothetical net return for resolved actionable predictions."""
        with self.lock:
            result = {}
            for horizon in (1, 5, 10):
                rows = self.connection.execute(
                    "SELECT entry_price, exit_price, action FROM predictions "
                    "WHERE horizon_minutes=? AND correct IS NOT NULL "
                    "AND action IN ('LONG','SHORT') AND exit_price IS NOT NULL",
                    (horizon,),
                ).fetchall()
                net_return = 0.0
                for entry, exit_price, action in rows:
                    move = ((exit_price - entry) / entry) if entry else 0.0
                    net_return += directional_pnl(move, action, cost)
                result[str(horizon)] = {
                    "net_return": round(net_return, 6),
                    "closed_calls": len(rows),
                }
            return result

    def archive_and_reset(self, cost: float) -> Dict[str, Any]:
        """Archive the complete active journal, analyze it, then start empty."""
        with self.lock:
            rows = self.connection.execute(
                "SELECT symbol,horizon_minutes,created_at,entry_price,action,probability,"
                "expected_return,resolved_at,exit_price,correct FROM predictions "
                "ORDER BY id"
            ).fetchall()
            columns = (
                "symbol", "horizon_minutes", "created_at", "entry_price", "action",
                "probability", "expected_return", "resolved_at", "exit_price", "correct",
            )
            records = [dict(zip(columns, row)) for row in rows]
            analysis = {
                "outcomes": self.summary(),
                "performance": self.performance_summary(cost),
                "symbols": sorted({record["symbol"] for record in records}),
                "actionable_calls": sum(
                    record["action"] in ("LONG", "SHORT") for record in records
                ),
            }
            archive_id = uuid.uuid4().hex
            self.connection.execute(
                "INSERT INTO prediction_archives "
                "(archive_id,created_at,prediction_count,analysis_json,predictions_json) "
                "VALUES(?,?,?,?,?)",
                (archive_id, utc_now(), len(records), json.dumps(analysis, sort_keys=True),
                 json.dumps(records, sort_keys=True)),
            )
            self.connection.execute("DELETE FROM predictions")
            self.connection.commit()
            return {
                "archive_id": archive_id,
                "prediction_count": len(records),
                "analysis": analysis,
            }

    def close(self) -> None:
        self.connection.close()
