# INTRA Trading Research Desktop status

## Current implementation

- Signal-only crypto scanner using public Binance USDⓈ-M data with Bybit fallback.
- 1-minute OHLCV collection and 1/5/10-minute feature horizons.
- Cost-aware `LONG`, `SHORT`, and `NO_TRADE` decisions.
- TradingView webhook receiver with shared-secret validation and no login handling.
- Local INTRA dashboard with editable safety label.
- Paper-only foundation with Windows-safe application directories and an append-only
  SQLite audit store.
- Local dashboard API bound to loopback and authenticated with a per-launch token.

## Deliberate safety boundaries

The permanent contract is **NO BROKER CREDENTIALS - NO LIVE ORDERS - PAPER MODE
ONLY**. The application does not expose private exchange endpoints, order
placement, withdrawals, TradingView login automation, password collection, or
credential storage.

## Remaining roadmap

The attached master specification describes a larger product: a resumable setup
wizard, multi-page research workspace, paper ledger, backtesting lab, persistent
automation scheduler, notifications, packaged Windows shell, and expanded
provider adapters. These are not represented as complete by the current
prototype and should be implemented incrementally behind tests.
