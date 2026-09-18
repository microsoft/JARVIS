# Crypto perpetual futures signal bot

This is a **signal-only** baseline scanner. It discovers active USDT-margined
Binance USDⓈ-M perpetual contracts (with automatic public Bybit and Kraken
spot fallback when derivatives endpoints are unavailable), downloads public
1-minute klines, aggregates
complete 5- and 10-minute bars, and emits calibrated-looking (bounded, volatility
adjusted) probabilities, expected returns, and `LONG`, `SHORT`, or `NO_TRADE`
signals. It never places orders and does not require credentials.

```bash
python -m pip install -r crypto_bot/requirements.txt
CRYPTO_BOT_MAX_SYMBOLS=10 python -m crypto_bot --once
```

## Desktop dashboard

Launch the local INTRA visual console:

```powershell
python -m crypto_bot --desktop
```

To run the desktop app with the terminal bridge:

```powershell
python -m crypto_bot --desktop --bridge
```

The bridge listens only on `127.0.0.1:8790`, prints a per-launch token, and
accepts only `status`, `scan`, `open_dashboard`, and `open_learning_lab`.
It intentionally has no shell execution, credential access, order submission,
or arbitrary computer-control API. Build a Windows package with
`powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1`; Inno
Setup 6 is required for the final `INTRA-Setup.exe`.

It opens `http://127.0.0.1:8788` in the default browser and provides a
full-screen dark telemetry view, scan controls, signal cards, cost thresholds,
exchange status, and TradingView readiness. It remains signal-only and paper
mode; browser controls never submit orders.

Use **SETTINGS** in the top bar to configure the browser workspace. The panel
covers public exchange routing, request limits, RSS news context, MCP/AI
endpoint and model preferences, signal horizons and cost assumptions, the
watchlist, dashboard layout/module visibility, refresh timing, reduced motion,
and the locked paper-only safety boundary. Preferences are stored locally in
the browser and do not replace server environment variables. The Google
control is a safe OAuth handoff: provide an HTTPS OAuth-start URL from a
separately configured host; INTRA never collects Google passwords or stores
provider secrets in the dashboard.

### Learning Lab

Open **LEARNING LAB** from the console for a separate research workspace at
`/learning.html`. It stores technique notes, observations, agent missions, and
upgrade proposals in the local application data directory. Agents are research
subsections, not autonomous trading processes; proposals require manual review
and are never executed as generated code. The upgrade console is deliberately a
proposal and audit surface rather than self-modifying execution.

The optional **OLLAMA CHAT** tab discovers models through the local Ollama HTTP
service at `http://127.0.0.1:11434` and sends paper-research prompts only. It
does not inspect, copy, or modify `C:\Users\relic\.ollama\models`, and it does
not place orders. Start Ollama separately if the model list is offline. Slow
local models may use up to 300 seconds by default; set
`INTRA_OLLAMA_TIMEOUT` to change that limit.

The **PINE SCRIPTS** tab accepts pasted TradingView Pine Script or a local
`.pine`/text file. INTRA safely reads the source as text and extracts the
version, declaration type, title, inputs, plots, alert calls, strategy calls,
line count, and review warnings. Scripts are stored in the local Learning Lab
file for later research. Pine is never evaluated, imported, compiled, or
executed; `strategy.entry`, `strategy.order`, and similar calls are metadata
only. Imported libraries and external `request.security` data are flagged for
manual mapping instead of being silently simulated.

### Holographic avatar visual direction

The dashboard includes a rendered, nonphysical INTRA geometric avatar with
cyan contour styling and scanline effects. This is a browser visual effect on a
normal monitor, not a physical optical hologram; it uses no webcam, microphone,
account credentials, or external model assets.

Useful environment variables include `CRYPTO_BOT_CANDLE_LIMIT` (default `180`),
`CRYPTO_BOT_EXCHANGE` (`auto`, `binance`, `bybit`, or `kraken`; default `auto`).
Kraken is a spot fallback, not a perpetual-futures feed.
By default the scanner focuses on the requested watchlist: MOODENG, NEAR, ONDO,
OP, PENDLE, PENGU, PEPE, PNUT, POL, POPCAT, PUMP, RENDER, S&P500, S, SHIB,
SILVER, SOL, STX, SUI, TAO, TIA, TRUMP, TRX, UNI, VIRTUAL, WIF, WLD, XLM,
XPL, XRP, XYZ100, ZEC, and ZRO. Crypto symbols are matched against active
USDT perpetual contracts, including multiplier contracts such as `1000PEPEUSDT`.
S&P500, SILVER, and XYZ100 are not Binance/Bybit perpetual contracts and are
reported as unavailable rather than being fabricated from another instrument.
Set `CRYPTO_BOT_WATCHLIST` to a comma-separated list to replace the default,
or set it to an empty value to scan all active symbols. `CRYPTO_BOT_MAX_SYMBOLS`
still limits unrestricted discovery, but never truncates an explicit watchlist.

`CRYPTO_BOT_MIN_PROBABILITY` (default `0.55`), `CRYPTO_BOT_FEE_BPS` (default `4`
per side), `CRYPTO_BOT_SPREAD_BPS` (default `2`), `CRYPTO_BOT_SLIPPAGE_BPS`
(default `1`), and `CRYPTO_BOT_POLL_SECONDS` (default `60`). Public candle
requests use bounded concurrency (`CRYPTO_BOT_SCAN_WORKERS`, default `8`),
rate-limit-safe request pacing (`CRYPTO_BOT_REQUEST_INTERVAL`, default
`0.08` seconds), and a finite timeout (`CRYPTO_BOT_TIMEOUT`, default `7`
seconds) so blocked providers do not stall the dashboard indefinitely. Set
`CRYPTO_BOT_MAX_SYMBOLS` to limit a scan; `0` means all active symbols.

The probabilities are a transparent heuristic baseline, not a promise of
profitable or perfect predictions. Validate with historical and forward paper
testing before relying on any signal.

## Prediction review and indicators

The desktop dashboard keeps a local SQLite paper journal for every 1M, 5M, and
10M prediction. After the matching horizon elapses, it compares the realized
price with the predicted direction, records a green **RIGHT** or red **WRONG**
outcome, and shows per-horizon counts. The next scan uses a small bounded
calibration adjustment from resolved outcomes; this is adaptive bookkeeping,
not machine-learning training or a guarantee of improvement. The feature
vector includes ADX trend strength, Awesome Oscillator momentum, RSI, EMA/MACD
distance, Bollinger-band position, ATR ratio, and VWAP distance, which are
also returned with each signal for inspection.

Each refresh also reuses resolved paper outcomes preserved by **ARCHIVE + FRESH
START**, applying a bounded per-symbol/per-horizon calibration offset. A bad
symbol response is isolated so the rest of the watchlist continues scanning.
This remains a heuristic rather than a perfect predictor: public-data latency,
market regime changes, slippage, and unmodeled execution conditions can still
produce wrong signals.

The dashboard also includes a text-only INTRA market brief. Its NEWS
SCOUT reads a small set of public crypto RSS feeds, labels headline tone as
constructive/cautious/mixed, and displays the strongest current paper setup.
Headlines are supplemental context only; they are not treated as confirmed
events, financial advice, or a guarantee of immediate price direction.

Actionable paper signals also include a suggested trigger price, stop-loss, and
take-profit. These levels are volatility- and cost-aware research guidance for
a hypothetical trigger order only; the application never submits orders or
connects to a broker account. `NO_TRADE` cards show reference levels only and
remain non-actionable.

When a signal's displayed time-to-move window expires, the dashboard marks that
symbol/timeframe `NO_TRADE` immediately, removes it from actionable rankings,
and starts a fresh public-data scan. It stays non-actionable until the new scan
publishes an updated decision.

The **TRACK THIS TRADE** control lets you record an order as `OPEN ORDER` or
`FILLED` with entry, stop, and take-profit prices. The local trade board monitors
the next live scans and highlights stop/take alerts in the dashboard. This is a
local paper monitor only: it cannot cancel, raise stops, or modify an order on
Breakout or any exchange.

For a manual Breakout handoff, click a signal, choose **PREPARE BREAKOUT
TICKET**, enter the HTTPS URL for the Breakout order page, and choose **COPY &
OPEN BREAKOUT**. INTRA copies a plain-text ticket containing the symbol, side, trigger, quantity,
stop, and take-profit, then opens the page for you to review and manually
fill/submit. **COPY FORM VALUES** is also available when you only want the
Breakout field values on the clipboard. This is intentionally not browser
automation: INTRA does not log in, inject values into a third-party page, or
submit an order. A true one-click fill would require a separately installed
browser extension/content script with explicit permission for the Breakout
domain.

Use **ARCHIVE + FRESH START** in the dashboard to analyze and preserve the
current paper prediction journal, reset active win/profit counters, and begin a
new scan. Archived records remain in the local SQLite database for future
review; no exchange orders or credentials are involved.

Use **BACKTEST & LEARN** to run a bounded walk-forward review over the currently
discovered/watchlisted pairs and the 1M, 5M, and 10M horizons. While it runs,
manual and automatic market scans pause. It evaluates historical public candles
with the same fee-aware signal decision, stores bounded global and per-currency-
pair/per-horizon calibration in the local SQLite journal, and displays a
closable summary of calls, accuracy, and net return. This is a research aid: it
is not a guarantee of future performance, and it does not place orders.

## TradingView alerts

To use TradingView as an input source, configure an alert with **Webhook URL**
`https://YOUR_PUBLIC_HOST/tradingview` and this JSON message:

```json
{"secret":"replace-with-a-long-random-value","symbol":"{{ticker}}","action":"LONG","timeframe":"{{interval}}","price":"{{close}}","message":"{{strategy.order.comment}}"}
```

Start the receiver locally with:

```powershell
$env:CRYPTO_BOT_TRADINGVIEW_SECRET="replace-with-a-long-random-value"
python -m crypto_bot --tradingview-webhook
```

TradingView must be able to reach the URL, so a local-only `127.0.0.1` listener
needs a secure public HTTPS tunnel or deployed host. The receiver validates the
shared secret, accepts only `LONG`, `SHORT`, and `NO_TRADE`, and only prints
normalized alerts; it never submits exchange orders. Do not put exchange API
keys in TradingView alert messages.

Signing in to TradingView does **not** give this app access to the TradingView
account, and the app intentionally never requests that access. To connect
TradingView, create an alert in TradingView, enable its webhook URL field, and
point it at a public HTTPS relay that forwards to this receiver. A local
`127.0.0.1` URL is not reachable by TradingView. The receiver is therefore
ready for local testing, but a user-controlled HTTPS relay or tunnel is
required for external alerts.
