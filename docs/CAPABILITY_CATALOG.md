# INTRA capability catalog

The attached 100-item research list is tracked here so implementation status
stays honest. Items marked **Implemented** are available in this worktree;
items marked **Planned** are requirements for subsequent product phases.

## Implemented foundation

1. Public real-time crypto market data
2. Cost-aware signal decisions
10. Live monitoring dashboard
12. Multi-provider fallback
13. Public market scanner
14. TradingView webhook intake
15. Multi-timeframe feature generation
17. In-app signal display
34. Long and short paper signals
43. Transaction-cost modeling
44. Slippage and spread modeling
73. Basic data-provider failure handling
80. SQLite audit event storage with secret redaction

## Planned capability groups

The remaining requested items are intentionally not represented by fake
controls. They require durable domain models, migrations, replayable datasets,
provider contracts, and tests before being exposed as working features:

- 3, 8, 9, 16, 18, 19, 21-42: risk engine, paper ledger, scheduler, strategy
  builder, indicators, regimes, research strategies, and notifications.
- 41-60: walk-forward/out-of-sample validation, realistic fills, Monte Carlo,
  stress testing, leakage controls, manifests, and experiment tracking.
- 61-80: risk limits, sizing, exposure/correlation controls, circuit breakers,
  anomaly/staleness/event controls, scenario analysis, and immutable risk audit.
- 81-100: news, filings, macro, on-chain/order-flow research, explainability,
  provenance, contradiction detection, scenarios, drift monitoring, model
  registry, approval, bounded research agent, and feedback reporting.

This catalog is a delivery checklist, not a claim that the planned items are
already complete.
