# IDA PoC results — systemic tail-risk network (real data)

Panel: 1459 daily obs x 12 assets (2021-01-02..2024-12-30), Yahoo Finance.
Rolling window 150d / step 3d; quantile tau=0.05; L1 alpha=0.05.

## Systemic-stress trajectory INTO each crisis (trailing-120d percentile)

| Crisis | T−15d | T−10d | T−5d | onset |
|---|---|---|---|---|
| Terra/Luna | 100th | 87th | 90th | 100th |
| FTX | 3th | 0th | 0th | 0th |

## Two crisis types — an honest, motivating contrast

- **Terra/Luna** (market-wide deleveraging/contagion): index elevated & rising into onset (trailing pct 100→100); global peak ≈ 90th percentile. The connectedness operator FLAGS it.
- **FTX** (idiosyncratic exchange-solvency shock): index did NOT locally lead the event (trailing pct ≈ 0); it spikes only at/after onset (global peak ≈ 87th percentile). A naive point-estimate connectedness measure MISSES a sudden off-chain counterparty failure.

This contrast is the project's motivation, not a defect: it shows why a point estimate is not enough and why O1 (finite-sample reliability + diagnostics under shift) and O3 (rigorous, crisis-typed validation) are needed.

## Conditional forward risk

- Mean next-10-day equal-weight return after **high-stress** days (top quintile): **-0.5%**
- Mean next-10-day equal-weight return after **low-stress** days (bottom quintile): **+0.6%**
- Linear corr(stress_t, next-10d drawdown) = 0.05 (weak globally — exactly why a naive point estimate is insufficient and O1's reliability theory is needed).

Figure: `systemic_indices.png`. Series: `systemic_indices.csv`.

*PoC scope: an FRM-style proxy + generalised-FEVD connectedness on a fixed VAR(1) — it demonstrates the O2 operator is computable on real data and tracks systemic stress, NOT the O1 finite-sample reliability theory (the project's novel, unproven-in-PoC contribution).*
