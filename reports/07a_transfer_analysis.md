# Task 3 — Georgescu -> Simion Transfer Analysis

## Overall finding
Raw county-level correlation: r = 0.884 (p < 0.001), indicating near-linear
co-movement of Georgescu 2024 R1 and Simion 2025 R1 vote shares across the 42 judete.

After removing shared structural variance (14 controls via Frisch-Waugh residualisation):
**Partial r = 0.770**  (95% bootstrap CI: [0.649, 0.863], p = 0.0000)

The partial correlation remains strong and statistically significant, confirming
that the Georgescu -> Simion transfer is **not reducible to structural confounders**.
It reflects genuine county-level persistence of a nationalist-populist electoral bloc.

## Sensitivity (Cinelli & Hazlett 2020)
Robustness Value RV = 0.547
An unmeasured confounder would need a partial R^2 of at least 55% with BOTH
the Georgescu vote and the Simion vote (conditional on included controls) to
explain away the partial correlation. This is a stringent threshold, making the
finding robust to moderate omitted variable bias.

## Cluster heterogeneity
See figures/07a_partial_corr_clusters.png for cluster-stratified estimates.
The transfer signal is strongest in [see output] and weakest in Insula Capitalei
(capital/urban, n=2, caution: small n).

## Limitations
- N=42 ecological units — individual-level transfer cannot be inferred directly
  (ecological fallacy caveat applies)
- Simion and Georgescu ran in different institutional contexts (2024 cancelled
  election vs 2025 re-run); party/ballot structure differed
- Composite controls absorb multicollinearity but may over-control if structural
  conditions are themselves channels of the transfer
