# Recurrence Patterns Correlation (RPC)

**Recurrence Pattern Correlation (RPC)** is a novel method designed to bridge the gap between qualitative Recurrence Plot (RP) inspection and traditional quantitative measures. While Recurrence Quantification Analysis (RQA) typically relies on global metrics like line-based statistics, RPC introduces a spatial-statistics-inspired approach that captures localized and system-specific structures in RPs.

## Overview

Recurrence Plots (RPs) are powerful tools for visualizing the dynamics of time series. However, conventional RQA may overlook localized or scale-dependent features. RPC addresses this by measuring the correlation of an RP to patterns of arbitrary shape and scale using a spatial autocorrelation metric derived from **Moran's I** \[1].

This approach enables the detection and quantification of long-range, structured correlations in RPs—revealing hidden dynamics such as the unstable manifolds of periodic orbits. Applications include visualizing bifurcations in the Logistic map, characterizing the mixed phase space in the Standard map, and tracking unstable periodic orbits in the Lorenz '63 system.

## Key Features

* ✅ Introduces a novel pattern-based quantifier inspired by **Moran’s I** \[1]
* 📊 Captures localized and long-range spatial correlations in binary recurrence plots
* 🔍 Detects system-specific structures across arbitrary shapes, scales, and time lags

## Use Cases

* Visualizing unstable manifolds in nonlinear dynamical systems
* Dissecting phase space complexity in mixed or chaotic regimes
* Tracking periodic orbits and structure formation in RPs
* Augmenting traditional RQA with more flexible, pattern-sensitive analysis

## References

\[1] H. Li, C. A. Calder, N. Cressie, *Beyond Moran’s I: Testing for Spatial Dependence Based on the Spatial Autoregressive Model*, Geogr. Anal. 39 (2007) 357–375.

\[2] N. Marwan, M. C. Romano, M. Thiel, J. Kurths, *Recurrence Plots for the Analysis of Complex Systems*, Phys. Rep. 438 (2007) 237–329.
