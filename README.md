# Lead–Lag and Triplet Research

Research on NSE stock relationships and prediction of a third stock from two input stocks.

## Folder guide

- `data/`: cleaned prices and daily log returns. The cleaned prices are processed downloads, not an untouched raw-data archive.
- `calculations/`: runnable pipeline scripts.
- `calculations/project_paths.py`: central registry of input and output locations.
- `results/triplets/`: full-universe screening results, both the complete search and initial survivors.
- `results/validation/`: validation results for all 177 screened candidates, including failures.
- `results/strategy/`: prototype backtest results.
- `results/summary/`: combined table of 34 candidates passing the current validation rule.

Existing virtual environments are preserved. The root `.venv/` is the environment used in the commands below; the older `data/.venv/` is also retained.

## Running the pipeline

Run these commands from the project root. Scripts also resolve data paths correctly when invoked by absolute path from another directory. Execute scripts directly as shown, rather than as Python modules.

```sh
.venv/bin/python calculations/fetcher.py
.venv/bin/python calculations/transform.py
.venv/bin/python calculations/triplet_screener_v2.py
.venv/bin/python calculations/triplet_validation.py
.venv/bin/python calculations/triplet_strategy.py
.venv/bin/python calculations/compile_results.py
```

Fetching needs internet and replaces the cleaned price dataset. Each subsequent stage replaces its corresponding result files. Full-universe screening and validation may take substantial time. Validation consumes the full-universe screener output.

The existing environment uses NumPy, pandas, statsmodels, and yfinance. There is no pinned dependency lockfile yet.

## Interpretation of existing results

The saved data covers January 2019 through December 2025. The full screen contains 27,600 triplet–horizon models, 177 initial survivors, and 34 passing the current two-of-three validation rule. None passes rolling stability.

The current triplet workflow and its existing results are retained. Earlier pair-screening methods and the five-anchor experiment were removed from the working folder; they remain recoverable in Git commit `f5ab5e9`. Known methodological issues remain: full-sample candidate selection precedes the holdout test; the backtest uses full-sample coefficients, records only the final daily return for a multi-day hold, and counts nonzero cash-flow days as trades. Treat trading metrics and automatically generated economic explanations as preliminary. The dashboard is a saved presentation snapshot.
