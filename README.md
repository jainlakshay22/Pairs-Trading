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

## Running the corrected prediction study

Use the existing return dataset; fetching and transformation are not required for this run.

```sh
.venv/bin/python -u calculations/triplet_screener_v2.py
.venv/bin/python -u calculations/triplet_validation.py
.venv/bin/python -u calculations/compile_results.py
```

Run each command only after the previous one completes successfully. Output files are saved directly under `results/training_only/`. Upload `summary.json`, `final_results_table.csv`, `training_screen_all.csv`, and `test_predictions.csv`; no ZIP is created. Previous result files remain historical and are not inputs to the corrected run.

Screening uses only 2019-2022 observations and targets. Candidate identifiers and training coefficients are frozen before evaluation. Validation evaluates every candidate against single-input and training-mean forecasts; it does not select models using 2023-2025 performance. Compilation retains every candidate and saves the final results table and summary metadata. File hashes prevent mixing data or results from different stages.

For another run, pass the same new `--output results/training_only_run2` argument to all three commands. Screening refuses an existing output folder. Validation uses 1,000 paired circular shifts by default; this differs from the old A-only shuffle. Multiple-testing-adjusted values and HAC diagnostics are reported separately from the nominal exploratory screen.

Do not run `triplet_strategy.py` as part of this study: it remains a legacy backtest with known coefficient and holding-period accounting issues. No trading results are compiled. Existing tables/dashboard are historical, not corrected outputs.

## Interpretation

This fixes training/test selection boundaries in the analysis, but 2023-2025 has already been examined, so evaluation is retrospective. The existing dataset's missing-value treatment and price adjustments still require an audit. Statistical diagnostics do not prove causality, stable regimes or profitability. Do not reuse the old 177/34 counts or the unverified HTML's 630/25 counts as corrected results; use the outputs of the new run.
