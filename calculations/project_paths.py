"""Project paths resolved independently of the current working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'final_results_table.csv': 'results/summary/final_results_table.csv',
    'nse_cleaned_prices.csv': 'data/nse_cleaned_prices.csv',
    'nse_log_returns.csv': 'data/nse_log_returns.csv',
    'triplet_full_all.csv': 'results/triplets/triplet_full_all.csv',
    'triplet_full_significant.csv': 'results/triplets/triplet_full_significant.csv',
    'triplet_strategy_results.csv': 'results/strategy/triplet_strategy_results.csv',
    'validated_triplets.csv': 'results/validation/validated_triplets.csv',
}

def project_file(name):
    """Locate a known dataset or result without changing the working directory."""
    return ROOT / FILES[name]
