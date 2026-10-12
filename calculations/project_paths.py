"""Project paths resolved independently of the current working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'all_method_combinations_leaderboard.csv': 'results/pairs/all_method_combinations_leaderboard.csv',
    'anchor_pair_lags.csv': 'results/pairs/anchor_pair_lags.csv',
    'final_results_table.csv': 'results/summary/final_results_table.csv',
    'nse_cleaned_prices.csv': 'data/nse_cleaned_prices.csv',
    'nse_log_prices.csv': 'data/nse_log_prices.csv',
    'nse_log_returns.csv': 'data/nse_log_returns.csv',
    'nse_normalized_prices.csv': 'data/nse_normalized_prices.csv',
    'pairs_method_comparison_leaderboard.csv': 'results/pairs/pairs_method_comparison_leaderboard.csv',
    'triplet_full_all.csv': 'results/triplets/triplet_full_all.csv',
    'triplet_full_significant.csv': 'results/triplets/triplet_full_significant.csv',
    'triplet_results_full.csv': 'results/triplets/triplet_results_full.csv',
    'triplet_results_significant.csv': 'results/triplets/triplet_results_significant.csv',
    'triplet_strategy_results.csv': 'results/strategy/triplet_strategy_results.csv',
    'validated_triplets.csv': 'results/validation/validated_triplets.csv',
}

def project_file(name):
    """Locate a known dataset or result without changing the working directory."""
    return ROOT / FILES[name]
