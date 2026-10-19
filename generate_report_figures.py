"""Place in the Pairs-Trading root and run: python generate_report_figures.py.
Requires: pandas, numpy, matplotlib. Reads saved CSVs; does not rerun models.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
root = args.project_root
out = root / 'results' / 'report_figures'
inputs = {
    'prices': root / 'data/nse_cleaned_prices.csv',
    'screen': root / 'results/triplets/triplet_full_significant.csv',
    'validation': root / 'results/validation/validated_triplets.csv',
    'summary': root / 'results/summary/final_results_table.csv',
}
for path in inputs.values():
    if not path.is_file():
        raise SystemExit(f'Missing {path}. Place this script in the repository root, or use --project-root PATH.')
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300})

def save(fig, name):
    fig.savefig(out / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(out / name)

prices = pd.read_csv(inputs['prices'], index_col=0, parse_dates=True)
screen = pd.read_csv(inputs['screen'])
validation = pd.read_csv(inputs['validation'])
summary = pd.read_csv(inputs['summary'])
accepted = validation[validation['Validated'].astype(str).str.lower().eq('true')]

fig, ax = plt.subplots(figsize=(9, 4.7), layout='constrained')
for stock in ['CIPLA.NS', 'SIEMENS.NS', 'HAL.NS']:
    normalized = prices[stock] / prices[stock].iloc[0] * 100
    ax.plot(prices.index, normalized, label=stock.replace('.NS', ''), linewidth=1.4)
ax.axvline(pd.Timestamp('2023-01-01'), color='black', linestyle='--', linewidth=1, label='2023 split')
ax.set(title='Normalized price paths for the leading candidate', xlabel='Date', ylabel='Price index (first observation = 100)')
ax.grid(alpha=.2); ax.legend(frameon=False)
save(fig, 'figure_1_normalized_prices.png')

fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis('off')
steps = [
    'Historical prices\nForward and backward filling',
    'Daily log returns\n25 stocks',
    'Full-universe regression screen\n27,600 model configurations',
    f'Initial candidates\n{len(screen)} configurations',
    'Validation checks\n252-day rolling windows | 1,000 shuffles | 2023 split',
    f'At least two of three checks passed\n{len(accepted)} accepted candidates',
    'Prototype strategy and combined summary\nTrading metrics remain preliminary',
]
ys = np.linspace(9.1, .9, len(steps))
for i, (y, label) in enumerate(zip(ys, steps)):
    ax.add_patch(FancyBboxPatch((.5, y-.42), 9, .84, boxstyle='round,pad=0.08',
                              facecolor='#f1f5f9', edgecolor='#334155'))
    ax.text(5, y, label, ha='center', va='center', fontsize=10.5)
    if i < len(steps)-1:
        ax.annotate('', xy=(5, ys[i+1]+.51), xytext=(5, y-.51),
                    arrowprops={'arrowstyle': '->', 'color': '#334155', 'lw': 1.3})
save(fig, 'figure_2_research_pipeline.png')

horizons = [1, 2, 3, 5]
a = screen.groupby('Horizon_k').size().reindex(horizons, fill_value=0)
b = accepted.groupby('Horizon_k').size().reindex(horizons, fill_value=0)
fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
x = np.arange(4)
for positions, counts, label, color in [(x-.19, a, 'Initial survivors', '#334155'), (x+.19, b, 'Accepted candidates', '#0d9488')]:
    bars = ax.bar(positions, counts, .38, label=label, color=color)
    ax.bar_label(bars, padding=3)
ax.set(xticks=x, xticklabels=horizons, xlabel='Prediction horizon (trading days)', ylabel='Number of configurations',
       title='Candidate counts by prediction horizon', ylim=(0, max(a.max(), b.max())*1.2))
ax.legend(frameon=False);ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
save(fig, 'figure_3_horizon_counts.png')

fig, ax = plt.subplots(figsize=(8, 5.1), layout='constrained')
colors = {1:'#2563eb', 2:'#0d9488', 3:'#d97706', 5:'#9333ea'}
for k, group in summary.groupby('Horizon_k'):
    ax.scatter(group['IS_R2'], group['OOS_R2'], label=f'k = {int(k)}', s=45,
               alpha=.85, color=colors[int(k)], edgecolor='white', linewidth=.5)
lo = min(0, summary.IS_R2.min(), summary.OOS_R2.min())-.002
hi = max(summary.IS_R2.max(), summary.OOS_R2.max())+.004
ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1, label='Equal fit')
ax.axhline(0, color='black', linewidth=.8)
best = summary.loc[summary.OOS_R2.idxmax()]
label = f"{best.Stock_A.replace('.NS','')} + {best.Stock_B.replace('.NS','')} → {best.Stock_C.replace('.NS','')}"
ax.annotate(label, (best.IS_R2, best.OOS_R2), xytext=(8, 12), textcoords='offset points', fontsize=9)
ax.set(xlim=(lo,hi), ylim=(lo,hi), xlabel='Training R²', ylabel='Holdout R²', title='Training and holdout fit for accepted candidates')
ax.legend(frameon=False, fontsize=9);ax.grid(alpha=.2)
fig.supxlabel('Candidate selection used the full sample; holdout results are selection-affected.', fontsize=9)
save(fig, 'figure_4_training_holdout_fit.png')

fig, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
rolling = validation.Rolling_Sig_Pct * 100
ax.hist(rolling, bins=np.arange(0, 66, 5), color='#334155', edgecolor='white')
ax.axvline(60, color='#b91c1c', linestyle='--', linewidth=1.8, label='Required stability: 60%')
ax.set(xlim=(0,65), xlabel='Windows with both slopes significant (%)', ylabel='Number of configurations',
       title='Rolling stability across all screened candidates')
ax.text(.98,.76, f'Maximum observed: {rolling.max():.1f}%\nNo candidate meets the criterion',
        transform=ax.transAxes, ha='right', va='top', fontsize=10)
ax.legend(frameon=False);ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
save(fig, 'figure_5_rolling_stability.png')
print('All five report figures generated.')
