"""Plot a saved, immutable measurements snapshot while training continues."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

folder = Path(__file__).parent / 'results/random-cnn/cifar10'
rows = json.loads((folder / 'test-accuracy-snapshot.json').read_text(encoding='utf-8-sig'))
rows = [r for r in rows if r['seed'] == 409]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
primary = sorted([r for r in rows if r['experiment'] == 'primary'], key=lambda r: r['total_parameters'])
axes[0].plot([r['total_parameters'] for r in primary], [100*r['test']['accuracy'] for r in primary], '-o', color='#1768ac', lw=2)
for r in primary:
    axes[0].annotate(f"w={r['width']}", (r['total_parameters'], 100*r['test']['accuracy']), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8)
baseline = next(r for r in rows if r['experiment'] == 'baseline')
for ax in axes:
    ax.axhline(100*baseline['test']['accuracy'], ls='--', color='#777777', lw=1.2, label=f"Root-only baseline: {100*baseline['test']['accuracy']:.2f}%")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x/1e6:g}M' if x >= 1e6 else f'{x/1e3:g}k'))
    ax.set_xlabel('Total deployed parameters (log scale; random CNN counted once)')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_ylim(44, 56)
    ax.grid(alpha=.2)
axes[0].set_title('Fixed depth 1: vary correction-node width')
axes[0].legend(loc='lower left', fontsize=8)
deep = [r for r in rows if r['experiment'] == 'deeper' and not r.get('reused_saturated_model')]
for width in sorted({r['width'] for r in deep}):
    group = sorted([r for r in deep if r['width'] == width], key=lambda r: r['depth_limit'])
    axes[1].plot([r['total_parameters'] for r in group], [100*r['test']['accuracy'] for r in group], '-o', lw=2, label=f'Width {width}')
    for r in group:
        axes[1].annotate(f"d={r['depth_limit']}", (r['total_parameters'], 100*r['test']['accuracy']), xytext=(5, 7), textcoords='offset points', fontsize=8)
axes[1].set_title('Deeper trees: vary depth within each width')
axes[1].legend(loc='lower left', fontsize=8)
fig.suptitle('CIFAR-10 | Frozen random CNN + correction trees\nCompleted configurations, seed 409; exploratory test results', fontsize=13)
fig.savefig(folder / 'parameters-vs-test-accuracy.png', dpi=180)
fig.savefig(folder / 'parameters-vs-test-accuracy.pdf')
print(f'Saved graph: {len(primary)} primary points, {len(deep)} deeper points.')
