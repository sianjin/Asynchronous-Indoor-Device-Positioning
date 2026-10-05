"""
Make the result figures and the result table of the paper from the files
written by run_experiments.py.

If several result folders are given (e.g. one per learning rate), every
experiment is taken from the folder in which its mean validation error over
seeds is lowest.
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Categorical colors in fixed order (validated for color-vision deficiency);
# every series also has its own line style, so identity never relies on color alone
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'
INK, MUTED, GRID = '#0b0b0b', '#898781', '#dddcd8'

plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
    'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelcolor': INK, 'ytick.labelcolor': INK, 'axes.labelcolor': INK,
    'grid.color': GRID, 'grid.linewidth': 0.5, 'lines.linewidth': 1.4,
    'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02
})
COLUMN_WIDTH = 3.45  # inches, one IEEE column


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make the figures and the table of the paper')
    parser.add_argument('--results_dirs', type=str, nargs='+',
                       default=[os.path.join(ROOT, 'results', 'camera_ready_200ep_randref'),
                                os.path.join(ROOT, 'results', 'camera_ready_200ep_randref_lr3e-4')],
                       help='Result folders; the best one on the validation set is used per experiment')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(os.path.dirname(ROOT), 'paper', 'figures'),
                       help='Directory to save the figures')
    parser.add_argument('--table_path', type=str,
                       default=os.path.join(ROOT, 'results', 'results_table.tex'),
                       help='Path of the LaTeX result table (its content is pasted into the paper)')
    parser.add_argument('--map_path', type=str,
                       default=os.path.join(os.path.dirname(ROOT), 'phy', 'office.stl'),
                       help='Path of the office map')
    parser.add_argument('--data_path', type=str, default=os.path.join(ROOT, 'data_test_nominal.mat'),
                       help='Path of the nominal test set (for the AP positions)')
    return parser.parse_args()


def load_runs(results_dirs):
    """
    Load all runs and select the result folder of every experiment.

    Returns:
        dict: experiment name -> list of runs (result dict with 'arrays' and 'dir'),
            sorted by seed, from the folder with the lowest mean validation error
    """
    candidates = {}
    for results_dir in results_dirs:
        for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
            with open(path) as f:
                run = json.load(f)
            run['arrays'] = path[:-len('.json')] + '.npz'
            run['dir'] = os.path.basename(os.path.normpath(results_dir))
            candidates.setdefault(run['name'], {}).setdefault(results_dir, []).append(run)

    runs = {}
    for name, by_dir in candidates.items():
        best_dir = min(by_dir, key=lambda d: np.mean([r['val_error'] for r in by_dir[d]]))
        runs[name] = sorted(by_dir[best_dir], key=lambda r: r['seed'])
    return runs


def stat(runs, key, sub=None):
    """Mean and standard deviation over seeds of a result entry."""
    values = [r[key] if sub is None else r[key][sub] for r in runs]
    return np.mean(values), np.std(values)


def read_stl_walls(path):
    """
    Read a binary STL file and return the floor-plan line segments.

    Only the vertical faces (walls, partitions, furniture sides) are kept; seen
    from above each of them is a line segment.
    """
    with open(path, 'rb') as f:
        f.read(80)
        num_triangles = np.frombuffer(f.read(4), dtype='<u4')[0]
        records = np.frombuffer(f.read(50 * num_triangles), dtype=np.dtype(
            [('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]))
    vertices = records['vertices']
    normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    vertical = np.abs(normal[:, 2]) < 0.05

    segments = []
    for triangle in vertices[vertical]:
        xy = triangle[:, :2]
        # A vertical triangle projects to a segment between its two extreme points
        d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        i, j = np.unravel_index(d.argmax(), d.shape)
        segments.append((xy[i], xy[j]))
    return segments


def plot_cdf(runs, path):
    """CDF of the positioning error, pooled over seeds."""
    series = [
        ('transformer_appos_complex', 'Proposed', BLUE, '-'),
        ('concat_complex', 'Fixed-order fusion', ORANGE, (0, (5, 1.5))),
        ('resnet_complex', 'ResNet', AQUA, (0, (3, 1, 1, 1))),
        ('cnn_complex', 'CNN', YELLOW, (0, (1, 1))),
        ('transformer_appos_magnitude', 'Proposed, magnitude', MAGENTA, (0, (6, 1.5, 1, 1.5, 1, 1.5))),
        ('knn_complex', 'kNN', GREEN, (0, (2, 2))),
    ]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.25))
    for name, label, color, style in series:
        if name not in runs:
            continue
        errors = np.sort(np.concatenate([np.load(r['arrays'])['errors'] for r in runs[name]]))
        mean, _ = stat(runs[name], 'test_mean_error')
        ax.plot(errors, np.arange(1, len(errors) + 1) / len(errors), color=color, linestyle=style,
                label=f'{label} ({mean:.2f} m)')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Distance error (m)')
    ax.set_ylabel('Cumulative probability')
    ax.grid(True)
    ax.legend(loc='lower right', handlelength=3.2, title='Method (mean error)', title_fontsize=7,
              borderaxespad=0.2, labelspacing=0.3)
    fig.savefig(path)
    plt.close(fig)


def plot_ap_count(runs, path):
    """Mean error versus the number of available APs, with and without AP dropout."""
    series = [
        ('transformer_appos_complex', 'Proposed', BLUE, '-', 'o'),
        ('apdropout_appos_complex', 'Proposed + AP dropout', BLUE, (0, (4, 1.5)), 'o'),
        ('concat_complex', 'Fixed-order fusion', ORANGE, '-', 's'),
        ('concat_apdropout_complex', 'Fixed-order fusion + AP dropout', ORANGE, (0, (4, 1.5)), 's'),
        ('cnn_complex', 'CNN', AQUA, '-', '^'),
        ('cnn_apdropout_complex', 'CNN + AP dropout', AQUA, (0, (4, 1.5)), '^'),
    ]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.25))
    for name, label, color, style, marker in series:
        if name not in runs:
            continue
        counts = sorted(runs[name][0]['error_by_num_anchors'], key=int)
        mean, std = np.array([stat(runs[name], 'error_by_num_anchors', c) for c in counts]).T
        dropout = 'dropout' in name
        ax.errorbar([int(c) for c in counts], mean, yerr=std, color=color, linestyle=style, marker=marker,
                    markersize=4, markerfacecolor='white' if dropout else color, markeredgewidth=1,
                    capsize=2, elinewidth=0.8, label=label)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0, None)
    ax.set_xlabel('Number of available APs')
    ax.set_ylabel('Mean distance error (m)')
    ax.grid(True)
    ax.legend(loc='upper right', handlelength=3.2)
    fig.savefig(path)
    plt.close(fig)


def plot_error_map(runs, map_path, data_path, path, name='transformer_appos_complex'):
    """Top view of the office with the test positions colored by the mean error."""
    arrays = [np.load(r['arrays']) for r in runs[name]]
    positions = arrays[0]['ground_truth_positions']
    errors = np.mean([a['errors'] for a in arrays], axis=0)  # mean over seeds

    # Mean over the SNRs of every test position
    unique, inverse = np.unique(np.round(positions, 3), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    error = np.bincount(inverse, weights=errors) / np.bincount(inverse)

    # One hue, light to dark, for magnitude
    cmap = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#5598e7', '#184f95', '#0d366b'])
    vmax = 1.0

    # The long side of the room (y) is drawn horizontally
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.1))
    for a, b in read_stl_walls(map_path):
        ax.plot([a[1], b[1]], [a[0], b[0]], color=MUTED, linewidth=0.5, zorder=1)
    order = np.argsort(error)
    points = ax.scatter(unique[order, 1], unique[order, 0], c=np.minimum(error[order], vmax), cmap=cmap,
                        vmin=0, vmax=vmax, s=9, edgecolors='white', linewidths=0.25, zorder=2)
    ap = scipy.io.loadmat(data_path)['apPositions']
    ax.scatter(ap[1], ap[0], marker='^', s=42, color=ORANGE, edgecolors=INK, linewidths=0.6, zorder=3,
               label='AP')
    ax.set_aspect('equal')
    ax.set_xlim(-0.15, 8.15)
    ax.set_ylim(-0.15, 5.15)
    ax.set_xlabel('y (m)')
    ax.set_ylabel('x (m)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.72, 1.16), handletextpad=0.2)
    # Extent of the conference room along the long side of the office
    ax.annotate('', xy=(0, 5.42), xytext=(2.75, 5.42), xycoords='data', annotation_clip=False,
                arrowprops=dict(arrowstyle='|-|,widthA=0.25,widthB=0.25', color=INK, linewidth=0.7))
    ax.text(1.375, 5.62, 'conference room', ha='center', va='bottom', fontsize=7, color=INK)
    colorbar = fig.colorbar(points, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    colorbar.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', r'$\geq$1'])
    colorbar.set_label('Mean distance error (m)')
    colorbar.outline.set_linewidth(0.4)
    fig.savefig(path)
    plt.close(fig)


def write_table(runs, path):
    """Write the LaTeX result table (mean and standard deviation over seeds)."""
    rows = [
        ('Complex-valued input', None),
        ('Proposed', 'transformer_appos_complex'),
        ('\\quad w/o anchor position', 'transformer_complex'),
        ('\\quad set pooling', 'deepsets_appos_complex'),
        ('\\quad attention pooling', 'attnpool_appos_complex'),
        ('Fixed-order fusion', 'concat_complex'),
        ('CNN~\\cite{MathWorks80211azDeepLearning}', 'cnn_complex'),
        ('ResNet', 'resnet_complex'),
        ('$k$NN', 'knn_complex'),
        ('Magnitude-only input', None),
        ('Proposed', 'transformer_appos_magnitude'),
        ('CNN~\\cite{MathWorks80211azDeepLearning}', 'cnn_magnitude'),
        ('$k$NN', 'knn_magnitude'),
    ]
    lines = ['\\begin{tabular}{lcccc}', '\\toprule',
             'Method & Param. & Mean & Median & 90th pct. \\\\', '\\midrule']
    for label, name in rows:
        if name is None:
            if len(lines) > 4:
                lines.append('\\midrule')
            lines.append(f'\\multicolumn{{5}}{{l}}{{\\emph{{{label}}}}} \\\\')
            continue
        if name not in runs:
            continue
        params = runs[name][0]['num_params']
        cells = ['--' if params == 0 else f'{params / 1e6:.2f}\\,M']
        for key in ['test_mean_error', 'test_median_error', 'test_90th_percentile']:
            mean, std = stat(runs[name], key)
            cells.append(f'${mean:.2f} \\pm {std:.2f}$')
        lines.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_condition_table(runs, path):
    """Write the LaTeX table of the test error under the impairment conditions."""
    conditions = [('synchronized', 'None'), ('clockOnly', 'Clock'), ('phaseOnly', 'Phase'),
                  ('nominal', 'Nominal'), ('severe', 'Severe')]
    rows = [
        ('Proposed', 'transformer_appos_complex'),
        ('Proposed, clean training', 'transformer_appos_complex_sync'),
        ('Fixed-order fusion', 'concat_complex'),
        ('ResNet', 'resnet_complex'),
        ('CNN', 'cnn_complex'),
    ]
    lines = ['\\begin{tabular}{l' + 'c' * len(conditions) + '}', '\\toprule',
             'Method & ' + ' & '.join(label for _, label in conditions) + ' \\\\', '\\midrule']
    for label, name in rows:
        if name not in runs:
            continue
        cells = [f"{stat(runs[name], 'error_by_condition', c)[0]:.2f}" for c, _ in conditions]
        lines.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    """Main function."""
    args = parse_args()
    runs = load_runs(args.results_dirs)
    if not runs:
        sys.exit(f"No results found in {args.results_dirs}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Result folder and number of seeds used per experiment:")
    for name, r in sorted(runs.items()):
        print(f"  {name:34s} {r[0]['dir']:28s} {len(r)} seeds, "
              f"validation {stat(r, 'val_error')[0]:.3f} m, test {stat(r, 'test_mean_error')[0]:.3f} m")

    plot_cdf(runs, os.path.join(args.output_dir, 'positioningError.pdf'))
    plot_ap_count(runs, os.path.join(args.output_dir, 'errorVersusAPs.pdf'))
    plot_error_map(runs, args.map_path, args.data_path, os.path.join(args.output_dir, 'errorMap.pdf'))
    write_table(runs, args.table_path)
    write_condition_table(runs, os.path.join(os.path.dirname(args.table_path), 'condition_table.tex'))
    print(f"Figures saved in {args.output_dir}, table saved as {args.table_path}")


if __name__ == '__main__':
    main()
