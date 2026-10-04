"""
Summarize the results written by run_experiments.py (mean and standard deviation over seeds).

If several result folders are given (e.g. one per learning rate), every
experiment is taken from the folder in which its mean validation error over
seeds is lowest.
"""

import argparse
import numpy as np

from make_figures import load_runs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Summarize experiment results over seeds')
    parser.add_argument('--results_dir', type=str, nargs='+', default=['results/camera_ready'],
                       help='Directories with the result .json files')
    return parser.parse_args()


def mean_std(values):
    """Format mean and standard deviation over seeds."""
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"


def main():
    """Main summary function."""
    args = parse_args()

    # Runs grouped by experiment name, from the best folder on the validation set
    runs = dict(sorted(load_runs(args.results_dir).items()))

    print(f"{'Experiment':34s} {'Folder':26s} {'Seeds':>5s} {'Params':>10s} {'Mean (m)':>12s} "
          f"{'Median (m)':>12s} {'90th (m)':>12s}")
    print("-" * 117)
    for name, results in runs.items():
        print(f"{name:34s} {results[0]['dir']:26s} {len(results):5d} {results[0]['num_params']:10,d} "
              f"{mean_std([r['test_mean_error'] for r in results]):>12s} "
              f"{mean_std([r['test_median_error'] for r in results]):>12s} "
              f"{mean_std([r['test_90th_percentile'] for r in results]):>12s}")

    # Breakdowns of the mean test error
    for key, title in [('error_by_num_anchors', 'number of available APs'),
                       ('error_by_snr', 'SNR (dB)'),
                       ('error_by_num_los', 'number of line-of-sight APs'),
                       ('error_by_condition', 'test impairment condition')]:
        rows = {name: results for name, results in runs.items() if key in results[0]}
        if not rows:
            continue
        columns = list(next(iter(rows.values()))[0][key].keys())
        print(f"\nMean test error (m) by {title}")
        print(f"{'Experiment':34s} " + " ".join(f"{c:>14s}" for c in columns))
        print("-" * (35 + 15 * len(columns)))
        for name, results in rows.items():
            print(f"{name:34s} " + " ".join(
                f"{mean_std([r[key][c] for r in results if c in r[key]]):>14s}" for c in columns))


if __name__ == '__main__':
    main()
