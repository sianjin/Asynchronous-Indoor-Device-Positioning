"""
Summarize the results written by run_experiments.py (mean and standard deviation over seeds).
"""

import os
import json
import glob
import argparse
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Summarize experiment results over seeds')
    parser.add_argument('--results_dir', type=str, default='results/camera_ready_v2',
                       help='Directory with the result .json files')
    return parser.parse_args()


def mean_std(values):
    """Format mean and standard deviation over seeds."""
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"


def main():
    """Main summary function."""
    args = parse_args()

    # Group runs by experiment name
    runs = {}
    for path in sorted(glob.glob(os.path.join(args.results_dir, '*.json'))):
        with open(path) as f:
            result = json.load(f)
        runs.setdefault(result['name'], []).append(result)

    print(f"{'Experiment':34s} {'Seeds':>5s} {'Params':>10s} {'Mean (m)':>12s} "
          f"{'Median (m)':>12s} {'90th (m)':>12s}")
    print("-" * 90)
    for name, results in runs.items():
        print(f"{name:34s} {len(results):5d} {results[0]['num_params']:10,d} "
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
