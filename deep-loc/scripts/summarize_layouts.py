"""
Summarize the results of the multi-layout study written by run_layout_experiments.py
(mean and standard deviation over seeds).

If several result folders are given (e.g. one per learning rate), every
experiment is taken from the folder in which its mean validation error over
seeds is lowest.
"""

import argparse
import numpy as np

from make_figures import load_runs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Summarize the multi-layout study over seeds')
    parser.add_argument('--results_dir', type=str, nargs='+', default=['results/layouts'],
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

    print(f"{'Experiment':22s} {'Folder':24s} {'Seeds':>5s} {'Params':>10s} {'Validation':>12s} "
          f"{'Unseen layouts':>15s} {'Seen layouts':>13s}   (mean error, m)")
    print("-" * 122)
    for name, results in runs.items():
        print(f"{name:22s} {results[0]['dir']:24s} {len(results):5d} {results[0]['num_params']:10,d} "
              f"{mean_std([r['val_error'] for r in results]):>12s} "
              f"{mean_std([r['test_unseen']['mean_error'] for r in results]):>15s} "
              f"{mean_std([r['test_seen']['mean_error'] for r in results]):>13s}")

    for test_set, title in [('test_unseen', 'unseen layouts'), ('test_seen', 'seen layouts')]:
        for key, column_title in [('median_error', None), ('error_by_num_aps', 'number of APs of the layout'),
                                  ('error_by_num_detected', 'number of detected APs'),
                                  ('error_by_snr', 'SNR (dB)')]:
            if column_title is None:
                continue
            columns = sorted({c for results in runs.values() for r in results for c in r[test_set][key]}, key=int)
            print(f"\nMean test error (m) on {title} by {column_title}")
            print(f"{'Experiment':22s} " + " ".join(f"{c:>14s}" for c in columns))
            print("-" * (23 + 15 * len(columns)))
            for name, results in runs.items():
                print(f"{name:22s} " + " ".join(
                    f"{mean_std([r[test_set][key][c] for r in results if c in r[test_set][key]]):>14s}"
                    if any(c in r[test_set][key] for r in results) else f"{'-':>14s}" for c in columns))


if __name__ == '__main__':
    main()
