"""
Run the full model comparison on the datasets written by phy/wifiPosGenerateData.m.

Usage (from the deep-loc folder):
    python scripts/run_all.py smoke     Quick test of every experiment (a few minutes)
    python scripts/run_all.py           Run everything, one seed per parallel job
    python scripts/run_all.py status    Show the progress of a run

The data folder and the number of CPU threads are detected automatically.
Experiments that already have a result file are skipped, so an interrupted
run can be restarted with the same command.
"""

import os
import sys
import glob
import time
import argparse
import platform
import threading
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, arguments of run_experiments.py), in order of priority
EXPERIMENTS = [
    # Main comparison: proposed model and CNN baseline, complex and magnitude input
    ('transformer_appos_complex', ['--model', 'transformer', '--use_anchor_position']),
    ('cnn_complex', ['--model', 'cnn']),
    ('transformer_appos_magnitude', ['--model', 'transformer', '--use_anchor_position', '--input', 'magnitude']),
    ('cnn_magnitude', ['--model', 'cnn', '--input', 'magnitude']),
    ('knn_complex', ['--model', 'knn']),
    ('knn_magnitude', ['--model', 'knn', '--input', 'magnitude']),

    # Ablations of the proposed model
    ('transformer_complex', ['--model', 'transformer']),
    ('deepsets_appos_complex', ['--model', 'transformer', '--use_anchor_position', '--num_layers', '0']),
    ('attnpool_appos_complex', ['--model', 'transformer', '--use_anchor_position', '--pooling', 'attention']),
    ('apdropout_appos_complex', ['--model', 'transformer', '--use_anchor_position', '--anchor_dropout', '0.25']),

    # Synchronized control: trained and tested without clock offset and phase noise
    ('transformer_appos_complex_sync', ['--model', 'transformer', '--use_anchor_position',
                                        '--train_condition', 'synchronized']),
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the full model comparison')
    parser.add_argument('mode', nargs='?', default='run', choices=['run', 'smoke', 'status'],
                       help='run (default), smoke (quick test) or status (show progress)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory with the data_*.mat files (default: detected automatically)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Seeds to run')
    parser.add_argument('--epochs', type=int, default=60, help='Maximum number of epochs per run')
    parser.add_argument('--threads', type=int, default=None,
                       help='CPU threads per parallel job (default: physical cores / number of seeds)')
    return parser.parse_args()


def find_data_dir():
    """Find the folder with the generated datasets."""
    for candidate in [ROOT, os.path.join(os.path.dirname(ROOT), 'phy')]:
        if os.path.isfile(os.path.join(candidate, 'data_train_nominal.mat')):
            return candidate
    sys.exit("Could not find data_train_nominal.mat in deep-loc or phy. "
             "Run phy/wifiPosGenerateData.m first, or pass --data_dir.")


def num_physical_cores():
    """Number of physical CPU cores."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count()
    except ImportError:
        # Without psutil, assume two logical cores per physical core except on Arm
        logical = os.cpu_count() or 1
        return logical if platform.machine().lower() in ('arm64', 'aarch64') else max(1, logical // 2)


def result_path(output_dir, name, seed):
    """Path of the result file of one experiment."""
    return os.path.join(output_dir, f'{name}_seed{seed}.json')


def log_path(output_dir, name, seed):
    """Path of the log file of one experiment."""
    return os.path.join(output_dir, 'logs', f'{name}_seed{seed}.log')


def last_line(path):
    """Last non-empty line of a text file."""
    try:
        with open(path, errors='replace') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines[-1] if lines else ''
    except OSError:
        return ''


def run_experiment(name, seed, extra_args, common_args, output_dir):
    """Run one experiment unless its result exists. Returns True on success."""
    if os.path.isfile(result_path(output_dir, name, seed)):
        return True
    command = [sys.executable, os.path.join(ROOT, 'scripts', 'run_experiments.py'),
               '--name', name, '--seed', str(seed), '--output_dir', output_dir] + extra_args + common_args
    with open(log_path(output_dir, name, seed), 'w') as log:
        code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return code == 0 and os.path.isfile(result_path(output_dir, name, seed))


def print_status(output_dir, seeds):
    """Print which experiments are done and what the unfinished ones are doing."""
    num_done = 0
    for name, _ in EXPERIMENTS:
        for seed in seeds:
            if os.path.isfile(result_path(output_dir, name, seed)):
                num_done += 1
                state = 'done'
            elif os.path.isfile(log_path(output_dir, name, seed)):
                state = last_line(log_path(output_dir, name, seed))[:90]
            else:
                state = 'waiting'
            print(f"  {name:32s} seed {seed}: {state}")
    print(f"{num_done} of {len(EXPERIMENTS) * len(seeds)} experiments done.")


def summarize(output_dir):
    """Print the summary table of all finished experiments."""
    subprocess.call([sys.executable, os.path.join(ROOT, 'scripts', 'summarize_results.py'),
                     '--results_dir', output_dir], cwd=ROOT)


def main():
    """Main function."""
    args = parse_args()
    smoke = args.mode == 'smoke'
    output_dir = os.path.join(ROOT, 'results', 'smoke_test' if smoke else 'camera_ready')
    seeds = [0] if smoke else args.seeds

    if args.mode == 'status':
        print_status(output_dir, seeds)
        return

    data_dir = args.data_dir if args.data_dir is not None else find_data_dir()
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    if smoke:
        # Always repeat the quick test from scratch
        for path in glob.glob(os.path.join(output_dir, '*.json')):
            os.remove(path)

    num_jobs = len(seeds)
    threads = args.threads if args.threads is not None else max(1, num_physical_cores() // num_jobs)
    common_args = ['--data_dir', data_dir, '--epochs', str(args.epochs),
                   '--warmup_epochs', '5', '--patience', '15', '--threads', str(threads)]
    if smoke:
        common_args.append('--smoke')

    print(f"Data folder:   {data_dir}")
    print(f"Results:       {output_dir}")
    print(f"Parallel jobs: {num_jobs} (one per seed), {threads} CPU threads each")
    print(f"Experiments:   {len(EXPERIMENTS) * len(seeds)}")

    # One worker per seed runs its experiments one after the other
    failed = []

    def worker(seed):
        for name, extra_args in EXPERIMENTS:
            if not run_experiment(name, seed, extra_args, common_args, output_dir):
                failed.append((name, seed))

    workers = [threading.Thread(target=worker, args=(seed,), daemon=True) for seed in seeds]
    start = time.time()
    for w in workers:
        w.start()
    while any(w.is_alive() for w in workers):
        time.sleep(5 if smoke else 300)
        num_done = len(glob.glob(os.path.join(output_dir, '*.json')))
        print(f"[{(time.time() - start) / 60:6.1f} min] {num_done} of "
              f"{len(EXPERIMENTS) * len(seeds)} experiments done", flush=True)

    print()
    if failed:
        print("FAILED experiments:")
        for name, seed in failed:
            print(f"  {name} seed {seed}: {last_line(log_path(output_dir, name, seed))}")
            print(f"    full log: {log_path(output_dir, name, seed)}")
        sys.exit(1)

    if smoke:
        print("SMOKE TEST PASSED. Start the full run with: python scripts/run_all.py")
    else:
        summarize(output_dir)
        print(f"\nALL DONE. Send back the folder {output_dir}")


if __name__ == '__main__':
    main()
