"""
Run the full model comparison on the datasets written by phy/wifiPosGenerateData.m.

Usage (from the deep-loc folder):
    python scripts/run_all.py smoke     Quick test of every experiment (a few minutes)
    python scripts/run_all.py           Run everything, several experiments in parallel
    python scripts/run_all.py status    Show the progress of a run

Add "--study layouts" to run the multi-layout study on the datasets written by
phy/wifiPosGenerateDataLayouts.m instead.

The data folder, the device (GPU if available, otherwise CPU) and the number
of parallel jobs and CPU threads are detected automatically.
Experiments that already have a result file are skipped, so an interrupted
run can be restarted with the same command.
"""

import os
import sys
import glob
import time
import queue
import argparse
import platform
import threading
import subprocess

import torch

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

    # Additional baselines: deeper residual CNN, order-dependent late fusion
    # (shared encoder, embeddings concatenated in the fixed AP order), and the
    # CNN trained with the same AP dropout as the proposed model
    ('resnet_complex', ['--model', 'resnet']),
    ('concat_complex', ['--model', 'transformer', '--num_layers', '0', '--pooling', 'concat']),
    ('cnn_apdropout_complex', ['--model', 'cnn', '--anchor_dropout', '0.25']),
    ('concat_apdropout_complex', ['--model', 'transformer', '--num_layers', '0', '--pooling', 'concat',
                                  '--anchor_dropout', '0.25']),

    # Receiver reference: uniformly random common phase per AP, and additionally a
    # random sub-sample timing offset per AP, in training and in testing
    ('transformer_appos_complex_randphase', ['--model', 'transformer', '--use_anchor_position',
                                             '--random_phase']),
    ('transformer_appos_complex_randphase_delay', ['--model', 'transformer', '--use_anchor_position',
                                                   '--random_phase', '--random_delay']),
    ('concat_complex_randphase_delay', ['--model', 'transformer', '--num_layers', '0', '--pooling', 'concat',
                                        '--random_phase', '--random_delay']),
    ('resnet_complex_randphase_delay', ['--model', 'resnet', '--random_phase', '--random_delay']),
    ('cnn_complex_randphase_delay', ['--model', 'cnn', '--random_phase', '--random_delay']),

    # Synchronized control: trained and tested without clock offset and phase noise
    ('transformer_appos_complex_sync', ['--model', 'transformer', '--use_anchor_position',
                                        '--train_condition', 'synchronized']),
]


# Experiments of the multi-layout study (run_layout_experiments.py): every model
# with and without the AP coordinates as input
LAYOUT_EXPERIMENTS = [
    ('transformer_appos', ['--model', 'transformer', '--use_anchor_position']),
    ('transformer_nopos', ['--model', 'transformer']),
    ('concat_appos', ['--model', 'transformer', '--num_layers', '0', '--pooling', 'concat',
                      '--use_anchor_position']),
    ('concat_nopos', ['--model', 'transformer', '--num_layers', '0', '--pooling', 'concat']),
    ('resnet_appos', ['--model', 'resnet', '--use_anchor_position']),
    ('resnet_appos_input', ['--model', 'resnet', '--use_anchor_position', '--position_at_input']),
    ('resnet_nopos', ['--model', 'resnet']),
    ('deepsets_appos', ['--model', 'transformer', '--num_layers', '0', '--use_anchor_position']),
    ('attnpool_appos', ['--model', 'transformer', '--pooling', 'attention', '--use_anchor_position']),
    ('cnn_appos', ['--model', 'cnn', '--use_anchor_position']),
    ('cnn_nopos', ['--model', 'cnn']),
    ('knn', ['--model', 'knn']),
]

# Named groups of experiments for --only
GROUPS = {
    'reference': ['transformer_appos_complex_randphase', 'transformer_appos_complex_randphase_delay',
                  'concat_complex_randphase_delay', 'resnet_complex_randphase_delay',
                  'cnn_complex_randphase_delay'],
}


# Script that runs one experiment (set by the study)
RUNNER = 'run_experiments.py'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the full model comparison')
    parser.add_argument('mode', nargs='?', default='run', choices=['run', 'smoke', 'status'],
                       help='run (default), smoke (quick test) or status (show progress)')
    parser.add_argument('--study', type=str, default='fixed', choices=['fixed', 'layouts'],
                       help='fixed: one AP layout (wifiPosGenerateData.m); '
                            'layouts: many AP layouts (wifiPosGenerateDataLayouts.m)')
    parser.add_argument('--data_prefix', type=str, default='',
                       help='Prefix of the data files of the layouts study ("smoke_" for the MATLAB smoke test)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory with the data_*.mat files (default: detected automatically)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='Seeds to run')
    parser.add_argument('--epochs', type=int, default=60, help='Maximum number of epochs per run')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience in epochs (set to the number of epochs to disable)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (default: 1e-3)')
    parser.add_argument('--only', type=str, nargs='+', default=None,
                       help='Run only the experiments with these names (or the group "reference")')
    parser.add_argument('--random_reference', action='store_true',
                       help='Train and test every experiment with a random common phase and a random '
                            'sub-sample delay per AP')
    parser.add_argument('--redo_early_stopped', action='store_true',
                       help='Repeat the finished experiments that stopped before the last epoch')
    parser.add_argument('--results_name', type=str, default=None,
                       help='Name of the results folder under results/ '
                            '(default: camera_ready, or layouts for the layouts study)')
    parser.add_argument('--jobs', type=int, default=None,
                       help='Experiments to run in parallel (default: 3)')
    parser.add_argument('--threads', type=int, default=None,
                       help='CPU threads per parallel job (default: physical cores / number of jobs)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda if available, otherwise cpu)')
    return parser.parse_args()


def find_data_dir(file_name, generator):
    """Find the folder with the generated datasets."""
    for candidate in [ROOT, os.path.join(os.path.dirname(ROOT), 'phy')]:
        if os.path.isfile(os.path.join(candidate, file_name)):
            return candidate
    sys.exit(f"Could not find {file_name} in deep-loc or phy. "
             f"Run phy/{generator} first, or pass --data_dir.")


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
    command = [sys.executable, os.path.join(ROOT, 'scripts', RUNNER),
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
    script = 'summarize_layouts.py' if RUNNER == 'run_layout_experiments.py' else 'summarize_results.py'
    subprocess.call([sys.executable, os.path.join(ROOT, 'scripts', script),
                     '--results_dir', output_dir], cwd=ROOT)


def main():
    """Main function."""
    global RUNNER
    args = parse_args()
    layouts = args.study == 'layouts'
    if layouts:
        RUNNER = 'run_layout_experiments.py'
        EXPERIMENTS[:] = LAYOUT_EXPERIMENTS
    if args.results_name is None:
        args.results_name = 'layouts' if layouts else 'camera_ready'
    if args.only is not None:
        args.only = [name for item in args.only for name in GROUPS.get(item, [item])]
        unknown = set(args.only) - {name for name, _ in EXPERIMENTS}
        if unknown:
            sys.exit(f"Unknown experiments: {sorted(unknown)}")
        EXPERIMENTS[:] = [e for e in EXPERIMENTS if e[0] in args.only]
    if args.random_reference:
        # The random reference applies to every experiment, so the dedicated ones are redundant
        EXPERIMENTS[:] = [e for e in EXPERIMENTS if 'randphase' not in e[0]]
    smoke = args.mode == 'smoke'
    output_dir = os.path.join(ROOT, 'results', 'smoke_test' if smoke else args.results_name)
    seeds = [0] if smoke else args.seeds

    if args.mode == 'status':
        print_status(output_dir, seeds)
        return

    if args.data_dir is not None:
        data_dir = args.data_dir
    elif layouts:
        data_dir = find_data_dir(args.data_prefix + 'data_layouts_train.mat', 'wifiPosGenerateDataLayouts.m')
    else:
        data_dir = find_data_dir('data_train_nominal.mat', 'wifiPosGenerateData.m')
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
    if smoke:
        # Always repeat the quick test from scratch
        for path in glob.glob(os.path.join(output_dir, '*.json')):
            os.remove(path)

    device = args.device if args.device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'
    num_jobs = args.jobs if args.jobs is not None else 3
    threads = args.threads if args.threads is not None else max(1, num_physical_cores() // num_jobs)
    common_args = ['--data_dir', data_dir, '--epochs', str(args.epochs),
                   '--warmup_epochs', '5', '--patience', str(args.patience), '--threads', str(threads),
                   '--device', device]
    if args.lr is not None:
        common_args += ['--lr', str(args.lr)]
    if layouts:
        # The layouts study always uses the random receiver reference
        common_args += ['--data_prefix', args.data_prefix]
    elif args.random_reference:
        common_args += ['--random_phase', '--random_delay']
    if smoke:
        common_args.append('--smoke')

    if args.redo_early_stopped:
        # Remove the results of the runs whose log has fewer epochs than requested
        for path in glob.glob(os.path.join(output_dir, '*.json')):
            log = os.path.join(output_dir, 'logs', os.path.basename(path)[:-len('.json')] + '.log')
            with open(log, errors='replace') as f:
                num_epochs = sum(line.startswith('Epoch') for line in f)
            if 0 < num_epochs < args.epochs:
                print(f"Repeating {os.path.basename(path)} (stopped after {num_epochs} epochs)")
                os.remove(path)

    print(f"Data folder:   {data_dir}")
    print(f"Results:       {output_dir}")
    print(f"Device:        {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ''))
    print(f"Parallel jobs: {num_jobs}, {threads} CPU threads each")
    print(f"Experiments:   {len(EXPERIMENTS) * len(seeds)}")

    # Every worker takes the next experiment from the queue (in order of priority)
    tasks = queue.Queue()
    for name, extra_args in EXPERIMENTS:
        for seed in seeds:
            tasks.put((name, seed, extra_args))
    failed = []

    def worker():
        while True:
            try:
                name, seed, extra_args = tasks.get_nowait()
            except queue.Empty:
                return
            if not run_experiment(name, seed, extra_args, common_args, output_dir):
                failed.append((name, seed))

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(num_jobs)]
    start = time.time()
    for w in workers:
        w.start()
    while any(w.is_alive() for w in workers):
        time.sleep(5 if smoke else 60 if device == 'cuda' else 300)
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
