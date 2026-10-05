# Asynchronous Indoor Device Positioning from Complex Wi-Fi Channels via Transformer Fusion

Code for the paper "Asynchronous Indoor Device Positioning from Complex Wi-Fi Channels via Transformer Fusion" (ISAC 2026).

A device estimates its 3D position from the complex channel impulse responses (CIRs) of IEEE 802.11az ranging packets sent by several access points (APs), without clock synchronization between the APs and the device.

- Each AP's CIR is encoded by a shared CNN and tagged with an embedding of the AP position.
- The AP tokens are fused by a Transformer without positional encoding, so the estimate does not depend on the order in which the APs are listed.
- APs whose packet is not detected are masked.

## Repository layout

```
phy/        MATLAB: ray tracing, 802.11az waveform, hardware impairments, dataset generation
deep-loc/   Python (PyTorch): models, training, evaluation, figures
```

## Requirements

- **Data generation:** MATLAB with WLAN Toolbox and the products it requires. Parallel Computing Toolbox is optional and speeds up the generation.
- **Training and evaluation:** Python 3 with the packages in `deep-loc/requirements.txt`. A GPU is strongly recommended: one 200-epoch run takes a few minutes on a GPU and several hours on a CPU.

## 1. Generate the datasets (MATLAB)

Run `phy/wifiPosGenerateData.m` from the `phy` folder. It ray-traces a 5 m × 8 m office (`office.stl`) with four APs in the room corners, simulates the packets and writes seven files:

| File | Content |
|---|---|
| `data_train_nominal.mat` | 3000 random device positions × 3 SNRs, nominal impairments |
| `data_train_synchronized.mat` | Same positions, no clock offset and no phase noise |
| `data_test_<condition>.mat` | 500 separately drawn positions × 3 SNRs, for the conditions `nominal`, `synchronized`, `clockOnly`, `phaseOnly` and `severe` |

No test position is used for training. Impairments (clock offset, CFO, phase noise, Doppler) and noise are drawn independently for every packet. The ray-tracing result is cached and finished files are skipped, so the script can be restarted.

The data files are not part of the repository because of their size.

To view the scenario (office, the four APs, one example device and the rays between them) in Site Viewer, run `phy/wifiPosShowScenario.m`.

## 2. Run the experiments (Python)

From the `deep-loc` folder:

```bash
pip install -r requirements.txt

python3 scripts/run_all.py smoke     # quick test of every experiment (a few minutes)
python3 scripts/run_all.py           # full run
python3 scripts/run_all.py status    # progress of a run
```

The script finds the data in `deep-loc` or `phy`, uses a GPU if one is available, runs several experiments in parallel and skips experiments that are already finished.

The results in the paper use 200 epochs without early stopping, five seeds, two learning rates and a random receiver reference:

```bash
python3 scripts/run_all.py --epochs 200 --patience 200 --seeds 0 1 2 3 4 --results_name camera_ready_200ep_randref --random_reference
python3 scripts/run_all.py --epochs 200 --patience 200 --seeds 0 1 2 3 4 --results_name camera_ready_200ep_randref_lr3e-4 --lr 3e-4 --random_reference
```

In the simulated channel estimates the first path lies on the receiver sampling grid and the carrier phase offset is zero at the packet start. `--random_reference` removes both idealizations: the CIR of every AP is delayed by a random fraction of a sample and rotated by a random common phase, with a new draw every time a training sample is used and one fixed draw for the validation and test samples.

Every model is trained on the training positions, the checkpoint with the lowest validation error is kept (15% of the training positions are held out for validation), and it is evaluated on the test positions. For every experiment, the learning rate with the lower validation error is reported.

### Experiments

| Name | Model |
|---|---|
| `transformer_appos_complex` | Proposed model |
| `transformer_complex` | Proposed model without the AP-position embedding |
| `deepsets_appos_complex` | Proposed model without the Transformer (mean pooling of the AP tokens) |
| `attnpool_appos_complex` | Proposed model with attention pooling instead of mean pooling |
| `apdropout_appos_complex` | Proposed model trained with AP dropout |
| `concat_complex` | Fixed-order fusion: shared encoder, AP embeddings concatenated in a fixed order |
| `concat_apdropout_complex` | Fixed-order fusion trained with AP dropout |
| `cnn_complex` | Early-fusion CNN: APs stacked as input channels |
| `cnn_apdropout_complex` | Early-fusion CNN trained with AP dropout |
| `resnet_complex` | Early-fusion ResNet (ResNet-18 layout at half width) |
| `knn_complex`, `knn_magnitude` | k-nearest-neighbor fingerprinting |
| `transformer_appos_magnitude`, `cnn_magnitude` | Proposed model and CNN with magnitude-only input |
| `transformer_appos_complex_sync` | Proposed model trained and tested without impairments |

Every trained model is also evaluated with only a subset of the APs available, per SNR, per number of line-of-sight APs, and under every test impairment condition.

## 3. Summaries, figures and complexity

```bash
python3 scripts/summarize_results.py --results_dir results/camera_ready_200ep_randref results/camera_ready_200ep_randref_lr3e-4
python3 scripts/make_figures.py --output_dir figures --table_path figures/results_table.tex
python3 scripts/measure_complexity.py
```

`make_figures.py` writes the error CDF, the error versus the number of available APs, the error map of the office and the LaTeX tables. It needs `data_test_nominal.mat` in `deep-loc` (or `--data_path`).

## Results

Distance error on the test positions, mean ± standard deviation over five seeds. Every model is trained and tested with a random sub-sample delay and a random common phase per AP (`--random_reference`):

| Model | Parameters | Mean (m) | Median (m) | 90th percentile (m) |
|---|---|---|---|---|
| Proposed | 1.60 M | 0.17 ± 0.01 | 0.11 | 0.30 |
| Fixed-order fusion | 0.58 M | 0.20 ± 0.00 | 0.14 | 0.40 |
| Early-fusion ResNet | 2.80 M | 0.20 ± 0.02 | 0.13 | 0.36 |
| Proposed without AP position | 1.53 M | 0.31 ± 0.02 | 0.16 | 0.51 |
| Early-fusion CNN | 1.79 M | 0.32 ± 0.00 | 0.22 | 0.65 |
| Set pooling (no Transformer) | 0.54 M | 0.36 ± 0.01 | 0.24 | 0.68 |
| Proposed, magnitude-only input | 1.60 M | 0.47 ± 0.01 | 0.34 | 0.93 |
| Early-fusion CNN, magnitude-only input | 1.78 M | 1.00 ± 0.01 | 0.81 | 1.90 |
| kNN fingerprinting | – | 1.94 ± 0.06 | 1.64 | 3.87 |

## Multi-layout study (in progress)

To test whether a model generalizes to AP layouts that it was not trained on, `phy/wifiPosGenerateDataLayouts.m` generates many layouts with 3 to 6 APs at random positions (set `smokeTest = true` at the top for a quick test). It writes `data_layouts_train.mat`, `data_layouts_test_unseen.mat` (layouts that are not in the training set) and `data_layouts_test_seen.mat` (training layouts with new device positions).

```bash
python3 scripts/run_all.py smoke --study layouts
python3 scripts/run_all.py --study layouts --epochs 200 --patience 200 --seeds 0 1 2 --results_name layouts_200ep
python3 scripts/run_all.py --study layouts --epochs 200 --patience 200 --seeds 0 1 2 --results_name layouts_200ep_lr3e-4 --lr 3e-4
python3 scripts/summarize_layouts.py --results_dir results/layouts_200ep results/layouts_200ep_lr3e-4
```

Every model is run with and without the AP coordinates as input (`_appos` / `_nopos`). For the models that depend on the AP order, the APs of every sample are sorted by their coordinates.

## Limitations

The evaluation uses ray-traced channels and a single AP layout. With one layout, the AP-position embedding identifies the APs; generalization to layouts that are not seen during training is not tested.
