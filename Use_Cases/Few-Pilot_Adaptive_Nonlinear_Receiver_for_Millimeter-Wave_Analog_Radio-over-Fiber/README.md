# Few-Pilot Adaptive Nonlinear Receiver for Millimeter-Wave Analog Radio-over-Fiber

Code and results supporting the submitted paper by Qianyi Zhao, Ampalavanapillai Nirmalathas, and Christina Lim. The repository root `LICENSE` applies.

## Contents

- `fewshot_equalizer/`, `receiver_helpers.py`, `run_receiver.py`: receiver and reproduction code.
- `model/source_prior.npz`: 32 source coefficient vectors and the trained PCA prior.
- `results/matched_baselines_*.csv`: per-capture and pooled results for the three
  receiver variants in Figure 2.
- `results/constellation_*.npz`: four Figure 2 constellation datasets
  (10 km, 29 GHz, 5 dBm).
- `results/budget_*.csv`: the four pilot/DD budgets reported in Section 4.
- `results/proposed_parameter_sweep/validation_summary.csv`: all 150 source-only
  configurations supporting the parameter-sensitivity statement in Section 4.
- `sweep_proposed_parameters.py`: reproduce that source-only sweep.

## Reproduce

Python 3.10 or newer. From this directory:

```bash
pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python run_receiver.py prepare
python run_receiver.py evaluate
python run_receiver.py budgets
```

The code reads the synchronized recordings and transmitted bits already in this
repository. It reuses the MATLAB-compatible signal/QAM utilities in
`../OFC_2026_baseline_ML_receiver/utils.py`; keep that sibling file available.
`prepare` creates `cache_2sps/`; evaluation also prepares missing target caches.
New results go to `reproduced/`, leaving the supplied reference results intact.
Generated caches and reruns are ignored by Git.

Optional source retraining and the reported parameter sweep:

```bash
python run_receiver.py train
python run_receiver.py evaluate --prior reproduced/source_prior.npz
# Requires source caches from the prepare command above.
python sweep_proposed_parameters.py
```

Training writes a new prior. The sweep uses the supplied prior for the 11-position
case and refits source teachers for other window lengths. It excludes all four
formats of one source physical setting per fold, across eight folds. The sweep
covers windows {7,9,11,13,15}, ranks {7,11,15,19,23}, and DD updates
{0,4,8,12,16,24}, with a common seven-symbol scoring guard. The best relative BER
improvement over (11,15,12) is approximately 0.54%.

## Paper configuration and results

Sources are {1 m, 5 km} × {28, 30 GHz} × {3, 7 dBm}, across four QAM formats
(32 captures). The remaining 76 captures are targets. Each source teacher uses
25,000 labels. The receiver has 89 complex coefficients: 11 positions × two
sample phases × direct/conjugate × linear/cubic, plus bias. The shared PCA prior
has 15 real coordinates and retains 97.281248% of source coefficient variance.
Eight uniformly selected pilots initialize the receiver (latent ridge 1e-3),
followed by 12 full-coefficient DD updates (ridge 2e-4; bias penalty factor 0.01).
The teacher-centered ridge comparator uses the fixed ridge value 1.

| Figure 2 receiver | Bit errors / 9,685,801 test bits | Pooled BER |
|---|---:|---:|
| Teacher-centered ridge + DD | 252,941 | 0.026114619 |
| Subspace, no DD | 171,793 | 0.017736582 |
| Subspace + DD | 35,970 | 0.003713684 |

| Pilots | DD rows | Pooled BER |
|---:|---:|---:|
| 4 | 8,965 | 0.009604162 |
| 8 | 8,965 | 0.003713684 |
| 16 | 8,965 | 0.003717813 |
| 8 | 4,096 | 0.003776249 |

BER is total errors divided by total transmitted bits. EVM is per-capture RMS
error divided by transmitted-symbol RMS, expressed as a percentage; format-level
EVM is the arithmetic mean across captures. The Method 3 comparison in the paper
is a cited external benchmark, not a receiver reproduced by this package.

## Indexing note

Zero-based, half-open symbol intervals are training `[0,59994)`, calibration
`[59994,69993)`, and test `[69993,99990)`. Five-symbol guards leave 29,987 test
symbols per target. Calibration skips its first 1,024 symbols and applies the
same guards, giving `9999 - 1024 - 5 - 5 = 8965` DD rows. Normalization uses the
full received calibration interval. Eight pilots are selected uniformly from
512 equally spaced candidate positions. The 4,096-row budget uniformly
subsamples DD rows while retaining the normalization region and pilot span.

**The implementation underlying the supplied results includes pilot positions
in the 8,965 DD rows, but uses no known labels during DD and no separate pilot
loss.** The submitted PDF's wording “excluding the pilot selections” does not
literally describe this indexing; excluding them would leave 8,957 rows and
would be a different experiment. The numerical results and receiver are
preserved here.

Synchronization is supplied by dataset preprocessing and is outside the pilot
budget. Known QAM order configures the slicer and demapper. Coefficients are frozen
for testing. The approximately 440-real-MAC/symbol estimate excludes filtering,
demapping, and calibration. Shared PRBS reference sequences do not establish
independent-sequence or new-acquisition-session generalization.
