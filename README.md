# mmWave-AROF-Dataset

## 🌟 Overview

This dataset provides a comprehensive collection of both **reference transmitted waveforms (Tx)** and **experimentally received waveforms (Rx)** from an Analog Radio-over-Fiber (ARoF) link. The primary purpose is to serve as a ground truth and experimental basis for signal processing, machine learning model training, and performance analysis in optical communication systems.

## Dataset Contents and Organization

The dataset is divided into two primary components:

### 1. Reference Transmitted Waveforms (Tx)

These serve as the ground truth for subsequent experiments. They are generated from a pseudorandom binary sequence (PRBS) and pulse-shaped.

| **Feature**              | **Description**                                              |
| ------------------------ | ------------------------------------------------------------ |
| **Modulation Formats**   | QPSK, 16-QAM, 32-QAM, 64-QAM (Gray coding, uniform spacing)  |
| **Symbols per Waveform** | 100,000                                                      |
| **Pulse Shaping**        | Root-Raised Cosine (RRC) Filter                              |
| **RRC Parameters**       | Roll-off factor=0.2 (Occupying Bandwidth 1.2 GHz), Filter span=10, Oversampling factor=10 |
| **Samples per Waveform** | 1,000,000 complex samples                                    |

**Storage Location:** `/Tx_Waveform_and_Bits`

- Contains the four reference complex waveforms and corresponding raw bit sequences.

### 2. Experimentally Received Waveforms (Rx)

These are the core of the experimental data, resulting from transmitting the Tx waveforms through the ARoF link under various conditions. The sweep of experimental parameters results in a total of 108 unique experimental scenarios.

| **Parameter Category**         | **Values Swept**                |
| ------------------------------ | ------------------------------- |
| **Modulation Schemes**         | QPSK, 16-QAM, 32-QAM, 64-QAM    |
| **Carrier Frequencies**        | 28 GHz, 29 GHz, 30 GHz          |
| **Transmission Fiber Lengths** | 1 m (back-to-back), 5 km, 10 km |
| **PD Received Optical Power**  | 3 dBm, 5 dBm, 7 dBm             |

------

## Data Format and File Organization

### File Format

- All waveforms (Tx and Rx) are provided in **`.txt` format** for broad compatibility.
- The **In-phase (I)** and **Quadrature (Q)** components of each complex waveform are saved in **separate files**.

### Rx Waveform Hierarchical Structure

The 108 sets of received waveforms (4 modulation schemes $\times$ 3 frequencies $\times$ 3 lengths $\times$ 3 powers $\rightarrow$ 108 scenarios) are organized in a clear folder structure:

```
[Fiber Length]/[Modulation Scheme]/[ModulationFormat_CarrierFrequency_PDInputPower_i/q.txt]
```

**Example File Path Breakdown:**

| **Path Component**    | **Example Value**  | **Description**                                              |
| --------------------- | ------------------ | ------------------------------------------------------------ |
| `[Fiber Length]`      | `/10km/`           | Transmission fiber length.                                   |
| `[Modulation Scheme]` | `/16QAM/`          | Modulation format used.                                      |
| `[File Name]`         | `16QAM_28_7_i.txt` | **16QAM** (Modulation), **28** (Carrier Frequency in GHz), **7** (PD Power in dBm), **i** (I-component). |

## Example Synchronization

The received (Rx) waveforms are NOT time-aligned with the reference transmitted (Tx) waveforms. An example waveform-level synchronization has been processed in `waveform_sync.ipynb` using autocorrelation in Python. Resulted synchronized waveforms have been saved in each folder under name `sync_[Modulation Scheme_ModulationFormat_CarrierFrequency_PDInputPower.txt]`, with Rx on the left column and Tx on the right column (label), which can be directly used for training.

An example MATLAB script is also provided in the dataset root directory demonstrating how to perform this synchronization using autocorrelation and subsequent symbol selection with matched filtering.

The baseline Python workflow now performs the same steps directly from the raw `.txt` waveform files, without requiring MATLAB `.mat` exports:

1. Load the reference Tx waveform from `Tx_Waveform_and_Bits/[Modulation]_i/q.txt`.
2. Load the experimental Rx waveform from `[Fiber Length]/[Modulation]/[Modulation_Carrier_PD_i/q.txt]`.
3. Estimate the waveform delay using FFT cross-correlation.
4. Circularly align the Rx waveform to the Tx waveform.
5. Apply the square-root raised-cosine receive filter with roll-off `0.2`, `sps=10`, filter span `10`, and decimation factor `10`.
6. Drop the first `filterSpan` output symbols, matching `filteredSymbols(filterSpan+1:end)` in `waveform_sync_and_matched_filtering.m`.
7. Generate Tx label symbols from the raw bit sequence using MATLAB-compatible Gray QAM mapping.
8. Save the processed arrays in Python-native `.npz` format.

For example, `ML_receiver_main.ipynb` builds or loads:

```
Use_Cases/OFC_Paper_Analysis/10km_16QAM_28_3.npz
```

with `input` as matched-filtered Rx symbols, `label` as the aligned transmitted symbols, and `tx_bits` as the original raw transmitted bit sequence.

You can also use the first 100 symbols (a known preamble sequence) for synchronization.

## Example Use Case - Baseline ML Equalizers

To validate the dataset and provide a baseline, we include several ML equalizers in the repository, all implemented for symbol regression.

- XGBoost - Tree model
- Multilayer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Transformer

For this demonstration, the Tx symbols (ground truth labels) and the Rx symbols (model inputs) are generated in Python by `ML_receiver_main.ipynb` from the raw waveform text files. The generated `.npz` cache replaces the previous `.mat`-based workflow.

Training Parameters (Deep Models):

- Input Window: 21 symbols
- Epochs: 100
- Hidden Layers: 2 (with size 64).

The models report MSE, R2 score, RMS EVM, and BER. BER is calculated against the actual raw transmitted bit sequence for each test symbol, not by re-demodulating the label symbols.

The models and Python scripts demonstrating this analysis are available in the `/Use_Cases/OFC_Paper_Analysis/` folder.

- `ML_receiver_main.ipynb`: The main Jupyter Notebook for running the analysis.
- Model implementations: `mlp_model.py`, `lstm_model.py`, `gru_model.py`, `transformer_model.py`, `xgboost_model.py`.
- `utils.py`: Utility functions for MATLAB-compatible QAM modulation/demodulation, RRC filtering, BER, and EVM.
- `verify_matlab_demod_rrc.ipynb`: A small notebook for verifying the Python QAM/RRC implementation against the provided 16QAM Tx waveform and raw bit file.

We welcome contributions and encourage researchers to upload their solutions.
