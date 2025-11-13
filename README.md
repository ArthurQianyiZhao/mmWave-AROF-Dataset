# mmWave-AROF-Dataset

## 🌟 Overview

This dataset provides a comprehensive collection of both **reference transmitted waveforms (Tx)** and **experimentally received waveforms (Rx)** from an Analog Radio-over-Fiber (ARoF) link. The primary purpose is to serve as a ground truth and experimental basis for signal processing, machine learning model training, and performance analysis in optical communication systems.

## Dataset Contents and Organization

The dataset is divided into two primary components:

### 1. Reference Transmitted Waveforms (Tx)

These serve as the **ground truth** for subsequent experiments. They are generated from a pseudorandom binary sequence (PRBS) and pulse-shaped.

| **Feature**              | **Description**                                              |
| ------------------------ | ------------------------------------------------------------ |
| **Modulation Formats**   | QPSK, 16-QAM, 32-QAM, 64-QAM (Gray coding, uniform spacing)  |
| **Symbols per Waveform** | 100,000                                                      |
| **Pulse Shaping**        | Root-Raised Cosine (RRC) Filter                              |
| **RRC Parameters**       | Roll-off factor=0.2 (Occupying Bandwidth 1.2 GHz), Filter span=10, Oversampling factor=10 |
| **Samples per Waveform** | 1,000,000 complex samples                                    |

**Storage Location:** `/Tx_Waveform_and_Bits`

- Contains the four reference complex waveforms and their corresponding bit sequences.

### 2. Experimentally Received Waveforms (Rx)

These are the core of the experimental data, resulting from transmitting the Tx waveforms through the ARoF link under various conditions. The sweep of experimental parameters results in a total of **108 unique experimental scenarios**.

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

The 81 sets of received waveforms (4 modulation schemes $\times$ 3 frequencies $\times$ 3 lengths $\times$ 3 powers $\rightarrow$ **108** scenarios) are organized in a clear folder structure:

```
[Fiber Length]/[Modulation Scheme]/[ModulationFormat_CarrierFrequency_PDInputPower_i/q.txt]
```

**Example File Path Breakdown:**

| **Path Component**    | **Example Value**  | **Description**                                              |
| --------------------- | ------------------ | ------------------------------------------------------------ |
| `[Fiber Length]`      | `/10km/`           | Transmission fiber length.                                   |
| `[Modulation Scheme]` | `/16QAM/`          | Modulation format used.                                      |
| `[File Name]`         | `16QAM_28_7_i.txt` | **16QAM** (Modulation), **28** (Carrier Frequency in GHz), **7** (PD Power in dBm), **i** (I-component). |

## Essential Pre-processing: Synchronization

**CRITICAL NOTE:** The received (Rx) waveforms are **NOT** time-aligned with the reference transmitted (Tx) waveforms.

- A **synchronization step** is essential before any signal processing, channel estimation, or model training can be performed.
- **Assistance:** An example **MATLAB script** is provided in the dataset root directory demonstrating how to perform this synchronization using **autocorrelation** and subsequent **symbol selection with matched filtering**.
