from __future__ import annotations
import numpy as np

def fractional_windows(rx: np.ndarray, indices: np.ndarray, taps: int) -> np.ndarray:
    """Return centered ``[example, taps, samples_per_symbol]`` views."""

    values = np.asarray(rx)
    indices = np.asarray(indices, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"rx must have shape [symbols, samples_per_symbol], got {values.shape}")
    if taps < 1 or taps % 2 != 1:
        raise ValueError("taps must be a positive odd number")
    half = taps // 2
    if indices.size and (indices.min() < half or indices.max() >= len(values) - half):
        raise ValueError("a requested window crosses the waveform boundary")
    view = np.lib.stride_tricks.sliding_window_view(values, (taps, values.shape[1]))
    view = view.reshape(len(values) - taps + 1, taps, values.shape[1])
    return np.asarray(view[indices - half])

