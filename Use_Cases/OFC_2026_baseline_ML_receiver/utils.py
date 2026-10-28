"""MATLAB-compatible QAM and RRC matched-filter utilities.

The QAM functions mirror the MATLAB calls used by this dataset:

    qammod(bits, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true)
    qamdemod(symbols, M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true)

Bits are grouped MSB-first, exactly like MATLAB's bit input mode.  The root
raised cosine receive filter mirrors:

    comm.RaisedCosineReceiveFilter(
        Shape='Square root',
        RolloffFactor=rolloff,
        InputSamplesPerSymbol=sps,
        FilterSpanInSymbols=filter_span,
        DecimationFactor=decimation_factor,
    )
"""

from __future__ import annotations

import math

import numpy as np


def _bits_per_symbol(M: int) -> int:
    M = int(M)
    if M < 2 or M & (M - 1):
        raise ValueError(f"M must be a power of two, got {M}")
    return int(math.log2(M))


def _as_bit_groups(bits, width: int) -> np.ndarray:
    bit_array = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bit_array.size % width:
        raise ValueError(
            f"Bit input length ({bit_array.size}) must be a multiple of "
            f"log2(M) ({width})."
        )
    if np.any((bit_array != 0) & (bit_array != 1)):
        raise ValueError("Bit input must contain only 0 and 1.")
    return bit_array.reshape(-1, width)


def _int_to_bits(values, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint64)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8)


def _bits_to_int(bit_groups: np.ndarray) -> np.ndarray:
    bit_groups = np.asarray(bit_groups, dtype=np.uint8)
    if bit_groups.ndim == 1:
        bit_groups = bit_groups.reshape(1, -1)
    width = bit_groups.shape[1]
    weights = (1 << np.arange(width - 1, -1, -1, dtype=np.uint64))
    return bit_groups.astype(np.uint64) @ weights


def _gray_to_binary(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64)
    out = values.copy()
    shift = 1
    while shift < 64:
        shifted = out >> shift
        if not np.any(shifted):
            break
        out ^= shifted
        shift <<= 1
    return out


def _axis_levels(n_levels: int) -> np.ndarray:
    return np.arange(-(n_levels - 1), n_levels, 2, dtype=float)


def _square_qam_constellation(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB Gray bit labels and unnormalized square-QAM symbols."""
    k = _bits_per_symbol(M)
    if k % 2:
        raise ValueError(f"M={M} is not square QAM.")

    axis_bits = k // 2
    n_levels = 1 << axis_bits
    levels = _axis_levels(n_levels)

    labels = _int_to_bits(np.arange(M), k)
    i_gray = _bits_to_int(labels[:, :axis_bits])
    q_gray = _bits_to_int(labels[:, axis_bits:])

    i_index = _gray_to_binary(i_gray).astype(np.int64)
    q_index = (n_levels - 1 - _gray_to_binary(q_gray)).astype(np.int64)

    symbols = levels[i_index] + 1j * levels[q_index]
    return labels, symbols


def _cross_32qam_constellation() -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB R2026a qammod(..., 32, 'gray') unnormalized symbols."""
    labels = _int_to_bits(np.arange(32), 5)

    # Natural MSB-first bit-label order 00000, 00001, ..., 11111.
    # MATLAB uses this cross-32QAM layout for qammod/qamdemod with M=32.
    points = np.array(
        [
            (-3, +5),
            (-1, +5),
            (-3, -5),
            (-1, -5),
            (-5, +3),
            (-5, +1),
            (-5, -3),
            (-5, -1),
            (-1, +3),
            (-1, +1),
            (-1, -3),
            (-1, -1),
            (-3, +3),
            (-3, +1),
            (-3, -3),
            (-3, -1),
            (+3, +5),
            (+1, +5),
            (+3, -5),
            (+1, -5),
            (+5, +3),
            (+5, +1),
            (+5, -3),
            (+5, -1),
            (+1, +3),
            (+1, +1),
            (+1, -3),
            (+1, -1),
            (+3, +3),
            (+3, +1),
            (+3, -3),
            (+3, -1),
        ],
        dtype=float,
    )
    symbols = points[:, 0] + 1j * points[:, 1]
    return labels, symbols


def _matlab_qam_constellation(M: int, unit_power: bool = True) -> tuple[np.ndarray, np.ndarray]:
    M = int(M)
    k = _bits_per_symbol(M)

    if M == 32:
        labels, symbols = _cross_32qam_constellation()
    elif k % 2 == 0:
        labels, symbols = _square_qam_constellation(M)
    else:
        raise ValueError(
            f"M={M} is not implemented. This utility supports MATLAB-compatible "
            "square QAM plus MATLAB cross-32QAM."
        )

    if unit_power:
        symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

    return labels, symbols


def generate_qam_constellation(M: int, unit_power: bool = True):
    """Generate a MATLAB-compatible Gray QAM constellation.

    Returns:
        constellation:
            Complex symbols ordered by natural MSB-first bit labels.  For
            example, index 0 is bit label ``0000`` for 16-QAM.
        bit_map:
            Dictionary mapping constellation index to its bit-label string.
    """
    labels, constellation = _matlab_qam_constellation(M, unit_power=unit_power)
    bit_map = {
        index: "".join(str(int(bit)) for bit in label)
        for index, label in enumerate(labels)
    }
    return constellation, bit_map


def qam_mod(bits, M: int, unit_power: bool = True) -> np.ndarray:
    """MATLAB-compatible ``qammod(..., 'gray', 'InputType', 'bit')``.

    Args:
        bits: Flat bit sequence. Bits are grouped MSB-first in blocks of
            ``log2(M)``.
        M: Modulation order. Supports square QAM and MATLAB cross-32QAM.
        unit_power: If true, match MATLAB ``UnitAveragePower=true``.

    Returns:
        Complex QAM symbols.
    """
    k = _bits_per_symbol(M)
    bit_groups = _as_bit_groups(bits, k)
    symbol_indices = _bits_to_int(bit_groups).astype(np.int64)
    _, constellation = _matlab_qam_constellation(M, unit_power=unit_power)
    return constellation[symbol_indices]


def qam_demod(symbols, M: int, unit_power: bool = True, normalize: bool = False) -> np.ndarray:
    """MATLAB-compatible hard QAM demodulation to bits.

    Args:
        symbols: Complex received symbols.
        M: Modulation order. Supports square QAM and MATLAB cross-32QAM.
        unit_power: If true, use the ``UnitAveragePower=true`` constellation.
        normalize: If true, scale the received symbols to the constellation
            average power before slicing. Leave false when symbols are already
            on the MATLAB unit-power constellation.

    Returns:
        Flat ``uint8`` bit array, equivalent to MATLAB ``OutputType='bit'``.
    """
    rx = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    labels, constellation = _matlab_qam_constellation(M, unit_power=unit_power)

    if normalize and rx.size:
        target_power = np.mean(np.abs(constellation) ** 2)
        rx_power = np.mean(np.abs(rx) ** 2)
        if rx_power > 0:
            rx = rx * np.sqrt(target_power / rx_power)

    distances = np.abs(rx[:, None] - constellation[None, :]) ** 2
    nearest = np.argmin(distances, axis=1)
    return labels[nearest].reshape(-1).astype(np.uint8)


def rrcosdesign(rolloff: float, filter_span: int, sps: int) -> np.ndarray:
    """MATLAB ``rcosdesign(rolloff, filter_span, sps, 'sqrt')`` equivalent."""
    beta = float(rolloff)
    span = int(filter_span)
    samples_per_symbol = int(sps)

    if not 0 <= beta <= 1:
        raise ValueError("rolloff must be in the range [0, 1].")
    if span <= 0:
        raise ValueError("filter_span must be positive.")
    if samples_per_symbol <= 0:
        raise ValueError("sps must be positive.")
    if (span * samples_per_symbol) % 2:
        raise ValueError("filter_span * sps must be even, matching MATLAB rcosdesign.")

    n = np.arange(-span * samples_per_symbol / 2, span * samples_per_symbol / 2 + 1)
    t = n / samples_per_symbol

    if beta == 0:
        h = np.sinc(t)
    else:
        h = np.empty_like(t, dtype=float)
        zero = np.isclose(t, 0.0)
        singular = np.isclose(np.abs(t), 1.0 / (4.0 * beta))
        regular = ~(zero | singular)

        h[zero] = 1.0 - beta + (4.0 * beta / np.pi)
        h[singular] = (
            beta
            / np.sqrt(2.0)
            * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        )
        tr = t[regular]
        h[regular] = (
            np.sin(np.pi * tr * (1.0 - beta))
            + 4.0 * beta * tr * np.cos(np.pi * tr * (1.0 + beta))
        ) / (np.pi * tr * (1.0 - (4.0 * beta * tr) ** 2))

    return h / np.sqrt(np.sum(h * h))


def raised_cosine_receive_filter(
    rx_waveform,
    rolloff: float = 0.2,
    sps: int = 10,
    filter_span: int = 10,
    decimation_factor: int = 10,
) -> np.ndarray:
    """Apply MATLAB-equivalent square-root raised-cosine receive filtering.

    This returns the direct equivalent of ``filteredSymbols = rxRrcFilter(x)``.
    To reproduce the dataset script's symbol alignment, use
    ``filtered_symbols[filter_span:]`` afterwards.
    """
    decimation_factor = int(decimation_factor)
    if decimation_factor <= 0:
        raise ValueError("decimation_factor must be positive.")

    x = np.asarray(rx_waveform, dtype=np.complex128).reshape(-1)
    if x.size == 0:
        return x.copy()

    h = rrcosdesign(rolloff, filter_span, sps)

    try:
        from scipy.signal import lfilter

        filtered = lfilter(h, [1.0], x)
    except Exception:
        filtered = np.convolve(x, h, mode="full")[: x.size]

    return filtered[::decimation_factor]


def matched_filter_symbols(
    rx_waveform,
    rolloff: float = 0.2,
    sps: int = 10,
    filter_span: int = 10,
    decimation_factor: int = 10,
) -> np.ndarray:
    """Return aligned symbols like MATLAB ``filteredSymbols(filterSpan+1:end)``."""
    filtered = raised_cosine_receive_filter(
        rx_waveform,
        rolloff=rolloff,
        sps=sps,
        filter_span=filter_span,
        decimation_factor=decimation_factor,
    )
    return filtered[int(filter_span) :]


def calculate_ber(tx_bits, rx_bits):
    """Calculate bit errors and BER over the common sequence length."""
    tx_bits = np.asarray(tx_bits, dtype=np.uint8).reshape(-1)
    rx_bits = np.asarray(rx_bits, dtype=np.uint8).reshape(-1)

    if tx_bits.size != rx_bits.size:
        n = min(tx_bits.size, rx_bits.size)
        tx_bits = tx_bits[:n]
        rx_bits = rx_bits[:n]

    if tx_bits.size == 0:
        return 0, 0.0

    num_errors = int(np.sum(tx_bits != rx_bits))
    return num_errors, num_errors / tx_bits.size


def calculate_evm(reference_symbols, measured_symbols, percent: bool = True) -> float:
    """Calculate RMS EVM normalized by reference RMS power.

    EVM = sqrt(mean(|measured - reference|^2) / mean(|reference|^2)).
    When ``percent`` is true, the returned value is multiplied by 100.
    """
    reference_symbols = np.asarray(reference_symbols, dtype=np.complex128).reshape(-1)
    measured_symbols = np.asarray(measured_symbols, dtype=np.complex128).reshape(-1)

    n = min(reference_symbols.size, measured_symbols.size)
    if n == 0:
        return 0.0

    reference_symbols = reference_symbols[:n]
    measured_symbols = measured_symbols[:n]
    reference_power = np.mean(np.abs(reference_symbols) ** 2)
    if reference_power == 0:
        return 0.0

    evm = np.sqrt(np.mean(np.abs(measured_symbols - reference_symbols) ** 2) / reference_power)
    return float(100.0 * evm if percent else evm)


def bits_for_symbol_indices(tx_bits, symbol_indices, M: int) -> np.ndarray:
    """Select raw transmitted bits for the given symbol indices.

    The bit file is stored as a flat MSB-first stream, so symbol ``n`` maps to
    ``tx_bits[n*log2(M):(n+1)*log2(M)]``.
    """
    tx_bits = np.asarray(tx_bits, dtype=np.uint8).reshape(-1)
    symbol_indices = np.asarray(symbol_indices, dtype=np.int64).reshape(-1)
    k = _bits_per_symbol(M)

    if symbol_indices.size == 0:
        return np.array([], dtype=np.uint8)
    if np.any(symbol_indices < 0):
        raise ValueError("symbol_indices must be non-negative.")

    bit_indices = symbol_indices[:, None] * k + np.arange(k, dtype=np.int64)
    if np.max(bit_indices) >= tx_bits.size:
        raise ValueError(
            "Requested symbol indices exceed the transmitted bit sequence "
            f"length ({tx_bits.size} bits)."
        )

    return tx_bits[bit_indices].reshape(-1)


# MATLAB-style aliases, useful for notebooks that use Communication Toolbox names.
qammod = qam_mod
qamdemod = qam_demod
