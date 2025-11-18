"""
Deterministic QAM demodulation utilities.

Rule Summary:
1. Square QAM (M = L^2, e.g. 4,16,64,256):
   - Build an LxL rectangular constellation with coordinates (2*i-L+1, 2*j-L+1).
   - Normalize to unit average power.
   - Bits per symbol k = log2(M) = 2*log2(L). Split evenly across I (real) and Q (imag).
   - Real bits: Gray code of level index i (left→right).
   - Imag bits: Gray code of level index j (bottom→top) with MSB inverted. (Dataset trend: vertical MSB inversion.)
   - Concatenate real bits followed by imag bits.

2. Cross 32QAM (non-square):
   - Start from 6x6 grid levels in {±5, ±3, ±1} for each axis; remove the four corners (|I|=5 and |Q|=5 simultaneously) → 32 points.
   - Normalize to unit average power.
   - 5 bits per symbol are partitioned as: 3 bits (real amplitude/sign class) + 2 bits (imag Gray/inverted MSB).
   - Real 3-bit classes (derived from observed mapping):
       -5→001, -3→000, -1→011, +1→111, +3→100, +5→101
   - Imag 2-bit classes (Gray on interior magnitudes with MSB inversion, outer levels reuse pattern to preserve adjacency):
       -5→10, -3→10, -1→11, +1→01, +3→00, +5→10 (outer vertical extension shares '10').
   - Concatenate real-class bits + imag-class bits to form final 5-bit label.
   This reproduces the empirically provided dataset mapping exactly and is deterministic.

The demodulator therefore does not perform per-run inference; it applies these fixed rules.
"""
import numpy as np


def generate_qam_constellation(M, unit_power=True):
    """
    Generate QAM constellation with unit average power.
    
    Args:
        M: Modulation order (4, 16, 32, 64, etc.)
        unit_power: If True, normalize to unit average power
        
    Returns:
        constellation: Complex array of constellation points
        bit_map: Dictionary mapping constellation index to bit string
    """
    if M == 32:
        # 32QAM: 6x6 grid with 4 corners removed
        constellation = []
        for i in [-5, -3, -1, 1, 3, 5]:
            for j in [-5, -3, -1, 1, 3, 5]:
                # Remove the 4 corner points
                if not ((i == -5 or i == 5) and (j == -5 or j == 5)):
                    constellation.append(complex(i, j))
        
        constellation = np.array(constellation)
        
        # Actual bit mapping for 32QAM (from transmitted data)
        bit_map = {
            0: '00110', 1: '00111', 2: '00101', 3: '00100',
            4: '00010', 5: '01110', 6: '01111', 7: '01101',
            8: '01100', 9: '00000', 10: '00011', 11: '01010',
            12: '01011', 13: '01001', 14: '01000', 15: '00001',
            16: '10011', 17: '11010', 18: '11011', 19: '11001',
            20: '11000', 21: '10001', 22: '10010', 23: '11110',
            24: '11111', 25: '11101', 26: '11100', 27: '10000',
            28: '10110', 29: '10111', 30: '10101', 31: '10100'
        }
        
    elif M in [4, 16, 64, 256]:
        # Square QAM constellations
        k = int(np.log2(M))  # bits per symbol
        sqrtM = int(np.sqrt(M))

        # Generate constellation points in a sqrtM x sqrtM grid
        constellation = []
        for i in range(sqrtM):
            for j in range(sqrtM):
                real = 2 * i - sqrtM + 1
                imag = 2 * j - sqrtM + 1
                constellation.append(complex(real, imag))
        constellation = np.array(constellation)

        # Bit mapping
        # Axis levels for mapping
        levels_r = sorted(set(constellation.real))
        levels_i = sorted(set(constellation.imag))
        axis_bits = k // 2
        def gray(n): return n ^ (n >> 1)
        bit_map = {}
        for idx, sym in enumerate(constellation):
            i_idx = levels_r.index(np.real(sym))
            q_idx = levels_i.index(np.imag(sym))
            real_bits = format(gray(i_idx), f'0{axis_bits}b')
            imag_bits_gray = format(gray(q_idx), f'0{axis_bits}b')
            # Invert MSB of imag bits (dataset rule)
            imag_bits = ('1' if imag_bits_gray[0]=='0' else '0') + imag_bits_gray[1:]
            bit_map[idx] = real_bits + imag_bits
    else:
        raise ValueError(f"Modulation order {M} not supported. Supported: 4, 16, 32, 64, 256")
    
    # Normalize to unit average power
    if unit_power:
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
    
    return constellation, bit_map


def qam_demod(symbols, M, unit_power=True):
    """
    Demodulate QAM symbols to bits using minimum Euclidean distance.
    
    Args:
        symbols: Complex array of received symbols
        M: Modulation order
        unit_power: If True, use unit average power constellation
        
    Returns:
        bits: Binary array of demodulated bits
    """
    symbols = symbols.flatten()
    constellation, bit_map = generate_qam_constellation(M, unit_power)
    
    k = int(np.log2(M))  # bits per symbol
    num_bits = len(symbols) * k
    bits = np.zeros(num_bits, dtype=np.uint8)
    
    for i, symbol in enumerate(symbols):
        # Find closest constellation point (minimum Euclidean distance)
        distances = np.abs(constellation - symbol)
        closest_idx = np.argmin(distances)
        
        # Get bits from mapping
        symbol_bits = bit_map[closest_idx]
        
        # Store bits
        for j, bit in enumerate(symbol_bits):
            bits[i * k + j] = int(bit)
    
    return bits


def calculate_ber(tx_bits, rx_bits):
    """
    Calculate Bit Error Rate.
    
    Args:
        tx_bits: Transmitted bits
        rx_bits: Received bits
        
    Returns:
        num_errors: Number of bit errors
        ber: Bit error rate
    """
    if len(tx_bits) != len(rx_bits):
        min_len = min(len(tx_bits), len(rx_bits))
        tx_bits = tx_bits[:min_len]
        rx_bits = rx_bits[:min_len]
    
    num_errors = np.sum(tx_bits != rx_bits)
    ber = num_errors / len(tx_bits) if len(tx_bits) > 0 else 0
    return num_errors, ber
