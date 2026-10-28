"""MATLAB-compatible preprocessing and globally leakage-safe datasets."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .compat import bits_for_symbol_indices, raised_cosine_receive_filter, qam_mod



MODULATIONS = ("QPSK", "16QAM", "32QAM", "64QAM")
MODULATION_ORDER = {"QPSK": 4, "16QAM": 16, "32QAM": 32, "64QAM": 64}
DOCUMENTED_CARRIERS = {"28", "29", "30"}
DOCUMENTED_POWERS = {"3", "5", "7"}
DOCUMENTED_FIBERS = {"1m", "5km", "10km"}
_SYNC_RE = re.compile(r"^sync_(QPSK|16QAM|32QAM|64QAM)_([^_]+)_([^_]+)\.txt$")


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    modulation: str
    carrier: str
    fiber: str
    power: str
    sync_path: Path
    bits_path: Path

    @property
    def order(self) -> int:
        return MODULATION_ORDER[self.modulation]

    @property
    def key(self) -> str:
        return f"{self.fiber}_{self.modulation}_{self.carrier}_{self.power}"

    def metadata(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario": self.key,
            "modulation": self.modulation,
            "carrier_ghz": self.carrier,
            "fiber": self.fiber,
            "power_dbm": self.power,
        }


def discover_scenarios(root: str | Path) -> list[Scenario]:
    """Discover the documented 3x4x3x3 grid from synchronized text files."""

    root = Path(root).resolve()
    found: list[tuple[str, str, str, str, Path]] = []
    for path in root.glob("*/*/sync_*.txt"):
        match = _SYNC_RE.match(path.name)
        if not match:
            continue
        modulation, carrier, power = match.groups()
        fiber = path.parents[1].name.lower()
        if (
            carrier in DOCUMENTED_CARRIERS
            and power in DOCUMENTED_POWERS
            and fiber in DOCUMENTED_FIBERS
            and path.parent.name.upper() == modulation.upper()
        ):
            found.append((fiber, modulation, carrier, power, path))
    fiber_rank = {"1m": 0, "5km": 1, "10km": 2}
    mod_rank = {name: i for i, name in enumerate(MODULATIONS)}
    found.sort(key=lambda x: (fiber_rank[x[0]], mod_rank[x[1]], int(x[2]), int(x[3])))
    scenarios = [
        Scenario(i, mod, carrier, fiber, power, path, root / "Tx_Waveform_and_Bits" / f"{mod}_bits.txt")
        for i, (fiber, mod, carrier, power, path) in enumerate(found)
    ]
    keys = {scenario.key for scenario in scenarios}
    if len(keys) != len(scenarios):
        raise RuntimeError("Duplicate scenarios discovered")
    return scenarios


def _read_sync(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=np.complex64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected two complex columns in {path}, got {data.shape}")
    return data[:, 0], data[:, 1]


def preprocess_scenario(
    scenario: Scenario,
    cache_dir: str | Path,
    force: bool = False,
    samples_per_symbol: int = 1,
) -> Path:
    """Create an aligned 1-SPS or fractionally-spaced cache.

    The synchronized repository waveform is sampled at 10 samples/symbol. As
    in the repository's 2-SPS notebook, the RRC transient is therefore
    ``filter_span * samples_per_symbol`` decimated samples.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{scenario.key}.npz"
    if output.exists() and not force:
        with np.load(output, allow_pickle=False) as cached:
            cached_sps = int(cached["samples_per_symbol"]) if "samples_per_symbol" in cached else 1
        if cached_sps != int(samples_per_symbol):
            raise ValueError(
                f"{output} contains {cached_sps} SPS, requested {samples_per_symbol}; "
                "use a separate cache directory or --force"
            )
        return output
    rx_wave, _ = _read_sync(scenario.sync_path)
    bits = np.loadtxt(scenario.bits_path, dtype=np.uint8).reshape(-1)
    tx = qam_mod(bits, scenario.order).astype(np.complex64)
    samples_per_symbol = int(samples_per_symbol)
    if samples_per_symbol < 1 or 10 % samples_per_symbol:
        raise ValueError("samples_per_symbol must be a positive divisor of 10")
    rx = raised_cosine_receive_filter(
        rx_wave,
        rolloff=0.2,
        sps=10,
        filter_span=10,
        decimation_factor=10 // samples_per_symbol,
    )[10 * samples_per_symbol :].astype(np.complex64)
    n = min(rx.size // samples_per_symbol, tx.size - 10)
    rx = rx[: n * samples_per_symbol].reshape(n, samples_per_symbol)
    if samples_per_symbol == 1:
        rx = rx[:, 0]
    tx = tx[:n]
    np.savez_compressed(
        output,
        rx=rx,
        tx=tx,
        bits=bits,
        symbol_indices=np.arange(n, dtype=np.int64),
        samples_per_symbol=np.array(samples_per_symbol, dtype=np.int64),
        delay_convention="pre-synchronized file; MATLAB RRC filterSpan drop",
        scenario=scenario.key,
    )
    return output


def prepare_scenarios(
    scenarios: Iterable[Scenario],
    cache_dir: str | Path,
    force: bool = False,
    samples_per_symbol: int = 1,
) -> list[Path]:
    return [
        preprocess_scenario(
            s, cache_dir, force=force, samples_per_symbol=samples_per_symbol
        )
        for s in scenarios
    ]


def global_regions(symbol_count: int, fractions: Sequence[float]) -> dict[str, tuple[int, int]]:
    """Create absolute regions shared by every scenario."""

    n = int(symbol_count)
    a = int(math.floor(n * float(fractions[0])))
    b = int(math.floor(n * float(fractions[0] + fractions[1])))
    return {"train": (0, a), "val": (a, b), "test": (b, n)}


def permuted_global_regions(
    symbol_count: int,
    fractions: Sequence[float],
    order: Sequence[str] = ("train", "val", "test"),
) -> dict[str, tuple[int, int]]:
    """Assign the standard contiguous block lengths in a requested role order.

    This is a leakage-safe alternative to randomly shuffling overlapping symbol
    windows. Every scenario receives the same absolute regions, each symbol is
    assigned to exactly one role, and equalizer guard edges are removed later.
    """

    names = tuple(str(name) for name in order)
    if len(names) != 3 or set(names) != {"train", "val", "test"}:
        raise ValueError("order must be a permutation of train,val,test")
    base = global_regions(symbol_count, fractions)
    lengths = {name: hi - lo for name, (lo, hi) in base.items()}
    regions: dict[str, tuple[int, int]] = {}
    start = 0
    for name in names:
        regions[name] = (start, start + lengths[name])
        start += lengths[name]
    if start != int(symbol_count):
        raise AssertionError("permuted regions do not cover the symbol sequence")
    return regions


