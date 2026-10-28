from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .compat import qam_demod, qam_mod
from .fractional import fractional_windows
from .teacher_calibration import d_optimal_indices, solve_real_latent

def real_vector(values: np.ndarray) -> np.ndarray:
    """Interleave real/imaginary parts of a complex vector."""

    values = np.asarray(values)
    return np.stack((values.real, values.imag), axis=-1).reshape(-1).astype(np.float32)


def complex_vector(values: np.ndarray) -> np.ndarray:
    """Invert :func:`real_vector`."""

    values = np.asarray(values)
    if values.size % 2:
        raise ValueError("an interleaved complex vector must have even length")
    pairs = values.reshape(-1, 2)
    return pairs[:, 0] + 1j * pairs[:, 1]


def normalize_from_region(
    rx: np.ndarray, region: tuple[int, int]
) -> tuple[np.ndarray, complex, float]:
    """Center and scale Rx using only an explicitly supplied unlabeled region."""

    values = np.asarray(rx, dtype=np.complex128)
    context = values[slice(*region)]
    if context.size == 0:
        raise ValueError("normalization region is empty")
    mean = complex(context.mean())
    scale = float(np.sqrt(np.mean(np.abs(context - mean) ** 2)).clip(1e-12))
    return (values - mean) / scale, mean, scale


def parallel_hammerstein_design(
    normalized_rx: np.ndarray,
    indices: np.ndarray,
    taps: int = 11,
    orders: tuple[int, ...] = (1, 3),
) -> np.ndarray:
    """Build direct/conjugate memory-polynomial features plus a complex bias."""

    orders = tuple(int(order) for order in orders)
    if not orders or any(order < 1 or order % 2 != 1 for order in orders):
        raise ValueError("orders must be nonempty positive odd integers")
    windows = fractional_windows(normalized_rx, indices, int(taps))
    x = windows.reshape(len(indices), -1).astype(np.complex128, copy=False)
    direct = np.concatenate(
        [x * np.abs(x) ** (order - 1) for order in orders], axis=1
    )
    return np.column_stack(
        (direct, direct.conj(), np.ones(len(indices), dtype=np.complex128))
    )


@dataclass(frozen=True)
class HammersteinTeacher:
    coefficient_vector: np.ndarray
    rx_mean: complex
    rx_scale: float
    train_indices: np.ndarray

    @property
    def coefficients(self) -> np.ndarray:
        return complex_vector(self.coefficient_vector)


def fit_hammerstein_teacher(
    rx: np.ndarray,
    tx: np.ndarray,
    region: tuple[int, int],
    taps: int = 11,
    orders: tuple[int, ...] = (1, 3),
    labels: int = 50_000,
    ridge: float = 1e-6,
) -> HammersteinTeacher:
    """Fit a leakage-safe source teacher by complex ridge regression."""

    taps = int(taps)
    half = taps // 2
    candidates = np.arange(region[0] + half, region[1] - half, dtype=np.int64)
    count = min(int(labels), len(candidates))
    if count < 1:
        raise ValueError("the fitting region contains no valid equalizer window")
    selected = np.unique(np.linspace(0, len(candidates) - 1, count, dtype=np.int64))
    indices = candidates[selected]
    normalized, mean, scale = normalize_from_region(rx, region)
    design = parallel_hammerstein_design(normalized, indices, taps, orders)
    target = np.asarray(tx, dtype=np.complex128)[indices]
    penalty = np.eye(design.shape[1], dtype=np.complex128) * float(ridge)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(
        design.conj().T @ design + penalty, design.conj().T @ target
    )
    return HammersteinTeacher(real_vector(weights), mean, scale, indices)


def complex_basis(real_basis: np.ndarray) -> np.ndarray:
    """Convert rows of an interleaved real coefficient basis to complex rows."""

    basis = np.asarray(real_basis)
    if basis.ndim != 2:
        raise ValueError("basis must be a matrix")
    return np.stack([complex_vector(row) for row in basis])


def calibrate_hammerstein_latent(
    design: np.ndarray,
    target: np.ndarray,
    candidate_indices: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    labels: int,
    ridge: float = 1e-3,
    selection: str = "uniform",
    prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Solve the real low-rank channel latent from a few complex pilots."""

    center_c = complex_vector(center)
    basis_c = complex_basis(basis)
    latent_dim = len(basis_c)
    if design.shape[0] != len(candidate_indices):
        raise ValueError("design rows and candidate indices do not match")
    if labels < 0 or labels > len(candidate_indices):
        raise ValueError("labels must fit within the candidate set")
    z_prior = np.zeros(latent_dim, dtype=np.float32) if prior is None else np.asarray(prior, dtype=np.float32)
    if z_prior.shape != (latent_dim,):
        raise ValueError("latent prior has the wrong shape")
    if labels == 0:
        return center_c + z_prior @ basis_c, z_prior, []
    responses = design @ basis_c.T
    if selection == "uniform":
        chosen = np.unique(
            np.linspace(0, len(candidate_indices) - 1, int(labels), dtype=np.int64)
        )
    elif selection == "d_optimal":
        chosen = d_optimal_indices(responses, int(labels))
    else:
        raise ValueError("selection must be 'uniform' or 'd_optimal'")
    residual = np.asarray(target)[candidate_indices[chosen]] - design[chosen] @ center_c
    z = solve_real_latent(responses[chosen], residual, z_prior, float(ridge))
    return center_c + z @ basis_c, z, candidate_indices[chosen].tolist()


def qam_decisions(prediction: np.ndarray, order: int) -> np.ndarray:
    """Map estimates to exact constellation points through repository utilities."""

    bits = qam_demod(np.asarray(prediction), int(order))
    return qam_mod(bits, int(order)).astype(np.complex128)


def decision_directed_adapt(
    initial: np.ndarray,
    unlabeled_design: np.ndarray,
    order: int,
    pilot_design: np.ndarray | None = None,
    pilot_target: np.ndarray | None = None,
    iterations: int = 12,
    confidence_fraction: float = 1.0,
    pilot_weight: float = 0.0,
    ridge: float = 2e-4,
) -> np.ndarray:
    """Normalized MAP/EM updates from unlabeled target decisions.

    ``pilot_weight`` weights the *mean* pilot loss relative to the mean
    pseudo-label loss, so changing the number of pilots does not silently
    change the objective.  Only the per-capture coefficient vector is updated;
    source bases and model parameters remain frozen.
    """

    design = np.asarray(unlabeled_design, dtype=np.complex128)
    weights = np.asarray(initial, dtype=np.complex128).copy()
    if design.ndim != 2 or design.shape[1] != weights.size:
        raise ValueError("unlabeled design and coefficient vector are incompatible")
    fraction = float(confidence_fraction)
    if not 0 < fraction <= 1:
        raise ValueError("confidence_fraction must lie in (0, 1]")
    if pilot_design is None:
        pilots = np.empty((0, weights.size), dtype=np.complex128)
        pilot_values = np.empty(0, dtype=np.complex128)
    else:
        pilots = np.asarray(pilot_design, dtype=np.complex128)
        pilot_values = np.asarray(pilot_target, dtype=np.complex128).reshape(-1)
        if pilots.shape != (pilot_values.size, weights.size):
            raise ValueError("pilot design and targets are incompatible")
    for _ in range(int(iterations)):
        prediction = design @ weights
        pseudo = qam_decisions(prediction, order)
        count = min(
            len(design),
            max(3 * weights.size, int(np.ceil(len(design) * fraction))),
        )
        if count < len(design):
            distance = np.abs(prediction - pseudo)
            chosen = np.argpartition(distance, count - 1)[:count]
            active, pseudo = design[chosen], pseudo[chosen]
        else:
            active = design
        gram = active.conj().T @ active / len(active)
        rhs = active.conj().T @ pseudo / len(active)
        if len(pilots):
            gram += float(pilot_weight) * (pilots.conj().T @ pilots / len(pilots))
            rhs += float(pilot_weight) * (pilots.conj().T @ pilot_values / len(pilots))
        penalty = np.eye(weights.size, dtype=np.complex128) * float(ridge)
        penalty[-1, -1] *= 0.01
        weights = np.linalg.solve(gram + penalty, rhs + penalty @ weights)
    return weights


def predict_hammerstein(
    normalized_rx: np.ndarray,
    indices: np.ndarray,
    coefficients: np.ndarray,
    taps: int = 11,
    orders: tuple[int, ...] = (1, 3),
    batch_size: int = 32_768,
) -> np.ndarray:
    """Predict a long capture without materializing its full feature matrix."""

    outputs = []
    for start in range(0, len(indices), int(batch_size)):
        batch = indices[start : start + int(batch_size)]
        design = parallel_hammerstein_design(normalized_rx, batch, taps, orders)
        outputs.append(design @ coefficients)
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.complex128)


def hammerstein_complexity(
    samples_per_symbol: int, taps: int, orders: tuple[int, ...]
) -> dict[str, int]:
    """Return documented deployed storage and real-MAC estimates."""

    complex_samples = int(samples_per_symbol) * int(taps)
    complex_coefficients = 2 * len(tuple(orders)) * complex_samples + 1
    dot_macs = 8 * len(tuple(orders)) * complex_samples
    feature_macs = 4 * complex_samples * sum(order > 1 for order in orders)
    return {
        "complex_coefficients": complex_coefficients,
        "deployed_real_parameters": 2 * complex_coefficients,
        "real_macs_per_symbol_approx": dot_macs + feature_macs,
    }

