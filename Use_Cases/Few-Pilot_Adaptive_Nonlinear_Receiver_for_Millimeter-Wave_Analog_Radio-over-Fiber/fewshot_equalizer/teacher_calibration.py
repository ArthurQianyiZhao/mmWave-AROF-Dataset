from __future__ import annotations
import numpy as np

def pca_basis(vectors: np.ndarray, latent_dim: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return center and scaled principal axes with unit-RMS latent scores."""

    values = np.asarray(vectors, dtype=np.float64)
    center = values.mean(axis=0)
    centered = values - center
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank = min(int(latent_dim), int(np.sum(singular > 1e-12)), vh.shape[0])
    basis = np.zeros((int(latent_dim), values.shape[1]), dtype=np.float32)
    scales = np.ones(int(latent_dim), dtype=np.float64)
    if rank:
        scores = centered @ vh[:rank].T
        scales[:rank] = np.sqrt(np.mean(scores**2, axis=0)).clip(1e-8)
        basis[:rank] = (vh[:rank] * scales[:rank, None]).astype(np.float32)
    total = float(np.sum(singular**2))
    explained = float(np.sum(singular[:rank] ** 2) / total) if total else 1.0
    metadata = {
        "effective_rank": rank,
        "explained_variance_fraction": explained,
        "singular_values": singular.tolist(),
        "score_scales": scales.tolist(),
    }
    return center.astype(np.float32), basis, metadata


def d_optimal_indices(features: np.ndarray, count: int, jitter: float = 1e-6) -> np.ndarray:
    """Select complex observations that best condition a real latent solve."""

    values = np.asarray(features)
    if values.ndim != 2 or not np.iscomplexobj(values):
        raise ValueError("features must be a [candidate, latent] complex array")
    if count < 1 or count > len(values):
        raise ValueError("count must be between one and the candidate count")
    information = np.eye(values.shape[1], dtype=np.float64) * float(jitter)
    remaining = list(range(len(values)))
    selected: list[int] = []
    for _ in range(int(count)):
        def score(index: int) -> float:
            row = values[index]
            update = np.outer(row.real, row.real) + np.outer(row.imag, row.imag)
            return float(np.linalg.slogdet(information + update)[1])

        best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
        row = values[best]
        information += np.outer(row.real, row.real) + np.outer(row.imag, row.imag)
    return np.asarray(selected, dtype=np.int64)


def solve_real_latent(
    features: np.ndarray,
    residual_targets: np.ndarray,
    prior: np.ndarray,
    ridge: float = 1e-2,
) -> np.ndarray:
    """Solve complex pilot equations for a real low-dimensional latent."""

    features = np.asarray(features)
    residual = np.asarray(residual_targets).reshape(-1)
    prior = np.asarray(prior, dtype=np.float64).reshape(-1)
    if features.shape != (residual.size, prior.size):
        raise ValueError("features, targets, and prior have incompatible shapes")
    design = np.concatenate((features.real, features.imag), axis=0)
    target = np.concatenate((residual.real, residual.imag), axis=0)
    gram = design.T @ design + float(ridge) * np.eye(prior.size)
    rhs = design.T @ target + float(ridge) * prior
    return np.linalg.solve(gram, rhs).astype(np.float32)

