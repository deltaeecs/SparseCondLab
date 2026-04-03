from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab, gmres


@dataclass(frozen=True)
class IterativeBenchmarkResult:
    method: str
    converged: bool
    iterations: int
    residual_norm: float
    elapsed_seconds: float


def _default_rhs(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.ones(matrix.shape[0], dtype=matrix.dtype)
    return np.ones(np.asarray(matrix).shape[0], dtype=np.asarray(matrix).dtype)


def _residual_norm(matrix: sparse.spmatrix | np.ndarray, solution: np.ndarray, rhs: np.ndarray) -> float:
    residual = matrix @ solution - rhs
    return float(np.linalg.norm(np.asarray(residual)))


def run_iterative_benchmark(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    method: str,
    rhs: np.ndarray | None = None,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> IterativeBenchmarkResult:
    working = matrix.tocsr() if sparse.issparse(matrix) else np.asarray(matrix)
    right_hand_side = np.asarray(rhs) if rhs is not None else _default_rhs(working)

    iteration_count = 0

    def _count_callback(_value: np.ndarray | float) -> None:
        nonlocal iteration_count
        iteration_count += 1

    start = perf_counter()
    if method == "gmres":
        solution, info = gmres(
            working,
            right_hand_side,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            callback=_count_callback,
            callback_type="legacy",
        )
    elif method == "bicgstab":
        solution, info = bicgstab(
            working,
            right_hand_side,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            callback=_count_callback,
        )
    else:
        raise ValueError(f"unsupported iterative method: {method}")
    elapsed_seconds = perf_counter() - start

    solution_array = np.asarray(solution)
    return IterativeBenchmarkResult(
        method=method,
        converged=info == 0,
        iterations=iteration_count,
        residual_norm=_residual_norm(working, solution_array, right_hand_side),
        elapsed_seconds=elapsed_seconds,
    )


def run_iterative_benchmarks(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    methods: Iterable[str] = ("gmres", "bicgstab"),
    rhs: np.ndarray | None = None,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> list[IterativeBenchmarkResult]:
    return [
        run_iterative_benchmark(
            matrix,
            method=method,
            rhs=rhs,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
        )
        for method in methods
    ]