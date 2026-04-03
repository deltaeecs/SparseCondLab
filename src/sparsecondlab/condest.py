from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu


def _one_norm(matrix: sparse.spmatrix | np.ndarray) -> float:
    if sparse.issparse(matrix):
        column_sums = np.abs(matrix).sum(axis=0)
        return float(np.asarray(column_sums).ravel().max(initial=0.0))
    return float(np.linalg.norm(np.asarray(matrix), 1))


def _solve_columns(lu: splu, values: np.ndarray, trans: str) -> np.ndarray:
    if values.ndim == 1:
        return lu.solve(values, trans=trans)
    solved = [lu.solve(column, trans=trans) for column in values.T]
    return np.column_stack(solved)


def condest_1(matrix: sparse.spmatrix | np.ndarray, *, exact_threshold: int = 64) -> float:
    """Estimate or compute the 1-norm condition number of a square matrix."""

    if sparse.issparse(matrix):
        working = matrix.tocsc()
    else:
        working = np.asarray(matrix)

    if working.ndim != 2 or working.shape[0] != working.shape[1]:
        raise ValueError("condest_1 requires a square matrix")

    if working.shape[0] <= exact_threshold:
        dense = working.toarray() if sparse.issparse(working) else working
        return float(np.linalg.cond(dense, p=1))

    sparse_working = working if sparse.issparse(working) else sparse.csc_matrix(working)
    lu = splu(sparse_working)
    trans = "H" if np.iscomplexobj(sparse_working.data if sparse.issparse(sparse_working) else sparse_working) else "T"

    def _matvec(values: np.ndarray) -> np.ndarray:
        return _solve_columns(lu, np.asarray(values), trans="N")

    def _rmatvec(values: np.ndarray) -> np.ndarray:
        return _solve_columns(lu, np.asarray(values), trans=trans)

    def _matmat(values: np.ndarray) -> np.ndarray:
        return _solve_columns(lu, np.asarray(values), trans="N")

    def _rmatmat(values: np.ndarray) -> np.ndarray:
        return _solve_columns(lu, np.asarray(values), trans=trans)

    inverse_operator = LinearOperator(
        shape=sparse_working.shape,
        matvec=_matvec,
        rmatvec=_rmatvec,
        matmat=_matmat,
        rmatmat=_rmatmat,
        dtype=sparse_working.dtype,
    )
    return float(_one_norm(sparse_working) * onenormest(inverse_operator))


def condest_2(matrix: sparse.spmatrix | np.ndarray) -> float:
    """Compute the exact 2-norm condition number of a square matrix."""

    if sparse.issparse(matrix):
        working = matrix.tocsc()
    else:
        working = np.asarray(matrix)

    if working.ndim != 2 or working.shape[0] != working.shape[1]:
        raise ValueError("condest_2 requires a square matrix")

    dense = working.toarray() if sparse.issparse(working) else working
    return float(np.linalg.cond(dense, p=2))