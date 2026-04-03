from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.io import mmread

MatrixSource = Any


def load_matrix(source: str | Path | MatrixSource) -> sparse.csr_matrix:
    """Load a sparse matrix from Matrix Market, NPZ, or an in-memory matrix."""

    if sparse.issparse(source):
        return source.tocsr()

    if isinstance(source, np.ndarray):
        if source.ndim != 2:
            raise ValueError("dense input must be two-dimensional")
        return sparse.csr_matrix(source)

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix == ".npz":
        return sparse.load_npz(path).tocsr()

    if suffix in {".mtx", ".mm", ".mtz"}:
        loaded = mmread(path, spmatrix=False)
        if sparse.issparse(loaded):
            return loaded.tocsr()
        dense = np.asarray(loaded)
        if dense.ndim != 2:
            raise ValueError("Matrix Market input must be two-dimensional")
        return sparse.csr_matrix(dense)

    raise ValueError(f"unsupported matrix source: {path}")