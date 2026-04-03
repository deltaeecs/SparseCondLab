from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from sparsecondlab.io import load_matrix


def test_load_matrix_from_real_matrix_market():
    matrix = load_matrix(Path("tests/data/real_example.mtx"))

    assert matrix.shape == (5, 5)
    assert matrix.nnz == 8
    assert matrix[0, 0] == 1.0
    assert matrix[4, 4] == 12.0


def test_load_matrix_from_dense_array():
    array = np.array([[1, 0], [0, 2]], dtype=float)

    matrix = load_matrix(array)

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.toarray().tolist() == [[1.0, 0.0], [0.0, 2.0]]