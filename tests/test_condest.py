"""Integration tests for condest_1 and condest_2 on real sparse-matrix files.

Note on test scope
------------------
real_example.mtx is a 5×5 matrix, which is below the exact_threshold=64 in
condest_1.  Both functions therefore take the dense exact path (np.linalg.cond)
for this input.  These tests verify:

  * The public API accepts a sparse matrix loaded from a .mtx file.
  * The result is finite and agrees with the reference np.linalg.cond value
    (i.e., the sparse → dense conversion does not corrupt the data).

They do NOT verify the accuracy of the onenormest large-matrix path or the
Krylov estimators against a known ground truth.  For that, see
test_condest_analytical.py, which uses diagonal matrices with closed-form κ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sparsecondlab.condest import condest_1, condest_2
from sparsecondlab.io import load_matrix


def test_condest_matches_dense_condition_number_on_real_matrix():
    matrix = load_matrix(Path("tests/data/real_example.mtx"))

    expected = np.linalg.cond(matrix.toarray(), p=1)
    actual = condest_1(matrix)

    assert np.isfinite(actual)
    assert actual == expected


def test_condest_2_matches_dense_condition_number_on_real_matrix():
    matrix = load_matrix(Path("tests/data/real_example.mtx"))

    expected = np.linalg.cond(matrix.toarray(), p=2)
    actual = condest_2(matrix)

    assert np.isfinite(actual)
    assert actual == expected