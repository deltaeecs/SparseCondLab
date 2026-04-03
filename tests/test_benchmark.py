from __future__ import annotations

from pathlib import Path

from sparsecondlab.benchmark import run_iterative_benchmarks
from sparsecondlab.io import load_matrix


def test_iterative_benchmarks_converge_on_real_matrix():
    matrix = load_matrix(Path("tests/data/real_example.mtx"))

    results = run_iterative_benchmarks(matrix, methods=("gmres", "bicgstab"))

    assert [result.method for result in results] == ["gmres", "bicgstab"]
    assert all(result.converged for result in results)
    assert all(result.iterations > 0 for result in results)
    assert all(result.residual_norm < 1e-8 for result in results)