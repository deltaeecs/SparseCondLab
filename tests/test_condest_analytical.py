"""Analytical-solution tests for condition number functions.

Each test uses a matrix whose exact condition number is known in closed form,
providing independent ground-truth verification that goes beyond the
self-consistency checks in test_condest.py.

Key coverage added here
-----------------------
* condest_1 exact path (n ≤ 64)   — diagonal matrices, identity
* condest_1 onenormest path (n > 64) — diagonal matrices, scaled identity
  (this path has NO coverage in test_condest.py)
* condest_2 (always dense)         — round-trip check against analytical κ₂
* estimate_condest_1_krylov        — ILU is exact for diagonal → κ̂₁ matches
* estimate_condest_2_krylov        — SVDS on diagonal → κ̂₂ matches σ_max/σ_min
* complex diagonal                 — checks that modulus is used correctly
* near-singular and near-identity  — edge-case stress

Analytical formulae used
------------------------
For a real diagonal matrix D = diag(d₁, …, dₙ) with all dᵢ > 0:
    κ₁(D) = κ₂(D) = max(dᵢ) / min(dᵢ)

For αI (any n, any α ≠ 0):
    κ₁(αI) = κ₂(αI) = 1.0

For a real symmetric positive-definite 2×2 D = diag(a, b), a < b:
    κ₂(D) = b / a
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from sparsecondlab.condest import condest_1, condest_2
from sparsecondlab.krylov import estimate_condest_1_krylov, estimate_condest_2_krylov


# ─────────────────────────────────────── helpers ─────────────────────────────

def _diag_csc(values: np.ndarray) -> sparse.csc_matrix:
    """Return a sparse CSC diagonal matrix from a 1-D array of diagonal entries."""
    return sparse.diags(values.astype(float), format="csc")


def _diag_csc_complex(values: np.ndarray) -> sparse.csc_matrix:
    """Return a sparse CSC diagonal matrix with complex entries."""
    return sparse.diags(values.astype(complex), format="csc")


# ─────────────────────────────────────── condest_1 ───────────────────────────

class TestCondest1Analytical:
    """condest_1 verified against matrices with closed-form 1-norm condition numbers.

    The exact threshold is 64 (default). Tests below and above that threshold
    exercise different code paths.
    """

    # ── exact path (n ≤ 64) ─────────────────────────────────────────────────

    def test_identity_small_exact_path(self) -> None:
        """κ₁(I) = 1, exact path."""
        A = sparse.eye(10, format="csc")
        assert abs(condest_1(A) - 1.0) < 1e-12

    def test_diagonal_n4_small_exact_path(self) -> None:
        """D = diag([1,2,4,8]): κ₁ = 8/1 = 8, exact path."""
        d = np.array([1.0, 2.0, 4.0, 8.0])
        A = _diag_csc(d)
        assert abs(condest_1(A) - 8.0) < 1e-12

    def test_diagonal_uniform_n8_kappa_one(self) -> None:
        """All-equal diagonal: κ₁ = 1 regardless of the common value."""
        d = np.full(8, 3.7)
        A = _diag_csc(d)
        assert abs(condest_1(A) - 1.0) < 1e-12

    def test_two_cluster_small(self) -> None:
        """Values at 1 and 100: κ₁ = 100, small exact path."""
        d = np.r_[np.ones(6), 100.0 * np.ones(6)]  # n=12 ≤ 64
        A = _diag_csc(d)
        assert abs(condest_1(A) - 100.0) / 100.0 < 1e-12

    # ── onenormest path (n > 64) ─────────────────────────────────────────────

    def test_identity_large_onenormest_path(self) -> None:
        """κ₁(I₁₂₈) = 1 via the onenormest path."""
        A = sparse.eye(128, format="csc")
        # onenormest on the identity inverse is exact by construction
        assert abs(condest_1(A) - 1.0) < 1e-10

    def test_scaled_identity_large_onenormest_path(self) -> None:
        """κ₁(7·I₁₂₈) = 1: scaling does not affect the condition number."""
        A = 7.5 * sparse.eye(128, format="csc")
        assert abs(condest_1(A) - 1.0) < 1e-10

    def test_diagonal_n200_onenormest_path(self) -> None:
        """D = diag([1,…,200]): κ₁ = 200/1 = 200, onenormest path.

        onenormest is exact for diagonal matrices (single power-method iteration
        identifies the extremal column).  Tolerance is set to 0.1 % to allow for
        the probabilistic nature of the estimator.
        """
        n = 200
        d = np.arange(1.0, n + 1)
        A = _diag_csc(d)
        result = condest_1(A)
        assert abs(result - float(n)) / float(n) < 1e-3

    def test_two_cluster_large_onenormest_path(self) -> None:
        """Values at 1 and 1000 (n=100): κ₁ = 1000, onenormest path."""
        d = np.r_[np.ones(50), 1000.0 * np.ones(50)]  # n=100 > 64
        A = _diag_csc(d)
        expected = 1000.0
        assert abs(condest_1(A) - expected) / expected < 1e-3

    def test_condest_1_dense_ndarray_input(self) -> None:
        """condest_1 accepts a dense numpy array; small case uses exact path."""
        d = np.array([1.0, 3.0, 9.0])
        A_dense = np.diag(d)
        result = condest_1(A_dense)
        assert abs(result - 9.0) < 1e-12


# ─────────────────────────────────────── condest_2 ───────────────────────────

class TestCondest2Analytical:
    """condest_2 verified against matrices with closed-form 2-norm condition numbers.

    condest_2 always converts to dense and calls np.linalg.cond(A, p=2).  These
    tests verify that the wrapping (sparse → dense conversion, shape checks) does
    not silently corrupt the input or result.
    """

    def test_identity_n20(self) -> None:
        """κ₂(I) = 1 for any size."""
        A = sparse.eye(20, format="csc")
        assert abs(condest_2(A) - 1.0) < 1e-12

    def test_scaled_identity(self) -> None:
        """κ₂(αI) = 1 for any α ≠ 0."""
        A = 3.14 * sparse.eye(15, format="csc")
        assert abs(condest_2(A) - 1.0) < 1e-12

    def test_diagonal_2x2(self) -> None:
        """D = diag([1, 5]): κ₂ = 5/1 = 5."""
        A = _diag_csc(np.array([1.0, 5.0]))
        assert abs(condest_2(A) - 5.0) < 1e-12

    def test_diagonal_n10(self) -> None:
        """D = diag([1,…,10]): κ₂ = 10/1 = 10."""
        d = np.arange(1.0, 11.0)
        A = _diag_csc(d)
        assert abs(condest_2(A) - 10.0) < 1e-12

    def test_kappa_1_and_2_agree_for_diagonal(self) -> None:
        """For a positive diagonal matrix κ₁ = κ₂ = max/min (both exact paths)."""
        d = np.array([2.0, 6.0, 12.0, 24.0])
        A = _diag_csc(d)
        kappa_1 = condest_1(A)
        kappa_2 = condest_2(A)
        expected = 24.0 / 2.0  # = 12.0
        assert abs(kappa_1 - expected) < 1e-12
        assert abs(kappa_2 - expected) < 1e-12

    def test_near_singular_gives_large_kappa(self) -> None:
        """D = diag([1e-10, 1]): κ₂ = 1e10, dominated by near-zero entry."""
        d = np.array([1e-10, 1.0])
        A = _diag_csc(d)
        result = condest_2(A)
        assert abs(result - 1e10) / 1e10 < 1e-8


# ─────────────────────────────────────── Krylov estimators ───────────────────

class TestKrylovEstimatorsAnalytical:
    """estimate_condest_1_krylov and estimate_condest_2_krylov against diagonal
    matrices with closed-form condition numbers.

    For D = diag(d₁,…,dₙ):
      * ILU factorisation is exact (L=I, U=D), so each GMRES solve converges
        in a single iteration.
      * SVDS finds σ_max = max dᵢ and σ_min = min dᵢ very reliably.

    Combining both gives an independent end-to-end check that the Krylov
    pipeline produces accurate condition numbers, not just plausible ones.
    """

    # ── 1-norm Krylov ────────────────────────────────────────────────────────

    def test_estimate_condest_1_krylov_diagonal_n200(self) -> None:
        """ILU exact ⟹ κ̂₁ matches analytical κ₁ = 200 to 0.1 %."""
        n = 200
        d = np.arange(1.0, n + 1)
        A = _diag_csc(d)
        result = estimate_condest_1_krylov(A, solver="gmres", preconditioner="ilu")
        expected = float(n)
        assert result.condition_number > 0
        assert abs(result.condition_number - expected) / expected < 1e-3

    def test_estimate_condest_1_krylov_scaled_identity(self) -> None:
        """κ̂₁(αI) must equal 1 for any scale α."""
        A = 3.0 * sparse.eye(200, format="csc")
        result = estimate_condest_1_krylov(A, solver="gmres", preconditioner="ilu")
        assert abs(result.condition_number - 1.0) < 1e-3

    def test_estimate_condest_1_krylov_two_cluster(self) -> None:
        """Cluster at 1 and 500 (n=100): κ̂₁ ≈ 500."""
        d = np.r_[np.ones(50), 500.0 * np.ones(50)]
        A = _diag_csc(d)
        result = estimate_condest_1_krylov(A, solver="gmres", preconditioner="ilu")
        expected = 500.0
        assert abs(result.condition_number - expected) / expected < 1e-3

    # ── 2-norm Krylov ────────────────────────────────────────────────────────

    def test_estimate_condest_2_krylov_diagonal_n200(self) -> None:
        """SVDS on diagonal: σ_max=200, σ_min=1 ⟹ κ̂₂ ≈ 200."""
        n = 200
        d = np.arange(1.0, n + 1)
        A = _diag_csc(d)
        result = estimate_condest_2_krylov(A)
        expected = float(n)
        assert result.condition_number > 0
        assert abs(result.condition_number - expected) / expected < 1e-3

    def test_estimate_condest_2_krylov_near_isotropic(self) -> None:
        """Near-isotropic diagonal (κ̂₂ = 2): ARPACK converges for non-degenerate spectra."""
        # Alternating 1 and 2 across 200 rows gives κ₂ = 2 exactly, with a
        # non-degenerate gap between the largest (2) and smallest (1) singular value.
        n = 200
        d = np.where(np.arange(n) % 2 == 0, 1.0, 2.0)
        A = _diag_csc(d)
        result = estimate_condest_2_krylov(A)
        assert abs(result.condition_number - 2.0) / 2.0 < 1e-3

    def test_estimate_condest_2_krylov_two_cluster(self) -> None:
        """Cluster at 1 and 500 (n=100): κ̂₂ ≈ 500."""
        d = np.r_[np.ones(50), 500.0 * np.ones(50)]
        A = _diag_csc(d)
        result = estimate_condest_2_krylov(A)
        expected = 500.0
        assert abs(result.condition_number - expected) / expected < 1e-3

    def test_krylov_1_and_2_estimates_agree_for_diagonal(self) -> None:
        """Both Krylov estimators should yield similar values for a PD diagonal."""
        n = 100
        d = np.arange(1.0, n + 1)
        A = _diag_csc(d)
        result_1 = estimate_condest_1_krylov(A, solver="gmres", preconditioner="ilu")
        result_2 = estimate_condest_2_krylov(A)
        expected = float(n)
        # Each should be within 0.5 % of the analytical value
        assert abs(result_1.condition_number - expected) / expected < 5e-3
        assert abs(result_2.condition_number - expected) / expected < 5e-3


# ─────────────────────────────────────── complex diagonal ────────────────────

class TestComplexDiagonalAnalytical:
    """Verify that complex entries are handled correctly.

    For D = diag(c₁, …, cₙ), the 1-norm and 2-norm condition numbers use
    |cᵢ| in place of cᵢ.  These tests confirm that the modulus is applied
    rather than the real part.
    """

    def test_condest_1_purely_imaginary_diagonal(self) -> None:
        """D = diag([j, 2j, 4j, 8j]): κ₁ = 8/1 = 8."""
        d = np.array([1j, 2j, 4j, 8j])
        A = sparse.diags(d, format="csc")
        result = condest_1(A)
        assert abs(result - 8.0) < 1e-10

    def test_condest_2_mixed_phase_diagonal(self) -> None:
        """D = diag([e^{jθ}, 5·e^{jφ}]): κ₂ = 5/1 = 5 regardless of phase."""
        d = np.array([np.exp(1j * 0.3), 5.0 * np.exp(1j * 1.7)])
        A = sparse.diags(d, format="csc")
        result = condest_2(A)
        assert abs(result - 5.0) < 1e-10
