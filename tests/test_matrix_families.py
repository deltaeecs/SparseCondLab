from __future__ import annotations

import numpy as np

from sparsecondlab.matrix_families import (
    build_generated_family_samples,
    build_generated_family_matrix,
    coupled_diffusion_2d_matrix,
    poisson_2d_condition_number_2,
    poisson_2d_dirichlet_matrix,
)


def test_poisson_2d_condition_number_matches_dense_reference():
    matrix = poisson_2d_dirichlet_matrix(4)
    expected = poisson_2d_condition_number_2(4)
    measured = float(np.linalg.cond(matrix.toarray(), p=2))

    assert abs(measured - expected) / expected < 1e-12


def test_coupled_diffusion_matrix_is_sparse_complex_and_square():
    matrix = coupled_diffusion_2d_matrix(5)

    assert matrix.shape == (50, 50)
    assert matrix.nnz > 0
    assert np.iscomplexobj(matrix.data)


def test_generated_family_samples_grow_with_grid_size():
    samples = build_generated_family_samples(
        family="coupled-diffusion-2d",
        min_grid_size=4,
        max_grid_size=16,
        growth_factor=2,
        max_dof=10_000,
    )

    names = [name for name, _matrix in samples]
    dofs = [matrix.shape[0] for _name, matrix in samples]

    assert names[0].startswith("generated:coupled-diffusion-2d:grid=4")
    assert dofs == sorted(dofs)
    assert len(dofs) >= 3


def test_convection_diffusion_family_is_square_sparse_and_nonsymmetric():
    matrix = build_generated_family_matrix("convection-diffusion-2d", 6)

    assert matrix.shape == (36, 36)
    assert matrix.nnz > 0
    assert (matrix - matrix.transpose()).nnz > 0