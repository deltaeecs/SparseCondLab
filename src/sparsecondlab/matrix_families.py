from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import sparse


def poisson_1d_dirichlet_matrix(size: int, *, scale: float = 1.0, shift: complex = 0.0) -> sparse.csr_matrix:
    """Build a 1D Dirichlet Poisson stencil with an optional diagonal shift."""

    if size < 2:
        raise ValueError("size must be at least 2")

    main = np.full(size, 2.0 * scale, dtype=complex if shift != 0 else float)
    off = np.full(size - 1, -1.0 * scale, dtype=complex if shift != 0 else float)
    matrix = sparse.diags((off, main, off), offsets=(-1, 0, 1), format="csr")
    if shift != 0:
        matrix = matrix.astype(complex) + shift * sparse.eye(size, format="csr", dtype=complex)
    return sparse.csr_matrix(matrix)


def convection_diffusion_1d_matrix(
    size: int,
    *,
    diffusion: float = 1.0,
    convection: float = 0.3,
    reaction: float = 0.05,
) -> sparse.csr_matrix:
    """Build a nonsymmetric 1D convection-diffusion-reaction operator."""

    if size < 2:
        raise ValueError("size must be at least 2")

    main = np.full(size, 2.0 * diffusion + convection + reaction, dtype=float)
    upper = np.full(size - 1, -diffusion, dtype=float)
    lower = np.full(size - 1, -diffusion - convection, dtype=float)
    return sparse.diags((lower, main, upper), offsets=(-1, 0, 1), format="csr")


def poisson_2d_dirichlet_matrix(
    grid_size: int,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    shift: complex = 0.0,
) -> sparse.csr_matrix:
    """Build a 2D Dirichlet Poisson operator on a tensor-product grid."""

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    x_operator = poisson_1d_dirichlet_matrix(grid_size, scale=scale_x)
    y_operator = poisson_1d_dirichlet_matrix(grid_size, scale=scale_y)
    identity = sparse.eye(grid_size, format="csr")
    matrix = sparse.kron(identity, x_operator, format="csr") + sparse.kron(y_operator, identity, format="csr")
    if shift != 0:
        matrix = matrix.astype(complex) + shift * sparse.eye(matrix.shape[0], format="csr", dtype=complex)
    return sparse.csr_matrix(matrix)


def poisson_2d_condition_number_2(grid_size: int, *, scale_x: float = 1.0, scale_y: float = 1.0, shift: float = 0.0) -> float:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if shift < 0.0:
        raise ValueError("shift must be non-negative for the analytical 2-norm formula")

    angle_min = np.pi / (2.0 * (grid_size + 1))
    angle_max = grid_size * np.pi / (2.0 * (grid_size + 1))
    # The 2D Dirichlet operator is assembled on the same grid in x/y, so the
    # smallest and largest eigenvalues are sums of the corresponding 1D edge
    # eigenvalues with different anisotropic scales.
    lambda_min = shift + 4.0 * scale_x * np.sin(angle_min) ** 2 + 4.0 * scale_y * np.sin(angle_min) ** 2
    lambda_max = shift + 4.0 * scale_x * np.sin(angle_max) ** 2 + 4.0 * scale_y * np.sin(angle_max) ** 2
    return float(lambda_max / lambda_min)


def coupled_diffusion_2d_matrix(
    grid_size: int,
    *,
    primary_shift: complex = 0.08 + 0.02j,
    secondary_shift: complex = 0.05 + 0.03j,
    forward_coupling: complex = 0.04 + 0.01j,
    backward_coupling: complex = 0.015 - 0.005j,
) -> sparse.csr_matrix:
    """Build a complex block-coupled diffusion system on a shared 2D grid."""

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    primary = poisson_2d_dirichlet_matrix(grid_size, scale_x=1.0, scale_y=0.45, shift=primary_shift).astype(complex)
    secondary = poisson_2d_dirichlet_matrix(grid_size, scale_x=0.7, scale_y=1.2, shift=secondary_shift).astype(complex)
    identity = sparse.eye(primary.shape[0], format="csr", dtype=complex)

    blocks = [
        [primary, forward_coupling * identity],
        [backward_coupling * identity, 1.35 * secondary],
    ]
    return sparse.csr_matrix(sparse.bmat(blocks, format="csr", dtype=complex))


def anisotropic_shifted_poisson_2d_matrix(grid_size: int) -> sparse.csr_matrix:
    """Build a sparse anisotropic shifted Poisson operator."""

    return poisson_2d_dirichlet_matrix(grid_size, scale_x=1.0, scale_y=0.2, shift=0.15)


def convection_diffusion_2d_matrix(grid_size: int) -> sparse.csr_matrix:
    """Build a real nonsymmetric 2D convection-diffusion operator."""

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    x_operator = convection_diffusion_1d_matrix(grid_size, diffusion=1.0, convection=0.35, reaction=0.04)
    y_operator = convection_diffusion_1d_matrix(grid_size, diffusion=0.65, convection=0.12, reaction=0.02)
    identity = sparse.eye(grid_size, format="csr")
    matrix = sparse.kron(identity, x_operator, format="csr") + sparse.kron(y_operator, identity, format="csr")
    return sparse.csr_matrix(matrix)


def _family_builder(family: str) -> Callable[[int], sparse.csr_matrix]:
    normalized = family.strip().lower()
    if normalized == "poisson-2d":
        return poisson_2d_dirichlet_matrix
    if normalized == "anisotropic-poisson-2d":
        return anisotropic_shifted_poisson_2d_matrix
    if normalized == "coupled-diffusion-2d":
        return coupled_diffusion_2d_matrix
    if normalized == "convection-diffusion-2d":
        return convection_diffusion_2d_matrix
    raise ValueError(f"unsupported generated matrix family: {family}")


def build_generated_family_matrix(family: str, grid_size: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(_family_builder(family)(grid_size))


def build_generated_family_samples(
    *,
    family: str,
    min_grid_size: int,
    max_grid_size: int,
    growth_factor: int = 2,
    max_dof: int = 1_000_000,
) -> list[tuple[str, sparse.csr_matrix]]:
    if min_grid_size < 2:
        raise ValueError("min_grid_size must be at least 2")
    if max_grid_size < min_grid_size:
        raise ValueError("max_grid_size must be greater than or equal to min_grid_size")
    if growth_factor < 2:
        raise ValueError("growth_factor must be at least 2")
    if max_dof < 1:
        raise ValueError("max_dof must be positive")

    samples: list[tuple[str, sparse.csr_matrix]] = []
    grid_size = min_grid_size

    while grid_size <= max_grid_size:
        matrix = build_generated_family_matrix(family, grid_size)
        if matrix.shape[0] > max_dof:
            break
        samples.append((f"generated:{family}:grid={grid_size}", matrix))
        next_grid_size = grid_size * growth_factor
        if next_grid_size == grid_size:
            break
        grid_size = next_grid_size

    return samples