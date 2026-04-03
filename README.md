# SparseCondLab

SparseCondLab is a toolkit for comparing the conditioning of large sparse complex matrices arising from different basis functions. This project explicitly avoids closed-source tools, promoting openness and collaboration in scientific computing.

## Features

- Supports various matrix input interfaces:
  - SciPy sparse matrices
  - Matrix Market files
  - NPZ files
  - MUMPS-style distributed matrix input

## Installation

To install the project dependencies, use the following command:

```bash
pip install scipy numpy pandas matplotlib[all]  
# Optional dependencies
pip install petsc4py slepc4py mpi4py
```

## Quick Start

After installing dependencies, run the compare CLI on one or more matrix files or shard manifests:

```bash
scl-compare path/to/matrix.mtx path/to/other_matrix.npz --methods gmres,bicgstab --format csv
```

Shard manifests are versioned JSON files with this shape:

```json
{
  "format": "sparsecondlab-shards",
  "version": 1,
  "shape": [10, 10],
  "shards": [
    {"path": "top.mtx", "row_offset": 0, "col_offset": 0},
    {"path": "bottom.mtx", "row_offset": 5, "col_offset": 5}
  ]
}
```

The compare command emits flat CSV or JSON rows with matrix dimensions, `condest_1`, and iterative benchmark results for each requested solver.

Both 1-norm and 2-norm condition numbers are available through the library API as `condest_1` and `condest_2`.

## Krylov Benchmark

Use the benchmark CLI to measure sampled FEM matrices and fit a time trend that can be extrapolated to 1e6 DOF:

```bash
scl-benchmark path/to/sample_1.mtx path/to/sample_2.mtx --solver gmres --preconditioner ilu --norms 1,2 --predict-dof 1000000 --format json
```

The benchmark does not run 1e6 DOF directly. It measures a family of smaller matrices, fits a log-log time trend for each norm order, and reports the predicted runtime at the target DOF.

For more realistic performance sampling, use a generated PDE-like family instead of repeating tiny input blocks. The `anisotropic-poisson-2d` family gives a sparse stiffness-like trend with increasing grid size, `coupled-diffusion-2d` adds complex block coupling, and `convection-diffusion-2d` provides a nonsymmetric transport-dominated sparse system.

| Family | Model class | Matrix character |
| --- | --- | --- |
| `poisson-2d` | isotropic Dirichlet diffusion | symmetric positive definite |
| `anisotropic-poisson-2d` | anisotropic shifted diffusion | symmetric sparse stiffness-like |
| `coupled-diffusion-2d` | complex block-coupled diffusion | complex nonsymmetric block system |
| `convection-diffusion-2d` | transport-dominated convection-diffusion | real nonsymmetric sparse system |

```bash
scl-benchmark \
  --generated-family anisotropic-poisson-2d \
  --solver gmres \
  --preconditioner ilu \
  --norms 1,2 \
  --predict-dof 1000000 \
  --min-grid-size 8 \
  --max-grid-size 1024 \
  --max-dof 2000000 \
  --max-sample-seconds 2.0 \
  --report-path docs/reports/2026-04-03-krylov-benchmark-report.md \
  --figure-path docs/reports/figures/krylov_benchmark_trend.png \
  --format json
```

If you still want to benchmark explicit input files, `--auto-scale` remains available and will build a larger block-diagonal family from the supplied matrices until the configured local limit is reached.

To produce one aggregated report across multiple generated families, repeat `--generated-family`. The CLI will emit a suite-level JSON report, optional markdown summary, and optional combined figure.

```bash
scl-benchmark \
  --generated-family anisotropic-poisson-2d \
  --generated-family coupled-diffusion-2d \
  --generated-family convection-diffusion-2d \
  --solver gmres \
  --preconditioner ilu \
  --norms 1,2 \
  --predict-dof 1000000 \
  --min-grid-size 4 \
  --max-grid-size 16 \
  --max-sample-seconds 5.0 \
  --report-path docs/reports/2026-04-03-krylov-benchmark-suite-report.md \
  --figure-path docs/reports/figures/krylov_benchmark_suite_trend.png \
  --format json
```

The generated report now embeds correctness-validation cases alongside the performance trend data. Those rows include both closed-form PDE-style references and exact dense-reference checks on moderate structured sparse systems, so the report includes both performance and accuracy evidence.

## Development

The test suite uses `pytest` and includes real matrix fixtures rather than only synthetic examples.

```bash
pytest
```
