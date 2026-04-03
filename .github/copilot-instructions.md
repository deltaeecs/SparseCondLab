# SparseCondLab Project Guidelines

## Code Style
- Use Python type hints on new or changed code.
- Prefer small, testable functions with explicit sparse-matrix types and shapes.
- Keep numerical routines deterministic and avoid hidden global state.
- Follow the existing project style in [README.md](../README.md) for project scope and supported matrix inputs.

## Architecture
- SparseCondLab is an open-source toolkit for comparing conditioning across large sparse complex matrices.
- Keep the pipeline modular: input adapters, shard assembly, conditioning metrics, iterative benchmarks, and report output.
- Treat Matrix Market, NPZ, SciPy sparse, and MUMPS-style shard input as first-class interfaces.
- Keep optional HPC integrations such as PETSc, SLEPc, and MPI isolated from the core path.

## Build and Test
- Core dependencies currently documented in [README.md](../README.md): `scipy`, `numpy`, `pandas`, and `matplotlib[all]`.
- Optional HPC dependencies: `petsc4py`, `slepc4py`, and `mpi4py`.
- When adding tests, prefer `pytest` and keep IO coverage alongside parser and assembly logic.
- If you introduce new commands or packaging metadata, document the exact invocation in the README.

## Conventions
- Do not introduce closed-source tooling or dependencies.
- Define shard schemas explicitly before implementing parsers or assembly helpers.
- Prefer CSV or JSON for benchmark outputs and keep them stable across runs.
- Link to dedicated docs instead of duplicating long explanations in agent instructions.
- Add tests for matrix IO, shard assembly, and numerical edge cases when those areas change.