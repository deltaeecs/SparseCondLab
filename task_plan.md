# Task Plan

## Goal
Bootstrap the initial SparseCondLab package, validate it against real matrix fixtures, and add Krylov benchmark trend prediction for 1e6 DOF extrapolation.

## Phases
- [x] Discover existing conventions and documentation
- [x] Read workspace-customization guidance
- [x] Draft and add workspace instructions
- [x] Review the generated instruction file for clarity and overlap
- [x] Confirm no extra workspace instruction file was added
- [x] Add package scaffold and project metadata
- [x] Implement Matrix Market and NPZ loading
- [x] Implement shard assembly and condest wrapper
- [x] Add CLI compare command
- [x] Formalize shard manifest schema
- [x] Add iterative benchmark runner and flat report output
- [x] Add extra real matrix sample and shard manifest fixture
- [x] Add condest_2 interface and test coverage
- [x] Remove hidden fallback from condest_2
- [x] Run tests against the real matrix fixture
- [x] Triage and fix any test failures
- [x] Add Krylov condition-number estimators
- [x] Add log-log benchmark trend fitting
- [x] Add `scl-benchmark` CLI entry point
- [x] Document benchmark extrapolation workflow
- [x] Verify the full test suite after the Krylov benchmark feature
- [x] Replace tiny block benchmark samples with realistic generated sparse families
- [x] Strengthen correctness validation with PDE-like and coupled sparse matrices

## Notes
- Tests should use the checked-in Matrix Market fixture rather than only synthetic arrays.
- Keep the initial implementation small and deterministic.
- The compare command now emits flat CSV/JSON records with per-solver benchmark metrics.
- `condest_2` now mirrors `condest_1` as a public API and compare output field.
- `condest_2` now uses a single exact dense path instead of switching algorithms behind the scenes.
- The Krylov benchmark flow measures smaller FEM samples and extrapolates runtime to 1e6 DOF instead of running that size directly.
- The benchmark flow now supports generated PDE-like matrix families (`anisotropic-poisson-2d`, `coupled-diffusion-2d`) in addition to explicit input files.
- Correctness validation now mixes analytical Poisson references with exact dense-reference checks on moderate structured sparse systems.

## Errors Encountered
- `test_condest.py` used an incorrect boolean assertion against the numeric result; corrected to check finiteness separately.