# Progress

## 2026-04-03
- Inspected the repository and confirmed it is an early-stage workspace with only a README.
- Added a project-wide workspace instruction file at [.github/copilot-instructions.md](.github/copilot-instructions.md).
- Created lightweight planning files to track the bootstrap task.
- Verified that no competing AGENTS.md or other workspace instruction file was introduced.
- Added the initial src-layout package, shard assembly, 1-norm condition estimation, and compare CLI.
- Added real-matrix pytest fixtures and tests that exercise Matrix Market parsing and shard assembly.
- Fixed a failing condest test assertion after the first real pytest run.
- Verified the suite with `python -m pytest` after the fix.
- Formalized the versioned shard manifest schema and added iterative benchmark/report output.
- Added a second real Matrix Market sample and a reusable block-diagonal shard manifest fixture.
- Added a public `condest_2` interface and verified it against `np.linalg.cond(..., p=2)`.
- Removed the hidden fallback path from `condest_2` so the 2-norm computation is explicit and single-path.
- Added Krylov-based condition-number estimators, log-log benchmark trend fitting, and the `scl-benchmark` CLI.
- Verified the full suite with `python -m pytest` after the Krylov benchmark feature.
- Re-ran the full suite after cleanup; all 12 tests still pass.
- Generated the final Krylov benchmark report and fitted chart under `docs/reports/`.
- Replaced the report benchmark source with a generated anisotropic Poisson family and regenerated the chart/report.
- Added structured sparse matrix family generators plus stronger correctness-validation cases covering Poisson and coupled complex systems.
- Verified the full suite after the realistic-family benchmark changes; all 46 tests pass.
- Added a nonsymmetric `convection-diffusion-2d` generated family for more realistic non-SPD-style benchmark coverage.
- Added multi-family Krylov benchmark suite APIs, markdown rendering, and combined figure output.
- Generated a single suite report covering anisotropic Poisson, coupled diffusion, and convection-diffusion families.
- Expanded `.gitignore` so Python caches, egg-info, build outputs, and local environments stay out of version control.
- Re-ran the full suite after the suite-report and review hardening changes; all 52 tests pass.
- Completed three consecutive project-wide review rounds with no new findings after the last fixes.