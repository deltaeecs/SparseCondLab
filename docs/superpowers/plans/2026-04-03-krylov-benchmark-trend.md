# Krylov Benchmark Trend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Krylov-based condition-number benchmark flows that measure 1-norm and 2-norm computation costs on sampled FEM sparse matrices and fit a time trend that can extrapolate to 1e6 DOF.

**Architecture:** Introduce a dedicated Krylov benchmark module for condition-number estimation, a small trend-fitting module for log-log time extrapolation, and a CLI entry point that runs the benchmark across sample matrices and emits JSON or CSV trend reports. Keep exact dense condition-number helpers separate so the new benchmark path is explicit and does not silently fall back to another algorithm.

**Tech Stack:** Python 3.11+, NumPy, SciPy sparse linear algebra, pandas, pytest.

---

### Task 1: Krylov estimators and trend fitting

**Files:** `src/sparsecondlab/krylov.py`, `tests/test_krylov_benchmark.py`

- [x] Added `estimate_condest_1_krylov` and `estimate_condest_2_krylov`.
- [x] Added `fit_time_trend` and `predict_time_from_trend`.
- [x] Verified the estimators on real matrix fixtures from `tests/data/real_example.mtx` and `tests/data/symmetric_example.mtx`.

### Task 2: Benchmark report and CLI

**Files:** `src/sparsecondlab/krylov.py`, `src/sparsecondlab/cli.py`, `tests/test_benchmark_cli.py`, `pyproject.toml`

- [x] Added `build_krylov_benchmark_report`.
- [x] Added `scl-benchmark` console script and `benchmark_main`.
- [x] Emitted JSON benchmark reports and CSV sample rows with trend predictions.

### Task 3: Documentation and validation

**Files:** `README.md`, `task_plan.md`, `progress.md`

- [x] Documented the benchmark workflow and 1e6 DOF extrapolation policy.
- [x] Ran `python -m pytest` and confirmed all tests pass.