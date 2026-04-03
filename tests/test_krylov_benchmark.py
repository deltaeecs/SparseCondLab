from __future__ import annotations

import json
from pathlib import Path

from sparsecondlab.condest import condest_1, condest_2
from sparsecondlab.krylov import (
    build_krylov_benchmark_report,
    build_generated_family_krylov_benchmark_report,
    build_generated_family_krylov_benchmark_suite,
    build_local_limit_krylov_benchmark_report,
    estimate_condest_1_krylov,
    estimate_condest_2_krylov,
    fit_time_trend,
    predict_time_from_trend,
)
from sparsecondlab.io import load_matrix


def test_krylov_condition_estimators_run_on_real_matrices():
    matrix = load_matrix(Path("tests/data/real_example.mtx"))

    result_1 = estimate_condest_1_krylov(matrix, solver="gmres", preconditioner="ilu")
    result_2 = estimate_condest_2_krylov(matrix)

    exact_1 = condest_1(matrix)
    exact_2 = condest_2(matrix)

    assert result_1.norm_order == 1
    assert result_1.condition_number > 0
    assert result_1.elapsed_seconds >= 0
    assert abs(result_1.condition_number - exact_1) / exact_1 < 1e-6

    assert result_2.norm_order == 2
    assert result_2.condition_number > 0
    assert result_2.elapsed_seconds >= 0
    assert abs(result_2.condition_number - exact_2) / exact_2 < 1e-6


def test_time_trend_prediction_uses_log_log_fit():
    samples = [
        {"dof": 1000, "elapsed_seconds": 1.0, "norm_order": 1},
        {"dof": 3000, "elapsed_seconds": 3.0, "norm_order": 1},
        {"dof": 9000, "elapsed_seconds": 9.0, "norm_order": 1},
    ]

    model = fit_time_trend(samples, x_key="dof", y_key="elapsed_seconds")
    predicted = predict_time_from_trend(model, 27000)

    assert abs(predicted - 27.0) < 1e-9


def test_krylov_benchmark_report_includes_trend_predictions():
    report = build_krylov_benchmark_report(
        [Path("tests/data/real_example.mtx"), Path("tests/data/symmetric_example.mtx")],
        norms=(1, 2),
        solver="gmres",
        preconditioner="ilu",
        target_dof=1_000_000,
    )

    assert report.target_dof == 1_000_000
    assert report.sample_count == 4
    assert len(report.trend_models) == 2
    assert all(model.predicted_seconds_at_target_dof > 0 for model in report.trend_models)


def test_local_limit_benchmark_auto_scales_until_dof_cap():
    report = build_local_limit_krylov_benchmark_report(
        [Path("tests/data/real_example.mtx"), Path("tests/data/symmetric_example.mtx")],
        norms=(1, 2),
        solver="gmres",
        preconditioner="ilu",
        target_dof=1_000_000,
        max_dof=35,
        max_scale=128,
        max_sample_seconds=60.0,
        growth_factor=2,
    )

    dofs = sorted({sample.dof for sample in report.samples})

    assert report.sample_strategy == "auto-scale"
    assert report.stop_reason == "max_dof"
    assert dofs == [10, 20]
    assert report.validation_records


def test_generated_family_benchmark_uses_structured_pde_like_matrices():
    report = build_generated_family_krylov_benchmark_report(
        family="coupled-diffusion-2d",
        norms=(1, 2),
        solver="gmres",
        preconditioner="ilu",
        min_grid_size=4,
        max_grid_size=8,
        growth_factor=2,
        target_dof=1_000_000,
        max_sample_seconds=60.0,
    )

    dofs = sorted({sample.dof for sample in report.samples})

    assert report.sample_strategy == "generated-family"
    assert report.stop_reason in {"max_grid_size", "max_sample_seconds", "max_dof"}
    assert dofs == [32, 128]
    assert all("generated:coupled-diffusion-2d" in sample.input for sample in report.samples)


def test_generated_family_benchmark_suite_aggregates_multiple_families():
    suite = build_generated_family_krylov_benchmark_suite(
        families=("anisotropic-poisson-2d", "convection-diffusion-2d"),
        norms=(1, 2),
        solver="gmres",
        preconditioner="ilu",
        min_grid_size=4,
        max_grid_size=8,
        growth_factor=2,
        target_dof=1_000_000,
        max_sample_seconds=60.0,
    )

    assert suite.family_count == 2
    assert [report.family_name for report in suite.reports] == ["anisotropic-poisson-2d", "convection-diffusion-2d"]
    assert all(report.trend_models for report in suite.reports)
