from __future__ import annotations

from datetime import date
from pathlib import Path

from sparsecondlab.krylov import (
    KrylovBenchmarkReport,
    KrylovBenchmarkSuiteReport,
    KrylovSampleRecord,
    build_generated_family_krylov_benchmark_report,
    build_generated_family_krylov_benchmark_suite,
    build_local_limit_krylov_benchmark_report,
)
from sparsecondlab.krylov_report import (
    build_correctness_validation_records,
    render_krylov_benchmark_markdown,
    render_krylov_benchmark_suite_markdown,
    write_krylov_benchmark_suite_figure,
)


def test_correctness_validation_records_have_small_relative_error():
    records = build_correctness_validation_records()

    assert records
    assert any(record.case_name == "poisson_2d_dirichlet_8x8" for record in records)
    assert any(record.case_name == "coupled_complex_diffusion_2d_6x6" for record in records)
    assert max(record.relative_error for record in records) < 1e-2


def test_rendered_benchmark_markdown_includes_validation_section():
    report = build_local_limit_krylov_benchmark_report(
        [Path("tests/data/real_example.mtx"), Path("tests/data/symmetric_example.mtx")],
        norms=(1, 2),
        solver="gmres",
        preconditioner="ilu",
        max_dof=35,
        max_scale=128,
        max_sample_seconds=60.0,
        target_dof=1_000_000,
    )

    markdown = render_krylov_benchmark_markdown(report, generated_on=date(2026, 4, 3))

    assert "## Correctness Validation" in markdown
    assert "poisson_2d_dirichlet_8x8" in markdown
    assert "auto-scale" in markdown


def test_rendered_benchmark_markdown_can_include_generated_family_name():
    report = build_generated_family_krylov_benchmark_report(
        family="coupled-diffusion-2d",
        norms=(1, 2),
        solver="gmres",
        preconditioner="ilu",
        min_grid_size=4,
        max_grid_size=8,
        target_dof=1_000_000,
        max_sample_seconds=60.0,
    )

    markdown = render_krylov_benchmark_markdown(report, generated_on=date(2026, 4, 3))

    assert "coupled-diffusion-2d" in markdown


def test_rendered_suite_markdown_includes_all_family_sections():
    suite = build_generated_family_krylov_benchmark_suite(
        families=("anisotropic-poisson-2d", "convection-diffusion-2d"),
        norms=(1, 2),
        solver="gmres",
        preconditioner="ilu",
        min_grid_size=4,
        max_grid_size=8,
        target_dof=1_000_000,
        max_sample_seconds=60.0,
    )

    markdown = render_krylov_benchmark_suite_markdown(suite, generated_on=date(2026, 4, 3))

    assert "# Krylov Benchmark Suite Report" in markdown
    assert "anisotropic-poisson-2d" in markdown
    assert "convection-diffusion-2d" in markdown


def test_suite_figure_writer_tolerates_missing_trend_models(tmp_path):
    report = KrylovBenchmarkReport(
        target_dof=1_000_000,
        samples=[
            KrylovSampleRecord(
                input="generated:test:grid=4",
                dof=16,
                nnz=64,
                norm_order=1,
                condition_number=10.0,
                elapsed_seconds=0.01,
                algorithm="onenormest+gmres",
                preconditioner="ilu",
                inner_iterations=6,
                solve_calls=6,
            )
        ],
        trend_models=[],
        sample_strategy="generated-family",
        stop_reason="max_grid_size",
        family_name="test-family",
        validation_records=[],
    )
    suite = KrylovBenchmarkSuiteReport(target_dof=1_000_000, reports=[report], validation_records=[])

    output_path = tmp_path / "suite.png"
    written = write_krylov_benchmark_suite_figure(suite, output_path)

    assert written == output_path
    assert output_path.exists()


def test_suite_markdown_tolerates_missing_family_name():
    report = KrylovBenchmarkReport(
        target_dof=1_000_000,
        samples=[],
        trend_models=[],
        sample_strategy="generated-family",
        stop_reason="max_grid_size",
        family_name=None,
        validation_records=[],
    )
    suite = KrylovBenchmarkSuiteReport(target_dof=1_000_000, reports=[report], validation_records=[])

    markdown = render_krylov_benchmark_suite_markdown(suite, generated_on=date(2026, 4, 3))

    assert "## Family: unknown" in markdown