from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from scipy import sparse

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .condest import condest_1, condest_2
from .krylov import (
    KrylovBenchmarkReport,
    KrylovBenchmarkSuiteReport,
    KrylovValidationRecord,
    estimate_condest_1_krylov,
    estimate_condest_2_krylov,
)
from .matrix_families import (
    anisotropic_shifted_poisson_2d_matrix,
    convection_diffusion_2d_matrix,
    coupled_diffusion_2d_matrix,
    poisson_2d_condition_number_2,
    poisson_2d_dirichlet_matrix,
)


def _analytical_diagonal_condition(values: np.ndarray) -> float:
    magnitudes = np.abs(np.asarray(values))
    return float(magnitudes.max() / magnitudes.min())


def _relative_error(expected: float, measured: float) -> float:
    return float(abs(measured - expected) / expected)


def build_correctness_validation_records() -> list[KrylovValidationRecord]:
    """Build shared correctness-validation rows used by benchmark reports."""

    records: list[KrylovValidationRecord] = []

    diagonal_values = np.arange(1.0, 129.0)
    diagonal_matrix = sparse.diags(diagonal_values, format="csc")
    diagonal_expected = _analytical_diagonal_condition(diagonal_values)
    for estimator, norm_order, measured in [
        ("condest_1", 1, condest_1(diagonal_matrix)),
        ("condest_2", 2, condest_2(diagonal_matrix)),
        ("estimate_condest_1_krylov", 1, estimate_condest_1_krylov(diagonal_matrix, solver="gmres", preconditioner="ilu").condition_number),
        ("estimate_condest_2_krylov", 2, estimate_condest_2_krylov(diagonal_matrix).condition_number),
    ]:
        records.append(
            KrylovValidationRecord(
                case_name="diagonal_1_to_128",
                estimator=estimator,
                norm_order=norm_order,
                dof=int(diagonal_matrix.shape[0]),
                expected_condition_number=diagonal_expected,
                measured_condition_number=float(measured),
                relative_error=_relative_error(diagonal_expected, float(measured)),
            )
        )

    poisson_matrix = poisson_2d_dirichlet_matrix(8)
    poisson_expected = poisson_2d_condition_number_2(8)
    for estimator, measured in [
        ("condest_2", condest_2(poisson_matrix)),
        ("estimate_condest_2_krylov", estimate_condest_2_krylov(poisson_matrix).condition_number),
    ]:
        records.append(
            KrylovValidationRecord(
                case_name="poisson_2d_dirichlet_8x8",
                estimator=estimator,
                norm_order=2,
                dof=int(poisson_matrix.shape[0]),
                expected_condition_number=poisson_expected,
                measured_condition_number=float(measured),
                relative_error=_relative_error(poisson_expected, float(measured)),
            )
        )

    anisotropic_matrix = anisotropic_shifted_poisson_2d_matrix(6)
    anisotropic_dense = anisotropic_matrix.toarray()
    anisotropic_reference_1 = float(np.linalg.cond(anisotropic_dense, p=1))
    anisotropic_reference_2 = float(np.linalg.cond(anisotropic_dense, p=2))
    for estimator, norm_order, expected, measured in [
        ("condest_1", 1, anisotropic_reference_1, condest_1(anisotropic_matrix)),
        ("condest_2", 2, anisotropic_reference_2, condest_2(anisotropic_matrix)),
        ("estimate_condest_2_krylov", 2, anisotropic_reference_2, estimate_condest_2_krylov(anisotropic_matrix).condition_number),
    ]:
        records.append(
            KrylovValidationRecord(
                case_name="anisotropic_shifted_poisson_2d_6x6",
                estimator=estimator,
                norm_order=norm_order,
                dof=int(anisotropic_matrix.shape[0]),
                expected_condition_number=expected,
                measured_condition_number=float(measured),
                relative_error=_relative_error(expected, float(measured)),
            )
        )

    coupled_matrix = coupled_diffusion_2d_matrix(6)
    coupled_dense = coupled_matrix.toarray()
    coupled_reference_1 = float(np.linalg.cond(coupled_dense, p=1))
    coupled_reference_2 = float(np.linalg.cond(coupled_dense, p=2))
    for estimator, norm_order, expected, measured in [
        ("condest_1", 1, coupled_reference_1, condest_1(coupled_matrix)),
        ("condest_2", 2, coupled_reference_2, condest_2(coupled_matrix)),
        ("estimate_condest_1_krylov", 1, coupled_reference_1, estimate_condest_1_krylov(coupled_matrix, solver="gmres", preconditioner="ilu").condition_number),
        ("estimate_condest_2_krylov", 2, coupled_reference_2, estimate_condest_2_krylov(coupled_matrix).condition_number),
    ]:
        records.append(
            KrylovValidationRecord(
                case_name="coupled_complex_diffusion_2d_6x6",
                estimator=estimator,
                norm_order=norm_order,
                dof=int(coupled_matrix.shape[0]),
                expected_condition_number=expected,
                measured_condition_number=float(measured),
                relative_error=_relative_error(expected, float(measured)),
            )
        )

    convection_matrix = convection_diffusion_2d_matrix(6)
    convection_dense = convection_matrix.toarray()
    convection_reference_1 = float(np.linalg.cond(convection_dense, p=1))
    convection_reference_2 = float(np.linalg.cond(convection_dense, p=2))
    for estimator, norm_order, expected, measured in [
        ("condest_1", 1, convection_reference_1, condest_1(convection_matrix)),
        ("condest_2", 2, convection_reference_2, condest_2(convection_matrix)),
        ("estimate_condest_1_krylov", 1, convection_reference_1, estimate_condest_1_krylov(convection_matrix, solver="gmres", preconditioner="ilu").condition_number),
        ("estimate_condest_2_krylov", 2, convection_reference_2, estimate_condest_2_krylov(convection_matrix).condition_number),
    ]:
        records.append(
            KrylovValidationRecord(
                case_name="convection_diffusion_2d_6x6",
                estimator=estimator,
                norm_order=norm_order,
                dof=int(convection_matrix.shape[0]),
                expected_condition_number=expected,
                measured_condition_number=float(measured),
                relative_error=_relative_error(expected, float(measured)),
            )
        )

    return records


def _plot_report_norm(axis: plt.Axes, report: KrylovBenchmarkReport, norm_order: int) -> None:
    samples = sorted((sample for sample in report.samples if sample.norm_order == norm_order), key=lambda sample: sample.dof)
    if not samples:
        axis.set_axis_off()
        return

    trend_by_norm = {model.norm_order: model for model in report.trend_models}
    x_values = [sample.dof for sample in samples]
    y_values = [sample.elapsed_seconds for sample in samples]
    colors = {1: "#0b6efd", 2: "#d63384"}
    color = colors.get(norm_order, "#198754")
    axis.loglog(x_values, y_values, "o", color=color, label="Measured")

    model = trend_by_norm.get(norm_order)
    if model is None:
        axis.set_title(f"{report.family_name or report.sample_strategy} / {norm_order}-norm")
        axis.set_xlabel("DOF")
        axis.set_ylabel("Elapsed seconds")
        axis.grid(True, which="both", linestyle=":", linewidth=0.6)
        return

    fit_x = [min(x_values), report.target_dof]
    fit_y = [model.predicted_seconds_at_target_dof * (x_value / report.target_dof) ** model.slope for x_value in fit_x]
    axis.loglog(fit_x, fit_y, "-", color=color, label=f"Fit: slope={model.slope:.3f}, R²={model.r_squared:.3f}")
    axis.axvline(report.target_dof, color="#666666", linestyle="--", linewidth=1)
    axis.set_title(f"{report.family_name or report.sample_strategy} / {norm_order}-norm")
    axis.set_xlabel("DOF")
    axis.set_ylabel("Elapsed seconds")
    axis.grid(True, which="both", linestyle=":", linewidth=0.6)
    axis.legend(fontsize=8)


def write_krylov_benchmark_figure(report: KrylovBenchmarkReport, output_path: str | Path) -> Path:
    """Write a figure for a single benchmark report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    norm_orders = sorted(set(sample.norm_order for sample in report.samples))
    figure, axes = plt.subplots(1, len(norm_orders), figsize=(6 * len(norm_orders), 5), constrained_layout=True)
    axes_sequence = [axes] if len(norm_orders) == 1 else list(axes)
    for axis, norm_order in zip(axes_sequence, norm_orders):
        _plot_report_norm(axis, report, norm_order)

    figure.suptitle(f"Krylov benchmark trend ({report.sample_strategy}, stop={report.stop_reason})")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def write_krylov_benchmark_suite_figure(suite: KrylovBenchmarkSuiteReport, output_path: str | Path) -> Path:
    """Write one combined figure for a multi-family benchmark suite."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    norm_orders = sorted({sample.norm_order for report in suite.reports for sample in report.samples})
    figure, axes = plt.subplots(
        len(suite.reports),
        len(norm_orders),
        figsize=(6 * len(norm_orders), 4.5 * len(suite.reports)),
        constrained_layout=True,
        squeeze=False,
    )

    for row_index, report in enumerate(suite.reports):
        for column_index, norm_order in enumerate(norm_orders):
            _plot_report_norm(axes[row_index][column_index], report, norm_order)

    figure.suptitle(f"Krylov benchmark suite trend ({suite.family_count} families)")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _render_samples_table(report: KrylovBenchmarkReport) -> str:
    rows = ["| Input | DOF | NNZ | Norm | Time (s) | Cond. |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for sample in sorted(report.samples, key=lambda item: (item.norm_order, item.dof, item.input)):
        input_name = sample.input.replace("\\", "/")
        rows.append(
            f"| {input_name} | {sample.dof} | {sample.nnz} | {sample.norm_order} | {sample.elapsed_seconds:.6f} | {sample.condition_number:.6g} |"
        )
    return "\n".join(rows)


def _render_trend_table(report: KrylovBenchmarkReport) -> str:
    rows = ["| Norm | Samples | Slope | R² | Predicted time at target DOF |", "| --- | ---: | ---: | ---: | ---: |"]
    for model in sorted(report.trend_models, key=lambda item: item.norm_order):
        rows.append(
            f"| {model.norm_order}-norm | {model.sample_count} | {model.slope:.4f} | {model.r_squared:.4f} | {model.predicted_seconds_at_target_dof:.6f} s |"
        )
    return "\n".join(rows)


def _render_validation_table(records: Sequence[KrylovValidationRecord]) -> str:
    rows = [
        "| Case | Estimator | Norm | DOF | Expected κ | Measured κ | Relative error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in sorted(records, key=lambda item: (item.case_name, item.estimator)):
        rows.append(
            f"| {record.case_name} | {record.estimator} | {record.norm_order} | {record.dof} | {record.expected_condition_number:.6g} | {record.measured_condition_number:.6g} | {record.relative_error:.3e} |"
        )
    return "\n".join(rows)


def _render_suite_summary_table(suite: KrylovBenchmarkSuiteReport) -> str:
    rows = [
        "| Family | Norm | Samples | Slope | R² | Predicted time at target DOF |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in suite.reports:
        family_name = report.family_name or "unknown"
        for model in sorted(report.trend_models, key=lambda item: item.norm_order):
            rows.append(
                f"| {family_name} | {model.norm_order} | {model.sample_count} | {model.slope:.4f} | {model.r_squared:.4f} | {model.predicted_seconds_at_target_dof:.6f} s |"
            )
    return "\n".join(rows)


def render_krylov_benchmark_markdown(
    report: KrylovBenchmarkReport,
    *,
    generated_on: date | None = None,
    figure_reference: str | None = None,
) -> str:
    """Render a markdown report for a single benchmark run."""

    stamp = generated_on or date.today()
    observed_dofs = sorted({sample.dof for sample in report.samples})
    validation_records = list(report.validation_records)
    max_validation_error = max((record.relative_error for record in validation_records), default=0.0)

    sections = [
        "# Krylov Benchmark Report",
        "",
        f"**Date:** {stamp.isoformat()}",
        "",
        "## Benchmark Setup",
        "",
        f"- Sampling strategy: {report.sample_strategy}",
        f"- Matrix family: {report.family_name or 'explicit-inputs'}",
        f"- Stop reason: {report.stop_reason}",
        f"- Observed DOF values: {', '.join(str(value) for value in observed_dofs)}",
        f"- Target DOF for extrapolation: {report.target_dof}",
        "",
        "## Measured Samples",
        "",
        _render_samples_table(report),
        "",
        "## Fitted Results",
        "",
        _render_trend_table(report),
        "",
        "## Correctness Validation",
        "",
        "The benchmark report now carries correctness-validation cases alongside performance data. The table below mixes closed-form PDE-style references (for example Dirichlet Poisson operators) with exact dense-reference checks on moderate structured sparse systems such as anisotropic diffusion and coupled complex block matrices.",
        "",
        _render_validation_table(validation_records),
        "",
        f"Maximum relative error across validation records: {max_validation_error:.3e}",
    ]

    if figure_reference is not None:
        sections.extend(["", "## Fitted Chart", "", f"![Krylov benchmark trend]({figure_reference})"])

    return "\n".join(sections) + "\n"


def render_krylov_benchmark_suite_markdown(
    suite: KrylovBenchmarkSuiteReport,
    *,
    generated_on: date | None = None,
    figure_reference: str | None = None,
) -> str:
    """Render a markdown report that summarizes multiple generated families."""

    stamp = generated_on or date.today()
    validation_records = list(suite.validation_records)
    max_validation_error = max((record.relative_error for record in validation_records), default=0.0)

    sections = [
        "# Krylov Benchmark Suite Report",
        "",
        f"**Date:** {stamp.isoformat()}",
        "",
        "## Benchmark Scope",
        "",
        f"- Family count: {suite.family_count}",
        f"- Families: {', '.join(report.family_name or 'unknown' for report in suite.reports)}",
        f"- Target DOF for extrapolation: {suite.target_dof}",
        "",
        "## Cross-Family Summary",
        "",
        _render_suite_summary_table(suite),
        "",
    ]

    for report in suite.reports:
        family_name = report.family_name or "unknown"
        sections.extend(
            [
                f"## Family: {family_name}",
                "",
                _render_trend_table(report),
                "",
                _render_samples_table(report),
                "",
            ]
        )

    sections.extend(
        [
            "## Correctness Validation",
            "",
            "The suite report reuses the shared correctness-validation records so each family trend is interpreted against the same 1-norm and 2-norm reference baseline.",
            "",
            _render_validation_table(validation_records),
            "",
            f"Maximum relative error across validation records: {max_validation_error:.3e}",
        ]
    )

    if figure_reference is not None:
        sections.extend(["", "## Fitted Chart", "", f"![Krylov benchmark suite trend]({figure_reference})"])

    return "\n".join(sections) + "\n"