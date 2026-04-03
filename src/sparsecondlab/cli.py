from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .report import build_compare_records, records_to_frame
from .krylov import (
    build_generated_family_krylov_benchmark_report,
    build_generated_family_krylov_benchmark_suite,
    build_krylov_benchmark_report,
    build_local_limit_krylov_benchmark_report,
)
from .krylov_report import (
    build_correctness_validation_records,
    render_krylov_benchmark_markdown,
    render_krylov_benchmark_suite_markdown,
    write_krylov_benchmark_figure,
    write_krylov_benchmark_suite_figure,
)


def _markdown_figure_reference(report_path: Path, figure_path: Path) -> str:
    """Return a markdown-friendly relative figure path from a report file."""

    return os.path.relpath(figure_path.resolve(), start=report_path.parent.resolve()).replace("\\", "/")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the compare CLI."""

    parser = argparse.ArgumentParser(prog="scl-compare", description="Compare matrix conditioning")
    parser.add_argument("inputs", nargs="+", help="Matrix files or shard manifests")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument(
        "--methods",
        default="gmres,bicgstab",
        help="Comma-separated iterative methods to run, for example gmres,bicgstab",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the compare CLI and emit CSV or JSON comparison rows."""

    parser = build_parser()
    args = parser.parse_args(argv)

    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    records = build_compare_records(args.inputs, methods=methods)
    frame = records_to_frame(records)

    if args.format == "csv":
        print(frame.to_csv(index=False).rstrip())
    else:
        print(json.dumps(frame.to_dict(orient="records"), indent=2))
    return 0


def benchmark_main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI for single-family, multi-family, or auto-scale reports."""

    parser = argparse.ArgumentParser(prog="scl-benchmark", description="Benchmark Krylov condition-number trends")
    parser.add_argument("inputs", nargs="*", help="Matrix files or shard manifests")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--solver", default="gmres")
    parser.add_argument("--preconditioner", default="ilu")
    parser.add_argument("--norms", default="1,2", help="Comma-separated norm orders to benchmark, for example 1,2")
    parser.add_argument("--predict-dof", type=int, default=1_000_000)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--auto-scale", action="store_true", help="Grow a block-diagonal benchmark family until a local limit is reached")
    parser.add_argument("--growth-factor", type=int, default=2)
    parser.add_argument("--max-scale", type=int, default=1 << 20)
    parser.add_argument("--max-dof", type=int, default=1_000_000)
    parser.add_argument("--max-sample-seconds", type=float, default=2.0)
    parser.add_argument(
        "--generated-family",
        action="append",
        choices=("poisson-2d", "anisotropic-poisson-2d", "coupled-diffusion-2d", "convection-diffusion-2d"),
    )
    parser.add_argument("--min-grid-size", type=int, default=4)
    parser.add_argument("--max-grid-size", type=int, default=1024)
    parser.add_argument("--report-path", help="Optional markdown report output path")
    parser.add_argument("--figure-path", help="Optional figure output path")
    args = parser.parse_args(argv)

    norms = tuple(int(item.strip()) for item in args.norms.split(",") if item.strip())
    validation_records = build_correctness_validation_records()
    generated_families = list(args.generated_family or [])
    if len(generated_families) > 1:
        suite = build_generated_family_krylov_benchmark_suite(
            families=generated_families,
            norms=norms,
            solver=args.solver,
            preconditioner=args.preconditioner,
            target_dof=args.predict_dof,
            rtol=args.rtol,
            atol=args.atol,
            maxiter=args.maxiter,
            min_grid_size=args.min_grid_size,
            max_grid_size=args.max_grid_size,
            growth_factor=args.growth_factor,
            max_dof=args.max_dof,
            max_sample_seconds=args.max_sample_seconds,
            validation_records=validation_records,
        )

        if args.figure_path:
            write_krylov_benchmark_suite_figure(suite, args.figure_path)

        if args.report_path:
            report_path = Path(args.report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            figure_reference = None
            if args.figure_path:
                figure_reference = _markdown_figure_reference(report_path, Path(args.figure_path))
            report_path.write_text(
                render_krylov_benchmark_suite_markdown(suite, figure_reference=figure_reference),
                encoding="utf-8",
            )

        if args.format == "csv":
            sample_rows: list[dict[str, object]] = []
            trend_rows: list[dict[str, object]] = []
            for report in suite.reports:
                family_name = report.family_name
                for sample in report.samples:
                    row = asdict(sample)
                    row["family_name"] = family_name
                    sample_rows.append(row)
                for model in report.trend_models:
                    row = asdict(model)
                    row["family_name"] = family_name
                    trend_rows.append(row)

            samples_frame = pd.DataFrame.from_records(sample_rows)
            trend_frame = pd.DataFrame.from_records(trend_rows)
            frame = samples_frame.merge(
                trend_frame,
                on=["family_name", "norm_order", "algorithm", "preconditioner"],
                how="left",
                suffixes=("", "_trend"),
            )
            print(frame.to_csv(index=False).rstrip())
        else:
            print(json.dumps(suite.to_dict(), indent=2))
        return 0

    if len(generated_families) == 1:
        report = build_generated_family_krylov_benchmark_report(
            family=generated_families[0],
            norms=norms,
            solver=args.solver,
            preconditioner=args.preconditioner,
            target_dof=args.predict_dof,
            rtol=args.rtol,
            atol=args.atol,
            maxiter=args.maxiter,
            min_grid_size=args.min_grid_size,
            max_grid_size=args.max_grid_size,
            growth_factor=args.growth_factor,
            max_dof=args.max_dof,
            max_sample_seconds=args.max_sample_seconds,
            validation_records=validation_records,
        )
    elif args.auto_scale:
        if not args.inputs:
            parser.error("inputs are required unless --generated-family is used")
        report = build_local_limit_krylov_benchmark_report(
            args.inputs,
            norms=norms,
            solver=args.solver,
            preconditioner=args.preconditioner,
            target_dof=args.predict_dof,
            rtol=args.rtol,
            atol=args.atol,
            maxiter=args.maxiter,
            growth_factor=args.growth_factor,
            max_scale=args.max_scale,
            max_dof=args.max_dof,
            max_sample_seconds=args.max_sample_seconds,
            validation_records=validation_records,
        )
    else:
        if not args.inputs:
            parser.error("inputs are required unless --generated-family is used")
        report = build_krylov_benchmark_report(
            args.inputs,
            norms=norms,
            solver=args.solver,
            preconditioner=args.preconditioner,
            target_dof=args.predict_dof,
            rtol=args.rtol,
            atol=args.atol,
            maxiter=args.maxiter,
            validation_records=validation_records,
        )

    if args.figure_path:
        write_krylov_benchmark_figure(report, args.figure_path)

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        figure_reference = None
        if args.figure_path:
            figure_reference = _markdown_figure_reference(report_path, Path(args.figure_path))
        report_path.write_text(
            render_krylov_benchmark_markdown(report, figure_reference=figure_reference),
            encoding="utf-8",
        )

    if args.format == "csv":
        samples_frame = pd.DataFrame.from_records([asdict(sample) for sample in report.samples])
        trend_frame = pd.DataFrame.from_records([asdict(model) for model in report.trend_models])
        frame = samples_frame.merge(
            trend_frame,
            on=["norm_order", "algorithm", "preconditioner"],
            how="left",
            suffixes=("", "_trend"),
        )
        print(frame.to_csv(index=False).rstrip())
    else:
        print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())