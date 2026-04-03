from __future__ import annotations

import json
from pathlib import Path

from sparsecondlab.cli import benchmark_main


def test_benchmark_cli_outputs_trend_report(capsys):
    exit_code = benchmark_main(
        [
            str(Path("tests/data/real_example.mtx")),
            str(Path("tests/data/symmetric_example.mtx")),
            "--solver",
            "gmres",
            "--preconditioner",
            "ilu",
            "--norms",
            "1,2",
            "--predict-dof",
            "1000000",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["target_dof"] == 1_000_000
    assert payload["sample_count"] == 4
    assert payload["trend_models"][0]["norm_order"] in (1, 2)
    assert payload["trend_models"][0]["predicted_seconds_at_target_dof"] > 0


def test_benchmark_cli_can_write_report_and_figure(tmp_path, capsys):
    report_path = tmp_path / "benchmark.md"
    figure_path = tmp_path / "trend.png"

    exit_code = benchmark_main(
        [
            str(Path("tests/data/real_example.mtx")),
            str(Path("tests/data/symmetric_example.mtx")),
            "--auto-scale",
            "--max-dof",
            "35",
            "--max-scale",
            "128",
            "--max-sample-seconds",
            "60",
            "--report-path",
            str(report_path),
            "--figure-path",
            str(figure_path),
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["sample_strategy"] == "auto-scale"
    assert report_path.exists()
    assert figure_path.exists()
    assert "Correctness Validation" in report_path.read_text(encoding="utf-8")


def test_benchmark_cli_supports_generated_family_mode(capsys):
    exit_code = benchmark_main(
        [
            "--generated-family",
            "coupled-diffusion-2d",
            "--min-grid-size",
            "4",
            "--max-grid-size",
            "8",
            "--growth-factor",
            "2",
            "--max-sample-seconds",
            "60",
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["sample_strategy"] == "generated-family"
    assert payload["family_name"] == "coupled-diffusion-2d"
    assert len({sample["dof"] for sample in payload["samples"]}) == 2


def test_benchmark_cli_can_emit_multi_family_suite_report(tmp_path, capsys):
    report_path = tmp_path / "benchmark-suite.md"

    exit_code = benchmark_main(
        [
            "--generated-family",
            "anisotropic-poisson-2d",
            "--generated-family",
            "convection-diffusion-2d",
            "--min-grid-size",
            "4",
            "--max-grid-size",
            "8",
            "--growth-factor",
            "2",
            "--max-sample-seconds",
            "60",
            "--report-path",
            str(report_path),
            "--format",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["family_count"] == 2
    assert report_path.exists()
    assert "convection-diffusion-2d" in report_path.read_text(encoding="utf-8")