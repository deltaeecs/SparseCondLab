from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .benchmark import IterativeBenchmarkResult, run_iterative_benchmarks
from .condest import condest_1, condest_2
from .io import load_matrix
from .shards import assemble_shards


@dataclass(frozen=True)
class CompareRecord:
    input: str
    rows: int
    cols: int
    condest_1: float
    condest_2: float
    method: str
    converged: bool
    iterations: int
    residual_norm: float
    elapsed_seconds: float


def _load_input(source: str | Path):
    path = Path(source)
    if path.suffix.lower() == ".json":
        return assemble_shards(path)
    return load_matrix(path)


def build_compare_records(
    inputs: Iterable[str | Path],
    *,
    methods: Iterable[str] = ("gmres", "bicgstab"),
) -> list[CompareRecord]:
    records: list[CompareRecord] = []
    for item in inputs:
        matrix = _load_input(item)
        condition_estimate = float(condest_1(matrix))
        condition_estimate_2 = float(condest_2(matrix))
        for benchmark in run_iterative_benchmarks(matrix, methods=methods):
            records.append(
                CompareRecord(
                    input=str(item),
                    rows=int(matrix.shape[0]),
                    cols=int(matrix.shape[1]),
                    condest_1=condition_estimate,
                    condest_2=condition_estimate_2,
                    method=benchmark.method,
                    converged=benchmark.converged,
                    iterations=benchmark.iterations,
                    residual_norm=benchmark.residual_norm,
                    elapsed_seconds=benchmark.elapsed_seconds,
                )
            )
    return records


def records_to_frame(records: Iterable[CompareRecord]) -> pd.DataFrame:
    return pd.DataFrame.from_records([asdict(record) for record in records])