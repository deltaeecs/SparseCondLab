from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import block_diag
from scipy.sparse.linalg import (
    ArpackError,
    ArpackNoConvergence,
    LinearOperator,
    bicgstab,
    gmres,
    onenormest,
    spilu,
    svds,
)

from .io import load_matrix
from .matrix_families import build_generated_family_matrix
from .shards import assemble_shards


class KrylovConditioningError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


@dataclass(frozen=True)
class KrylovConditionResult:
    norm_order: int
    condition_number: float
    elapsed_seconds: float
    inner_iterations: int
    solve_calls: int
    algorithm: str
    preconditioner: str


@dataclass(frozen=True)
class KrylovSampleRecord:
    input: str
    dof: int
    nnz: int
    norm_order: int
    condition_number: float
    elapsed_seconds: float
    algorithm: str
    preconditioner: str
    inner_iterations: int
    solve_calls: int


@dataclass(frozen=True)
class KrylovTrendModel:
    norm_order: int
    algorithm: str
    preconditioner: str
    slope: float
    intercept: float
    r_squared: float
    predicted_seconds_at_target_dof: float
    sample_count: int


@dataclass(frozen=True)
class KrylovValidationRecord:
    case_name: str
    estimator: str
    norm_order: int
    dof: int
    expected_condition_number: float
    measured_condition_number: float
    relative_error: float


@dataclass(frozen=True)
class KrylovBenchmarkReport:
    target_dof: int
    samples: list[KrylovSampleRecord]
    trend_models: list[KrylovTrendModel]
    sample_strategy: str = "manual"
    stop_reason: str = "input_exhausted"
    family_name: str | None = None
    validation_records: list[KrylovValidationRecord] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_dof": self.target_dof,
            "sample_count": self.sample_count,
            "sample_strategy": self.sample_strategy,
            "stop_reason": self.stop_reason,
            "family_name": self.family_name,
            "samples": [asdict(sample) for sample in self.samples],
            "trend_models": [asdict(model) for model in self.trend_models],
            "validation_records": [asdict(record) for record in self.validation_records],
        }


@dataclass(frozen=True)
class KrylovBenchmarkSuiteReport:
    target_dof: int
    reports: list[KrylovBenchmarkReport]
    validation_records: list[KrylovValidationRecord] = field(default_factory=list)

    @property
    def family_count(self) -> int:
        return len(self.reports)

    def to_dict(self) -> dict[str, object]:
        return {
            "target_dof": self.target_dof,
            "family_count": self.family_count,
            "families": [report.family_name for report in self.reports],
            "reports": [report.to_dict() for report in self.reports],
            "validation_records": [asdict(record) for record in self.validation_records],
        }


def _load_input(source: str | Path | sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    if sparse.issparse(source):
        return source.tocsr()

    if isinstance(source, np.ndarray):
        if source.ndim != 2:
            raise ValueError("dense input must be two-dimensional")
        return sparse.csr_matrix(source)

    path = Path(source)
    if path.suffix.lower() == ".json":
        return assemble_shards(path)
    return load_matrix(path)


def _one_norm(matrix: sparse.spmatrix | np.ndarray) -> float:
    if sparse.issparse(matrix):
        column_sums = np.abs(matrix).sum(axis=0)
        return float(np.asarray(column_sums).ravel().max(initial=0.0))
    return float(np.linalg.norm(np.asarray(matrix), 1))


def _build_preconditioner_pair(matrix: sparse.csr_matrix, preconditioner: str) -> tuple[LinearOperator, LinearOperator]:
    if preconditioner == "none":
        identity = LinearOperator(matrix.shape, matvec=lambda values: values, rmatvec=lambda values: values, dtype=matrix.dtype)
        return identity, identity

    if preconditioner != "ilu":
        raise ValueError(f"unsupported preconditioner: {preconditioner}")

    try:
        forward = spilu(matrix.tocsc())
        transpose = spilu(matrix.conjugate().transpose().tocsc())
    except Exception as exc:  # pragma: no cover - SciPy error types vary by matrix and version
        raise KrylovConditioningError("preconditioner", f"failed to build ILU preconditioner: {exc}") from exc

    def forward_apply(values: np.ndarray) -> np.ndarray:
        return forward.solve(values)

    def transpose_apply(values: np.ndarray) -> np.ndarray:
        return transpose.solve(values)

    forward_operator = LinearOperator(matrix.shape, matvec=forward_apply, rmatvec=transpose_apply, dtype=matrix.dtype)
    transpose_operator = LinearOperator(matrix.shape, matvec=transpose_apply, rmatvec=forward_apply, dtype=matrix.dtype)
    return forward_operator, transpose_operator


def _solve_with_krylov(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    *,
    solver: str,
    preconditioner_operator: LinearOperator,
    transpose_preconditioner_operator: LinearOperator,
    rtol: float,
    atol: float,
    maxiter: int | None,
    transposed: bool,
) -> tuple[np.ndarray, int]:
    operator = matrix.conjugate().transpose() if transposed else matrix
    preconditioner = transpose_preconditioner_operator if transposed else preconditioner_operator
    inner_iterations = 0

    def callback(_value: np.ndarray | float) -> None:
        nonlocal inner_iterations
        inner_iterations += 1

    if solver == "gmres":
        solution, info = gmres(
            operator,
            rhs,
            M=preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            callback=callback,
            callback_type="legacy",
        )
    elif solver == "bicgstab":
        solution, info = bicgstab(
            operator,
            rhs,
            M=preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            callback=callback,
        )
    else:
        raise ValueError(f"unsupported solver: {solver}")

    if info != 0:
        raise KrylovConditioningError("solve", f"{solver} did not converge (info={info})")

    return np.asarray(solution), inner_iterations


def _inverse_operator(
    matrix: sparse.csr_matrix,
    *,
    solver: str,
    preconditioner: str,
    rtol: float,
    atol: float,
    maxiter: int | None,
) -> tuple[LinearOperator, dict[str, int]]:
    forward_preconditioner, transpose_preconditioner = _build_preconditioner_pair(matrix, preconditioner)
    metrics = {"solve_calls": 0, "inner_iterations": 0}

    def matvec(values: np.ndarray) -> np.ndarray:
        metrics["solve_calls"] += 1
        solution, iterations = _solve_with_krylov(
            matrix,
            np.asarray(values),
            solver=solver,
            preconditioner_operator=forward_preconditioner,
            transpose_preconditioner_operator=transpose_preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            transposed=False,
        )
        metrics["inner_iterations"] += iterations
        return solution

    def rmatvec(values: np.ndarray) -> np.ndarray:
        metrics["solve_calls"] += 1
        solution, iterations = _solve_with_krylov(
            matrix,
            np.asarray(values),
            solver=solver,
            preconditioner_operator=forward_preconditioner,
            transpose_preconditioner_operator=transpose_preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            transposed=True,
        )
        metrics["inner_iterations"] += iterations
        return solution

    def matmat(values: np.ndarray) -> np.ndarray:
        columns = [matvec(column) for column in np.asarray(values).T]
        return np.column_stack(columns)

    def rmatmat(values: np.ndarray) -> np.ndarray:
        columns = [rmatvec(column) for column in np.asarray(values).T]
        return np.column_stack(columns)

    operator = LinearOperator(
        matrix.shape,
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=matrix.dtype,
    )
    return operator, metrics


def estimate_condest_1_krylov(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    solver: str = "gmres",
    preconditioner: str = "ilu",
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> KrylovConditionResult:
    """Estimate the sparse 1-norm condition number using onenormest and Krylov solves."""

    working = _load_input(matrix)
    if working.shape[0] != working.shape[1]:
        raise ValueError("estimate_condest_1_krylov requires a square matrix")

    start = perf_counter()
    inverse_operator, metrics = _inverse_operator(
        working,
        solver=solver,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
    )
    inverse_norm = float(onenormest(inverse_operator))
    elapsed_seconds = perf_counter() - start

    return KrylovConditionResult(
        norm_order=1,
        condition_number=float(_one_norm(working) * inverse_norm),
        elapsed_seconds=elapsed_seconds,
        inner_iterations=metrics["inner_iterations"],
        solve_calls=metrics["solve_calls"],
        algorithm=f"onenormest+{solver}",
        preconditioner=preconditioner,
    )


def estimate_condest_2_krylov(
    matrix: sparse.spmatrix | np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> KrylovConditionResult:
    """Estimate the sparse 2-norm condition number from extremal singular values."""

    working = _load_input(matrix)
    if working.shape[0] != working.shape[1]:
        raise ValueError("estimate_condest_2_krylov requires a square matrix")

    start = perf_counter()
    sparse_working = working.tocsc()

    try:
        largest = float(np.asarray(svds(sparse_working, k=1, which="LM", return_singular_vectors=False, tol=rtol, maxiter=maxiter)).ravel()[0])
        smallest = float(np.asarray(svds(sparse_working, k=1, which="SM", return_singular_vectors=False, tol=rtol, maxiter=maxiter)).ravel()[0])
    except (ArpackNoConvergence, ArpackError) as exc:
        # ArpackError covers e.g. error 3 (no shifts could be applied), which
        # occurs for perfectly degenerate singular values (scaled identity matrices).
        raise KrylovConditioningError("svds", f"failed to converge: {exc}") from exc

    if smallest == 0.0:
        condition_number = float(np.inf)
    else:
        condition_number = float(largest / smallest)

    elapsed_seconds = perf_counter() - start
    return KrylovConditionResult(
        norm_order=2,
        condition_number=condition_number,
        elapsed_seconds=elapsed_seconds,
        inner_iterations=0,
        solve_calls=0,
        algorithm="svds",
        preconditioner="none",
    )


def _sample_value(sample: Mapping[str, object] | object, key: str) -> object:
    if isinstance(sample, Mapping):
        return sample[key]
    return getattr(sample, key)


def _sample_float(sample: Mapping[str, object] | object, key: str) -> float:
    return float(_sample_value(sample, key))


def _sample_text(sample: Mapping[str, object] | object, key: str, default: str = "unknown") -> str:
    if isinstance(sample, Mapping):
        return str(sample.get(key, default))
    return str(getattr(sample, key, default))


def fit_time_trend(samples: Sequence[Mapping[str, object] | object], *, x_key: str, y_key: str) -> KrylovTrendModel:
    """Fit a log-log runtime trend model from benchmark samples."""

    if len(samples) < 2:
        raise ValueError("fit_time_trend requires at least two samples")

    x_values = np.asarray([_sample_float(sample, x_key) for sample in samples], dtype=float)
    y_values = np.asarray([_sample_float(sample, y_key) for sample in samples], dtype=float)

    if np.any(x_values <= 0):
        raise ValueError("trend x values must be positive")
    if np.any(y_values <= 0):
        raise ValueError("trend y values must be positive")

    log_x = np.log(x_values)
    log_y = np.log(y_values)
    x_mean = float(np.mean(log_x))
    y_mean = float(np.mean(log_y))
    centered_x = log_x - x_mean
    centered_y = log_y - y_mean
    denominator = float(np.sum(centered_x ** 2))
    if denominator == 0.0:
        slope = 0.0
    else:
        slope = float(np.sum(centered_x * centered_y) / denominator)
    intercept = y_mean - slope * x_mean
    fitted = slope * log_x + intercept

    total_variance = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
    if total_variance == 0.0:
        r_squared = 1.0
    else:
        residual_variance = float(np.sum((log_y - fitted) ** 2))
        r_squared = 1.0 - residual_variance / total_variance

    return KrylovTrendModel(
        norm_order=int(_sample_float(samples[0], "norm_order")),
        algorithm=_sample_text(samples[0], "algorithm"),
        preconditioner=_sample_text(samples[0], "preconditioner"),
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
        predicted_seconds_at_target_dof=0.0,
        sample_count=len(samples),
    )


def predict_time_from_trend(model: KrylovTrendModel, x_value: float) -> float:
    """Predict runtime at a new DOF using a fitted log-log trend model."""

    if x_value <= 0:
        raise ValueError("x_value must be positive")
    return float(np.exp(model.intercept + model.slope * np.log(float(x_value))))


def _load_benchmark_input(source: str | Path | sparse.spmatrix | np.ndarray) -> tuple[str, sparse.csr_matrix]:
    if isinstance(source, (sparse.spmatrix, np.ndarray)):
        return ("<memory>", _load_input(source))

    path = Path(source)
    if path.suffix.lower() == ".json":
        return (str(path), assemble_shards(path))
    return (str(path), load_matrix(path))


def _run_norm_estimators(
    input_name: str,
    matrix: sparse.csr_matrix,
    *,
    norms: Iterable[int],
    solver: str,
    preconditioner: str,
    rtol: float,
    atol: float,
    maxiter: int | None,
) -> list[KrylovSampleRecord]:
    records: list[KrylovSampleRecord] = []
    for norm_order in norms:
        if norm_order == 1:
            result = estimate_condest_1_krylov(
                matrix,
                solver=solver,
                preconditioner=preconditioner,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
        elif norm_order == 2:
            result = estimate_condest_2_krylov(
                matrix,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
        else:
            raise ValueError(f"unsupported norm order: {norm_order}")

        records.append(
            KrylovSampleRecord(
                input=input_name,
                dof=int(matrix.shape[0]),
                nnz=int(matrix.nnz),
                norm_order=result.norm_order,
                condition_number=result.condition_number,
                elapsed_seconds=result.elapsed_seconds,
                algorithm=result.algorithm,
                preconditioner=result.preconditioner,
                inner_iterations=result.inner_iterations,
                solve_calls=result.solve_calls,
            )
        )
    return records


def _build_trend_models(records: Sequence[KrylovSampleRecord], *, target_dof: int) -> list[KrylovTrendModel]:
    trend_models: list[KrylovTrendModel] = []
    for norm_order in sorted(set(record.norm_order for record in records)):
        norm_samples = [record for record in records if record.norm_order == norm_order]
        model = fit_time_trend(norm_samples, x_key="dof", y_key="elapsed_seconds")
        predicted = predict_time_from_trend(model, target_dof)
        trend_models.append(
            KrylovTrendModel(
                norm_order=norm_order,
                algorithm=model.algorithm,
                preconditioner=model.preconditioner,
                slope=model.slope,
                intercept=model.intercept,
                r_squared=model.r_squared,
                predicted_seconds_at_target_dof=predicted,
                sample_count=model.sample_count,
            )
        )
    return trend_models


def _build_scaled_block_diagonal_matrix(base_matrices: Sequence[sparse.csr_matrix], scale: int) -> sparse.csr_matrix:
    if scale <= 0:
        raise ValueError("scale must be positive")
    pieces = list(base_matrices) * scale
    return sparse.csr_matrix(block_diag(pieces, format="csr"))


def build_krylov_benchmark_report(
    inputs: Iterable[str | Path | sparse.spmatrix | np.ndarray],
    *,
    norms: Iterable[int] = (1, 2),
    solver: str = "gmres",
    preconditioner: str = "ilu",
    target_dof: int = 1_000_000,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    validation_records: Sequence[KrylovValidationRecord] | None = None,
) -> KrylovBenchmarkReport:
    """Benchmark one or more explicit inputs and fit 1-/2-norm runtime trends."""

    if validation_records is None:
        from .krylov_report import build_correctness_validation_records

        validation_records = build_correctness_validation_records()

    records: list[KrylovSampleRecord] = []
    for source in inputs:
        input_name, matrix = _load_benchmark_input(source)
        records.extend(
            _run_norm_estimators(
                input_name,
                matrix,
                norms=norms,
                solver=solver,
                preconditioner=preconditioner,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
        )

    return KrylovBenchmarkReport(
        target_dof=target_dof,
        samples=records,
        trend_models=_build_trend_models(records, target_dof=target_dof),
        sample_strategy="manual",
        stop_reason="input_exhausted",
        family_name=None,
        validation_records=list(validation_records or []),
    )


def build_local_limit_krylov_benchmark_report(
    base_inputs: Iterable[str | Path | sparse.spmatrix | np.ndarray],
    *,
    norms: Iterable[int] = (1, 2),
    solver: str = "gmres",
    preconditioner: str = "ilu",
    target_dof: int = 1_000_000,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    growth_factor: int = 2,
    max_scale: int = 1 << 20,
    max_dof: int = 1_000_000,
    max_sample_seconds: float = 2.0,
    min_scale_samples: int = 2,
    validation_records: Sequence[KrylovValidationRecord] | None = None,
) -> KrylovBenchmarkReport:
    """Benchmark auto-scaled block-diagonal inputs until a local resource limit is hit."""

    if validation_records is None:
        from .krylov_report import build_correctness_validation_records

        validation_records = build_correctness_validation_records()

    if growth_factor < 2:
        raise ValueError("growth_factor must be at least 2")
    if max_scale < 1:
        raise ValueError("max_scale must be positive")
    if max_dof < 1:
        raise ValueError("max_dof must be positive")
    if max_sample_seconds <= 0.0:
        raise ValueError("max_sample_seconds must be positive")

    loaded_base_inputs = [_load_benchmark_input(source) for source in base_inputs]
    if not loaded_base_inputs:
        raise ValueError("build_local_limit_krylov_benchmark_report requires at least one base input")

    base_names = [name for name, _matrix in loaded_base_inputs]
    base_matrices = [matrix for _name, matrix in loaded_base_inputs]

    records: list[KrylovSampleRecord] = []
    scale = 1
    sampled_scales = 0
    stop_reason = "input_exhausted"

    while True:
        matrix = _build_scaled_block_diagonal_matrix(base_matrices, scale)
        if matrix.shape[0] > max_dof:
            stop_reason = "max_dof"
            break

        input_name = f"auto-scale(scale={scale};bases={','.join(base_names)})"
        scale_records = _run_norm_estimators(
            input_name,
            matrix,
            norms=norms,
            solver=solver,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
        )
        records.extend(scale_records)
        sampled_scales += 1

        if sampled_scales >= min_scale_samples and max(record.elapsed_seconds for record in scale_records) >= max_sample_seconds:
            stop_reason = "max_sample_seconds"
            break

        next_scale = scale * growth_factor
        if next_scale > max_scale:
            stop_reason = "max_scale"
            break
        scale = next_scale

    if len({record.dof for record in records}) < 2:
        raise ValueError("local-limit benchmark requires at least two sampled DOF values for trend fitting")

    return KrylovBenchmarkReport(
        target_dof=target_dof,
        samples=records,
        trend_models=_build_trend_models(records, target_dof=target_dof),
        sample_strategy="auto-scale",
        stop_reason=stop_reason,
        family_name="block-diagonal-from-inputs",
        validation_records=list(validation_records or []),
    )


def build_generated_family_krylov_benchmark_report(
    *,
    family: str,
    norms: Iterable[int] = (1, 2),
    solver: str = "gmres",
    preconditioner: str = "ilu",
    target_dof: int = 1_000_000,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    min_grid_size: int = 4,
    max_grid_size: int = 1024,
    growth_factor: int = 2,
    max_dof: int = 1_000_000,
    max_sample_seconds: float = 2.0,
    min_grid_samples: int = 2,
    validation_records: Sequence[KrylovValidationRecord] | None = None,
) -> KrylovBenchmarkReport:
    """Benchmark one generated matrix family across grid sizes and fit trend models."""

    if validation_records is None:
        from .krylov_report import build_correctness_validation_records

        validation_records = build_correctness_validation_records()

    if min_grid_size < 2:
        raise ValueError("min_grid_size must be at least 2")
    if max_grid_size < min_grid_size:
        raise ValueError("max_grid_size must be greater than or equal to min_grid_size")
    if growth_factor < 2:
        raise ValueError("growth_factor must be at least 2")
    if max_dof < 1:
        raise ValueError("max_dof must be positive")
    if max_sample_seconds <= 0.0:
        raise ValueError("max_sample_seconds must be positive")

    records: list[KrylovSampleRecord] = []
    stop_reason = "max_grid_size"
    grid_size = min_grid_size
    sampled_grids = 0

    while True:
        matrix = build_generated_family_matrix(family, grid_size)
        if matrix.shape[0] > max_dof:
            stop_reason = "max_dof"
            break

        input_name = f"generated:{family}:grid={grid_size}"
        grid_records = _run_norm_estimators(
            input_name,
            matrix,
            norms=norms,
            solver=solver,
            preconditioner=preconditioner,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
        )
        records.extend(grid_records)
        sampled_grids += 1

        if sampled_grids >= min_grid_samples and max(record.elapsed_seconds for record in grid_records) >= max_sample_seconds:
            stop_reason = "max_sample_seconds"
            break

        next_grid_size = grid_size * growth_factor
        if next_grid_size > max_grid_size:
            stop_reason = "max_grid_size"
            break
        grid_size = next_grid_size

    if len({record.dof for record in records}) < 2:
        raise ValueError("generated-family benchmark requires at least two sampled DOF values for trend fitting")

    return KrylovBenchmarkReport(
        target_dof=target_dof,
        samples=records,
        trend_models=_build_trend_models(records, target_dof=target_dof),
        sample_strategy="generated-family",
        stop_reason=stop_reason,
        family_name=family,
        validation_records=list(validation_records or []),
    )


def build_generated_family_krylov_benchmark_suite(
    *,
    families: Iterable[str],
    norms: Iterable[int] = (1, 2),
    solver: str = "gmres",
    preconditioner: str = "ilu",
    target_dof: int = 1_000_000,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    min_grid_size: int = 4,
    max_grid_size: int = 1024,
    growth_factor: int = 2,
    max_dof: int = 1_000_000,
    max_sample_seconds: float = 2.0,
    min_grid_samples: int = 2,
    validation_records: Sequence[KrylovValidationRecord] | None = None,
) -> KrylovBenchmarkSuiteReport:
    """Benchmark multiple generated matrix families and collect them in one suite report."""

    normalized_families = [family for family in families]
    if not normalized_families:
        raise ValueError("build_generated_family_krylov_benchmark_suite requires at least one family")

    if validation_records is None:
        from .krylov_report import build_correctness_validation_records

        validation_records = build_correctness_validation_records()

    reports = [
        build_generated_family_krylov_benchmark_report(
            family=family,
            norms=norms,
            solver=solver,
            preconditioner=preconditioner,
            target_dof=target_dof,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            min_grid_size=min_grid_size,
            max_grid_size=max_grid_size,
            growth_factor=growth_factor,
            max_dof=max_dof,
            max_sample_seconds=max_sample_seconds,
            min_grid_samples=min_grid_samples,
            validation_records=validation_records,
        )
        for family in normalized_families
    ]

    return KrylovBenchmarkSuiteReport(
        target_dof=target_dof,
        reports=reports,
        validation_records=list(validation_records),
    )
