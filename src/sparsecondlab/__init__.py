"""SparseCondLab core package."""

from .benchmark import IterativeBenchmarkResult, run_iterative_benchmark, run_iterative_benchmarks
from .condest import condest_1, condest_2
from .krylov import (
	KrylovBenchmarkReport,
	KrylovBenchmarkSuiteReport,
	KrylovConditionResult,
	KrylovConditioningError,
	KrylovSampleRecord,
	KrylovTrendModel,
	KrylovValidationRecord,
	build_generated_family_krylov_benchmark_report,
	build_generated_family_krylov_benchmark_suite,
	build_krylov_benchmark_report,
	build_local_limit_krylov_benchmark_report,
	estimate_condest_1_krylov,
	estimate_condest_2_krylov,
	fit_time_trend,
	predict_time_from_trend,
)
from .matrix_families import (
	build_generated_family_matrix,
	build_generated_family_samples,
	convection_diffusion_2d_matrix,
	coupled_diffusion_2d_matrix,
	poisson_1d_dirichlet_matrix,
	poisson_2d_condition_number_2,
	poisson_2d_dirichlet_matrix,
)
from .krylov_report import (
	build_correctness_validation_records,
	render_krylov_benchmark_markdown,
	render_krylov_benchmark_suite_markdown,
	write_krylov_benchmark_figure,
	write_krylov_benchmark_suite_figure,
)
from .io import load_matrix
from .report import CompareRecord, build_compare_records, records_to_frame
from .shards import assemble_shards, load_shard_manifest

__all__ = [
	"CompareRecord",
	"KrylovBenchmarkReport",
	"KrylovBenchmarkSuiteReport",
	"KrylovConditionResult",
	"KrylovConditioningError",
	"IterativeBenchmarkResult",
	"KrylovSampleRecord",
	"KrylovTrendModel",
	"KrylovValidationRecord",
	"assemble_shards",
	"build_generated_family_krylov_benchmark_report",
	"build_generated_family_krylov_benchmark_suite",
	"build_generated_family_matrix",
	"build_generated_family_samples",
	"build_correctness_validation_records",
	"build_compare_records",
	"build_krylov_benchmark_report",
	"build_local_limit_krylov_benchmark_report",
	"condest_1",
	"condest_2",
	"convection_diffusion_2d_matrix",
	"coupled_diffusion_2d_matrix",
	"estimate_condest_1_krylov",
	"estimate_condest_2_krylov",
	"fit_time_trend",
	"load_matrix",
	"load_shard_manifest",
	"poisson_1d_dirichlet_matrix",
	"poisson_2d_condition_number_2",
	"poisson_2d_dirichlet_matrix",
	"predict_time_from_trend",
	"render_krylov_benchmark_markdown",
	"render_krylov_benchmark_suite_markdown",
	"records_to_frame",
	"run_iterative_benchmark",
	"run_iterative_benchmarks",
	"write_krylov_benchmark_figure",
	"write_krylov_benchmark_suite_figure",
]