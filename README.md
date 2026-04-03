# SparseCondLab

SparseCondLab 是一个用于比较大规模稀疏复数矩阵条件数特性的工具集，面向由不同基函数产生的线性系统。项目明确避免依赖闭源工具，以促进科学计算中的开放性与协作。

## 功能特性

- 支持多种矩阵输入接口：
  - SciPy 稀疏矩阵
  - Matrix Market 文件
  - NPZ 文件
  - MUMPS 风格的分布式矩阵输入

## 安装

使用以下命令安装项目依赖：

```bash
pip install scipy numpy pandas matplotlib[all]  
# Optional dependencies
pip install petsc4py slepc4py mpi4py
```

## 快速开始

安装依赖后，可以对一个或多个矩阵文件或分片清单运行 compare CLI：

```bash
scl-compare path/to/matrix.mtx path/to/other_matrix.npz --methods gmres,bicgstab --format csv
```

如果你只是想先验证“工具能不能正确读入我的矩阵”，最推荐先从 `scl-compare` 开始，而不是直接跑 benchmark。因为 compare 会先给出矩阵维度、条件数结果和迭代求解器表现，能更快暴露输入格式、矩阵是否为方阵、求解是否收敛等基础问题。

## 如何测试你自己的矩阵

大多数用户真正关心的是“怎样把自己的矩阵喂给工具”，下面按最常见场景整理。

### 1. 先确认你的输入是否适合测试

在 CLI 工作流里，最稳妥的输入是已经落盘的稀疏矩阵文件或分片清单。实际使用前建议先确认：

- 你的矩阵是二维方阵。条件数计算要求输入必须是方阵。
- 稀疏格式优先。非常大的稠密矩阵不适合直接走这个工作流。
- 实数和复数矩阵都可以；项目就是面向大规模稀疏复数矩阵设计的。
- 如果矩阵来自 Python 内存对象，建议先导出成 `.npz` 或 `.mtx` 再用 CLI 测试。

### 2. 你可以用哪几种方式输入自己的矩阵

SparseCondLab 当前最常见的三种 CLI 输入方式如下：

#### 方式 A：Matrix Market 文件

适合已经有 `.mtx`、`.mm`、`.mtz` 文件的情况。

```bash
scl-compare path/to/your_matrix.mtx --format json
```

如果你在 Python 里有一个 SciPy 稀疏矩阵，可以这样导出：

```python
from scipy.io import mmwrite

mmwrite("your_matrix.mtx", matrix)
```

#### 方式 B：SciPy NPZ 文件

适合你自己在 Python 里生成了 SciPy 稀疏矩阵，并希望直接保存后测试。

```bash
scl-compare path/to/your_matrix.npz --format csv
```

导出方式示例：

```python
from scipy import sparse

sparse.save_npz("your_matrix.npz", matrix)
```

#### 方式 C：分片清单 JSON

适合矩阵天然分块存储，或你已经有多个 shard 需要按偏移组装成总矩阵的情况。CLI 会先按照清单组装矩阵，再进行比较或 benchmark。

分片清单是带版本信息的 JSON 文件，结构如下：

```json
{
  "format": "sparsecondlab-shards",
  "version": 1,
  "shape": [10, 10],
  "shards": [
    {"path": "top.mtx", "row_offset": 0, "col_offset": 0},
    {"path": "bottom.mtx", "row_offset": 5, "col_offset": 5}
  ]
}
```

运行方式：

```bash
scl-compare path/to/manifest.json --format json
```

如果清单里的 `path` 不是绝对路径，它会相对于 manifest 文件所在目录解析。

### 3. 推荐的用户工作流

如果你是第一次把自己的矩阵接入，建议按下面顺序来：

1. 先用 `scl-compare your_matrix.mtx --format json` 确认文件能被正确读取。
2. 检查输出中的 `rows`、`cols`、`condest_1`、`condest_2` 是否符合预期量级。
3. 再看迭代求解器的 `converged`、`iterations`、`residual_norm`，判断矩阵数值行为是否合理。
4. 只有当基础读入和 compare 结果看起来正常时，再继续跑 `scl-benchmark` 做性能趋势测试。

这一顺序的好处是：如果你的矩阵有格式问题、维度问题、偏移拼装问题，通常会在 compare 阶段就暴露出来，不必先跑一长串 benchmark 才发现输入不对。

### 4. compare 输出里应该重点看什么

`scl-compare` 输出的是扁平记录。第一次接入自己的矩阵时，最值得优先看的字段通常不是 benchmark 表格，而是下面这些：

- `rows` 和 `cols`：确认读入后的矩阵维度是否正确。
- `condest_1`：1-范数条件数结果，适合先做数量级判断。
- `condest_2`：2-范数条件数结果，适合和你已有参考值或经验值对照。
- `converged`：GMRES / BiCGSTAB 是否收敛。
- `iterations`：是否出现异常高的迭代次数。
- `residual_norm`：残差量级是否可接受。

如果你只想快速判断“我的矩阵是不是明显病态”或者“输入有没有读错”，通常这几个字段已经比完整 benchmark 图表更重要。

`compare` 命令会输出扁平化的 CSV 或 JSON 记录，其中包含矩阵维度、`condest_1` 以及每个所选求解器的迭代 benchmark 结果。

库接口同时提供 1-范数和 2-范数条件数计算，对应 API 为 `condest_1` 和 `condest_2`。

### 5. 如果你希望直接在 Python 里测试

如果你不想先写文件，也可以在 Python 中直接调用库接口。对于脚本验证、小规模试算或预处理流程联调，这种方式更直接：

```python
import numpy as np
from scipy import sparse

from sparsecondlab import condest_1, condest_2

matrix = sparse.load_npz("your_matrix.npz")
print(condest_1(matrix))
print(condest_2(matrix))
```

其中 `load_matrix()` 支持 Matrix Market、NPZ，以及内存中的 SciPy 稀疏矩阵或 NumPy 二维数组；但对于大矩阵的实际使用，仍然更建议先走稀疏文件输入路径。

## Krylov Benchmark

可以使用 benchmark CLI 对采样得到的 FEM 矩阵进行测量，并拟合可外推到 1e6 自由度的时间趋势：

```bash
scl-benchmark path/to/sample_1.mtx path/to/sample_2.mtx --solver gmres --preconditioner ilu --norms 1,2 --predict-dof 1000000 --format json
```

对于用户自己的矩阵，更实际的理解方式是：`scl-benchmark` 主要回答“如果我对这一类矩阵持续放大规模，1-范数和 2-范数条件数估计各自大概要花多久”。因此它更适合在你已经确认输入正确之后再使用。

### 什么时候该用哪种 benchmark 模式

#### 模式 A：直接 benchmark 你的真实输入文件

如果你已经有多份真实矩阵样本，例如不同网格密度、不同模型尺寸、不同频点对应的系统矩阵，优先直接把这些文件传给 `scl-benchmark`：

```bash
scl-benchmark sample_1.mtx sample_2.mtx sample_3.mtx --norms 1,2 --format json
```

这是最贴近实际工程输入的方式，也是最值得优先相信的 benchmark 来源。

#### 模式 B：`--auto-scale`

如果你手头只有一个或几个较小的真实矩阵，但又想粗看放大后的趋势，可以使用 `--auto-scale`。它会基于你给的矩阵构造更大的块对角矩阵族。

```bash
scl-benchmark your_matrix.mtx --auto-scale --norms 1,2 --max-dof 200000 --format json
```

这适合“趋势摸底”，但它不是对真实物理离散系统的严格替代，因为块对角扩展的结构比真实网格加密更简单。

#### 模式 C：`--generated-family`

如果你暂时没有可直接公开或共享的真实矩阵，也可以先用内置生成矩阵族测试工具链和量级趋势：

```bash
scl-benchmark --generated-family anisotropic-poisson-2d --norms 1,2 --format json
```

这一模式更适合验证算法路径、生成示例报告、做趋势参考，而不是替代你自己的真实矩阵数据。

该 benchmark 不会直接在 1e6 DOF 上运行，而是测量一组较小规模的矩阵，对每个范数阶次拟合对数-对数时间趋势，并给出目标自由度下的预测耗时。

为了让性能采样更接近真实问题，建议使用生成式 PDE 风格矩阵族，而不是简单重复很小的输入块。`anisotropic-poisson-2d` 会给出随网格增大而扩展的稀疏刚度矩阵趋势，`coupled-diffusion-2d` 引入复数块耦合，而 `convection-diffusion-2d` 则提供一个以输运主导的非对称稀疏系统。

| 矩阵族 | 模型类别 | 矩阵特性 |
| --- | --- | --- |
| `poisson-2d` | 各向同性 Dirichlet 扩散 | 对称正定 |
| `anisotropic-poisson-2d` | 各向异性带偏移扩散 | 对称、近似刚度矩阵型稀疏系统 |
| `coupled-diffusion-2d` | 复数块耦合扩散 | 复数非对称块系统 |
| `convection-diffusion-2d` | 输运主导对流-扩散 | 实数非对称稀疏系统 |

```bash
scl-benchmark \
  --generated-family anisotropic-poisson-2d \
  --solver gmres \
  --preconditioner ilu \
  --norms 1,2 \
  --predict-dof 1000000 \
  --min-grid-size 8 \
  --max-grid-size 1024 \
  --max-dof 2000000 \
  --max-sample-seconds 2.0 \
  --report-path docs/reports/2026-04-03-krylov-benchmark-report.md \
  --figure-path docs/reports/figures/krylov_benchmark_trend.png \
  --format json
```

如果你仍然希望直接对显式输入文件做 benchmark，也可以继续使用 `--auto-scale`。它会基于给定矩阵构造更大的块对角矩阵族，直到触达到配置的本地资源上限。

如果希望把多个生成矩阵族汇总到一份报告中，可以重复传入 `--generated-family`。CLI 会输出 suite 级别的 JSON 报告，并可选生成 markdown 汇总和组合图表。

```bash
scl-benchmark \
  --generated-family anisotropic-poisson-2d \
  --generated-family coupled-diffusion-2d \
  --generated-family convection-diffusion-2d \
  --solver gmres \
  --preconditioner ilu \
  --norms 1,2 \
  --predict-dof 1000000 \
  --min-grid-size 4 \
  --max-grid-size 16 \
  --max-sample-seconds 5.0 \
  --report-path docs/reports/2026-04-03-krylov-benchmark-suite-report.md \
  --figure-path docs/reports/figures/krylov_benchmark_suite_trend.png \
  --format json
```

当前生成的报告会把正确性验证用例与性能趋势数据一起写入。其中既包含 PDE 风格的闭式参考值，也包含对中等规模结构化稀疏系统的精确稠密参考检查，因此报告同时给出性能证据和精度证据。

### benchmark 输出该怎么读

如果你只扫一眼 benchmark 记录，最值得先看的字段通常是：

- 每个 `norm_order` 对应的 `elapsed_seconds`：当前样本上的实际耗时。
- `predicted_seconds_at_target_dof`：外推到目标自由度后的预测耗时。
- `r_squared`：趋势拟合是否可靠。越接近 1，说明这条拟合线越稳定。
- `sample_count`：用于拟合的样本点数量。样本太少时，预测值应保守看待。

但在真正使用时，更推荐你先确认“我的矩阵已经被正确读入并且 compare 结果合理”，再去解读这些 benchmark 趋势数据。

## 开发

测试套件使用 `pytest`，并且包含真实矩阵 fixture，而不只是合成示例。

```bash
pytest
```
