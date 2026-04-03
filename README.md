# SparseCondLab

SparseCondLab is a toolkit for comparing the conditioning of large sparse complex matrices arising from different basis functions. This project explicitly avoids closed-source tools, promoting openness and collaboration in scientific computing.

## Features

- Supports various matrix input interfaces:
  - SciPy sparse matrices
  - Matrix Market files
  - NPZ files
  - MUMPS-style distributed matrix input

## Installation

To install the project dependencies, use the following command:

```bash
pip install scipy numpy pandas matplotlib[all]  
# Optional dependencies
pip install petsc4py slepc4py mpi4py
```
