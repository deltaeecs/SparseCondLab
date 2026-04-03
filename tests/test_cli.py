from __future__ import annotations

import csv
import io
from pathlib import Path

from scipy import sparse

from sparsecondlab.cli import main
from sparsecondlab.io import load_matrix


def test_compare_cli_outputs_csv_for_real_matrix(tmp_path, capsys):
    matrix_path = Path("tests/data/real_example.mtx")
    matrix = load_matrix(matrix_path)

    shard_path = tmp_path / "shard.npz"
    sparse.save_npz(shard_path, matrix)

    exit_code = main([str(matrix_path), str(shard_path), "--methods", "gmres", "--format", "csv"])

    reader = csv.DictReader(io.StringIO(capsys.readouterr().out))
    rows = list(reader)

    assert exit_code == 0
    assert len(rows) == 2
    assert rows[0]["method"] == "gmres"
    assert rows[0]["rows"] == "5"
    assert rows[1]["input"].endswith("shard.npz")
    assert "condest_2" in rows[0]