from __future__ import annotations

import json
from pathlib import Path

from scipy import sparse

from sparsecondlab.io import load_matrix
from sparsecondlab.shards import assemble_shards, load_shard_manifest


def test_load_shard_manifest_requires_versioned_schema(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "shape": [5, 5],
                "shards": [{"path": "tests/data/real_example.mtx", "row_offset": 0, "col_offset": 0}],
            }
        ),
        encoding="utf-8",
    )

    try:
        load_shard_manifest(manifest_path)
    except ValueError as exc:
        assert "format" in str(exc) or "version" in str(exc)
    else:
        raise AssertionError("expected the manifest parser to reject unversioned schema")


def test_assemble_shards_from_real_matrix(tmp_path):
    original = load_matrix(Path("tests/data/real_example.mtx"))
    other = load_matrix(Path("tests/data/symmetric_example.mtx"))

    sparse.save_npz(tmp_path / "top.npz", original)
    sparse.save_npz(tmp_path / "bottom.npz", other)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "sparsecondlab-shards",
                "version": 1,
                "shape": [10, 10],
                "shards": [
                    {"path": "top.npz", "row_offset": 0, "col_offset": 0},
                    {"path": "bottom.npz", "row_offset": 5, "col_offset": 5},
                ],
            }
        ),
        encoding="utf-8",
    )

    assembled = assemble_shards(manifest_path)

    assert assembled.shape == (10, 10)
    assert assembled.nnz >= original.nnz
    assert assembled[0, 0] == original[0, 0]