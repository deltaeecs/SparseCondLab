from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from .io import load_matrix

MANIFEST_FORMAT = "sparsecondlab-shards"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ShardSpec:
    path: Path
    row_offset: int = 0
    col_offset: int = 0


@dataclass(frozen=True)
class ShardManifest:
    format: str
    version: int
    shape: tuple[int, int]
    shards: tuple[ShardSpec, ...]


def load_shard_manifest(source: str | Path) -> ShardManifest:
    path = Path(source)
    data = json.loads(path.read_text(encoding="utf-8"))

    manifest_format = data.get("format")
    if manifest_format != MANIFEST_FORMAT:
        raise ValueError(f"manifest format must be {MANIFEST_FORMAT!r}")

    version = int(data.get("version", -1))
    if version != MANIFEST_VERSION:
        raise ValueError(f"manifest version must be {MANIFEST_VERSION}")

    shape_data = data.get("shape")
    if not isinstance(shape_data, list) or len(shape_data) != 2:
        raise ValueError("manifest shape must have two dimensions")

    shape = tuple(int(value) for value in shape_data)
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("manifest shape must contain positive integers")

    raw_shards = data.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("manifest shards must be a list")

    shards: list[ShardSpec] = []
    for raw_shard in raw_shards:
        if not isinstance(raw_shard, dict):
            raise ValueError("each shard entry must be a mapping")

        shard_path = Path(raw_shard["path"])
        if not shard_path.is_absolute():
            shard_path = path.parent / shard_path
        shards.append(
            ShardSpec(
                path=shard_path,
                row_offset=int(raw_shard.get("row_offset", 0)),
                col_offset=int(raw_shard.get("col_offset", 0)),
            )
        )

    return ShardManifest(format=manifest_format, version=version, shape=shape, shards=tuple(shards))


def assemble_shards(source: str | Path | ShardManifest) -> sparse.csr_matrix:
    if isinstance(source, ShardManifest):
        manifest = source
    else:
        manifest = load_shard_manifest(source)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    for shard in manifest.shards:
        matrix = load_matrix(shard.path).tocoo()
        row_indices = matrix.row + shard.row_offset
        col_indices = matrix.col + shard.col_offset

        if row_indices.size and (row_indices.max() >= manifest.shape[0] or row_indices.min() < 0):
            raise ValueError(f"row offsets in {shard.path} exceed the target shape")
        if col_indices.size and (col_indices.max() >= manifest.shape[1] or col_indices.min() < 0):
            raise ValueError(f"column offsets in {shard.path} exceed the target shape")

        rows.append(row_indices.astype(np.intp, copy=False))
        cols.append(col_indices.astype(np.intp, copy=False))
        data.append(matrix.data)

    if not rows:
        return sparse.csr_matrix(manifest.shape)

    stacked_rows = np.concatenate(rows)
    stacked_cols = np.concatenate(cols)
    stacked_data = np.concatenate(data)
    assembled = sparse.coo_matrix((stacked_data, (stacked_rows, stacked_cols)), shape=manifest.shape)
    assembled.sum_duplicates()
    return assembled.tocsr()