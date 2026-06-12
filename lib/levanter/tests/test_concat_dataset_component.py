# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for `ConcatDatasetComponent` — virtual cache concat without rebuild.

The motivation is shard-union training: you have N pre-built TreeCaches
(e.g. tokenized shards of the same corpus) and want to train on their
union without paying the physical-consolidation cost of
`UrlDatasetSourceConfig(train_urls=[g0,g1,…])`.

These tests verify:

1. Construction validates inputs (non-empty children, homogeneous formats).
2. `build_token_datasets` over a `ConcatDatasetComponent` produces an
   `AsyncDataset` whose `len` equals the sum of child lengths.
3. Row-equality vs the by-hand baseline of reading each child cache in order.
4. `block_shuffle` over the concat dataset mixes across child boundaries (no
   shard is "early" or "late" beyond what its position in the cumulative
   layout implies once a sufficiently large window is applied).
"""

import tempfile

import numpy as np
import pytest

import haliax as hax

from levanter.data.text import (
    ConcatDatasetComponent,
    DatasetComponent,
    LmDataConfig,
    PrebuiltLmDatasetFormat,
    SupervisedLmDatasetFormat,
    BlockShuffleConfig,
)
from levanter.store.cache import SerialCacheWriter, TreeCache


def _build_synthetic_caches(
    root: str,
    *,
    child_rows: dict[str, int],
    doc_len: int = 32,
    vocab_size: int = 1024,
    seed: int = 0,
) -> dict[str, TreeCache]:
    """Build one TreeCache per child name with `child_rows[name]` rows.

    Each row's `input_ids` is filled with the row's GLOBAL ordinal so we can
    later verify which child a row came from and at what offset. The
    cache's `cache_dir` is `<root>/<name>/`.

    Returns a dict `{name: TreeCache}` keyed by child name.
    """
    rng = np.random.default_rng(seed)
    exemplar = {"input_ids": np.zeros((doc_len,), dtype=np.int32)}
    caches: dict[str, TreeCache] = {}
    global_ordinal = 0
    for name, n_rows in child_rows.items():
        cache_dir = f"{root}/{name}/train"
        with SerialCacheWriter(cache_dir, exemplar) as writer:
            for _ in range(n_rows):
                # First slot encodes the row's global ordinal; the rest is
                # random padding so the cache exercises non-trivial offsets.
                row = rng.integers(0, vocab_size, size=(doc_len,), dtype=np.int32)
                row[0] = global_ordinal
                writer.write_batch([{"input_ids": row}])
                global_ordinal += 1
        caches[name] = writer.result()
    return caches


def _child_component(cache_dir: str) -> DatasetComponent:
    """Source-less DatasetComponent that just points at a pre-built cache."""
    return DatasetComponent(
        source=None,
        cache_dir=cache_dir,
        format=PrebuiltLmDatasetFormat(input_ids_key="input_ids"),
    )


def test_construction_rejects_empty_children():
    with pytest.raises(ValueError, match="at least one child"):
        ConcatDatasetComponent(children={})


def test_construction_rejects_heterogeneous_formats():
    a = DatasetComponent(
        source=None,
        cache_dir="/tmp/a",
        format=PrebuiltLmDatasetFormat(input_ids_key="input_ids"),
    )
    b = DatasetComponent(
        source=None,
        cache_dir="/tmp/b",
        format=SupervisedLmDatasetFormat(),
    )
    with pytest.raises(ValueError, match="share a format type"):
        ConcatDatasetComponent(children={"a": a, "b": b})


def test_concat_dataset_length_sums_children():
    """len(concat_ds) == sum of child cache rows."""
    rows = {"a": 50, "b": 30, "c": 20}
    with tempfile.TemporaryDirectory() as tmp:
        child_caches = _build_synthetic_caches(tmp, child_rows=rows)
        flat_caches: dict[str, TreeCache] = {
            f"union/{name}": cache for name, cache in child_caches.items()
        }
        component = ConcatDatasetComponent(
            children={name: _child_component(cache.cache_dir) for name, cache in child_caches.items()},
        )
        config = LmDataConfig(
            components={"union": component},
            vocab_size=1024,
            tokenizer="passthrough",
            shuffle=False,
        )
        Pos = hax.Axis("position", 32)
        datasets = config.build_token_datasets(flat_caches, Pos, split="train")
        ds = datasets["union"]
        assert len(ds.as_sync_dataset()) == sum(rows.values())


def test_concat_dataset_row_equality_vs_baseline():
    """Reading the concat row-by-row matches concat-of-each-child by hand."""
    rows = {"a": 12, "b": 8, "c": 5}
    with tempfile.TemporaryDirectory() as tmp:
        child_caches = _build_synthetic_caches(tmp, child_rows=rows)
        flat_caches: dict[str, TreeCache] = {
            f"union/{name}": cache for name, cache in child_caches.items()
        }
        component = ConcatDatasetComponent(
            children={name: _child_component(cache.cache_dir) for name, cache in child_caches.items()},
        )
        config = LmDataConfig(
            components={"union": component},
            vocab_size=1024,
            tokenizer="passthrough",
            shuffle=False,
        )
        Pos = hax.Axis("position", 32)
        datasets = config.build_token_datasets(flat_caches, Pos, split="train")
        union_ds = datasets["union"].as_sync_dataset()

        # Each row's `input_ids[..., 0]` is its global ordinal; under
        # `shuffle=False` the concat presents rows in
        # children-declaration order (a, then b, then c), with ordinals
        # 0..49, 50..79, 80..99 respectively (matching the synthetic
        # ordinal we wrote).
        for i in range(sum(rows.values())):
            row = union_ds[i]
            assert int(np.asarray(row.tokens)[0]) == i, (
                f"row {i}: expected global-ordinal {i}, got {row.tokens[0]}"
            )


def test_concat_dataset_block_shuffle_mixes_children():
    """`BlockShuffleConfig` over the concat reaches all children's rows.

    With `window_blocks` covering the entire union, the shuffled stream
    should pull rows from every child within the first batch-worth of reads
    (not stay pinned to child `a` for the first N reads, which is what a
    too-small window would do over a concat'd stream).
    """
    # Three equal-size children so a "uniform mix" baseline is easy to state.
    rows = {"a": 64, "b": 64, "c": 64}
    n_total = sum(rows.values())
    doc_len = 8
    with tempfile.TemporaryDirectory() as tmp:
        child_caches = _build_synthetic_caches(
            tmp, child_rows=rows, doc_len=doc_len
        )
        flat_caches: dict[str, TreeCache] = {
            f"union/{name}": cache for name, cache in child_caches.items()
        }
        component = ConcatDatasetComponent(
            children={name: _child_component(cache.cache_dir) for name, cache in child_caches.items()},
        )
        config = LmDataConfig(
            components={"union": component},
            vocab_size=1024,
            tokenizer="passthrough",
            shuffle=BlockShuffleConfig(io_block_size=8, window_blocks=24),  # whole union
        )
        Pos = hax.Axis("position", doc_len)
        datasets = config.build_token_datasets(flat_caches, Pos, split="train")
        import jax as _jax
        shuffled = datasets["union"].block_shuffle(
            io_block_size=8, window_blocks=24, key=_jax.random.PRNGKey(0)
        ).as_sync_dataset()

        # Read the first half (n_total // 2) of the shuffled stream; with a
        # whole-union window, each child should contribute ~1/3 of those rows
        # (not 100% from `a`, which is the failure mode of a too-small window
        # over a deterministic concat).
        first_half_ords = [int(np.asarray(shuffled[i].tokens)[0]) for i in range(n_total // 2)]
        per_child = {"a": 0, "b": 0, "c": 0}
        for o in first_half_ords:
            if o < 64:
                per_child["a"] += 1
            elif o < 128:
                per_child["b"] += 1
            else:
                per_child["c"] += 1
        # Each child should have contributed at least 5% of the half-stream —
        # easy bar to clear if the window genuinely spans the union; impossible
        # if the shuffle is per-child.
        expected_floor = (n_total // 2) * 0.05
        assert per_child["a"] >= expected_floor, f"child a only contributed {per_child['a']}/{n_total // 2}"
        assert per_child["b"] >= expected_floor, f"child b only contributed {per_child['b']}/{n_total // 2}"
        assert per_child["c"] >= expected_floor, f"child c only contributed {per_child['c']}/{n_total // 2}"
