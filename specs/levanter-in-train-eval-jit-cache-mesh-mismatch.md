# Levanter in-train eval crash: `stack_tree` jit-cache CPU↔TPU mesh mismatch

**Status**: Option A implemented (drop `@jax.jit` from `stack_tree`) — `21605b6492` on `rw/levanter-eval-mesh-wrap-fix`. PENDING multi-host TPU verification. See "Update 2026-06-14 (marin session)" below.
**Filed**: 2026-06-14 from tomat session — see https://wandb.ai/open-athena/tomat-lmq-P19/runs/train-mg-kl-bin5-cos-cont for the most recent live repro.
**Branch**: `rw/levanter-eval-mesh-wrap-fix` (committed but ineffective on multi-host TPU); see `lib/levanter/src/levanter/eval.py:493-505`.

## TL;DR

On multi-host TPU, **in-training eval** crashes intermittently with:

```
ValueError: Received incompatible devices for jitted computation.
Got argument individual_datums[0][1] of stack_tree with shape int32[1]
  on platform TPU
and jit's context mesh with device ids [<CPU id>] on platform CPU
```

Crash site: `lib/levanter/src/levanter/data/loader.py:377` — the module-level `@jax.jit` decorated `stack_tree` function, invoked from inside `jax.make_array_from_callback`'s `data_callback`.

**The intermittent part is the killer**: in our most recent multi-host bin5 run (`steps_per_eval=22500`, 28 trainer restarts over 67500 steps), evals at step 22500 and 45000 **succeeded**; step 67500 **crashed**. Whether `stack_tree`'s jit-cache locks to a TPU-compatible mesh or a CPU-only mesh depends on which thread happens to trace it first across restarts.

A wrap of `evaluator.evaluate(model)` in `hax.partitioning.set_mesh + hax.axis_mapping` was tried (`b52ab82d` on `rw/levanter-eval-mesh-wrap-fix`) — it deterministically reproduces the workaround used in tomat's `scripts/backfill_vl_modal.py:340-348`. **It works on single-host H200 but does NOT hold on multi-host TPU** — verified live on a fresh v5p-16 cos-cont fire on 2026-06-13 (crashed at step 73500 with the same signature).

## Update 2026-06-14 (marin session)

Implemented **Option A** (drop `@jax.jit` from `stack_tree`, `loader.py`) as
`21605b6492`. Lint + `lib/levanter/tests/test_new_loader.py` pass. Two findings
from the marin side that revise this spec:

1. **`jax.set_mesh` is thread-local, not global** (verified on JAX 0.10 here): a
   concurrent thread holding `set_mesh(meshB)` does NOT change another thread's
   active mesh. So the "producer's `local_cpu_mesh()` leaks into the consumer
   thread" sub-theory is **ruled out**. The shared-jit-cache framing is the
   surviving explanation and is consistent with the code.

2. **This cannot be reproduced or verified off multi-host TPU** — so the
   "CPU 2-device mesh unit test" proposed below is **not feasible**. I confirmed
   it: two disjoint CPU meshes cross-calling `stack_tree` *succeed*, because (as
   noted at the end of "Why the wrap-evaluate fix doesn't help") CPU permits the
   device transfer the failure depends on. There is no local fail-before /
   pass-after test. **Only the live tomat v5p repro can verify this fix.**

**Why Option A is still right regardless of the exact mechanism**: removing the
`@jax.jit` removes the compiled context-mesh binding and the cross-mesh cache
entirely, so the fan-in `hax.stack`/`jnp.stack` calls dispatch eagerly on each
input's own devices — nothing to mismatch.

**Caveat for the v5p check**: `stack_tree` runs not only in the background
producer but also at `loader.py:377` in `_batchify_local_data`, which is
per-step on the main trainer thread. So glance at MFU on the v5p run, not just
correctness — the lost fusion is almost certainly negligible (a few stacks per
batch) but should be confirmed, not assumed.

### Rebase onto `m/main` (#6331) — commits dropped

The comprehensive tomat branch (`rw/levanter-eval-mesh-wrap-fix`) was rebased
onto current `m/main`. Dropped during the rebase:

- The 3 in-flight fixes that **merged upstream** (PRs #6316 flash-None-mask,
  #6319 cache-fallthrough, #6320 jax_init-idempotent) — now in `m/main`.
- The **#5594 reverts** (`undo hoist of iris imports` + the test-side revert).
  Verified unnecessary: in a fresh process `import levanter.distributed` on
  `m/main` (which fires the module-top `iris.*` + `levanter.megascale` imports)
  leaves `xla_bridge.backends_are_initialized() == False` — the imports never
  touch the XLA backend, so they cannot break `jax.distributed.initialize()`.
  The revert's premise is false (and the test-side revert no longer applied —
  upstream refactored that test away).
- The **eval-wrap** (`b52ab82d`, `_evaluate_under_mesh`). It was a workaround
  for this same jit-cache root cause and is confirmed ineffective on multi-host
  TPU; the `stack_tree` de-jit fixes the root cause on all platforms (eager
  dispatch has no cached compilation to mismatch, so it also covers the
  single-host H200 path the wrap targeted), making the wrap redundant. It also
  conflicted with upstream's refactored `eval.py`.

Kept: BUILD_DATE stamp, the two open-PR fixes (#6317 BackgroundIterator, #6318
PassthroughTokenizer), `ConcatDatasetComponent`, `e9ca207` (set_mesh-config
snapshot), and the `stack_tree` de-jit. Pre-rebase state preserved at
`backup/eval-mesh-wrap-fix-pre-rebase`.

## Diagnosis

### What `stack_tree` does

```python
# lib/levanter/src/levanter/data/loader.py:564-572
@functools.partial(jax.jit, static_argnums=(0,))
def stack_tree(batch_name, individual_datums):
    def _stack_leaves_unchecked(*leaves):
        if is_named_array(leaves[0]):
            return hax.stack(batch_name, leaves)
        else:
            return jnp.stack(leaves)
    return jax.tree.map(_stack_leaves_unchecked, *individual_datums, is_leaf=is_named_array)
```

Module-level `@jax.jit`. JAX caches the compiled version keyed by `(traced-axes, input-shape, mesh)`. Once cached against a particular mesh, subsequent calls re-use that compilation **unless the cache-key changes**.

### Who traces it under which mesh

There are two sites that invoke `stack_tree` during eval:

1. **Producer thread** (`_JaxCpuBackgroundIterator._fill_queue_with_batches` → `_produce_batches`): enters `with local_cpu_mesh():` (`lib/levanter/src/levanter/utils/jax_utils.py:54`) which routes through `haliax.partitioning.set_mesh(cpu_mesh)`. On JAX ≥ 0.5, that's `jax.set_mesh(cpu_mesh)` — sets the active mesh to a 1-CPU-device mesh. When `stack_tree` is invoked here, JAX caches a CPU-mesh compilation.

2. **Consumer thread** (main trainer): inside `_batchify_local_data` → `make_global_array_for_leaf` → `jax.make_array_from_callback(..., data_callback)`. The `data_callback` runs in the consumer thread and eventually calls `get_local_batch(begin, end)[leaf_index]` → which invokes `stack_tree` again. At this point the consumer is inside the trainer's `with use_device_mesh():` block (TPU mesh).

If the producer thread's first `stack_tree` trace beats the consumer to it (which happens after some restarts, not others — race on thread-startup order + first-touch), the JAX cache locks to CPU. The consumer's later invocation under TPU mesh trips the validator: cached compilation's context-mesh = CPU, current jit-context = TPU → `Received incompatible devices`.

### Why the wrap-evaluate fix doesn't help

The wrap I added (`b52ab82d`) puts `hax.partitioning.set_mesh(self.device_mesh)` + `hax.axis_mapping(self.axis_mapping)` around `evaluator.evaluate(model)`. That sets the **consumer thread's** active mesh during the eval call. But:

- The producer thread (which already exists from the data loader) is **outside** this scope. It's still inside `local_cpu_mesh`.
- The cache key for `stack_tree` is determined by what mesh is active when the trace happens — **once per cache entry**. If the producer has already traced under CPU (which it does, since the BG iter prefetches batches eagerly), the consumer's TPU-mesh re-invocation is a cache **hit** against the CPU-traced version, NOT a new trace.
- So the wrap is a no-op for the actual `stack_tree` cache — the cache is already polluted.

This is why `backfill_vl_modal.py`'s same pattern works on single-host H200 but ours doesn't on multi-host TPU: on single-host with `jax.local_devices(backend="cpu")` returning a CPU device that doesn't collide with TPU device-IDs, the consumer's TPU-resident inputs can be **transferred** into the CPU-compiled function without raising. On multi-host TPU, device-ID partitioning makes that transfer impossible to fake.

## Proposed fixes (ranked)

### Option A: drop `@jax.jit` from `stack_tree` *(simplest, recommended)*

`stack_tree` is just `jax.tree.map(_stack_leaves_unchecked, ...)` over `hax.stack` / `jnp.stack`. Both `hax.stack` and `jnp.stack` are already efficient — the outer `@jax.jit` exists to avoid Python-side tree-traversal overhead per batch, not for fusion benefits (the contents are bounded fan-in stacks).

Drop the `@jax.jit` (or guard it with a `with_sharding_constraint` that re-binds mesh per-call). No more cache → no more mesh mismatch.

Cost: ~Python overhead of `jax.tree.map` per batch. Worth measuring vs the maintenance cost of the bug.

**This is what I'd try first.** A 10-line diff plus a single benchmark.

### Option B: `device_get` inputs to CPU before `stack_tree`

In `get_local_batch` (`loader.py:377`), call `jax.device_get(local_data)` before passing to `stack_tree`. Forces the input data onto CPU; `stack_tree`'s CPU-mesh compilation is then a legitimate match. Loses any benefit from device-resident batching but is a localized fix.

Cost: an extra D2H transfer per batch in the data path. Small for the loader's modest tensors.

### Option C: remove `local_cpu_mesh` from `_JaxCpuBackgroundIterator`

Stop scoping the producer thread's mesh to CPU. The intent — keeping data-prep on CPU — was a JAX 0.3-era pattern; modern JAX handles backend selection for `jnp.stack` etc. on its own.

Cost: producer ops that used to be CPU-pinned may end up on TPU. Could re-introduce a different memory pressure issue if the loader does large CPU work. Needs benchmark.

### Option D: propagate consumer's mesh into producer's `local_cpu_mesh`

Make the BG iterator's producer thread snapshot the parent thread's mesh at iterator construction time (which `lib/levanter/src/levanter/utils/background_iterable.py` partially does via the `e9ca207` set_mesh-config-snapshot commit), then enter THAT mesh instead of `local_cpu_mesh`'s default CPU mesh.

Cost: more invasive — restructures the loader's CPU-isolation contract. Likely needs design discussion.

## Reproducer

1. Multi-host TPU (v5p-16 / v5p-32 / v6e-32 — anything with multiple iris tasks).
2. Levanter trainer with `steps_per_eval` set and `val_seqs > 0`.
3. Some preemptions / restarts to randomize the first-trace race (or just keep firing — eventually the race resolves "wrong").

Direct repro from this session:

```
./tomat train train-mg-kl-bin5-cos-cont --resume \
  --from-ckpt gs://marin-us-east5/tomat/results/train-mg-kl-bin5-fs-tpu/checkpoints/train-mg-kl-bin5-fs-tpu/step-73000 \
  -T v5p-16 --zone us-east5-a -s 90000 -b 128 --seed 42 \
  -D 'train-full-v3,train-full-v3-shard1,train-full-v3-shard2,train-full-v3-shard3' \
  -m 200M --lr 3e-4 --lr-schedule constant --warmup 0.0 \
  --val-seqs 256 --steps-per-eval 500 \
  --bucket gs://marin-us-east5/tomat ...
```

Crashes at the first eval boundary (~13 min in on v5p-16).

## What's already on the branches

- **`rw/levanter-partial-cache-fallthrough`** (current tomat marin pin): contains `e9ca207` — BackgroundIterator snapshots parent's `_jax_config.{abstract_mesh_context_manager, device_context}` and re-applies in producer thread. Real fix for a *related* bug (codex bot's PR #6317 review) but does NOT fix the `_JaxCpuBackgroundIterator`-overrides-it case.
- **`rw/levanter-eval-mesh-wrap-fix`**: contains `b52ab82d` — wraps `evaluator.evaluate(model)` in `hax.partitioning.set_mesh + hax.axis_mapping`. Live-verified ineffective on multi-host TPU (see above).

Both branches should stay merged-in for now (neither hurts) while we decide on A/B/C/D.

## Recommendation

Land **Option A** as a 10-line diff against `lib/levanter/src/levanter/data/loader.py` + a unit test that uses a CPU 2-device mesh to repro the cache-mismatch on `stack_tree`. Benchmark the per-batch overhead on a v5p-16 200M run for one epoch — if it's under ~1% MFU impact (very likely), ship it. If it's bigger, fall back to Option B (`device_get` localized fix).

Either way, the codex-bot fix on `BackgroundIterator` (parent-thread mesh-config snapshot) should stay — it's correct for the parent-mesh-propagation case and the codex review is sound.

## Related

- PR #6317 (BackgroundIterator parent-mesh ContextVars + thread_resources.stack snapshot — landed locally as `d133891306` and `cb9abf1dbe` on the OA fork, plus codex review's set_mesh-config snapshot as `e9ca207`).
- tomat `scripts/backfill_vl_modal.py:340-348` — the workaround that works for single-host H200 Modal and motivated `rw/levanter-eval-mesh-wrap-fix`.
- tomat memory `feedback_dont_misframe_experiments` — verification needs a multi-host TPU run, single-host is not sufficient.
