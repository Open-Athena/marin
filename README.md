# Marin + Levanter Workspace Migration

This directory contains migration scripts for the [uv workspace migration plan][#1773].

**Repository**: The workspace-migration scripts live in [Open-Athena/marin/tree/rw/wm][wm-branch], not the old Gist.

## Overview

- **Step 1 🚧**: Initialize workspace
  - Move `marin` package under `lib/marin/`
  - Root project contains `experiments/` (depends on `lib/marin`)
  - `data_browser` stays independent (not a workspace member)
  - Draft PR: [#1690]

- **Step 2 🚧**: Add Levanter as workspace member
  - Merge Levanter repo with full Git history
  - Move Levanter to `lib/levanter/`
  - Migrate workflows to monorepo structure
  - Draft PR: [#1723]

- **Step 3 ✋**: Haliax integration (coming soon)

## Contents

- [Step 1: Initialize Workspace](#step-1)
  - [Usage](#step-1-usage)
  - [What It Does](#step-1-what-it-does)
  - [Files](#step-1-files)
  - [Testing](#step-1-testing)
  - [Result](#step-1-result)
- [Step 2: Levanter Integration](#step-2)
  - [Usage](#step-2-usage)
  - [What It Does](#step-2-what-it-does)
  - [Files](#step-2-files)
  - [Testing](#step-2-testing)
  - [Result](#step-2-result)
- [Step 3: Haliax Integration](#step-3)

---

## Step 1: Initialize Workspace <a id="step-1"></a>

Convert the marin repo to a uv workspace with `lib/marin/` as the main package.

**Draft PR**: [#1690]

### Usage <a id="step-1-usage"></a>

```bash
# From repo root
./workspace-migration/step1/main.py
```

The script is idempotent and hermetic - it creates the workspace structure from the current state.

### What It Does <a id="step-1-what-it-does"></a>

1. **Move package**: `src/` → `lib/marin/src/`
2. **Create workspace root**: Transform root `pyproject.toml` to workspace config
3. **Create member**: Create `lib/marin/pyproject.toml` for the marin package
4. **Update paths**: Fix imports and references in:
   - `.github/workflows/*.yaml` - CI workflow paths
   - `Makefile` - Build script paths
   - `mkdocs.yml` - Documentation paths
   - Documentation files - GitHub blob URLs
5. **Update lockfile**: Run `uv sync` (preserves package versions)
6. **Commit**: Create migration commit

### Files <a id="step-1-files"></a>

#### [`step1/main.py`]
Main migration script. Pure Python implementation that orchestrates the entire step 1 migration.

#### [`step1/transform_pyprojects.py`]
Transforms `pyproject.toml` files using `tomlkit` to:
- Convert root to workspace config
- Create `lib/marin/pyproject.toml` from original
- Preserve formatting and comments

#### [`step1/update_paths.py`]
Updates file paths throughout the codebase:
- CI workflows
- Documentation
- Build scripts

#### [`step1/update_ci_docs.py`]
Updates CI and documentation configs for workspace structure.

#### [`step1/test.sh`]
Test harness that verifies step 1 is reproducible:
1. Creates ephemeral test branch from parent commit
2. Runs `main.py`
3. Compares resulting git tree with expected state
4. Cleans up on success

### Testing <a id="step-1-testing"></a>

```bash
# Test reproducibility
./workspace-migration/step1/test.sh

# Manual verification
cd path/to/migrated/repo
uv sync                                    # Should complete successfully
uv run python -c "import marin; print('✓')"  # Should work
uv run pytest tests/                       # Should pass
make check                                 # Should pass
```

### Result <a id="step-1-result"></a>

After step 1:

```
marin/
  pyproject.toml        # Workspace root
  experiments/          # Stays at root, imports from lib/marin
  lib/
    marin/              # Workspace member (marin package)
      pyproject.toml
      src/marin/
```

**Note**: `data_browser` stays independent (not moved to `lib/`).

---

## Step 2: Levanter Integration <a id="step-2"></a>

Merge Levanter repository with full Git history and integrate as workspace member.

**Draft PR**: [#1723]

### Usage <a id="step-2-usage"></a>

```bash
# From repo root, starting from ws branch (after step 1)
./workspace-migration/step2/main.sh [options] [levanter-repo-path] [levanter-ref]

# Options:
#   -l, --lock-ref REF    Use uv.lock from specified git ref instead of re-resolving
#                         (saves 5-10 minutes during testing)

# Examples:
./workspace-migration/step2/main.sh                           # Default: ../levanter, HEAD
./workspace-migration/step2/main.sh --lock-ref ws-2          # Skip uv sync, use existing lock
./workspace-migration/step2/main.sh ../levanter main         # Specific ref
```

**Standalone workflow migration** (if you already have the merge commit):
```bash
./workspace-migration/step2/migrate_workflows_standalone.sh
```

### What It Does <a id="step-2-what-it-does"></a>

**Part 1**: Prepare Levanter branch
1. Create `levanter-pkg` branch from Levanter repo
2. Move all Levanter files to `lib/levanter/`
3. Preserve full Git history

**Part 2**: Merge into workspace
1. Merge `levanter-pkg` into current branch
2. Resolve `pyproject.toml` conflicts (workspace vs package config)
3. Update workspace `pyproject.toml` to include `lib/levanter`
4. Update `uv.lock` for new structure (or use `--lock-ref` to skip)

**Part 3**: Migrate workflows
1. Rename Marin workflows: `*.yaml` → `marin-*.yaml`
2. Move Levanter workflows: `lib/levanter/.github/workflows/*.yaml` → `.github/workflows/levanter-*.yaml`
3. Update workflow content:
   - Add "Marin - " / "Levanter - " prefixes to workflow names
   - Add `working-directory: lib/levanter` to Levanter jobs
   - Add path filters to trigger only on relevant changes
   - Update `uv` commands to use `--package levanter`
4. Update TPU setup scripts for monorepo structure

### Files <a id="step-2-files"></a>

#### [`step2/main.sh`]
Main step 2 migration script. Hermetic bash script that orchestrates all 3 parts.

#### [`step2/migrate_workflows_standalone.sh`]
Standalone script for Part 3 (workflow migration). Useful for:
- Re-running just the workflow migration
- Testing workflow transformations
- Comparing with upstream

#### [`step2/update_workflows.py`]
Updates GitHub Actions workflows using [yaya] for YAML transformations:
- Uses `insert_key_between()` API for safe, verified insertions
- Handles all GitHub Actions job structures (strategy, env, permissions, etc.)
- Zero conflicts on all Levanter workflows

#### [`step2/update_tpu_setup.py`]
Updates TPU setup scripts (`lib/levanter/infra/helpers/setup-tpu-vm*.sh`):
- Change repo URL to marin monorepo
- Add `--package levanter` to uv commands
- Create `venv_path.txt` for monorepo venv location

#### [`step2/update_workspace_toml.py`]
Updates workspace `pyproject.toml` to add `lib/levanter` member.

#### [`step2/sync.sh`]
Sync Levanter updates from upstream (for keeping `lib/levanter` in sync).

#### [`step2/test.sh`]
Test harness for verifying step 2 reproducibility.

#### [`step2/STEP2_ISSUES.md`]
Documents issues encountered during manual step 2 completion and their fixes.

### Testing <a id="step-2-testing"></a>

```bash
# Test reproducibility
./workspace-migration/step2/test.sh

# Test with lock file reuse (faster)
./workspace-migration/step2/main.sh --lock-ref ws-2

# Manual verification
uv sync                                      # Should complete
uv run --package marin pytest tests/        # Marin tests
uv run --package levanter pytest lib/levanter/tests/  # Levanter tests
```

### Result <a id="step-2-result"></a>

After step 2:

```
marin/
  pyproject.toml        # Workspace root
  experiments/
  .github/workflows/
    marin-*.yaml        # Marin workflows
    levanter-*.yaml     # Levanter workflows
  lib/
    marin/              # Workspace member
      pyproject.toml
      src/marin/
    levanter/           # Workspace member (NEW)
      pyproject.toml
      src/levanter/
      infra/
```

**Git history**: Full Levanter commit history is preserved in the merged branch.

---

## Step 3: Haliax Integration <a id="step-3"></a>

**Status**: Not yet implemented

See the [uv workspace migration plan][#1773] for details.

Expected structure after step 3:

```
marin/
  pyproject.toml
  experiments/
  lib/
    marin/
    levanter/
    haliax/           # NEW
```

---

## Technical Notes

### Dependencies

- **uv**: Workspace and package management
- **lossless-yaml** (v0.1.0+): YAML transformations (for workflow migrations)
  - Published on PyPI as `lossless-yaml`, imported as `yaya`
  - Uses `insert_key_between()` API for safe ordered insertions
  - Uses `replace_key()` with list indices for `.readthedocs.yaml` updates
  - Fixes GitHub Actions jinja2 expression handling
- **tomlkit**: TOML transformations (preserves formatting)

### Key Improvements

**Step 2 workflow migration** (vs. old approach):
- Uses `insert_key_between()` instead of `add_key_after`
- Verifies key adjacency before inserting
- Handles all GitHub Actions structures: `strategy`, `env`, `permissions`, `needs`, etc.
- **Result**: 0 conflicts (was 9 conflicts before)

### Reproducibility

Both step 1 and step 2 are designed to be:
- **Hermetic**: Don't depend on external state
- **Idempotent**: Can be run multiple times safely
- **Testable**: Included test harnesses verify reproducibility

---

## Links

[#1773]: https://github.com/marin-community/marin/issues/1773
[#1690]: https://github.com/marin-community/marin/pull/1690
[#1723]: https://github.com/marin-community/marin/pull/1723
[wm-branch]: https://github.com/Open-Athena/marin/tree/rw%2Fwm
[yaya]: https://github.com/ryan-williams/yaya

### File Links

<!-- Step 1 files on rw/ws branch (PR #1690) -->
[`step1/main.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/main.py
[`step1/transform_pyprojects.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/transform_pyprojects.py
[`step1/update_paths.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/update_paths.py
[`step1/update_ci_docs.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/update_ci_docs.py
[`step1/test.sh`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/test.sh

<!-- Step 2 files on rw/ws-2 branch (PR #1723) -->
[`step2/main.sh`]: https://github.com/Open-Athena/marin/blob/rw%2Fws-2/workspace-migration/step2/main.sh
[`step2/migrate_workflows_standalone.sh`]: https://github.com/Open-Athena/marin/blob/rw%2Fws-2/workspace-migration/step2/migrate_workflows_standalone.sh
[`step2/update_workflows.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws-2/workspace-migration/step2/update_workflows.py
[`step2/update_tpu_setup.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws-2/workspace-migration/step2/update_tpu_setup.py
[`step2/update_workspace_toml.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws-2/workspace-migration/step2/update_workspace_toml.py
[`step2/sync.sh`]: https://github.com/Open-Athena/marin/blob/rw%2Fws-2/workspace-migration/step2/sync.sh
[`step2/test.sh`]: https://github.com/Open-Athena/marin/blob/rw%2Fws-2/workspace-migration/step2/test.sh
[`step2/STEP2_ISSUES.md`]: https://github.com/Open-Athena/marin/blob/rw%2Fws-2/workspace-migration/step2/STEP2_ISSUES.md
