# Workspace Migration - Step 1

This directory contains the migration script and notes for **Step 1** of the uv workspace migration plan: initializing the workspace and folding `marin` and `data_browser` members into `lib/`.

## Gist Contents

- [`step-1.sh`](#step-1sh) - Main migration script for workspace initialization
- [`test-step-1.sh`](#test-step-1sh) - Test script to verify migration reproducibility
- [`pyproject-root.patch`](#pyproject-rootpatch) - Patch to transform root pyproject.toml to workspace root
- [`pyproject-lib-marin.patch`](#pyproject-lib-marinpatch) - Patch to create lib/marin/pyproject.toml
- [`ray_deps.patch`](#ray_depspatch) - Patch to fix ray_deps.py for workspace structure
- [`resolve-conflicts.sh`](#resolve-conflictssh) - Helper for resolving merge conflicts during migration
- [`step-2*.sh`](#step-2-scripts) - Scripts for Step 2 (Levanter integration)
- [`README.md`](#readmemd) - This file

## Migration Script

Run `./workspace-migration/step-1.sh [REFERENCE_BRANCH]` from the repo root to replay the workspace restructuring on a clean branch.

The script accepts an optional "reference branch" argument, which it can copy from (for files whose content is essentially new in "step 1", not a modification of existing files). If not provided, it will fall back to `rw/ws` on a `marin-community/marin` remote.

## Files Updated by Script

The migration script automatically handles all path updates:

1. **`.github/workflows/build-docker-images.yaml`** - Update `src/marin/cluster/config.py` → `lib/marin/src/marin/cluster/config.py`
2. **`.github/workflows/update-leaderboard.yml`** - Update `src/marin/speedrun/` → `lib/marin/src/marin/speedrun/`
3. **`Makefile`** - Update both macOS and Linux sed commands for config.py path
4. **`mkdocs.yml`** - Update `paths: [".", "src"]` → `paths: [".", "lib/marin/src"]`
5. **`.gitignore`** - Remove `lib/` and `CLAUDE.md` exclusions
6. **`pyproject.toml`** - Create workspace root and `lib/marin/pyproject.toml` from reference branch
7. **`CLAUDE.md`** - Create from reference branch
8. **Documentation files** - Update GitHub URLs from `/blob/main/src/marin/` → `/blob/<doc-branch>/lib/marin/src/marin/` (uses `MARIN_DOC_BRANCH` env var, defaults to `main`)

## Testing the Migration

Run `./workspace-migration/test-step-1.sh` to verify the migration is reproducible. This creates an ephemeral test branch from the parent commit, replays the migration, and compares the resulting tree. On success, it cleans up and returns to the original branch.

### Not Impacted

The following do NOT need updates because they work with the new structure:

- **Import statements** (`from marin.X import Y`) - These work unchanged because the package is still named `marin` and installed as an editable package
- **`PYTHONPATH=tests:.`** - The `.` still refers to the repo root, and with the workspace the `marin` package is properly installed
- **Tests** - Import `marin` as a package, which works with the workspace setup
- **Experiments** - Import `from marin.X`, which works because the workspace root depends on the `marin` member

## Testing Checklist

After migration:

- [ ] `uv sync` completes successfully
- [ ] `uv run python -c "import marin; print('Success')"` works
- [ ] `uv run pytest tests/` passes
- [ ] `make check` passes
- [ ] `uv run mkdocs build` generates docs correctly
- [ ] Docker builds work (test locally if possible)
- [ ] Experiments can still import from `marin` package

## Result

After this step, the structure is:

```
marin/
  pyproject.toml        # Workspace root (marin-root)
  experiments/          # Stays at root
  lib/
    marin/              # Workspace member (marin package)
      pyproject.toml
      src/marin/
    data_browser/       # Workspace member
      pyproject.toml
      ...
```

## File Details

### step-1.sh
Main migration script that:
1. Moves `src/` → `lib/marin/src/` and `data_browser/` → `lib/data_browser/`
2. Applies patches to transform pyproject.toml files
3. Updates file paths in CI workflows, docs, and other config files
4. Runs `uv sync` to update lockfile for workspace structure (preserving package versions)
5. Commits the migration

### test-step-1.sh
Test harness that verifies step-1.sh is reproducible:
1. Creates ephemeral test branch from parent commit
2. Runs step-1.sh
3. Compares resulting git tree hash with original migration commit
4. Cleans up on success or leaves test branch for inspection on failure

### pyproject-root.patch
Transforms root `pyproject.toml` from package config to workspace root config. Key changes:
- Adds `[tool.uv.workspace]` with `members = ["lib/*"]`
- Preserves dependencies as workspace root dependencies
- Updates package name to `marin-root`

### pyproject-lib-marin.patch
Creates `lib/marin/pyproject.toml` from original root pyproject.toml. Key changes:
- Keeps `marin` as package name
- Moves package-specific dependencies and extras
- Updates paths for new structure

### ray_deps.patch
Fixes `lib/marin/src/marin/run/ray_deps.py` to work with workspace structure:
- Adds `--package marin` flag to `uv export` command
- Required because workspace root contains multiple packages and uv needs to know which package's extras to use

### resolve-conflicts.sh
Helper script for resolving merge conflicts during migration replays. Used when applying migration on branches that have diverged from main.

### Step 2 scripts
Scripts for Step 2 of the migration plan (Levanter integration):
- `step-2-init.sh` - Initialize Levanter as workspace member
- `step-2-sync.sh` - Sync Levanter updates from upstream
- `step-2.sh` - Main Step 2 migration script

## Next Steps

See the [uv workspace migration plan](https://github.com/marin-community/marin/blob/ws/CLAUDE.md#repo-reorg) for the complete roadmap:
- **Step 2**: Add Levanter as a workspace member
- **Step 3**: Add Haliax as a workspace member
- **Step Omega**: Further split into `marin-core`, `marin-crawl`, `ray_tpu`, `rl`, `thalas` packages
