# Step 1: Initialize Workspace

Convert the marin repo to a uv workspace with `lib/marin/` as the main package.

**Status**: Complete
**PR**: [#1690]

## Usage

```bash
# From repo root
./workspace-migration/step1/main.py
```

The script is idempotent and hermetic - it creates the workspace structure from the current state.

## What It Does

1. **Move package**: `src/marin` → `lib/marin/src/marin`
2. **Transform pyproject.toml files**:
   - Root becomes workspace config + experiments package
   - Create `lib/marin/pyproject.toml` with all deps/extras
3. **Update file paths**:
   - `.github/workflows/*.yaml` - CI workflow paths
   - `Makefile` - Build script paths
   - `mkdocs.yml` - Documentation paths
   - Documentation files - GitHub blob URLs
4. **Update configs**:
   - CI: Use `--extra=marin:cpu` syntax
   - ReadTheDocs: Install from `lib/marin`
   - `ray_deps.py`: Change PYTHONPATH from `["src", "experiments"]` to `["lib/marin/src", "experiments"]`
5. **Update lockfile**: Run `uv lock` (preserves package versions)
6. **Commit**: Create migration commit

## Files

### [`main.py`]
Main orchestration script. Pure Python implementation that runs all transformation steps.

### [`transform_pyprojects.py`]
Transforms `pyproject.toml` files using [tomlkit]:
- Convert root to workspace config
- Create `lib/marin/pyproject.toml` from original
- Preserve formatting and comments

### [`update_paths.py`]
Updates file paths throughout the codebase:
- CI workflows (GitHub Actions)
- Documentation (mkdocs, README links)
- Build scripts (Makefile)

### [`update_ci_docs.py`]
Updates CI and documentation configs for workspace structure:
- GitHub Actions: `--extra=marin:cpu` syntax
- ReadTheDocs: Install `lib/marin`

### [`update_ray_deps.py`]
Updates `ray_deps.py` PYTHONPATH for workspace structure:
- Changes `["src", "experiments"]` to `["lib/marin/src", "experiments"]`
- Fixes `ModuleNotFoundError` when running Ray jobs

### [`test.sh`]
Test harness that verifies step 1 is reproducible:
1. Creates ephemeral test branch from parent commit
2. Runs `main.py`
3. Compares resulting git tree with expected state
4. Cleans up on success

## Testing

```bash
# Test reproducibility
./workspace-migration/step1/test.sh

# Manual verification
cd path/to/migrated/repo
uv sync                                     # Should complete successfully
uv run python -c "import marin; print('✓')" # Should work
uv run pytest tests/                        # Should pass
make check                                  # Should pass
```

## Result

After step 1:

```
marin/
  pyproject.toml        # Workspace root + experiments package
  experiments/          # Stays at root, imports from lib/marin
  lib/
    marin/              # Workspace member (marin package)
      pyproject.toml    # All deps/extras, no tool configs
      src/marin/        # Marin source code
```

**Notes**:
- Tool configs (black, ruff, mypy, pytest) stay at workspace root
- `data_browser/` stays independent (not a workspace member)
- `experiments/` imports from `lib/marin` via workspace dependency

## Next Steps

After step 1 completes, proceed to [step 2](../step2/README.md) to integrate Levanter.

[#1690]: https://github.com/marin-community/marin/pull/1690
[tomlkit]: https://github.com/sdispater/tomlkit
[`main.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/main.py
[`transform_pyprojects.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/transform_pyprojects.py
[`update_paths.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/update_paths.py
[`update_ci_docs.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/update_ci_docs.py
[`update_ray_deps.py`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/update_ray_deps.py
[`test.sh`]: https://github.com/Open-Athena/marin/blob/rw%2Fws/workspace-migration/step1/test.sh
