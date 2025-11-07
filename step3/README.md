# Step 3: Haliax Integration

Merge Haliax repository with full Git history and integrate as workspace member.

**Status**: In development

## Usage

```bash
# From repo root, after steps 1 & 2 (workspace with lib/marin/ and lib/levanter/)
./workspace-migration/step3/main.sh [options]

# Options:
#   -l, --lock-ref REF        Use uv.lock from specified git ref instead of re-resolving
#                             (saves 5-10 minutes during testing)
#   -r, --haliax-repo PATH    Path to Haliax repo (default: ../haliax)

# Examples:
./workspace-migration/step3/main.sh                          # Default
./workspace-migration/step3/main.sh --lock-ref ws-3         # Skip uv sync
./workspace-migration/step3/main.sh -r ~/haliax             # Custom path
```

## What It Does

**Prerequisites**: Steps 1 and 2 must be complete (workspace with `lib/marin/` and `lib/levanter/`)

### Part 1: Prepare Haliax Branch

1. Extract Haliax SHA from `uv.lock` (currently used version)
2. Create `haliax-pkg` branch from extracted SHA
3. Move all Haliax files to `lib/haliax/`
4. Preserve full Git history

### Part 2: Merge into Workspace

1. Merge `haliax-pkg` into current branch (with `--allow-unrelated-histories`)
2. Resolve conflicts:
   - `pyproject.toml`: Keep workspace root version
   - `.github/`: Keep workspace version (Haliax workflows migrated in Part 5)
3. Update workspace configuration:
   - Add `lib/haliax` to workspace members
   - Add `haliax` to root dependencies (experiments may import)
   - Change marin's and levanter's haliax dependencies from PyPI to workspace reference
4. Update `uv.lock` (or use `--lock-ref` to skip)

### Part 3: Migrate Workflows (if any)

1. **Rename workflows**: Prefix with `haliax-`
2. **Update paths**: Change relative paths to `lib/haliax/...`
3. **Add path restrictions**: Only trigger on `lib/haliax/**` changes
4. **Remove** `lib/haliax/.github/`

## Dependencies

Same as step 2:
- **tomlkit**: TOML transformations
- **Python 3.11+**: For `tomllib`

## Key Differences from Step 2

- **Simpler**: Haliax is a pure library (no workflows, infra, etc.)
- **No dolma**: Only need to handle Haliax itself
- **No ReadTheDocs**: Haliax documentation stays where it is
- **Fewer conflicts**: Smaller footprint = fewer merge conflicts

## Testing

```bash
# Test on a clean branch
git checkout -b test-ws-3 ws-2  # Start from step 2 result
./workspace-migration/step3/main.sh

# Verify structure
ls -la lib/haliax/
cat pyproject.toml  # Check workspace members
uv tree | grep haliax  # Verify workspace dependency

# Clean up
git reset --hard ws-2
```

## Expected Result

```
marin/
  pyproject.toml        # Workspace root + experiments
  experiments/
  .github/workflows/
    marin-*.yaml
    levanter-*.yaml
    haliax-*.yaml       # If any
  lib/
    marin/              # Workspace member
    levanter/           # Workspace member
    haliax/             # Workspace member (NEW)
      pyproject.toml
      src/haliax/
      tests/
      README.md
```

## Links

- **Issue**: [#1773]
- **Previous**: [Step 2](../step2/README.md) - Levanter integration
- **Next**: [Step 4](../step4/README.md) - Thalas/executor integration

[#1773]: https://github.com/marin-community/marin/issues/1773
