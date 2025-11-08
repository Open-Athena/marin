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

### Part 3: Update Workspace Configuration

1. Add `lib/haliax` to workspace members
2. Add `haliax` workspace source
3. Add `haliax` to root dependencies (experiments may import)
4. **Alphabetize** workspace members, sources, and dependencies
5. Change levanter's haliax dependency from PyPI to workspace reference
   - **Preserves formatting**: Uses tomlkit in-place modification to keep multi-line arrays and comments

### Part 4: Update uv.lock

1. Run `uv sync` to regenerate lock with workspace haliax
2. Or use `--lock-ref` to skip (copies from specified git ref)

### Part 5: Migrate Workflows

1. **Rename workflows**: Prefix with `haliax-`
2. **Update workflow names**: Add "Haliax - " prefix
3. **Update for workspace**: Adjust paths and working directories
4. **Remove** `lib/haliax/.github/`

### Part 6: Commit Workspace Integration

1. Commit all workspace integration changes
2. Include updated `uv.lock` with workspace haliax reference

### Part 7: Update Pre-commit Configuration

1. Add Haliax-specific pre-commit rules to `infra/pre-commit.py`:
   - Separate Black config (line-length 119 vs Marin's 121)
   - Separate license header (Haliax vs Marin/Levanter)
   - Exclude `lib/haliax/` from general Python config
2. Run `./infra/pre-commit.py --all-files --fix`
3. Commit pre-commit config updates

## Technical Details

### TOML Formatting Preservation

Critical improvement over naive approach:
- **In-place modification**: Modifies tomlkit arrays directly instead of replacing with Python lists
- **Preserves formatting**: Multi-line dependency arrays, inline comments, whitespace all preserved
- **Example**:
  ```python
  # Bad: Replaces array, loses formatting
  deps = doc["project"]["dependencies"]
  new_deps = [d for d in deps if ...]
  doc["project"]["dependencies"] = new_deps  # ❌ Collapses to single line

  # Good: Modifies in-place
  deps = doc["project"]["dependencies"]
  for i, dep in enumerate(deps):
      if needs_update(dep):
          deps[i] = updated_value  # ✅ Preserves formatting
  ```

### Alphabetization

Workspace configuration is automatically alphabetized for maintainability:
- Workspace members: `lib/haliax`, `lib/levanter`, `lib/marin`, `lib/zephyr`
- Root dependencies: Same alphabetical order
- Workspace sources: Alphabetized dict keys

## Dependencies

- **tomlkit**: TOML transformations with formatting preservation
- **Python 3.11+**: For `tomllib` (read-only parsing)

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
