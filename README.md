# Marin Workspace Migration Scripts

Migration scripts for the [uv workspace migration plan][#1773].

## Quick Start

From an arbitrary Marin `main`:

**Step 1 ([#1690])** - Initialize workspace (move marin to `lib/marin/`):
```bash
./workspace-migration/step1/main.py
```
See [step1/README.md](step1/README.md) for details.

**Step 2 ([#1723])** - Add Levanter as workspace member:
```bash
./workspace-migration/step2/main.sh
```
See [step2/README.md](step2/README.md) for details.

## Overview

### Step 1: Initialize Workspace ✅
**Status**: Merged ([#1690])

Convert marin to uv workspace with `lib/marin/` as main package member.

**Structure after**:
```
marin/
  pyproject.toml        # Workspace root + experiments package
  experiments/          # Root package, imports from lib/marin
  lib/
    marin/              # Workspace member
      pyproject.toml
      src/marin/
```

### Step 2: Levanter Integration ✅
**Status**: Merged ([#1723])

Merge [Levanter] repo (with full Git history) as `lib/levanter/` workspace member.

**Structure after**:
```
marin/
  pyproject.toml        # Workspace root + experiments package
  experiments/
  .github/workflows/
    marin-*.yaml        # Marin workflows
    levanter-*.yaml     # Levanter workflows
  lib/
    marin/              # Workspace member
    levanter/           # Workspace member (NEW)
      src/levanter/
      infra/
```

### Step 3: Thalas, Haliax, etc. ⏸️
**Status**: Not started

Future workspace members. See [#1773] for details.

## Technical Details

### Dependencies

- **uv**: Workspace and package management
- **[lossless-yaml]** (a.k.a. [yaya]): YAML transformations (workflow migrations)
- **[tomlkit]**: TOML transformations (preserves formatting/comments)

### Design Principles

All scripts are:
- **Hermetic**: No external state dependencies
- **Idempotent**: Safe to run multiple times
- **Testable**: Test harnesses verify reproducibility

### Directory Structure

```
workspace-migration/
  README.md             # This file
  step1/
    README.md           # Step 1 documentation
    main.py             # Step 1 main script
    *.py                # Step 1 helper scripts
    test.sh             # Step 1 test harness
  step2/
    README.md           # Step 2 documentation
    main.sh             # Step 2 main script
    *.py                # Step 2 helper scripts
    test.sh             # Step 2 test harness
```

## Links

- **Issue**: [#1773] - Workspace migration plan
- **PRs**: [#1690] (step 1), [#1723] (step 2)
- **Branch**: [rw/wm] - Migration scripts (this directory)

[#1773]: https://github.com/marin-community/marin/issues/1773
[#1690]: https://github.com/marin-community/marin/pull/1690
[#1723]: https://github.com/marin-community/marin/pull/1723
[#1690]: https://github.com/marin-community/marin/pull/1690
[#1723]: https://github.com/marin-community/marin/pull/1723
[rw/wm]: https://github.com/Open-Athena/marin/tree/rw%2Fwm
[yaya]: https://github.com/Open-Athena/yaya
[lossless-yaml]: https://pypi.org/project/lossless-yaml/
[tomlkit]: https://github.com/sdispater/tomlkit
[Levanter]: https://github.com/marin-community/levanter
