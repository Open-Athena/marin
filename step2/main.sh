#!/usr/bin/env bash
# Workspace Migration - Step 2
#
# Hermetic script that merges Levanter (with preserved Git history) into workspace as lib/levanter/
# Run from repo root: ./workspace-migration/step2/main.sh [options]
#
# Process:
#   1. Extract Levanter and dolma SHAs from lib/marin/pyproject.toml
#   2. Merge Levanter with preserved history into lib/levanter/
#
# Note: Assumes step1 has already applied dependency updates (m/rw/deps cherry-pick)
#
# Prerequisites:
#   - Should be on ws branch (or branch with step 1 applied)
#   - Levanter repo cloned (default: ../levanter)
#
# Options:
#   -l, --lock-ref REF      Use uv.lock from specified git ref instead of re-resolving
#   -r, --levanter-repo PATH  Path to Levanter repo (default: ../levanter)

set -e

# Show help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    cat << 'EOF'
Workspace Migration - Step 2
============================

Merges Levanter (with preserved Git history) into workspace as lib/levanter/

Process:
  1. Extract Levanter and dolma SHAs from lib/marin/pyproject.toml
  2. Merge Levanter with preserved history into lib/levanter/

Note: Assumes step1 has already applied dependency updates (m/rw/deps).

Usage:
    ./workspace-migration/step2/main.sh [options]

Prerequisites:
    - Should be on ws branch (or branch with step 1 applied)
    - Levanter repo cloned (default: ../levanter)

Options:
    -l, --lock-ref REF        Use uv.lock from specified git ref instead of re-resolving
                              (saves 5-10 minutes during testing)
    -r, --levanter-repo PATH  Path to Levanter repo (default: ../levanter)
    -h, --help                Show this help message

Examples:
    # Default: extract SHAs from lib/marin/pyproject.toml, re-resolve lockfile
    ./workspace-migration/step2/main.sh

    # Use existing lockfile from ws-2 (faster for testing)
    ./workspace-migration/step2/main.sh --lock-ref ws-2

See workspace-migration/README.md for more details.
EOF
    exit 0
fi

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Parse options
LOCK_REF=""
LEVANTER_REPO="../levanter"

while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--lock-ref)
            LOCK_REF="$2"
            shift 2
            ;;
        -r|--levanter-repo)
            LEVANTER_REPO="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Helper function for cross-platform sed in-place editing
sed_inplace() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

echo "Workspace Migration - Step 2"
echo "Merging Levanter with preserved Git history into lib/levanter/"
echo ""

# Verify we're on a branch with workspace structure
if [ ! -d "lib/marin" ]; then
    echo "ERROR: lib/marin not found. Run step-1.sh first."
    exit 1
fi

# Verify levanter repo exists
if [ ! -d "$LEVANTER_REPO/.git" ]; then
    echo "ERROR: Levanter repository not found at $LEVANTER_REPO"
    exit 1
fi

# Extract Levanter and dolma SHAs from lib/marin/pyproject.toml
PYPROJECT="lib/marin/pyproject.toml"

echo "Extracting dependency SHAs from $PYPROJECT..."

LEVANTER_SHA=$(grep -o 'levanter.*@ git+https://[^@]*@[a-f0-9]\{40\}' "$PYPROJECT" | grep -o '[a-f0-9]\{40\}' || true)
if [ -z "$LEVANTER_SHA" ]; then
    echo "ERROR: Could not extract Levanter SHA from $PYPROJECT"
    echo "Ensure step1 has been run and m/rw/deps has been applied"
    exit 1
fi
echo "  Levanter SHA: $LEVANTER_SHA"

DOLMA_SHA=$(grep -o 'dolma.*@ git+https://[^@]*@[a-f0-9]\{40\}' "$PYPROJECT" | grep -o '[a-f0-9]\{40\}' || true)
if [ -z "$DOLMA_SHA" ]; then
    echo "ERROR: Could not extract dolma SHA from $PYPROJECT"
    echo "Ensure step1 has been run and m/rw/deps has been applied"
    exit 1
fi
echo "  dolma SHA: $DOLMA_SHA"
echo ""

# Clean up lib/levanter if it only contains untracked files
# (git checkout can leave empty dirs behind, IDEs can create .idea, etc.)
if [ -d "lib/levanter" ]; then
    # Check if lib/levanter contains any tracked files
    TRACKED_FILES=$(git ls-files lib/levanter 2>/dev/null || true)
    if [ -n "$TRACKED_FILES" ]; then
        echo "ERROR: lib/levanter already exists with tracked files:"
        echo "$TRACKED_FILES"
        echo "Please remove it manually or commit/stash your changes"
        exit 1
    fi

    # Has only untracked files (or is empty) - safe to remove
    if [ -n "$(ls -A lib/levanter 2>/dev/null)" ]; then
        echo "Removing lib/levanter with untracked files (e.g., .idea/)..."
    else
        echo "Removing empty lib/levanter directory structure..."
    fi
    rm -rf lib/levanter
fi

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"
echo ""

#
# Part 1: Initialize levanter-pkg branch
#

echo "=== Part 1: Preparing levanter-pkg branch ==="
echo ""

# Add levanter as remote if not already present
if ! git remote | grep -q "^levanter$"; then
    echo "Adding levanter as remote..."
    git remote add levanter "$LEVANTER_REPO"
fi

# Fetch levanter
echo "Fetching levanter..."
git fetch levanter

# Verify Levanter SHA exists
if ! git cat-file -e "$LEVANTER_SHA" 2>/dev/null; then
    echo "ERROR: Levanter SHA not found: $LEVANTER_SHA"
    echo "Make sure you've fetched from the Levanter remote"
    exit 1
fi

echo "Using Levanter SHA: $LEVANTER_SHA"
echo ""

# Delete levanter-pkg branch if it exists
if git rev-parse --verify levanter-pkg >/dev/null 2>&1; then
    echo "Deleting existing levanter-pkg branch..."
    git branch -D levanter-pkg
fi

# Create levanter-pkg branch from specified SHA
echo "Creating levanter-pkg branch from $LEVANTER_SHA..."
git checkout -b levanter-pkg "$LEVANTER_SHA"

# Get list of all files at root
FILES=$(git ls-tree --name-only HEAD)

# Create lib/levanter/ directory
mkdir -p lib/levanter

# Move all files to lib/levanter/
echo "Moving files to lib/levanter/..."
for file in $FILES; do
    if [ "$file" != "lib" ]; then
        git mv "$file" "lib/levanter/"
    fi
done

# Update workspace configuration in levanter-pkg branch
echo "Creating root pyproject.toml for levanter-pkg branch..."

# Create minimal root pyproject.toml for levanter-pkg branch
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "levanter-pkg"
version = "0.1.0"
description = "Levanter workspace member"
requires-python = ">=3.11"

[tool.uv.workspace]
members = ["lib/levanter"]

[tool.uv.sources]
levanter = { workspace = true }
EOF

git add pyproject.toml

# Commit the restructuring
echo "Committing restructuring..."
git commit -m "Move Levanter to lib/levanter/ for workspace integration

Restructure Levanter repository for integration as workspace member.
All files moved to lib/levanter/ subdirectory.

This branch preserves full Levanter Git history and will be merged
into the main Marin workspace migration."

echo ""
echo "✓ levanter-pkg branch created from Levanter SHA $LEVANTER_SHA"
echo ""

#
# Part 2: Merge levanter-pkg into workspace
#

echo "=== Part 2: Merging levanter-pkg into $CURRENT_BRANCH ==="
echo ""

# Return to original branch
git checkout "$CURRENT_BRANCH"

# Merge levanter-pkg branch
echo "Merging levanter-pkg branch..."
git merge levanter-pkg --allow-unrelated-histories --no-commit || {
    echo "Merge has conflicts, resolving..."

    # Handle pyproject.toml conflict (use ours, which is the workspace root)
    if git diff --name-only --diff-filter=U | grep -q "^pyproject.toml$"; then
        echo "Resolving pyproject.toml conflict (keeping workspace root version)..."
        git checkout --ours pyproject.toml
        git add pyproject.toml
    fi

    # Handle .github conflicts (we'll handle workflows separately after merge)
    if git diff --name-only --diff-filter=U | grep -q "^\.github/"; then
        echo "Resolving .github conflicts..."
        # Take theirs for .github (Levanter's workflows)
        git diff --name-only --diff-filter=U | grep "^\.github/" | while read file; do
            git checkout --theirs "$file"
            git add "$file"
        done
    fi

    # Check if all conflicts are resolved
    REMAINING=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
    if [ -n "$REMAINING" ]; then
        echo "ERROR: Unexpected conflicts remain:"
        echo "$REMAINING"
        exit 1
    fi
}

# Update workspace configuration using tomlkit
echo "Updating workspace configuration..."
"$SCRIPT_DIR/update_workspace_toml.py" --dolma-sha "$DOLMA_SHA"

# Update uv.lock for new workspace structure
if [ -n "$LOCK_REF" ]; then
    echo "Extracting uv.lock from $LOCK_REF..."
    if ! git rev-parse --verify "$LOCK_REF" >/dev/null 2>&1; then
        echo "ERROR: Lock ref '$LOCK_REF' not found"
        exit 1
    fi
    git show "$LOCK_REF:uv.lock" > uv.lock
    echo "✓ Using uv.lock from $LOCK_REF"
else
    echo "Updating uv.lock for workspace structure..."
    uv sync -q
fi

# Stage all changes
echo "Staging changes..."
git add pyproject.toml lib/marin/pyproject.toml uv.lock

# Commit the merge
git commit -m "Merge Levanter into lib/levanter/

Ingest Levanter repository as workspace member with full Git history.
Levanter files are now in lib/levanter/ subdirectory.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo ""
echo "✓ Merge complete!"
echo ""

#
# Part 3: Migrate GitHub Actions workflows
#

echo "=== Part 3: Migrating GitHub Actions workflows ==="
echo ""

# Rename Marin workflows with marin- prefix
echo "Renaming Marin workflows with marin- prefix..."
for workflow in .github/workflows/*.yaml .github/workflows/*.yml; do
    if [ -f "$workflow" ]; then
        basename=$(basename "$workflow")
        if [[ ! "$basename" =~ ^marin- ]]; then
            new_name="marin-${basename}"
            git mv "$workflow" ".github/workflows/$new_name"
            echo "  $basename -> $new_name (Marin)"
        fi
    fi
done

# Move Levanter workflows from lib/levanter/.github/workflows/ to root with levanter- prefix
if [ -d "lib/levanter/.github/workflows" ]; then
    echo "Moving Levanter workflows to root with levanter- prefix..."
    for workflow in lib/levanter/.github/workflows/*.yaml lib/levanter/.github/workflows/*.yml; do
        if [ -f "$workflow" ]; then
            basename=$(basename "$workflow")
            new_name="levanter-${basename}"
            git mv "$workflow" ".github/workflows/$new_name"
            echo "  $basename -> $new_name (Levanter)"
        fi
    done
fi

# Move dependabot.yml if it exists in lib/levanter/.github/
if [ -f "lib/levanter/.github/dependabot.yml" ]; then
    echo "Moving dependabot.yml with levanter- prefix..."
    git mv lib/levanter/.github/dependabot.yml .github/levanter-dependabot.yml
fi

# Apply workflow content updates using ruamel.yaml
echo "Updating workflow content..."
"$SCRIPT_DIR/update_workflows.py"

# Update TPU setup scripts for monorepo structure
echo "Updating TPU setup scripts..."
"$SCRIPT_DIR/update_tpu_setup.py"

# Update pre-commit config to exclude lib/levanter/ from license insertion
echo "Updating pre-commit config..."
"$SCRIPT_DIR/update_precommit.py"

# Update ReadTheDocs config to use --frozen
echo "Updating ReadTheDocs config..."
"$SCRIPT_DIR/update_readthedocs.py"

# Commit workflow and TPU setup changes
git add .github/ lib/levanter/infra/helpers/ .pre-commit-config.yaml .readthedocs.yaml lib/levanter/.readthedocs.yaml

git commit -m "Migrate workflows to monorepo structure

- Rename Marin workflows with marin- prefix for clarity
- Rename Levanter workflows with levanter- prefix
- Add 'Marin - ' and 'Levanter - ' prefixes to workflow names
- Update Levanter workflows for uv workspace structure:
  - Add path filters to trigger only on relevant changes
  - Set working-directory: lib/levanter
  - Use --package levanter for uv commands
  - Add --frozen to uv sync/run to prevent git dependency updates
  - Update TPU SSH commands to use marin/lib/levanter paths
- Update TPU setup scripts for monorepo structure:
  - Clone marin monorepo instead of levanter repo
  - Use marin directory name instead of levanter
  - Add --package levanter --frozen to uv sync commands
- Update ReadTheDocs config:
  - Add --frozen to uv sync and uv run commands
- Exclude lib/levanter/ from Marin license insertion (preserve Levanter licenses)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo ""
echo "✓ Workflow migration complete!"
echo ""

#
# Done!
#

echo "========================================="
echo "✓ Step 2 complete!"
echo "========================================="
echo ""
echo "Levanter has been merged into lib/levanter/ with full Git history preserved."
echo "GitHub Actions workflows have been migrated to monorepo structure."
echo ""
echo "Next: Test the integration and create PR for step 2"
