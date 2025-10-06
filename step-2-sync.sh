#!/usr/bin/env bash
# Workspace Migration - Step 2 Sync
#
# Syncs upstream Levanter changes into the workspace.
# Updates levanter-pkg branch and merges into current branch.
#
# Run from repo root: ./workspace-migration/step-2-sync.sh

set -e

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Workspace Migration - Step 2 Sync"
echo "Syncing upstream Levanter changes"
echo ""

# Verify levanter remote exists
if ! git remote | grep -q "^levanter$"; then
    echo "ERROR: levanter remote not found"
    echo "Run step-2-init.sh first"
    exit 1
fi

# Verify levanter-pkg branch exists
if ! git rev-parse --verify levanter-pkg >/dev/null 2>&1; then
    echo "ERROR: levanter-pkg branch not found"
    echo "Run step-2-init.sh and step-2.sh first"
    exit 1
fi

# Verify lib/levanter exists
if [ ! -d "lib/levanter" ]; then
    echo "ERROR: lib/levanter not found"
    echo "Run step-2.sh first to complete initial Levanter ingestion"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Fetch latest levanter
echo "Fetching latest from levanter remote..."
git fetch levanter

# Get current levanter HEAD
LEVANTER_HEAD=$(git rev-parse levanter/main)
echo "Latest Levanter commit: $LEVANTER_HEAD"
echo ""

# Switch to levanter-pkg branch
echo "Updating levanter-pkg branch..."
git checkout levanter-pkg

# Find the commit where we did the initial restructuring
# (the commit that moved everything to lib/levanter/)
RESTRUCTURE_COMMIT=$(git log --grep="Move Levanter to lib/levanter/" --format=%H -1)

if [ -z "$RESTRUCTURE_COMMIT" ]; then
    echo "ERROR: Could not find restructuring commit on levanter-pkg"
    exit 1
fi

# Rebase levanter-pkg onto latest levanter/main
# This replays our "move to lib/levanter" commit on top of new upstream changes
echo "Rebasing levanter-pkg onto levanter/main..."
echo "This will replay the restructuring commit on top of latest upstream"

# Reset to levanter/main
git reset --hard levanter/main

# Get list of all files at root
FILES=$(git ls-tree --name-only HEAD)

# Recreate the restructuring
mkdir -p lib/levanter

echo "Moving files to lib/levanter/..."
for file in $FILES; do
    if [ "$file" != "lib" ]; then
        git mv "$file" "lib/levanter/" 2>/dev/null || true
    fi
done

# Update root pyproject.toml
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

# Commit if there are changes
if ! git diff --cached --quiet; then
    git commit -m "Sync Levanter to lib/levanter/ ($(git rev-parse --short levanter/main))

Updated Levanter to latest upstream commit.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    echo "✓ levanter-pkg branch updated"
else
    echo "No changes to commit (already up to date)"
fi

# Switch back to original branch
echo ""
echo "Switching back to $CURRENT_BRANCH..."
git checkout "$CURRENT_BRANCH"

# Merge updated levanter-pkg
echo "Merging updated levanter-pkg into $CURRENT_BRANCH..."
git merge levanter-pkg -m "Sync Levanter to latest upstream ($(git rev-parse --short levanter/main))

Merge latest Levanter changes from upstream.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Update uv.lock
echo "Updating uv.lock..."
uv sync

# Commit lockfile update
echo "Committing lockfile update..."
git add uv.lock
git commit --amend --no-edit

echo ""
echo "✓ Sync complete!"
echo "Levanter updated to $(git rev-parse --short levanter/main)"
