#!/usr/bin/env bash
# Workspace Migration - Step 2 Initialization
#
# One-time setup to prepare Levanter for ingestion with full Git history.
# Creates a levanter-pkg branch with all files moved to lib/levanter/
#
# Run from repo root: ./workspace-migration/step-2-init.sh [levanter-repo-path]

set -e

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Levanter repo path (default to sibling directory)
LEVANTER_REPO="${1:-../levanter}"

echo "Workspace Migration - Step 2 Initialization"
echo "Preparing Levanter for ingestion with Git history preservation"
echo ""

# Verify levanter repo exists
if [ ! -d "$LEVANTER_REPO/.git" ]; then
    echo "ERROR: Levanter repository not found at $LEVANTER_REPO"
    exit 1
fi

# Add levanter as remote if not already present
if ! git remote | grep -q "^levanter$"; then
    echo "Adding levanter as remote..."
    git remote add levanter "$LEVANTER_REPO"
fi

# Fetch levanter
echo "Fetching levanter..."
git fetch levanter

# Get levanter HEAD commit
LEVANTER_HEAD=$(git rev-parse levanter/main)
echo "Levanter HEAD: $LEVANTER_HEAD"
echo ""

# Create levanter-pkg branch from levanter/main
if git rev-parse --verify levanter-pkg >/dev/null 2>&1; then
    echo "WARNING: levanter-pkg branch already exists"
    echo "To recreate, delete it first: git branch -D levanter-pkg"
    exit 1
fi

echo "Creating levanter-pkg branch from levanter/main..."
git checkout -b levanter-pkg levanter/main

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
echo "Updating pyproject.toml for workspace..."

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
echo "✓ Initialization complete!"
echo ""
echo "levanter-pkg branch created from Levanter commit $LEVANTER_HEAD"
echo "Next step: Run ./workspace-migration/step-2.sh to merge into ws branch"
