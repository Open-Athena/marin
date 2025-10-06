#!/usr/bin/env bash
# Workspace Migration - Step 2
#
# Merges Levanter (with preserved Git history) into workspace as lib/levanter/
# Run from repo root: ./workspace-migration/step-2.sh
#
# Prerequisites:
#   - Run step-2-init.sh first to create levanter-pkg branch
#   - Should be on ws branch (or branch with step 1 applied)

set -e

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "Workspace Migration - Step 2"
echo "Merging Levanter with preserved Git history into lib/levanter/"
echo ""

# Verify we're on a branch with workspace structure
if [ ! -d "lib/marin" ]; then
    echo "ERROR: lib/marin not found. Run step-1.sh first."
    exit 1
fi

# Verify levanter-pkg branch exists
if ! git rev-parse --verify levanter-pkg >/dev/null 2>&1; then
    echo "ERROR: levanter-pkg branch not found"
    echo "Run step-2-init.sh first to create it"
    exit 1
fi

# Get current branch name
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Verify lib/levanter doesn't already exist
if [ -d "lib/levanter" ]; then
    echo "ERROR: lib/levanter already exists"
    exit 1
fi

# Merge levanter-pkg branch
echo "Merging levanter-pkg branch..."
git merge levanter-pkg --allow-unrelated-histories --no-commit || {
    echo "Merge has conflicts, resolving..."

    # Handle pyproject.toml conflict (use ours, which is the workspace root)
    if git diff --name-only --diff-filter=U | grep -q "pyproject.toml"; then
        echo "Resolving pyproject.toml conflict (keeping workspace root version)..."
        git checkout --ours pyproject.toml
        git add pyproject.toml
    fi

    # Check if all conflicts are resolved
    REMAINING=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
    if [ -n "$REMAINING" ]; then
        echo "ERROR: Unexpected conflicts remain:"
        echo "$REMAINING"
        exit 1
    fi
}

# Commit the merge
git commit -m "Merge Levanter into lib/levanter/

Ingest Levanter repository as workspace member with full Git history.
Levanter files are now in lib/levanter/ subdirectory.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Update workspace root pyproject.toml to include levanter
echo "Updating workspace configuration..."

# Add levanter to workspace members
if ! grep -q '"lib/levanter"' pyproject.toml; then
    # Use sed to add levanter to workspace members
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' '/^members = \[/,/^\]/ s/\("lib\/data_browser",\)/\1\n    "lib\/levanter",/' pyproject.toml
    else
        sed -i '/^members = \[/,/^\]/ s/\("lib\/data_browser",\)/\1\n    "lib\/levanter",/' pyproject.toml
    fi
fi

# Add levanter to workspace sources
if ! grep -q 'levanter = { workspace = true }' pyproject.toml; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' '/marin = { workspace = true }/a\
levanter = { workspace = true }
' pyproject.toml
    else
        sed -i '/marin = { workspace = true }/a levanter = { workspace = true }' pyproject.toml
    fi
fi

# Update marin's dependency on levanter to use workspace
echo "Updating marin to use workspace levanter..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's|"levanter\[serve\] @ git+https://github.com/marin-community/levanter.git"|"levanter[serve]"|' lib/marin/pyproject.toml
else
    sed -i 's|"levanter\[serve\] @ git+https://github.com/marin-community/levanter.git"|"levanter[serve]"|' lib/marin/pyproject.toml
fi

# Update uv.lock for new workspace structure
echo "Updating uv.lock for workspace structure..."
uv sync

# Stage all changes
echo "Staging changes..."
git add pyproject.toml lib/marin/pyproject.toml uv.lock

# Amend the merge commit with workspace configuration updates
echo "Amending merge commit with workspace configuration..."
git commit --amend --no-edit

echo ""
echo "✓ Step 2 complete!"
echo ""
echo "Levanter has been merged into lib/levanter/ with full Git history preserved."
echo "Next: Create step-2-sync.sh for syncing upstream Levanter changes."
