#!/usr/bin/env bash
# Conflict Resolution Script for Workspace Migration
#
# Handles common conflicts when rebasing/cherry-picking the workspace migration
# onto branches with upstream changes.

set -e

echo "Resolving workspace migration conflicts..."
echo ""

# Get list of conflicted files
CONFLICTS=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
if [ -z "$CONFLICTS" ]; then
    echo "No conflicts to resolve."
    exit 0
fi

echo "Conflicted files:"
echo "$CONFLICTS"
echo ""

# Handle "added by us" (AU) conflicts - files added in new lib/ location
# Git should have already placed them correctly, just need to stage them
AU_FILES=$(git status --porcelain | grep "^AU" | awk '{print $2}' || true)
if [ -n "$AU_FILES" ]; then
    echo "Resolving 'added by us' conflicts (files moved to lib/)..."
    echo "$AU_FILES" | while read -r file; do
        if [ -n "$file" ]; then
            echo "  - $file (staging as-is)"
            git add "$file"
        fi
    done
    echo ""
fi

# Handle uv.lock conflicts - regenerate from scratch
if echo "$CONFLICTS" | grep -q "^uv.lock$"; then
    echo "Resolving uv.lock conflict by regenerating..."
    # Use theirs as base, then regenerate
    git checkout --theirs uv.lock 2>/dev/null || rm -f uv.lock
    echo "Running: uv sync"
    uv sync
    git add uv.lock
    echo "  ✓ uv.lock regenerated and staged"
    echo ""
fi

# Check if all conflicts resolved
REMAINING=$(git diff --name-only --diff-filter=U 2>/dev/null || true)
if [ -z "$REMAINING" ]; then
    echo "✓ All conflicts resolved!"
    echo ""
    echo "Next steps:"
    echo "  git rebase --continue"
    echo "  # or"
    echo "  git cherry-pick --continue"
else
    echo "⚠ Remaining conflicts that need manual resolution:"
    echo "$REMAINING"
    echo ""
    echo "After resolving manually:"
    echo "  git add <files>"
    echo "  git rebase --continue  # or git cherry-pick --continue"
    exit 1
fi
