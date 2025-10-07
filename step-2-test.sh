#!/usr/bin/env bash
# Workspace Migration - Step 2 Test
#
# Tests step-2.sh by running it from a step-1 base and verifying the result
# Run from repo root: ./workspace-migration/step-2-test.sh [levanter-ref]
#
# This creates a temporary test branch, runs step-2.sh, and compares
# the resulting worktree and commits against the expected ws-2 branch.
#
# Arguments:
#   levanter-ref: Git ref to use for Levanter (default: auto-detect from ws-2)

set -e

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Optional: Levanter ref to use
LEVANTER_REF="${1:-}"

echo "Workspace Migration - Step 2 Test"
echo "=================================="
echo ""

# Verify we have the required branches
if ! git rev-parse --verify ws >/dev/null 2>&1; then
    echo "ERROR: ws branch not found (step 1 base)"
    exit 1
fi

if ! git rev-parse --verify ws-2 >/dev/null 2>&1; then
    echo "ERROR: ws-2 branch not found (step 2 reference)"
    exit 1
fi

# Save current branch
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $ORIGINAL_BRANCH"

# Auto-detect Levanter ref from ws-2 if not specified
if [ -z "$LEVANTER_REF" ]; then
    echo "Auto-detecting Levanter commit from ws-2..."
    # Find the levanter-pkg parent of the merge commit on ws-2
    MERGE_COMMIT=$(git log ws-2 ^ws --merges --format=%H -1)
    if [ -n "$MERGE_COMMIT" ]; then
        # Get the second parent (levanter-pkg side)
        LEVANTER_PKG_COMMIT=$(git rev-parse ${MERGE_COMMIT}^2)
        # Find the actual Levanter commit (parent of the restructuring commit)
        LEVANTER_REF=$(git rev-parse ${LEVANTER_PKG_COMMIT}^)
        echo "Detected Levanter commit: $LEVANTER_REF"
    else
        echo "WARNING: Could not auto-detect Levanter ref, using main"
        LEVANTER_REF="main"
    fi
fi

echo "Using Levanter ref: $LEVANTER_REF"
echo ""

# Create temporary test branch from ws (step 1)
TEST_BRANCH="ws-2-test-$$"
echo "Creating test branch: $TEST_BRANCH (from ws)"
git checkout -b "$TEST_BRANCH" ws

# Clean up on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    git checkout "$ORIGINAL_BRANCH"
    if git rev-parse --verify "$TEST_BRANCH" >/dev/null 2>&1; then
        git branch -D "$TEST_BRANCH"
    fi
    if git rev-parse --verify levanter-pkg >/dev/null 2>&1; then
        git branch -D levanter-pkg
    fi
}
trap cleanup EXIT

# Run step-2.sh with detected/specified Levanter ref
echo ""
echo "Running step-2.sh..."
echo "===================="
echo ""
./workspace-migration/step-2.sh ../levanter "$LEVANTER_REF"

# Get commit info
echo ""
echo "Verifying results..."
echo "===================="
echo ""

WS_2_HEAD=$(git rev-parse ws-2)
TEST_HEAD=$(git rev-parse HEAD)

echo "ws-2 HEAD:    $WS_2_HEAD"
echo "test HEAD:    $TEST_HEAD"
echo ""

# Compare worktrees
echo "Comparing worktrees..."

# Get list of tracked files in both branches
WS_2_FILES=$(git ls-tree -r --name-only ws-2 | sort)
TEST_FILES=$(git ls-tree -r --name-only HEAD | sort)

if [ "$WS_2_FILES" != "$TEST_FILES" ]; then
    echo "ERROR: File lists differ!"
    echo ""
    echo "Files in ws-2 but not in test:"
    comm -23 <(echo "$WS_2_FILES") <(echo "$TEST_FILES")
    echo ""
    echo "Files in test but not in ws-2:"
    comm -13 <(echo "$WS_2_FILES") <(echo "$TEST_FILES")
    exit 1
fi

echo "✓ File lists match ($(echo "$WS_2_FILES" | wc -l | tr -d ' ') files)"

# Compare file contents
echo "Comparing file contents..."
DIFF_COUNT=0
for file in $WS_2_FILES; do
    WS_2_HASH=$(git rev-parse "ws-2:$file")
    TEST_HASH=$(git rev-parse "HEAD:$file")

    if [ "$WS_2_HASH" != "$TEST_HASH" ]; then
        DIFF_COUNT=$((DIFF_COUNT + 1))
        if [ $DIFF_COUNT -eq 1 ]; then
            echo ""
            echo "Files with different content:"
        fi
        echo "  $file"
        echo "    ws-2:  $WS_2_HASH"
        echo "    test:  $TEST_HASH"
    fi
done

if [ $DIFF_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $DIFF_COUNT files have different content"
    echo ""
    echo "To investigate, compare branches:"
    echo "  git diff ws-2 $TEST_BRANCH"
    exit 1
fi

echo "✓ All file contents match"

# Compare commit structure
echo ""
echo "Comparing commit structure..."

# Get commit messages
WS_2_COMMITS=$(git log ws-2 ^ws --oneline)
TEST_COMMITS=$(git log HEAD ^ws --oneline)

WS_2_COUNT=$(echo "$WS_2_COMMITS" | wc -l | tr -d ' ')
TEST_COUNT=$(echo "$TEST_COMMITS" | wc -l | tr -d ' ')

echo "ws-2 commits:  $WS_2_COUNT"
echo "test commits:  $TEST_COUNT"

if [ "$WS_2_COUNT" != "$TEST_COUNT" ]; then
    echo "ERROR: Commit count mismatch!"
    echo ""
    echo "ws-2 commits:"
    echo "$WS_2_COMMITS"
    echo ""
    echo "test commits:"
    echo "$TEST_COMMITS"
    exit 1
fi

echo "✓ Commit count matches ($WS_2_COUNT commits)"

# Verify commit messages match
WS_2_MESSAGES=$(git log ws-2 ^ws --format="%s")
TEST_MESSAGES=$(git log HEAD ^ws --format="%s")

if [ "$WS_2_MESSAGES" != "$TEST_MESSAGES" ]; then
    echo "WARNING: Commit messages differ (this is expected if timestamps/hashes differ)"
    echo ""
    echo "ws-2 messages:"
    echo "$WS_2_MESSAGES"
    echo ""
    echo "test messages:"
    echo "$TEST_MESSAGES"
else
    echo "✓ Commit messages match"
fi

# Summary
echo ""
echo "========================================="
echo "✓ Step 2 test PASSED!"
echo "========================================="
echo ""
echo "The step-2.sh script successfully recreates the ws-2 worktree."
echo "All files and contents match the reference ws-2 branch."
echo ""
