#!/usr/bin/env bash
# Workspace Migration - Step 2 Test
#
# Tests step-2.sh by running it from a step-1 base and verifying the result
# Run from repo root: ./workspace-migration/step-2-test.sh [options] [reference] [levanter-ref]
#
# This creates a temporary test branch, runs step-2.sh, and compares
# the resulting worktree against the reference (current HEAD by default).
#
# Options:
#   -L, --allow-lock-diffs: Allow uv.lock to differ (it's a derived artifact)
#   -l, --lock-ref REF: Use uv.lock from specified git ref (typically the reference being tested)
#
# Arguments:
#   reference: Git ref to compare against (default: HEAD, can use branch name like ws-2)
#   levanter-ref: Git ref to use for Levanter (default: auto-detect from reference)

set -e

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Parse options
ALLOW_LOCK_DIFFS=false
LOCK_REF=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -L|--allow-lock-diffs)
            ALLOW_LOCK_DIFFS=true
            shift
            ;;
        -l|--lock-ref)
            LOCK_REF="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Reference to compare against (default to current HEAD)
REFERENCE="${1:-HEAD}"
# Optional: Levanter ref to use
LEVANTER_REF="${2:-}"

echo "Workspace Migration - Step 2 Test"
echo "=================================="
echo ""

# Verify we have the required branches
if ! git rev-parse --verify ws >/dev/null 2>&1; then
    echo "ERROR: ws branch not found (step 1 base)"
    exit 1
fi

if ! git rev-parse --verify "$REFERENCE" >/dev/null 2>&1; then
    echo "ERROR: Reference '$REFERENCE' not found"
    exit 1
fi

# Save current branch
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)
REFERENCE_COMMIT=$(git rev-parse "$REFERENCE")
REFERENCE_NAME=$(git rev-parse --abbrev-ref "$REFERENCE" 2>/dev/null || echo "$REFERENCE_COMMIT")

echo "Current branch: $ORIGINAL_BRANCH"
echo "Reference: $REFERENCE_NAME ($REFERENCE_COMMIT)"
echo ""

# Auto-detect Levanter ref from reference if not specified
if [ -z "$LEVANTER_REF" ]; then
    echo "Auto-detecting Levanter commit from $REFERENCE_NAME..."
    # Find the levanter-pkg parent of the merge commit on reference
    MERGE_COMMIT=$(git log "$REFERENCE" ^ws --merges --format=%H -1)
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

# Run step2/main.sh with detected/specified Levanter ref and optional lock ref
echo ""
echo "Running step2/main.sh..."
echo "===================="
echo ""
if [ -n "$LOCK_REF" ]; then
    echo "Using uv.lock from: $LOCK_REF"
    ./workspace-migration/step2/main.sh --lock-ref "$LOCK_REF" ../levanter "$LEVANTER_REF"
else
    ./workspace-migration/step2/main.sh ../levanter "$LEVANTER_REF"
fi

# Get commit info
echo ""
echo "Verifying results..."
echo "===================="
echo ""

REF_HEAD=$(git rev-parse "$REFERENCE")
TEST_HEAD=$(git rev-parse HEAD)

echo "Reference HEAD: $REF_HEAD"
echo "Test HEAD:      $TEST_HEAD"
echo ""

# Compare worktrees
echo "Comparing worktrees..."

# Get list of tracked files in both branches
REF_FILES=$(git ls-tree -r --name-only "$REFERENCE" | sort)
TEST_FILES=$(git ls-tree -r --name-only HEAD | sort)

if [ "$REF_FILES" != "$TEST_FILES" ]; then
    echo "ERROR: File lists differ!"
    echo ""
    echo "Files in $REFERENCE_NAME but not in test:"
    comm -23 <(echo "$REF_FILES") <(echo "$TEST_FILES")
    echo ""
    echo "Files in test but not in $REFERENCE_NAME:"
    comm -13 <(echo "$REF_FILES") <(echo "$TEST_FILES")
    exit 1
fi

echo "✓ File lists match ($(echo "$REF_FILES" | wc -l | tr -d ' ') files)"

# Compare file contents
echo "Comparing file contents..."
DIFF_COUNT=0
SKIPPED_COUNT=0
for file in $REF_FILES; do
    # Skip uv.lock if --allow-lock-diffs is set
    if [ "$ALLOW_LOCK_DIFFS" = true ] && [ "$file" = "uv.lock" ]; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    REF_HASH=$(git rev-parse "$REFERENCE:$file")
    TEST_HASH=$(git rev-parse "HEAD:$file")

    if [ "$REF_HASH" != "$TEST_HASH" ]; then
        DIFF_COUNT=$((DIFF_COUNT + 1))
        if [ $DIFF_COUNT -eq 1 ]; then
            echo ""
            echo "Files with different content:"
        fi
        echo "  $file"
        echo "    $REFERENCE_NAME: $REF_HASH"
        echo "    test:            $TEST_HASH"
    fi
done

if [ $DIFF_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $DIFF_COUNT files have different content"
    echo ""
    echo "To investigate, compare branches:"
    echo "  git diff $REFERENCE $TEST_BRANCH"
    exit 1
fi

if [ $SKIPPED_COUNT -gt 0 ]; then
    echo "✓ All file contents match (skipped $SKIPPED_COUNT files: uv.lock)"
else
    echo "✓ All file contents match"
fi

# Compare commit structure
echo ""
echo "Comparing commit structure..."

# Get commit messages
REF_COMMITS=$(git log "$REFERENCE" ^ws --oneline)
TEST_COMMITS=$(git log HEAD ^ws --oneline)

REF_COUNT=$(echo "$REF_COMMITS" | wc -l | tr -d ' ')
TEST_COUNT=$(echo "$TEST_COMMITS" | wc -l | tr -d ' ')

echo "$REFERENCE_NAME commits: $REF_COUNT"
echo "test commits:             $TEST_COUNT"

if [ "$REF_COUNT" != "$TEST_COUNT" ]; then
    echo "ERROR: Commit count mismatch!"
    echo ""
    echo "$REFERENCE_NAME commits:"
    echo "$REF_COMMITS"
    echo ""
    echo "test commits:"
    echo "$TEST_COMMITS"
    exit 1
fi

echo "✓ Commit count matches ($REF_COUNT commits)"

# Verify commit messages match
REF_MESSAGES=$(git log "$REFERENCE" ^ws --format="%s")
TEST_MESSAGES=$(git log HEAD ^ws --format="%s")

if [ "$REF_MESSAGES" != "$TEST_MESSAGES" ]; then
    echo "WARNING: Commit messages differ (this is expected if timestamps/hashes differ)"
    echo ""
    echo "$REFERENCE_NAME messages:"
    echo "$REF_MESSAGES"
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
echo "The step2/main.sh script successfully recreates the $REFERENCE_NAME worktree."
echo "All files and contents match the reference."
echo ""
