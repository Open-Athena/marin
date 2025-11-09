#!/bin/bash
# Step 3: Integrate Haliax as workspace member
# Merge Haliax repository with full Git history and integrate as workspace member
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths
HALIAX_REPO="../haliax"
HALIAX_REF=""
LOCK_REF=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--haliax-ref)
            HALIAX_REF="$2"
            shift 2
            ;;
        -l|--lock-ref)
            LOCK_REF="$2"
            shift 2
            ;;
        -r|--haliax-repo)
            HALIAX_REPO="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -b, --haliax-ref REF      Use specific Haliax ref instead of uv.lock SHA"
            echo "  -l, --lock-ref REF        Use uv.lock from specified git ref"
            echo "  -r, --haliax-repo PATH    Path to Haliax repo (default: ../haliax)"
            echo "  -h, --help                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

echo "=== Step 3: Haliax Integration ==="
echo ""
echo "Configuration:"
echo "  Haliax repo: $HALIAX_REPO"
echo "  Haliax ref: ${HALIAX_REF:-<from uv.lock>}"
echo "  Lock ref: ${LOCK_REF:-<will resolve>}"
echo ""

# Verify we're on appropriate branch (should be after step 2, with levanter integrated)
if [ ! -d "lib/levanter" ]; then
    echo "Error: lib/levanter not found. Run step 2 first."
    exit 1
fi

# Part 1: Extract Haliax info and prepare branch
echo "Part 1: Preparing Haliax branch..."
echo ""

# Get Haliax version from Levanter's dependencies (it's the more restrictive one)
HALIAX_VERSION=$(grep 'haliax>=' lib/levanter/pyproject.toml | head -1 | sed -E 's/.*haliax>=([^"]+).*/\1/')
echo "Current Haliax version constraint: >=$HALIAX_VERSION"

# Get Haliax commit SHA - use flag if provided, otherwise from uv.lock
if [ -n "$HALIAX_REF" ]; then
    HALIAX_SHA="$HALIAX_REF"
    echo "Using Haliax ref from --haliax-ref flag: $HALIAX_SHA"
else
    HALIAX_SHA=$(python3 -c "
import tomllib
with open('uv.lock', 'rb') as f:
    lock = tomllib.load(f)
for pkg in lock.get('package', []):
    if pkg.get('name') == 'haliax':
        source = pkg.get('source', {})
        if 'git' in source:
            print(source['git'].split('#')[-1] if '#' in source['git'] else 'HEAD')
        else:
            # PyPI source - use HEAD
            print('HEAD')
        break
")

    if [ -z "$HALIAX_SHA" ]; then
        echo "Error: Could not find Haliax in uv.lock"
        exit 1
    fi

    echo "Using Haliax SHA from uv.lock: $HALIAX_SHA"
fi
echo ""

# Verify Haliax repo exists
if [ ! -d "$HALIAX_REPO/.git" ]; then
    echo "Error: Haliax repo not found at $HALIAX_REPO"
    exit 1
fi

# Create haliax-pkg branch
echo "Creating haliax-pkg branch from $HALIAX_SHA..."
HALIAX_BRANCH="haliax-pkg-$(date +%s)"

(
    cd "$HALIAX_REPO"
    # Fetch from any available remote
    git fetch $(git remote | head -1) || true
    git checkout -b "$HALIAX_BRANCH" "$HALIAX_SHA"

    # Move everything to lib/haliax/
    echo "Moving Haliax files to lib/haliax/..."
    mkdir -p lib/haliax

    # Move all files except .git to lib/haliax/
    for item in *; do
        if [ "$item" != ".git" ] && [ "$item" != "lib" ]; then
            git mv "$item" lib/haliax/ 2>/dev/null || true
        fi
    done

    # Move hidden files (except .git)
    for item in .[!.]*; do
        if [ "$item" != ".git" ] && [ -e "$item" ]; then
            git mv "$item" lib/haliax/ 2>/dev/null || true
        fi
    done

    git commit -m "Move Haliax to lib/haliax/ for workspace integration"
)

echo "✓ Prepared Haliax branch: $HALIAX_BRANCH"
echo ""

# Part 2: Merge into workspace
echo "Part 2: Merging Haliax into workspace..."
echo ""

git remote add haliax-temp "$HALIAX_REPO" || true
git fetch haliax-temp "$HALIAX_BRANCH"

echo "Merging haliax-temp/$HALIAX_BRANCH (with --allow-unrelated-histories)..."
# Disable rename detection to avoid confusing Marin's root files with Haliax's
git -c merge.renames=false merge "haliax-temp/$HALIAX_BRANCH" --allow-unrelated-histories -m "Merge Haliax as lib/haliax/" --no-edit
MERGE_STATUS=$?
if [ $MERGE_STATUS -ne 0 ]; then
    echo ""
    echo "Merge conflicts detected. Resolving..."

    # Keep workspace root files (not moved to lib/haliax/)
    # Git's rename detection is confusing root files with lib/haliax/ files
    echo "  Keeping workspace root files in place..."
    git reset HEAD .github/ docs/ tests/ 2>/dev/null || true
    git checkout --ours -- .github/ docs/ tests/ pyproject.toml README.md .gitignore 2>/dev/null || true
    git add .github/ docs/ tests/ pyproject.toml README.md .gitignore 2>/dev/null || true

    # Remove "both deleted" workflow conflicts
    for wf in publish_dev.yaml run_quick_levanter_tests.yaml run_tests.yaml; do
        if git diff --name-only --diff-filter=U | grep -q "^.github/workflows/$wf$"; then
            echo "  Removing deleted workflow: .github/workflows/$wf"
            git rm ".github/workflows/$wf" 2>/dev/null || true
        fi
    done

    # Accept all Haliax files into lib/haliax/
    echo "  Accepting Haliax files into lib/haliax/..."
    git checkout --theirs -- lib/haliax/ 2>/dev/null || true
    git add lib/haliax/ 2>/dev/null || true

    git commit -m "Merge Haliax as lib/haliax/" --no-edit
fi

git remote remove haliax-temp

echo "✓ Merged Haliax"
echo ""

# Part 3: Update workspace configuration
echo "Part 3: Updating workspace configuration..."
echo ""

"$SCRIPT_DIR/update_workspace.py"
echo "✓ Updated pyproject.toml for Haliax workspace member"
echo ""

# Part 4: Update uv.lock
if [ -n "$LOCK_REF" ]; then
    echo "Part 4: Using uv.lock from $LOCK_REF..."
    git show "$LOCK_REF:uv.lock" > uv.lock
    git add uv.lock
    echo "✓ Updated uv.lock from $LOCK_REF"
else
    echo "Part 4: Resolving dependencies (this may take 5-10 minutes)..."
    uv sync
    git add uv.lock
    echo "✓ Resolved and locked dependencies"
fi
echo ""

# Part 5: Migrate Haliax workflows (if any)
echo "Part 5: Checking for Haliax workflows..."
if [ -d "lib/haliax/.github/workflows" ]; then
    echo "Migrating Haliax workflows..."
    "$SCRIPT_DIR/update_workflows.py"
    echo "✓ Migrated Haliax workflows"
else
    echo "No Haliax workflows to migrate"
fi
echo ""

# Part 5.5: Apply compatibility patches to Haliax tests
echo "Part 5.5: Applying compatibility patches..."
if [ -f "$SCRIPT_DIR/test_named_ref.patch" ]; then
    echo "Applying test_named_ref.patch for JAX compatibility..."
    git apply "$SCRIPT_DIR/test_named_ref.patch"
    echo "✓ Applied test_named_ref.patch"
else
    echo "! Warning: test_named_ref.patch not found"
fi
echo ""

# Part 6: Commit workspace integration changes
echo "Part 6: Committing workspace integration..."
git add -u
git add .github/workflows/haliax-*.yaml 2>/dev/null || true

git commit -m "$(cat <<'EOF'
Integrate Haliax as workspace member

- Add \`lib/haliax\` to workspace members
- Convert marin and levanter haliax dependencies to workspace references
- Migrate Haliax workflows to \`.github/workflows/haliax-*.yaml\`
- Update \`uv.lock\` with workspace haliax

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

echo ""
echo "✓ Haliax workspace integration committed!"
echo ""

# Part 7: Update pre-commit config
echo "Part 7: Updating pre-commit config for Haliax..."
"$SCRIPT_DIR/update_precommit.py"
git add -u
git commit -m "Add Haliax to pre-commit config with separate lint/format rules

- Add HALIAX_LICENSE and HALIAX_BLACK_CONFIG constants
- Add lib/haliax/**/*.py config using Haliax's own settings
- Exclude lib/haliax/** from Marin's general Python config
- Run ./infra/pre-commit.py --all-files --fix to apply formatting

This keeps Haliax's license headers and formatting separate (like Levanter),
avoiding duplicate license headers and preserving line-length 119.

Generated with Claude Code (https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo ""
echo "✓ Pre-commit config updated!"
echo ""

#
# Done!
#

echo "========================================="
echo "✓ Step 3 complete!"
echo "========================================="
echo ""
echo "Haliax has been merged into lib/haliax/ with full Git history preserved."
echo "Workspace configuration updated for haliax as workspace member."
echo ""
echo "Next: Test the integration and create PR for step 3"
