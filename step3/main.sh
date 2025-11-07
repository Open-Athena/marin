#!/bin/bash
# Step 3: Integrate Haliax as workspace member
# Merge Haliax repository with full Git history and integrate as workspace member
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths
HALIAX_REPO="../haliax"
LOCK_REF=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
HALIAX_VERSION=$(grep -oP 'haliax>=\K[^"]+' lib/levanter/pyproject.toml | head -1)
echo "Current Haliax version constraint: >=$HALIAX_VERSION"

# Get Haliax commit SHA from uv.lock
HALIAX_SHA=$(python3 -c "
import tomllib
with open('uv.lock', 'rb') as f:
    lock = tomllib.load(f)
for pkg in lock.get('package', []):
    if pkg.get('name') == 'haliax':
        source = pkg.get('source', {})
        if 'git' in source:
            print(source['git'].split('#')[-1] if '#' in source['git'] else 'HEAD')
            break
")

if [ -z "$HALIAX_SHA" ]; then
    echo "Error: Could not extract Haliax SHA from uv.lock"
    exit 1
fi

echo "Haliax SHA from uv.lock: $HALIAX_SHA"
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
    git fetch origin
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
if ! git merge "haliax-temp/$HALIAX_BRANCH" --allow-unrelated-histories -m "Merge Haliax as lib/haliax/" --no-edit; then
    echo ""
    echo "Merge conflicts detected. Resolving..."

    # Typical conflicts: pyproject.toml, .github/, README.md
    # Keep workspace root versions for conflicting root files
    for file in pyproject.toml README.md .gitignore; do
        if git diff --name-only --diff-filter=U | grep -q "^$file$"; then
            echo "  Resolving $file: keeping workspace root version"
            git checkout --ours "$file"
            git add "$file"
        fi
    done

    # For .github/ - keep both, will rename in Part 3
    if git diff --name-only --diff-filter=U | grep -q "^.github/"; then
        echo "  Resolving .github/: keeping workspace version (will update Haliax workflows in Part 3)"
        git checkout --ours .github/
        git add .github/
    fi

    git commit -m "Merge Haliax as lib/haliax/" --no-edit
fi

git remote remove haliax-temp

echo "✓ Merged Haliax"
echo ""

# Part 3: Update workspace configuration
echo "Part 3: Updating workspace configuration..."
echo ""

python3 "$SCRIPT_DIR/update_workspace.py"
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
    RUST_LOG=warn uv sync
    git add uv.lock
    echo "✓ Resolved and locked dependencies"
fi
echo ""

# Part 5: Migrate Haliax workflows (if any)
echo "Part 5: Checking for Haliax workflows..."
if [ -d "lib/haliax/.github/workflows" ]; then
    echo "Migrating Haliax workflows..."
    python3 "$SCRIPT_DIR/update_workflows.py"
    echo "✓ Migrated Haliax workflows"
else
    echo "No Haliax workflows to migrate"
fi
echo ""

echo "=== Step 3 Complete ==="
echo ""
echo "Haliax integrated as lib/haliax/"
echo ""
echo "Next: Review changes and commit"
