#!/usr/bin/env bash
# Workspace Migration - Step 2 Part 3: Workflow Migration (Standalone)
#
# Migrate GitHub Actions workflows to monorepo structure.
# Run from repo root after step2 merge is complete.
#
# This script is idempotent - it can be run multiple times safely.
# It will skip already-renamed files and update workflow content based on current state.

set -e

# Change to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "========================================="
echo "Migrating GitHub Actions workflows"
echo "========================================="
echo ""

# Rename Marin workflows with marin- prefix
echo "Renaming Marin workflows with marin- prefix..."
renamed_count=0
for workflow in .github/workflows/*.yaml .github/workflows/*.yml; do
    if [ -f "$workflow" ]; then
        basename=$(basename "$workflow")
        if [[ ! "$basename" =~ ^marin- && ! "$basename" =~ ^levanter- ]]; then
            new_name="marin-${basename}"
            git mv "$workflow" ".github/workflows/$new_name"
            echo "  $basename -> $new_name (Marin)"
            renamed_count=$((renamed_count + 1))
        fi
    fi
done
if [ $renamed_count -eq 0 ]; then
    echo "  (All Marin workflows already have prefix)"
fi
echo ""

# Move Levanter workflows from lib/levanter/.github/workflows/ to root with levanter- prefix
if [ -d "lib/levanter/.github/workflows" ]; then
    echo "Moving Levanter workflows to root with levanter- prefix..."
    renamed_count=0
    for workflow in lib/levanter/.github/workflows/*.yaml lib/levanter/.github/workflows/*.yml; do
        if [ -f "$workflow" ]; then
            basename=$(basename "$workflow")
            new_name="levanter-${basename}"
            git mv "$workflow" ".github/workflows/$new_name"
            echo "  $basename -> $new_name (Levanter)"
            renamed_count=$((renamed_count + 1))
        fi
    done
    if [ $renamed_count -eq 0 ]; then
        echo "  (All Levanter workflows already moved)"
    fi
    echo ""
fi

# Move dependabot.yml if it exists in lib/levanter/.github/
if [ -f "lib/levanter/.github/dependabot.yml" ]; then
    echo "Moving dependabot.yml with levanter- prefix..."
    git mv lib/levanter/.github/dependabot.yml .github/levanter-dependabot.yml
    echo ""
elif [ -f ".github/levanter-dependabot.yml" ]; then
    echo "dependabot.yml already moved"
    echo ""
fi

# Apply workflow content updates
echo "Updating workflow content..."
"$SCRIPT_DIR/update_workflows.py"
echo ""

# Update TPU setup scripts for monorepo structure
echo "Updating TPU setup scripts..."
"$SCRIPT_DIR/update_tpu_setup.py"
echo ""

# Stage changes
echo "Staging changes..."
git add .github/ lib/levanter/infra/helpers/
echo ""

# Show what will be committed
echo "========================================="
echo "Changes ready to commit:"
echo "========================================="
git status --short
echo ""

# Commit
read -p "Commit these changes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "Migrate workflows to monorepo structure

- Rename Marin workflows with marin- prefix for clarity
- Rename Levanter workflows with levanter- prefix
- Add 'Marin - ' and 'Levanter - ' prefixes to workflow names
- Update Levanter workflows for uv workspace structure:
  - Add path filters to trigger only on relevant changes
  - Set working-directory: lib/levanter
  - Use --package levanter for uv commands
  - Update TPU SSH commands to use marin/lib/levanter paths
- Update TPU setup scripts for monorepo structure:
  - Clone marin monorepo instead of levanter repo
  - Use marin directory name instead of levanter
  - Add --package levanter to uv sync commands

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

    echo ""
    echo "✓ Workflow migration complete!"
else
    echo ""
    echo "Changes staged but not committed. Run 'git commit' when ready."
fi
