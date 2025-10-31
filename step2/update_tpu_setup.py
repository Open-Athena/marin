#!/usr/bin/env -S uv run
# /// script
# dependencies = []
# ///
"""
Update TPU setup scripts for step 2 workspace migration.

Updates:
1. Change repo URL from stanford-crfm/levanter to marin-community/marin
2. Add CLONE_DIR variable and use it for cloning
3. Update uv sync commands to use --package levanter
"""

import re
import sys
from pathlib import Path


def update_tpu_setup_script(script_path: Path) -> bool:
    """
    Update a TPU setup script for monorepo structure.

    Returns:
        True if the script was updated, False if no changes needed
    """
    original = script_path.read_text()
    modified = original

    print(f"  Processing {script_path.name}...")

    # 1. Update repo URL
    old_repo = 'REPO="https://github.com/stanford-crfm/levanter.git"'
    new_repo = 'REPO="https://github.com/marin-community/marin.git"'
    if old_repo in modified:
        modified = modified.replace(old_repo, new_repo, 1)
        print(f"    ✓ Updated repo URL to marin-community/marin")

    # 2. Add CLONE_DIR variable after BRANCH=main
    if 'CLONE_DIR=' not in modified:
        modified = re.sub(
            r'(BRANCH=main\n)',
            r'\1CLONE_DIR="marin"\n',
            modified,
            count=1
        )
        print(f"    ✓ Added CLONE_DIR variable")

    # 3. Update clone directory references from levanter to $CLONE_DIR
    # Match: # clone levanter
    if '# clone levanter' in modified:
        modified = modified.replace('# clone levanter', '# clone marin monorepo', 1)
        print(f"    ✓ Updated clone comment")

    # Match: if [ -d levanter ]; then
    if '[ -d levanter ]' in modified:
        modified = modified.replace('[ -d levanter ]', '[ -d $CLONE_DIR ]', 1)
        print(f"    ✓ Updated directory check to use $CLONE_DIR")

    # Match: echo "Levanter directory already exists...
    if 'echo "Levanter directory already exists' in modified:
        modified = modified.replace(
            'echo "Levanter directory already exists',
            'echo "$CLONE_DIR directory already exists',
            1
        )
        print(f"    ✓ Updated echo message")

    # Match: cd levanter (in the if block)
    # Match: git clone $REPO levanter
    # Match: cd levanter (after clone)
    modified = re.sub(
        r'cd levanter \|\| exit 1',
        r'cd $CLONE_DIR || exit 1',
        modified
    )
    modified = re.sub(
        r'git clone \$REPO levanter \|\| exit 1',
        r'git clone $REPO $CLONE_DIR || exit 1',
        modified,
        count=1
    )
    print(f"    ✓ Updated cd and clone commands to use $CLONE_DIR")

    # 4. Update uv sync to use --package levanter
    if 'uv sync' in modified and '--package levanter' not in modified:
        modified = re.sub(
            r'uv sync --extra',
            r'uv sync --package levanter --extra',
            modified
        )
        print(f"    ✓ Updated uv sync to use --package levanter")

    # 5. Add --group test to uv sync for test dependencies (pytest, etc.)
    if 'uv sync' in modified and '--group test' not in modified:
        modified = re.sub(
            r'uv sync --package levanter --extra tpu(?! --group)',
            r'uv sync --package levanter --extra tpu --group test',
            modified
        )
        print(f"    ✓ Added --group test to uv sync")

    # 6. Add --frozen to uv sync commands
    if 'uv sync' in modified and '--frozen' not in modified:
        modified = re.sub(
            r'uv sync --package levanter --extra tpu --group test(?! --frozen)',
            r'uv sync --package levanter --extra tpu --group test --frozen',
            modified
        )
        print(f"    ✓ Added --frozen to uv sync")

    # 7. Add venv_path.txt creation for run.sh to find venv
    if 'venv_path.txt' not in modified:
        # Add after uv sync command
        venv_path_line = '\n# Create venv_path.txt so run.sh can find the venv at monorepo root\necho "$(pwd)/.venv" > lib/levanter/infra/venv_path.txt'
        modified = re.sub(
            r'(uv sync --package levanter --extra tpu --group test --frozen\n)',
            r'\1' + venv_path_line + '\n',
            modified
        )
        print(f"    ✓ Added venv_path.txt creation")

    # Save if modified
    if modified != original:
        script_path.write_text(modified)
        return True

    return False


def main():
    """Update TPU setup scripts."""
    cwd = Path.cwd()
    setup_scripts_dir = cwd / "lib/levanter/infra/helpers"

    if not setup_scripts_dir.exists():
        print(f"ERROR: Setup scripts directory not found: {setup_scripts_dir}")
        sys.exit(1)

    print("Updating TPU setup scripts...")
    print(f"Working directory: {cwd}")
    print()

    updated_count = 0
    for script_name in ["setup-tpu-vm.sh", "setup-tpu-vm-tests.sh"]:
        script_path = setup_scripts_dir / script_name
        if script_path.exists():
            if update_tpu_setup_script(script_path):
                updated_count += 1
        else:
            print(f"  ! Skipping {script_name} (not found)")

    print()
    if updated_count > 0:
        print(f"✓ Updated {updated_count} TPU setup scripts!")
    else:
        print("✓ All TPU setup scripts already up to date!")


if __name__ == "__main__":
    main()
