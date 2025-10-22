#!/usr/bin/env -S uv run
# /// script
# dependencies = ["ruamel.yaml>=0.17.0"]
# ///
"""
Update GitHub Actions workflows for step 2 workspace migration.

Uses lossless byte-level string replacement for maximum preservation of
original formatting.

Updates:
1. Marin workflows: Add "Marin - " prefix to workflow name
2. Levanter workflows:
   - Add "Levanter - " prefix to workflow name
   - Add path filters to trigger only on relevant changes
   - Set defaults.run.working-directory: lib/levanter
   - Add working-directory to astral-sh/setup-uv step
   - Use --package levanter for uv commands
"""

import sys
from pathlib import Path

# Add lossless-yaml to path
sys.path.insert(0, str(Path.home() / "c/lossless-yaml/src"))

from lossless_yaml import LosslessYAML


def update_marin_workflow(workflow_path: Path) -> bool:
    """
    Update a Marin workflow with name prefix using lossless YAML.

    Returns:
        True if the workflow was updated, False if no changes needed
    """
    doc = LosslessYAML.load(workflow_path)

    if "name" not in doc.data:
        print(f"  ! Skipping {workflow_path.name} (no name field)")
        return False

    old_name = doc.data["name"]
    if old_name.startswith("Marin - "):
        print(f"  - {workflow_path.name}: already has prefix")
        return False

    new_name = f"Marin - {old_name}"

    # Use replace_in_values for byte-level replacement
    doc.replace_in_values(old_name, new_name)
    doc.save()

    print(f"  ✓ {workflow_path.name}: {old_name} -> {new_name}")
    return True


def update_levanter_workflow(workflow_path: Path) -> bool:
    """
    Update a Levanter workflow for workspace structure using lossless edits.

    Returns:
        True if the workflow was updated, False if no changes needed
    """
    # Read original bytes
    original = workflow_path.read_text()
    modified = original

    print(f"  Processing {workflow_path.name}...")

    # 1. Update workflow name with "Levanter - " prefix
    # Match: name: <anything>
    import re
    name_match = re.search(r'^name:\s*(.+)$', modified, re.MULTILINE)
    if name_match:
        old_name = name_match.group(1)
        if not old_name.startswith("Levanter - "):
            new_name = f"Levanter - {old_name}"
            modified = modified.replace(
                f"name: {old_name}",
                f"name: {new_name}",
                1
            )
            print(f"    ✓ Updated name: {old_name} -> {new_name}")

    # 2. Update triggers with path filters
    # Match: on: [push, pull_request]
    if "on: [push, pull_request]" in modified:
        paths_section = f'''on:
  push:
    branches:
      - main
    paths:
      - 'lib/levanter/**'
      - 'uv.lock'
      - '.github/workflows/{workflow_path.name}'
  pull_request:
    paths:
      - 'lib/levanter/**'
      - 'uv.lock'
      - '.github/workflows/{workflow_path.name}' '''

        modified = modified.replace("on: [push, pull_request]", paths_section.rstrip(), 1)
        print(f"    ✓ Added path filters")

    # 3. Add defaults.run.working-directory after runs-on
    # Match the pattern and insert defaults after runs-on line, preserving blank line before strategy
    if "defaults:" not in modified or "working-directory: lib/levanter" not in modified:
        # Find runs-on line and add defaults after it (before strategy section)
        runs_on_pattern = r'(\n    runs-on: [^\n]+\n)(    strategy:)'
        defaults_section = r'''\1    defaults:
      run:
        working-directory: lib/levanter
\2'''

        if re.search(runs_on_pattern, modified):
            modified = re.sub(runs_on_pattern, defaults_section, modified, count=1)
            print(f"    ✓ Added defaults.run.working-directory")

    # 4. Add working-directory to astral-sh/setup-uv step
    # Find the setup-uv section and add working-directory if not present
    setup_uv_section = re.search(r'(uses: astral-sh/setup-uv@[^\n]+\n\s+with:\n)((?:\s+[^\n]+\n)*?)(\s+- name:)',  modified, re.MULTILINE)
    if setup_uv_section and "working-directory: lib/levanter" not in setup_uv_section.group(0):
        # Extract the indent level from the last line in 'with'
        indent = "          "  # 10 spaces to match other 'with' items
        replacement = setup_uv_section.group(1) + setup_uv_section.group(2) + f"{indent}working-directory: lib/levanter\n" + setup_uv_section.group(3)
        modified = modified[:setup_uv_section.start()] + replacement + modified[setup_uv_section.end():]
        print(f"    ✓ Added working-directory to setup-uv")

    # 5. Update uv commands to include --package levanter
    uv_commands_updated = False

    # Check if uv sync needs updating
    if "uv sync" in modified:
        new_modified = re.sub(
            r'\buv sync(?! --package)',
            'uv sync --package levanter',
            modified
        )
        if new_modified != modified:
            modified = new_modified
            uv_commands_updated = True

    # Check if uv run needs updating
    if "uv run" in modified:
        new_modified = re.sub(
            r'\buv run(?! --package)',
            'uv run --package levanter',
            modified
        )
        if new_modified != modified:
            modified = new_modified
            uv_commands_updated = True

    if uv_commands_updated:
        print(f"    ✓ Updated uv commands")

    # Save if modified
    if modified != original:
        workflow_path.write_text(modified)
        return True

    return False


def main():
    """Update all workflows."""
    cwd = Path.cwd()
    workflows_dir = cwd / ".github/workflows"

    if not workflows_dir.exists():
        print(f"ERROR: Workflows directory not found: {workflows_dir}")
        sys.exit(1)

    print("Updating GitHub Actions workflows with lossless byte-level edits...")
    print(f"Working directory: {cwd}")
    print()

    # Update Marin workflows
    print("Updating Marin workflows:")
    marin_count = 0
    for pattern in ["marin-*.yaml", "marin-*.yml"]:
        for workflow_path in sorted(workflows_dir.glob(pattern)):
            if update_marin_workflow(workflow_path):
                marin_count += 1
    print(f"  Updated {marin_count} Marin workflows")
    print()

    # Update Levanter workflows
    print("Updating Levanter workflows:")
    levanter_count = 0
    for pattern in ["levanter-*.yaml", "levanter-*.yml"]:
        for workflow_path in sorted(workflows_dir.glob(pattern)):
            if update_levanter_workflow(workflow_path):
                levanter_count += 1
    print(f"  Updated {levanter_count} Levanter workflows")
    print()

    total = marin_count + levanter_count
    if total > 0:
        print(f"✓ Updated {total} workflows!")
    else:
        print("✓ All workflows already up to date!")


if __name__ == "__main__":
    main()
