#!/usr/bin/env -S uv run
# /// script
# dependencies = ["ruamel.yaml>=0.17.0"]
# ///
"""
Update GitHub Actions workflows for step 2 workspace migration.

Uses lossless YAML editing for value changes and targeted string replacement
for structural modifications.

Updates:
1. Marin workflows: Add "Marin - " prefix to workflow name
2. Levanter workflows:
   - Add "Levanter - " prefix to workflow name
   - Add path filters to trigger only on relevant changes
   - Set defaults.run.working-directory: lib/levanter
   - Add working-directory to astral-sh/setup-uv step
   - Use --package levanter for uv commands
   - Update TPU SSH commands to use marin/lib/levanter paths
"""

import re
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
    Update a Levanter workflow for workspace structure.

    Uses lossless YAML for value replacements and string operations for
    structural changes.

    Returns:
        True if the workflow was updated, False if no changes needed
    """
    print(f"  Processing {workflow_path.name}...")

    # Phase 1: Use LosslessYAML for value replacements
    doc = LosslessYAML.load(workflow_path)
    value_changes_made = False

    # 1. Update workflow name with "Levanter - " prefix
    if "name" in doc.data:
        old_name = doc.data["name"]
        if not old_name.startswith("Levanter - "):
            new_name = f"Levanter - {old_name}"
            doc.replace_in_values(old_name, new_name)
            value_changes_made = True
            print(f"    ✓ Updated name: {old_name} -> {new_name}")

    # 2. Update TPU SSH paths in command strings
    # Use replace_in_values for paths within command strings
    if "tpu" in workflow_path.name.lower():
        # Check if there are SSH commands with old paths
        original_text = workflow_path.read_text()
        if "levanter/tests" in original_text or "levanter/infra" in original_text:
            # These replacements work because they're in string values
            doc.replace_in_values("levanter/tests", "marin/lib/levanter/tests")
            doc.replace_in_values("levanter/infra", "marin/lib/levanter/infra")
            value_changes_made = True
            print(f"    ✓ Updated SSH paths: levanter/ -> marin/lib/levanter/")

    # Save value changes if any were made
    if value_changes_made:
        doc.save()

    # Phase 2: Structural modifications using string operations
    # These changes add/modify YAML structure, which lossless-yaml doesn't handle
    original = workflow_path.read_text()
    modified = original
    structural_changes_made = False

    # 3. Add path filters to triggers
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
        structural_changes_made = True
        print(f"    ✓ Added path filters")
    elif "on: [push]" in modified:
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
        modified = modified.replace("on: [push]", paths_section.rstrip(), 1)
        structural_changes_made = True
        print(f"    ✓ Added path filters")

    # 4. Add defaults.run.working-directory after runs-on
    if "defaults:" not in modified or "working-directory: lib/levanter" not in modified:
        # Try pattern 1: runs-on followed by strategy
        runs_on_pattern = r'(\n    runs-on: [^\n]+\n)(    strategy:)'
        defaults_section = r'''\1    defaults:
      run:
        working-directory: lib/levanter
\2'''

        if re.search(runs_on_pattern, modified):
            modified = re.sub(runs_on_pattern, defaults_section, modified, count=1)
            structural_changes_made = True
            print(f"    ✓ Added defaults.run.working-directory")
        else:
            # Try pattern 2: runs-on followed by env
            runs_on_env_pattern = r'(\n    runs-on: [^\n]+\n)(    env:)'
            if re.search(runs_on_env_pattern, modified):
                modified = re.sub(runs_on_env_pattern, defaults_section, modified, count=1)
                structural_changes_made = True
                print(f"    ✓ Added defaults.run.working-directory")
            else:
                # Try pattern 3: runs-on followed by steps or if
                runs_on_steps_pattern = r'(\n    runs-on: [^\n]+\n)(    if:|\n    steps:)'
                if re.search(runs_on_steps_pattern, modified):
                    modified = re.sub(runs_on_steps_pattern, defaults_section, modified, count=1)
                    structural_changes_made = True
                    print(f"    ✓ Added defaults.run.working-directory")

    # 5. Add working-directory to astral-sh/setup-uv step
    setup_uv_section = re.search(
        r'(uses: astral-sh/setup-uv@[^\n]+\n\s+with:\n)((?:\s+[^\n]+\n)*?)(\s+- name:)',
        modified,
        re.MULTILINE
    )
    if setup_uv_section and "working-directory: lib/levanter" not in setup_uv_section.group(0):
        indent = "          "  # 10 spaces to match other 'with' items
        replacement = (
            setup_uv_section.group(1) +
            setup_uv_section.group(2) +
            f"{indent}working-directory: lib/levanter\n" +
            setup_uv_section.group(3)
        )
        modified = modified[:setup_uv_section.start()] + replacement + modified[setup_uv_section.end():]
        structural_changes_made = True
        print(f"    ✓ Added working-directory to setup-uv")

    # 6. Update uv commands to include --package levanter
    # Check if uv sync needs updating
    if "uv sync" in modified:
        new_modified = re.sub(
            r'\buv sync(?! --package)',
            'uv sync --package levanter',
            modified
        )
        if new_modified != modified:
            modified = new_modified
            structural_changes_made = True
            print(f"    ✓ Updated uv sync commands")

    # Check if uv run needs updating
    if "uv run" in modified:
        new_modified = re.sub(
            r'\buv run(?! --package)',
            'uv run --package levanter',
            modified
        )
        if new_modified != modified:
            modified = new_modified
            structural_changes_made = True
            print(f"    ✓ Updated uv run commands")

    # Save structural changes if any were made
    if structural_changes_made:
        workflow_path.write_text(modified)

    return value_changes_made or structural_changes_made


def main():
    """Update all workflows."""
    cwd = Path.cwd()
    workflows_dir = cwd / ".github/workflows"

    if not workflows_dir.exists():
        print(f"ERROR: Workflows directory not found: {workflows_dir}")
        sys.exit(1)

    print("Updating GitHub Actions workflows...")
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
