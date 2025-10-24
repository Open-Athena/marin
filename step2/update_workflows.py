#!/usr/bin/env -S uv run
# /// script
# dependencies = ["ruamel.yaml>=0.17.0"]
# ///
"""
Update GitHub Actions workflows for step 2 workspace migration.

Uses lossless YAML editing to preserve byte-level formatting while making
structural and value changes.

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
    doc.replace_in_values(old_name, new_name)
    doc.save()

    print(f"  ✓ {workflow_path.name}: {old_name} -> {new_name}")
    return True


def update_levanter_workflow(workflow_path: Path) -> bool:
    """
    Update a Levanter workflow for workspace structure using lossless YAML.

    Returns:
        True if the workflow was updated, False if no changes needed
    """
    print(f"  Processing {workflow_path.name}...")

    doc = LosslessYAML.load(workflow_path)
    modified = False

    # 1. Update workflow name with "Levanter - " prefix
    if "name" in doc.data:
        old_name = doc.data["name"]
        if not old_name.startswith("Levanter - "):
            new_name = f"Levanter - {old_name}"
            doc.replace_in_values(old_name, new_name)
            modified = True
            print(f"    ✓ Updated name: {old_name} -> {new_name}")

    # 2. Expand trigger with path filters
    # Check if on is a simple list like [push] or [push, pull_request]
    if doc["on"] in (["push"], ["push", "pull_request"]):
        doc.replace_key("on", {
            "push": {
                "branches": ["main"],
                "paths": [
                    "lib/levanter/**",
                    "uv.lock",
                    f".github/workflows/{workflow_path.name}"
                ]
            },
            "pull_request": {
                "paths": [
                    "lib/levanter/**",
                    "uv.lock",
                    f".github/workflows/{workflow_path.name}"
                ]
            }
        })
        modified = True
        print(f"    ✓ Added path filters")

    # 3. Add defaults.run.working-directory to each job
    for job_name, job in doc["jobs"].items():
        try:
            doc.assert_absent(f"jobs.{job_name}.defaults")
            # Add defaults after runs-on
            doc.add_key_after(
                f"jobs.{job_name}.runs-on",
                "defaults",
                {"run": {"working-directory": "lib/levanter"}}
            )
            modified = True
            print(f"    ✓ Added defaults.run.working-directory to {job_name}")
        except (AssertionError, KeyError):
            # Already exists or runs-on not found, skip
            pass

    # 4. Add working-directory to setup-uv steps
    for job_name, job in doc["jobs"].items():
        for i, step in enumerate(job.get("steps", [])):
            if "uses" in step and "astral-sh/setup-uv" in step["uses"]:
                if "with" in step:
                    try:
                        doc.assert_absent(f"jobs.{job_name}.steps[{i}].with.working-directory")
                        # Would need add_key for dicts, but we can use a workaround
                        # by getting the with dict and updating it
                        step["with"]["working-directory"] = "lib/levanter"
                        modified = True
                        print(f"    ✓ Added working-directory to setup-uv in {job_name}")
                    except AssertionError:
                        pass

    # 5. Update uv commands to include --package levanter
    # Use regex replacement for command strings
    original_text = workflow_path.read_text()
    if "uv sync" in original_text or "uv run" in original_text:
        doc.replace_in_values_regex(r'\buv sync(?! --package)', 'uv sync --package levanter')
        doc.replace_in_values_regex(r'\buv run(?! --package)', 'uv run --package levanter')
        modified = True
        print(f"    ✓ Updated uv commands")

    # 6. Update TPU SSH paths in command strings
    if "tpu" in workflow_path.name.lower():
        original_text = workflow_path.read_text()
        if "levanter/tests" in original_text or "levanter/infra" in original_text:
            doc.replace_in_values("levanter/tests", "marin/lib/levanter/tests")
            doc.replace_in_values("levanter/infra", "marin/lib/levanter/infra")
            modified = True
            print(f"    ✓ Updated SSH paths: levanter/ -> marin/lib/levanter/")

    if modified:
        doc.save()

    return modified


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
