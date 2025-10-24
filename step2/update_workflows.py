#!/usr/bin/env -S uv run
# /// script
# dependencies = ["ruamel.yaml>=0.17.0"]
# ///
"""
Update GitHub Actions workflows for step 2 workspace migration.

More robust version with better conflict detection and semantic checks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "c/lossless-yaml/src"))

from lossless_yaml import LosslessYAML


class WorkflowConflict(Exception):
    """Raised when a workflow has unexpected structure requiring manual intervention."""
    pass


def update_levanter_workflow(workflow_path: Path) -> bool:
    """
    Update a Levanter workflow with semantic conflict detection.

    Raises WorkflowConflict if manual intervention is needed.
    """
    print(f"  Processing {workflow_path.name}...")

    doc = LosslessYAML.load(workflow_path)
    modified = False

    # 1. Update workflow name
    if "name" in doc.data:
        old_name = doc.data["name"]
        if not old_name.startswith("Levanter - "):
            new_name = f"Levanter - {old_name}"
            doc.replace_in_values(old_name, new_name)
            modified = True
            print(f"    ✓ Updated name: {old_name} -> {new_name}")

    # 2. Expand trigger with path filters - with conflict detection
    on_value = doc["on"]

    if on_value in (["push"], ["push", "pull_request"]):
        # Simple case: exactly what we expect
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
    elif isinstance(on_value, dict):
        # Already expanded - check if it has our path filters
        if "push" in on_value and isinstance(on_value["push"], dict):
            push_paths = on_value["push"].get("paths", [])
            if "lib/levanter/**" in push_paths:
                # Already has our paths, good
                print(f"    - Already has path filters")
            else:
                # Has dict structure but different paths - conflict!
                raise WorkflowConflict(
                    f"{workflow_path.name}: 'on.push' is a dict but doesn't have expected path filters. "
                    f"Current paths: {push_paths}. Manual merge needed."
                )
        else:
            # Dict but no push, or push is not a dict - unexpected
            raise WorkflowConflict(
                f"{workflow_path.name}: 'on' has unexpected structure: {on_value}. "
                f"Expected list or dict with push/pull_request."
            )
    elif isinstance(on_value, list):
        # List but not one we recognize
        raise WorkflowConflict(
            f"{workflow_path.name}: 'on' is {on_value}, not the expected [push] or [push, pull_request]. "
            f"Manual intervention needed to add path filters."
        )
    else:
        raise WorkflowConflict(
            f"{workflow_path.name}: 'on' has unexpected type {type(on_value).__name__}: {on_value}"
        )

    # 3. Add defaults.run.working-directory with better conflict handling
    for job_name, job in doc["jobs"].items():
        try:
            # Check if working-directory is already set correctly
            existing_wd = doc.get_path(f"jobs.{job_name}.defaults.run.working-directory")
            if existing_wd == "lib/levanter":
                print(f"    - {job_name} already has working-directory")
            else:
                raise WorkflowConflict(
                    f"{workflow_path.name}: Job {job_name} has defaults.run.working-directory={existing_wd!r}, "
                    f"expected 'lib/levanter'. Manual merge needed."
                )
        except KeyError:
            # working-directory not set, need to add it

            # Check if defaults.run exists
            try:
                doc.get_path(f"jobs.{job_name}.defaults.run")
                # Exists but no working-directory - add it
                # TODO: Need add_key API for this
                # For now, this is a limitation
                raise WorkflowConflict(
                    f"{workflow_path.name}: Job {job_name} has defaults.run but no working-directory. "
                    f"Need add_key API to merge this properly."
                )
            except KeyError:
                pass

            # Check if defaults exists at all
            try:
                doc.get_path(f"jobs.{job_name}.defaults")
                # Exists but no .run - conflict
                raise WorkflowConflict(
                    f"{workflow_path.name}: Job {job_name} has defaults but not defaults.run. "
                    f"Manual intervention needed."
                )
            except KeyError:
                pass

            # No defaults at all - add the whole thing
            try:
                doc.add_key_after(
                    f"jobs.{job_name}.runs-on",
                    "defaults",
                    {"run": {"working-directory": "lib/levanter"}}
                )
                modified = True
                print(f"    ✓ Added defaults.run.working-directory to {job_name}")
            except KeyError:
                # No runs-on - this is unusual and should be flagged
                raise WorkflowConflict(
                    f"{workflow_path.name}: Job {job_name} has no runs-on key. "
                    f"Unusual structure, manual intervention needed."
                )

    # 4. Add working-directory to setup-uv steps
    for job_name, job in doc["jobs"].items():
        for i, step in enumerate(job.get("steps", [])):
            if "uses" in step and "astral-sh/setup-uv" in step["uses"]:
                if "with" in step:
                    try:
                        existing = doc.get_path(f"jobs.{job_name}.steps[{i}].with.working-directory")
                        if existing != "lib/levanter":
                            raise WorkflowConflict(
                                f"{workflow_path.name}: setup-uv in {job_name} has working-directory={existing!r}, "
                                f"expected 'lib/levanter'."
                            )
                        print(f"    - setup-uv in {job_name} already has working-directory")
                    except KeyError:
                        # Not set, add it
                        # NOTE: Direct mutation - not ideal, but works for now
                        # TODO: Need add_key API for existing dicts
                        step["with"]["working-directory"] = "lib/levanter"
                        modified = True
                        print(f"    ✓ Added working-directory to setup-uv in {job_name}")

    # 5. Update uv commands - check if actually needed
    original_text = workflow_path.read_text()
    needs_uv_update = False

    # Check if there are uv commands without --package
    if "uv sync" in original_text and "uv sync --package levanter" not in original_text:
        needs_uv_update = True
    if "uv run" in original_text and "uv run --package levanter" not in original_text:
        needs_uv_update = True

    if needs_uv_update:
        doc.replace_in_values_regex(r'\buv sync(?! --package)', 'uv sync --package levanter')
        doc.replace_in_values_regex(r'\buv run(?! --package)', 'uv run --package levanter')
        modified = True
        print(f"    ✓ Updated uv commands")
    elif "uv sync" in original_text or "uv run" in original_text:
        print(f"    - uv commands already have --package")

    # 6. Update TPU SSH paths - check if actually needed
    if "tpu" in workflow_path.name.lower():
        needs_path_update = (
            "levanter/tests" in original_text or
            "levanter/infra" in original_text
        )

        if needs_path_update:
            doc.replace_in_values("levanter/tests", "marin/lib/levanter/tests")
            doc.replace_in_values("levanter/infra", "marin/lib/levanter/infra")
            modified = True
            print(f"    ✓ Updated SSH paths: levanter/ -> marin/lib/levanter/")
        elif "marin/lib/levanter" in original_text:
            print(f"    - SSH paths already updated")

    if modified:
        doc.save()

    return modified


def main():
    """Update all workflows with conflict detection."""
    cwd = Path.cwd()
    workflows_dir = cwd / ".github/workflows"

    if not workflows_dir.exists():
        print(f"ERROR: Workflows directory not found: {workflows_dir}")
        sys.exit(1)

    print("Updating Levanter workflows...")
    print(f"Working directory: {cwd}")
    print()

    updated_count = 0
    error_count = 0

    for pattern in ["levanter-*.yaml", "levanter-*.yml"]:
        for workflow_path in sorted(workflows_dir.glob(pattern)):
            try:
                if update_levanter_workflow(workflow_path):
                    updated_count += 1
            except WorkflowConflict as e:
                print(f"  ⚠️  CONFLICT: {e}")
                error_count += 1
            except Exception as e:
                print(f"  ❌ ERROR in {workflow_path.name}: {e}")
                error_count += 1
                raise

    print()
    if error_count > 0:
        print(f"⚠️  {error_count} conflicts detected - manual intervention needed")
        sys.exit(1)
    elif updated_count > 0:
        print(f"✓ Updated {updated_count} workflows!")
    else:
        print("✓ All workflows already up to date!")


if __name__ == "__main__":
    main()
