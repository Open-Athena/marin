#!/usr/bin/env -S uv run
# /// script
# dependencies = ["lossless-yaml==0.1.0"]
# ///
"""
Update GitHub Actions workflows for step 2 workspace migration.

More robust version with better conflict detection and semantic checks.
"""

import sys
from pathlib import Path
from yaya import YAYA


class WorkflowConflict(Exception):
    """Raised when a workflow has unexpected structure requiring manual intervention."""
    pass


def update_marin_workflow(workflow_path: Path) -> bool:
    """
    Update a Marin workflow - add 'Marin - ' prefix to workflow name.

    Returns:
        True if the workflow was updated, False if no changes needed

    Raises:
        WorkflowConflict if workflow is not in expected list
    """
    # Whitelist of expected Marin workflows from ws branch (step 1)
    EXPECTED_MARIN_WORKFLOWS = {
        "marin-build-docker-images.yaml",
        "marin-codeql.yml",
        "marin-docs.yaml",
        "marin-lint-and-format.yaml",
        "marin-metrics.yaml",
        "marin-quickstart.yaml",
        "marin-tpu-tests.yaml",
        "marin-unit-tests.yaml",
        "marin-update-leaderboard.yml",
    }

    if workflow_path.name not in EXPECTED_MARIN_WORKFLOWS:
        raise WorkflowConflict(
            f"{workflow_path.name}: Unknown Marin workflow, not in expected list from ws branch (step 1). "
            f"Add to EXPECTED_MARIN_WORKFLOWS list after reviewing."
        )

    doc = YAYA.load(workflow_path)

    if "name" not in doc.data:
        print(f"  ! Skipping {workflow_path.name} (no name field)")
        return False

    old_name = doc.data["name"]
    if old_name.startswith("Marin - "):
        print(f"  - {workflow_path.name}: already has prefix")
        return False

    new_name = f"Marin - {old_name}"

    doc.replace_key("name", new_name)
    doc.save()

    print(f"  ✓ {workflow_path.name}: {old_name} -> {new_name}")
    return True


def update_levanter_workflow(workflow_path: Path) -> bool:
    """
    Update a Levanter workflow with semantic conflict detection.

    Raises WorkflowConflict if manual intervention is needed.
    """
    print(f"  Processing {workflow_path.name}...")

    doc = YAYA.load(workflow_path)
    modified = False

    # 1. Update workflow name
    if "name" in doc.data:
        old_name = doc.data["name"]
        if not old_name.startswith("Levanter - "):
            new_name = f"Levanter - {old_name}"
            doc.replace_key("name", new_name)
            modified = True
            print(f"    ✓ Updated name: {old_name} -> {new_name}")

    # 2. Expand trigger with path filters - with conflict detection
    # Assert exact expected trigger for each workflow from Levanter 77aa5913d
    # Map workflow name (without levanter- prefix) to expected trigger
    EXPECTED_TRIGGERS = {
        "check_lockfile.yaml": ["push", "pull_request"],
        "docker-base-image.yaml": "workflow_run",
        "docker-cluster-image.yaml": "workflow_run",
        "gpt2_small_itest.yaml": "workflow_run",
        "launch_small_fast.yaml": "workflow_run",
        "publish_dev.yaml": "workflow_run",
        "run_entry_tests.yaml": ["push", "pull_request"],
        "run_pre_commit.yaml": ["push", "pull_request"],
        "run_ray_tests.yaml": ["push"],
        "run_tests.yaml": ["push", "pull_request"],
        "tpu_unit_tests.yaml": {"pull_request", "workflow_dispatch"},  # Has both triggers as of 07372de8
    }

    # Get workflow basename without levanter- prefix
    workflow_basename = workflow_path.name.replace("levanter-", "")
    expected_trigger = EXPECTED_TRIGGERS.get(workflow_basename)

    if expected_trigger is None:
        raise WorkflowConflict(
            f"{workflow_path.name}: Unknown workflow, not in Levanter 95ab586e2. "
            f"Add to EXPECTED_TRIGGERS map."
        )

    on_value = doc["on"]

    if expected_trigger == "workflow_run":
        # Assert it has workflow_run trigger
        if isinstance(on_value, dict) and "workflow_run" in on_value:
            print(f"    - Skipping workflow_run trigger (doesn't need path filters)")
        else:
            raise WorkflowConflict(
                f"{workflow_path.name}: Expected workflow_run trigger, got: {on_value}"
            )
    elif isinstance(expected_trigger, set):
        # Handle workflows with multiple trigger types (e.g., pull_request + workflow_dispatch)
        # We add path filters only to pull_request, leave others alone
        if not isinstance(on_value, dict):
            raise WorkflowConflict(
                f"{workflow_path.name}: Expected dict with triggers {expected_trigger}, got: {on_value}"
            )

        # Check that we have the expected triggers
        actual_triggers = set(on_value.keys())
        if actual_triggers != expected_trigger:
            raise WorkflowConflict(
                f"{workflow_path.name}: Expected triggers {expected_trigger}, got: {actual_triggers}"
            )

        # Add path filters to pull_request if present
        if "pull_request" in on_value:
            pr_value = on_value["pull_request"]
            if pr_value is None:
                # Simple 'pull_request:' trigger - expand it with path filters
                # Can't use replace_key_path (doesn't exist), so reconstruct the whole 'on' dict
                new_on = dict(on_value)
                new_on["pull_request"] = {
                    "paths": [
                        "lib/levanter/**",
                        ".github/workflows/" + workflow_path.name
                    ]
                }
                doc.replace_key("on", new_on)
                modified = True
                print(f"    ✓ Added path filters to pull_request trigger")
            elif isinstance(pr_value, dict):
                # Already a dict - check if it has our paths
                existing_paths = pr_value.get("paths", [])
                if "lib/levanter/**" in existing_paths:
                    print(f"    - pull_request trigger already has path filters")
                else:
                    raise WorkflowConflict(
                        f"{workflow_path.name}: pull_request has unexpected paths: {existing_paths}"
                    )
            else:
                raise WorkflowConflict(
                    f"{workflow_path.name}: Unexpected pull_request value: {pr_value}"
                )

        # Leave other triggers (like workflow_dispatch) untouched
        print(f"    - Preserving other triggers: {actual_triggers - {'pull_request'}}")

    elif isinstance(expected_trigger, list):
        # Assert exact list match, then add path filters
        if on_value == expected_trigger:
            # Matches expected - add path filters
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
                    print(f"    - Already has path filters")
                else:
                    raise WorkflowConflict(
                        f"{workflow_path.name}: Has expanded trigger but wrong paths. "
                        f"Expected lib/levanter/** in paths, got: {push_paths}"
                    )
            else:
                raise WorkflowConflict(
                    f"{workflow_path.name}: Expected {expected_trigger}, got dict: {on_value}"
                )
        else:
            raise WorkflowConflict(
                f"{workflow_path.name}: Expected {expected_trigger}, got: {on_value}. "
                f"Levanter main may have changed - update EXPECTED_TRIGGERS."
            )

    # 3. Add defaults.run.working-directory with better conflict handling
    # EXCEPTION: TPU workflows should NOT have working-directory because they:
    # - Run infra scripts relative to repo root
    # - SSH into VMs that clone the full repo
    # - Reference paths from repo root in SSH commands
    is_tpu_workflow = "tpu" in workflow_path.name.lower()

    for job_name, job in doc["jobs"].items():
        # Skip adding working-directory for TPU workflows
        if is_tpu_workflow:
            print(f"    - Skipping working-directory for TPU workflow")
            continue

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
            # Try to insert between various job keys and 'steps'
            # Order matters: try more specific keys first (if, strategy, env, permissions) before runs-on
            inserted = False
            for prev_key in ["if", "strategy", "env", "permissions", "needs", "runs-on"]:
                try:
                    doc.insert_key_between(
                        f"jobs.{job_name}",
                        prev_key=prev_key,
                        next_key="steps",
                        new_key="defaults",
                        value={"run": {"working-directory": "lib/levanter"}}
                    )
                    modified = True
                    inserted = True
                    print(f"    ✓ Added defaults.run.working-directory to {job_name} (between {prev_key} and steps)")
                    break
                except (KeyError, ValueError):
                    continue

            if not inserted:
                # None of the expected structures worked
                raise WorkflowConflict(
                    f"{workflow_path.name}: Job {job_name} has unusual structure. "
                    f"Cannot insert defaults before 'steps'. Try adding it manually."
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
                        doc.ensure_key(f"jobs.{job_name}.steps[{i}].with.working-directory", "lib/levanter")
                        modified = True
                        print(f"    ✓ Added working-directory to setup-uv in {job_name}")

    # 5. Update uv commands - check if actually needed
    original_text = workflow_path.read_text()
    needs_uv_update = False

    # Check if there are uv commands without --package or --frozen
    if "uv sync" in original_text and "uv sync --package levanter" not in original_text:
        needs_uv_update = True
    if "uv run" in original_text and "uv run --package levanter" not in original_text:
        needs_uv_update = True
    if "uv sync" in original_text and "--frozen" not in original_text:
        needs_uv_update = True

    if needs_uv_update:
        # Add --package levanter to uv sync/run commands
        doc.replace_in_values_regex(r'\buv sync(?! --package)', 'uv sync --package levanter')
        doc.replace_in_values_regex(r'\buv run(?! --package)', 'uv run --package levanter')
        # Add --frozen to uv sync and uv run commands (use lockfile, don't re-resolve or update git deps)
        # Handle both "uv sync --package levanter" and "uv sync --package levanter --dev"
        doc.replace_in_values_regex(r'\buv sync --package levanter( --dev)?(?! --frozen)', r'uv sync --package levanter\1 --frozen')
        # Add --frozen to uv run commands (handle optional --with flag with quoted value)
        # Match --with followed by either: quoted string (with spaces) or unquoted word
        doc.replace_in_values_regex(r'\buv run --package levanter( --with (?:"[^"]+"|[^\s]+))?(?! --frozen)', r'uv run --package levanter --frozen\1')
        modified = True
        print(f"    ✓ Updated uv commands")
    elif "uv sync" in original_text or "uv run" in original_text:
        print(f"    - uv commands already have --package")

    # 6. Update TPU workflow paths
    # TPU workflows run from repo root (no working-directory), so they need:
    # - Script paths: lib/levanter/infra/... (for bash commands)
    # - SSH paths: marin/lib/levanter/... (for commands on TPU VM)
    if is_tpu_workflow:
        needs_script_update = "infra/spin-up-vm.sh" in original_text or "infra/helpers/" in original_text
        needs_ssh_update = "levanter/tests" in original_text or "levanter/infra" in original_text

        if needs_script_update:
            # Update script paths to lib/levanter/infra/...
            doc.replace_in_values_regex(r'\binfra/(spin-up-vm\.sh|helpers/[^\s]+)', r'lib/levanter/infra/\1')
            modified = True
            print(f"    ✓ Updated script paths: infra/ -> lib/levanter/infra/")

        if needs_ssh_update:
            # Update SSH command paths to marin/lib/levanter/...
            # Use negative lookbehind to avoid matching lib/levanter/ (from script path updates)
            doc.replace_in_values_regex(r'(?<!lib/)levanter/(tests|infra)\b', r'marin/lib/levanter/\1')
            modified = True
            print(f"    ✓ Updated SSH paths: levanter/ -> marin/lib/levanter/")

        if not needs_script_update and "lib/levanter/infra" in original_text:
            print(f"    - Script paths already updated")
        if not needs_ssh_update and "marin/lib/levanter" in original_text:
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

    print("Updating GitHub Actions workflows...")
    print(f"Working directory: {cwd}")
    print()

    # Update Marin workflows
    print("Updating Marin workflows:")
    marin_count = 0
    error_count = 0

    for pattern in ["marin-*.yaml", "marin-*.yml"]:
        for workflow_path in sorted(workflows_dir.glob(pattern)):
            try:
                if update_marin_workflow(workflow_path):
                    marin_count += 1
            except WorkflowConflict as e:
                print(f"  ⚠️  CONFLICT: {e}")
                error_count += 1
            except Exception as e:
                print(f"  ❌ ERROR in {workflow_path.name}: {e}")
                error_count += 1
                raise
    print()

    # Update Levanter workflows
    print("Updating Levanter workflows:")
    levanter_count = 0

    for pattern in ["levanter-*.yaml", "levanter-*.yml"]:
        for workflow_path in sorted(workflows_dir.glob(pattern)):
            try:
                if update_levanter_workflow(workflow_path):
                    levanter_count += 1
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
    else:
        total = marin_count + levanter_count
        if total > 0:
            print(f"✓ Updated {total} workflows ({marin_count} Marin, {levanter_count} Levanter)!")
        else:
            print("✓ All workflows already up to date!")


if __name__ == "__main__":
    main()
