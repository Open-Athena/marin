#!/usr/bin/env -S uv run
# /// script
# dependencies = ["ruamel.yaml>=0.17.0"]
# ///
"""
Update GitHub Actions workflows for step 2 workspace migration.

Uses ruamel.yaml to preserve formatting, comments, and whitespace while
making structural changes.

Updates:
1. Marin workflows: Add "Marin - " prefix to workflow name
2. Levanter workflows:
   - Add "Levanter - " prefix to workflow name
   - Add path filters to trigger only on relevant changes
   - Set defaults.run.working-directory: lib/levanter
   - Add working-directory to astral-sh/setup-uv step
   - Use --package levanter for uv commands
"""

from pathlib import Path
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq


def update_marin_workflow(workflow_path: Path) -> bool:
    """
    Update a Marin workflow with name prefix.

    Returns:
        True if the workflow was updated, False if no changes needed
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False

    with open(workflow_path, "r") as f:
        doc = yaml.load(f)

    if "name" not in doc:
        print(f"  ! Skipping {workflow_path.name} (no name field)")
        return False

    old_name = doc["name"]
    if old_name.startswith("Marin - "):
        print(f"  - {workflow_path.name}: already has prefix")
        return False

    new_name = f"Marin - {old_name}"
    doc["name"] = new_name

    with open(workflow_path, "w") as f:
        yaml.dump(doc, f)

    print(f"  ✓ {workflow_path.name}: {old_name} -> {new_name}")
    return True


def update_levanter_workflow(workflow_path: Path) -> bool:
    """
    Update a Levanter workflow for workspace structure.

    Returns:
        True if the workflow was updated, False if no changes needed
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    yaml.width = 4096  # Avoid line wrapping

    with open(workflow_path, "r") as f:
        doc = yaml.load(f)

    modified = False

    # 1. Update workflow name
    if "name" in doc:
        old_name = doc["name"]
        if not old_name.startswith("Levanter - "):
            new_name = f"Levanter - {old_name}"
            doc["name"] = new_name
            print(f"  ✓ {workflow_path.name}: {old_name} -> {new_name}")
            modified = True
        else:
            print(f"  - {workflow_path.name}: name already has prefix")

    # 2. Update triggers with path filters
    if "on" in doc:
        on_config = doc["on"]
        paths = [
            "lib/levanter/**",
            "uv.lock",
            f".github/workflows/{workflow_path.name}",
        ]

        # Handle list format: on: [push, pull_request]
        if isinstance(on_config, list):
            new_on = CommentedMap()

            for trigger in on_config:
                if trigger == "push":
                    push_config = CommentedMap()
                    push_config["branches"] = ["main"]
                    push_config["paths"] = paths
                    new_on["push"] = push_config
                elif trigger == "pull_request":
                    pr_config = CommentedMap()
                    pr_config["paths"] = paths
                    new_on["pull_request"] = pr_config
                else:
                    new_on[trigger] = CommentedMap()

            doc["on"] = new_on
            print(f"    ✓ Added path filters to triggers")
            modified = True

    # 3. Update jobs
    if "jobs" in doc:
        for job_name, job_config in doc["jobs"].items():
            if not isinstance(job_config, dict):
                continue

            # Add defaults.run.working-directory (insert after runs-on)
            if job_config.get("defaults", {}).get("run", {}).get("working-directory") != "lib/levanter":
                defaults = CommentedMap()
                run_map = CommentedMap()
                run_map["working-directory"] = "lib/levanter"
                defaults["run"] = run_map

                # Insert defaults after runs-on to maintain order
                if "runs-on" in job_config:
                    # Create new ordered job_config
                    new_config = CommentedMap()
                    for key, value in job_config.items():
                        new_config[key] = value
                        if key == "runs-on":
                            new_config["defaults"] = defaults

                    # Replace job config with ordered version
                    doc["jobs"][job_name] = new_config
                    job_config = new_config
                else:
                    job_config["defaults"] = defaults

                print(f"    ✓ Added defaults.run.working-directory to {job_name}")
                modified = True

            # Update steps
            if "steps" in job_config:
                for step in job_config["steps"]:
                    if not isinstance(step, dict):
                        continue

                    # Add working-directory to astral-sh/setup-uv
                    if "uses" in step and "astral-sh/setup-uv" in step["uses"]:
                        if "with" not in step:
                            step["with"] = CommentedMap()

                        if step["with"].get("working-directory") != "lib/levanter":
                            step["with"]["working-directory"] = "lib/levanter"
                            print(f"    ✓ Added working-directory to setup-uv step")
                            modified = True

                    # Update uv commands
                    if "run" in step and isinstance(step["run"], str):
                        run_cmd = step["run"]
                        original_cmd = run_cmd

                        # Add --package levanter to uv sync
                        if "uv sync" in run_cmd and "--package levanter" not in run_cmd:
                            run_cmd = run_cmd.replace("uv sync", "uv sync --package levanter")

                        # Add --package levanter to uv run (handle multi-line)
                        if "uv run" in run_cmd and "--package levanter" not in run_cmd:
                            lines = run_cmd.split("\n")
                            new_lines = []
                            for line in lines:
                                if "uv run" in line and "--package levanter" not in line:
                                    line = line.replace("uv run", "uv run --package levanter")
                                new_lines.append(line)
                            run_cmd = "\n".join(new_lines)

                        if run_cmd != original_cmd:
                            step["run"] = run_cmd
                            print(f"    ✓ Updated uv commands")
                            modified = True

    if modified:
        with open(workflow_path, "w") as f:
            yaml.dump(doc, f)

    return modified


def main():
    """Update all workflows."""
    cwd = Path.cwd()
    workflows_dir = cwd / ".github/workflows"

    if not workflows_dir.exists():
        print(f"ERROR: Workflows directory not found: {workflows_dir}")
        import sys
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
