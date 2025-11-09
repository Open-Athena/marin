#!/usr/bin/env -S uv run
# /// script
# dependencies = ["lossless-yaml==0.2.0"]
# ///
"""Migrate Haliax workflows to monorepo structure."""

from pathlib import Path
import shutil
from yaya import YAYA


def main():
    repo_root = Path(__file__).parent.parent.parent
    haliax_workflows = repo_root / "lib/haliax/.github/workflows"
    root_workflows = repo_root / ".github/workflows"

    if not haliax_workflows.exists():
        print("No Haliax workflows to migrate")
        return

    # Skip workflows that are redundant in the monorepo
    SKIP_WORKFLOWS = {
        "run_pre_commit.yaml",  # Handled by central marin-lint-and-format.yaml
        "publish_dev.yaml",     # Unnecessary - users can pip install from GitHub
    }

    print("Migrating Haliax workflows...")

    # Rename and move Haliax workflows to root .github/workflows/
    for workflow_file in haliax_workflows.glob("*.y*ml"):
        # Skip workflows that are redundant
        if workflow_file.name in SKIP_WORKFLOWS:
            print(f"  {workflow_file.name} -> SKIPPED (redundant in monorepo)")
            continue

        # Prefix with haliax-
        new_name = f"haliax-{workflow_file.name}"
        dest = root_workflows / new_name

        print(f"  {workflow_file.name} -> {new_name}")

        # Load workflow with yaya for lossless editing
        doc = YAYA.load(workflow_file)

        # 1. Update triggers with path restrictions
        if "on" in doc.data:
            on_val = doc.data["on"]

            # Handle on: [push, pull_request] or similar list format
            if isinstance(on_val, list):
                doc.replace_key("on", {
                    "push": {
                        "branches": ["main"],
                        "paths": [
                            "lib/haliax/**",
                            "uv.lock",
                            ".github/workflows/haliax-*.yaml",
                        ],
                    },
                    "pull_request": {
                        "paths": [
                            "lib/haliax/**",
                            "uv.lock",
                            ".github/workflows/haliax-*.yaml",
                        ],
                    },
                })
            # Handle on: { pull_request: { branches: [main] } }
            elif isinstance(on_val, dict) and "pull_request" in on_val:
                doc.replace_key("on", {
                    "push": {
                        "branches": ["main"],
                        "paths": [
                            "lib/haliax/**",
                            "uv.lock",
                            ".github/workflows/haliax-*.yaml",
                        ],
                    },
                    "pull_request": {
                        "paths": [
                            "lib/haliax/**",
                            "uv.lock",
                            ".github/workflows/haliax-*.yaml",
                        ],
                    },
                })

        # 2. Add working-directory to each job
        if "jobs" in doc.data:
            for job_name in doc.data["jobs"].keys():
                # Check if defaults already exists
                try:
                    existing_wd = doc.get_path(f"jobs.{job_name}.defaults.run.working-directory")
                    if existing_wd == "lib/haliax":
                        # Already has correct working-directory
                        pass
                    else:
                        print(f"    ! {job_name} has unexpected working-directory: {existing_wd}")
                except KeyError:
                    # No defaults - add it
                    # Try to insert between various job keys and 'steps'
                    inserted = False
                    for prev_key in ["if", "strategy", "env", "permissions", "needs", "runs-on"]:
                        try:
                            doc.insert_key_between(
                                f"jobs.{job_name}",
                                prev_key=prev_key,
                                next_key="steps",
                                new_key="defaults",
                                value={"run": {"working-directory": "lib/haliax"}},
                            )
                            inserted = True
                            break
                        except (KeyError, ValueError):
                            continue

                    if not inserted:
                        # Fallback: just set it in place (will appear at end)
                        doc.data["jobs"][job_name]["defaults"] = {
                            "run": {"working-directory": "lib/haliax"}
                        }

        # Save with yaya to preserve formatting and structure
        doc.save(dest)

        # 3. Now do string replacements for command-level changes
        # Unfortunately yaya doesn't have an API for replacing deeply nested values,
        # so we need to do text replacements. But we must be careful to only replace
        # within the run: blocks to avoid breaking the structure we just added.
        content = dest.read_text()

        # Update uv commands to use --package haliax
        content = content.replace("uv sync\n", "uv sync --package haliax --dev\n")
        content = content.replace("uv run pytest", "uv run --package haliax pytest")
        content = content.replace("uv run python", "uv run --package haliax python")

        # Add -c pyproject.toml to pytest commands
        content = content.replace("pytest tests", "pytest -c pyproject.toml tests")
        content = content.replace(
            'pytest tests -m "not entry and not slow"',
            'pytest -c pyproject.toml tests -m "not entry and not slow"',
        )

        dest.write_text(content)

    # Remove Haliax's .github directory
    shutil.rmtree(repo_root / "lib/haliax/.github")
    print("  ✓ Removed lib/haliax/.github/")

    print("Workflow migration complete")


if __name__ == "__main__":
    main()
