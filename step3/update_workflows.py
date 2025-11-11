#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "ruamel.yaml",
# ]
# ///
"""Migrate Haliax workflows to monorepo structure.

Uses local yaya directly from source (not via file:// dependency).
This ensures we get the absolute latest code including blank line fixes.
"""

from pathlib import Path
import shutil
import os
import sys

# Add yaya source directory FIRST so we get latest code
sys.path.insert(0, str(Path.home() / "c/yaya/src"))

# Set yaya list indentation to match GitHub Actions convention (2-space offset)
os.environ['YAYA_LIST_OFFSET'] = '2'

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
        "run_quick_levanter_tests.yaml",  # Designed for standalone repo, doesn't work in monorepo
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

        # Set list indentation style to match GitHub Actions convention (2-space offset)
        doc.set_list_indent_style(offset=2)

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

        # 2. For run_tests.yaml: We'll add strategy/defaults via string replacement
        # after saving, because yaya has issues with creating nested structures

        # 3. Add working-directory to each job (skip for run_tests.yaml since handled above)
        if "jobs" in doc.data and workflow_file.name != "run_tests.yaml":
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

        # 4. Now do string replacements for command-level changes
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

        # For run_tests.yaml: add strategy/defaults/matrix via string replacement
        if workflow_file.name == "run_tests.yaml":
            # Insert strategy and defaults after runs-on, before steps
            strategy_defaults = """    strategy:
      matrix:
        python-version: ["3.11"]
        jax-version: ["0.6.2", "0.7.2"]

    defaults:
      run:
        working-directory: lib/haliax

"""
            content = content.replace(
                "    runs-on: ubuntu-latest\n\n    steps:",
                f"    runs-on: ubuntu-latest\n\n{strategy_defaults}    steps:"
            )

            # Use matrix python-version variable
            content = content.replace(
                "python-version: 3.11",
                "python-version: ${{ matrix.python-version }}"
            )
            content = content.replace(
                "Set up Python 3.11",
                "Set up Python ${{ matrix.python-version }}"
            )

            # Add --with flag for JAX version and update step name
            content = content.replace(
                "- name: Test with pytest\n        run: |\n          XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=tests:src:. uv run --package haliax pytest -c pyproject.toml tests",
                "- name: Test with pytest (JAX ${{ matrix.jax-version }})\n        run: |\n          # Test with specific JAX version\n          XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=tests:src:. uv run --package haliax --with \"jax[cpu]==${{ matrix.jax-version }}\" pytest -c pyproject.toml tests"
            )

        dest.write_text(content)

    # Remove Haliax's .github directory
    shutil.rmtree(repo_root / "lib/haliax/.github")
    print("  ✓ Removed lib/haliax/.github/")

    print("Workflow migration complete")


if __name__ == "__main__":
    main()
