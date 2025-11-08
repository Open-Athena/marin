#!/usr/bin/env -S uv run
# /// script
# ///
"""Migrate Haliax workflows to monorepo structure."""

from pathlib import Path
import shutil


def main():
    repo_root = Path(__file__).parent.parent.parent
    haliax_workflows = repo_root / "lib/haliax/.github/workflows"
    root_workflows = repo_root / ".github/workflows"

    if not haliax_workflows.exists():
        print("No Haliax workflows to migrate")
        return

    print("Migrating Haliax workflows...")

    # Rename and move Haliax workflows to root .github/workflows/
    for workflow_file in haliax_workflows.glob("*.y*ml"):
        # Prefix with haliax-
        new_name = f"haliax-{workflow_file.name}"
        dest = root_workflows / new_name

        print(f"  {workflow_file.name} -> {new_name}")

        # Copy and update paths
        content = workflow_file.read_text()

        # Add working-directory to jobs that run in Haliax context
        # Find the jobs section and add defaults with working-directory
        if "jobs:" in content:
            lines = content.split("\n")
            new_lines = []
            in_job = False
            job_indent = None
            added_defaults = set()

            for i, line in enumerate(lines):
                new_lines.append(line)

                # Detect job start (e.g., "  build:" or "  test:")
                if line.strip() and line.strip().endswith(":") and not line.strip().startswith("#"):
                    if line.startswith("  ") and not line.startswith("    "):
                        # This is a job name
                        job_name = line.strip().rstrip(":")
                        if job_name != "jobs":
                            # Add working-directory after job name if not already added
                            if job_name not in added_defaults:
                                new_lines.append("    defaults:")
                                new_lines.append("      run:")
                                new_lines.append("        working-directory: lib/haliax")
                                new_lines.append("")
                                added_defaults.add(job_name)

            content = "\n".join(new_lines)

        # Update uv commands to use --package haliax
        content = content.replace("uv sync", "uv sync --package haliax --dev")
        content = content.replace("uv run pytest", "uv run --package haliax pytest")
        content = content.replace("uv run python", "uv run --package haliax python")

        # Add path restrictions to only run on Haliax changes
        if "on:" in content and "pull_request:" in content:
            # Add paths filter if not present
            if "paths:" not in content.split("pull_request:")[1].split("push:")[0]:
                content = content.replace(
                    "pull_request:",
                    """pull_request:
    paths:
      - lib/haliax/**
      - .github/workflows/haliax-*.yaml"""
                )

        dest.write_text(content)

    # Remove Haliax's .github directory
    shutil.rmtree(repo_root / "lib/haliax/.github")
    print("  ✓ Removed lib/haliax/.github/")

    print("Workflow migration complete")


if __name__ == "__main__":
    main()
