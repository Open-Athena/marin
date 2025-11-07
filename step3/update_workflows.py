#!/usr/bin/env python3
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

        # Update paths to be relative to repo root
        # This is a simple replacement - might need more sophisticated handling
        content = content.replace("src/", "lib/haliax/src/")
        content = content.replace("tests/", "lib/haliax/tests/")

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
