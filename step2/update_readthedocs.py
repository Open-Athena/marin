#!/usr/bin/env -S uv run
# /// script
# dependencies = ["yaya @ file:///Users/ryan/c/yaya"]
# ///
"""
Update .readthedocs.yaml for step 2 workspace migration.

Adds --frozen flags to uv sync and uv run commands to prevent git
dependency updates during ReadTheDocs builds.
"""

import sys
from pathlib import Path
from yaya import YAYA


def main():
    """Update .readthedocs.yaml with --frozen flags."""
    cwd = Path.cwd()
    rtd_yaml = cwd / ".readthedocs.yaml"

    if not rtd_yaml.exists():
        print(f"! .readthedocs.yaml not found at {rtd_yaml}")
        return

    print("Updating .readthedocs.yaml...")
    doc = YAYA.load(rtd_yaml)

    # Get the commands list
    commands = doc.get_path("build.commands")
    if not commands or not isinstance(commands, list):
        print("  ! No build.commands found")
        return

    modified = False

    # Update each command
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, str):
            continue

        # Add --frozen to uv sync
        if "uv sync" in cmd and "--frozen" not in cmd:
            new_cmd = cmd.replace("uv sync --package marin", "uv sync --package marin --frozen")
            doc.set_path(f"build.commands[{i}]", new_cmd)
            modified = True
            print(f"  ✓ Added --frozen to uv sync")

        # Add --frozen to uv run
        if "uv run" in cmd and "--frozen" not in cmd:
            new_cmd = cmd.replace("uv run ", "uv run --frozen ")
            doc.set_path(f"build.commands[{i}]", new_cmd)
            modified = True
            print(f"  ✓ Added --frozen to uv run")

    if modified:
        doc.save()
        print("✓ .readthedocs.yaml updated!")
    else:
        print("  - Already has --frozen flags")


if __name__ == "__main__":
    main()
