#!/usr/bin/env -S uv run
# /// script
# dependencies = ["lossless-yaml==0.1.0"]
# ///
"""
Update .readthedocs.yaml files for step 2 workspace migration.

Updates both root (Marin) and lib/levanter/ (Levanter) ReadTheDocs configs:
- Marin: Adds --frozen flags to uv commands
- Levanter: Converts to workspace-aware build using uv from root
"""

import sys
from pathlib import Path
from yaya import YAYA


def update_marin_rtd(rtd_yaml: Path) -> bool:
    """Update root .readthedocs.yaml (Marin docs) with --frozen flags."""
    print("Updating .readthedocs.yaml (Marin docs)...")
    doc = YAYA.load(rtd_yaml)

    # Get the commands list
    commands = doc.get_path("build.commands")
    if not commands or not isinstance(commands, list):
        print("  ! No build.commands found")
        return False

    modified = False

    # Update each command
    for i, cmd in enumerate(commands):
        if not isinstance(cmd, str):
            continue

        # Add --frozen to uv sync
        if "uv sync" in cmd and "--frozen" not in cmd:
            new_cmd = cmd.replace("uv sync --package marin", "uv sync --package marin --frozen")
            doc.replace_key(f"build.commands[{i}]", new_cmd)
            modified = True
            print(f"  ✓ Added --frozen to uv sync")

        # Add --frozen to uv run
        if "uv run" in cmd and "--frozen" not in cmd:
            new_cmd = cmd.replace("uv run ", "uv run --frozen ")
            doc.replace_key(f"build.commands[{i}]", new_cmd)
            modified = True
            print(f"  ✓ Added --frozen to uv run")

    if modified:
        doc.save()
        print("✓ .readthedocs.yaml updated!")
    else:
        print("  - Already has --frozen flags")

    return modified


def update_levanter_rtd(rtd_yaml: Path) -> bool:
    """Update lib/levanter/.readthedocs.yaml for workspace structure."""
    print("\nUpdating lib/levanter/.readthedocs.yaml (Levanter docs)...")
    doc = YAYA.load(rtd_yaml)

    # Check if already converted to commands-based build
    try:
        commands = doc.get_path("build.commands")
        if commands:
            print("  - Already using commands-based build")
            return False
    except KeyError:
        pass

    # Check for old-style config
    has_mkdocs = False
    has_python = False
    try:
        doc.get_path("mkdocs")
        has_mkdocs = True
    except KeyError:
        pass

    try:
        doc.get_path("python")
        has_python = True
    except KeyError:
        pass

    if not (has_mkdocs or has_python):
        print("  ! Unexpected config structure")
        return False

    # Replace with workspace-aware build
    # We can't delete keys in YAYA, but we can replace the entire build section
    # The old mkdocs/python keys will be replaced by the commands approach

    # Add commands section
    # Note: We install from docs/requirements.txt to pick up any upstream changes
    # to mkdocs plugins/dependencies automatically
    commands = [
        "# Install uv and sync levanter package from workspace root",
        "pip install uv",
        "cd $READTHEDOCS_CHECKOUT && uv sync --package levanter --frozen",
        "# Install mkdocs dependencies from Levanter's requirements",
        "uv pip install -r lib/levanter/docs/requirements.txt",
        "# Build docs from lib/levanter/ subdirectory",
        "cd lib/levanter && uv run --frozen mkdocs build --strict --site-dir $READTHEDOCS_OUTPUT/html",
    ]

    # Add commands after build.tools
    doc.add_key_after(
        existing_path="build.tools",
        new_key="commands",
        value=commands,
    )

    doc.save()
    print("  ✓ Converted to workspace-aware build")
    print("  ✓ Added uv sync --package levanter --frozen")
    print("  ✓ Added mkdocs build from lib/levanter/")
    return True


def main():
    """Update ReadTheDocs configs for workspace."""
    cwd = Path.cwd()

    # Update Marin RTD config
    marin_rtd = cwd / ".readthedocs.yaml"
    if marin_rtd.exists():
        update_marin_rtd(marin_rtd)
    else:
        print(f"! .readthedocs.yaml not found at {marin_rtd}")

    # Update Levanter RTD config
    levanter_rtd = cwd / "lib/levanter/.readthedocs.yaml"
    if levanter_rtd.exists():
        update_levanter_rtd(levanter_rtd)
    else:
        print(f"! lib/levanter/.readthedocs.yaml not found at {levanter_rtd}")


if __name__ == "__main__":
    main()
