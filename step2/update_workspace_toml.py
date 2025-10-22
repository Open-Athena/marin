#!/usr/bin/env -S uv run
# /// script
# dependencies = ["tomlkit"]
# ///
"""
Update workspace TOML files for step 2.

Updates:
1. Root pyproject.toml: Add levanter to workspace members and sources
2. lib/marin/pyproject.toml: Change levanter dependency from git URL to workspace
"""

from pathlib import Path

import tomlkit


def update_root_pyproject(path: Path = Path("pyproject.toml")) -> None:
    """Update root pyproject.toml to include levanter in workspace."""
    print(f"Updating {path}...")

    with open(path, encoding="utf-8") as f:
        doc = tomlkit.parse(f.read())

    # Add levanter to workspace members if not present
    workspace = doc.get("tool", {}).get("uv", {}).get("workspace", {})
    members = workspace.get("members", [])

    if "lib/levanter" not in members:
        members.append("lib/levanter")
        print("  ✓ Added lib/levanter to workspace members")
    else:
        print("  - lib/levanter already in workspace members")

    # Add levanter to workspace sources if not present
    sources = doc.get("tool", {}).get("uv", {}).get("sources", {})

    if "levanter" not in sources:
        # Need to add as a new table section
        if "tool" not in doc:
            doc["tool"] = tomlkit.table()
        if "uv" not in doc["tool"]:
            doc["tool"]["uv"] = tomlkit.table()
        if "sources" not in doc["tool"]["uv"]:
            doc["tool"]["uv"]["sources"] = tomlkit.table()

        # Add levanter source as a new table (not inline)
        levanter_source = tomlkit.table()
        levanter_source["workspace"] = True

        # We need to insert this as a separate [tool.uv.sources.levanter] section
        # tomlkit doesn't make this easy, so we'll use a workaround
        doc_str = tomlkit.dumps(doc)

        # Find the [tool.uv.sources.marin] section and add levanter after it
        import re
        pattern = r'(\[tool\.uv\.sources\.marin\]\nworkspace = true)'
        replacement = r'\1\n\n[tool.uv.sources.levanter]\nworkspace = true'
        doc_str = re.sub(pattern, replacement, doc_str)

        # Re-parse to validate
        doc = tomlkit.parse(doc_str)
        print("  ✓ Added levanter to workspace sources")
    else:
        print("  - levanter already in workspace sources")

    # Write back
    content = tomlkit.dumps(doc)
    # Ensure single trailing newline
    content = content.rstrip() + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✓ Updated {path}")


def update_marin_pyproject(path: Path = Path("lib/marin/pyproject.toml")) -> None:
    """Update lib/marin/pyproject.toml to use workspace levanter dependency."""
    print(f"Updating {path}...")

    with open(path, encoding="utf-8") as f:
        doc = tomlkit.parse(f.read())

    # Find and update levanter dependency
    dependencies = doc.get("project", {}).get("dependencies", [])

    updated = False
    for i, dep in enumerate(dependencies):
        if isinstance(dep, str) and dep.startswith("levanter"):
            # Check if it's a git URL
            if "@" in dep and "git+" in dep:
                # Replace with workspace dependency
                dependencies[i] = "levanter[serve]"
                updated = True
                print(f"  ✓ Changed levanter dependency from git URL to workspace")
                break

    if not updated:
        print("  - levanter dependency already uses workspace or not found")

    # Write back
    content = tomlkit.dumps(doc)
    # Ensure single trailing newline
    content = content.rstrip() + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✓ Updated {path}")


def main():
    """Update workspace TOML files."""
    import sys
    import os

    # Use current directory (assumes script is run from repo root)
    # Script is designed to be called from step2/main.sh which is already in repo root
    cwd = Path.cwd()

    print("Updating workspace TOML configuration...")
    print(f"Working directory: {cwd}")
    print()

    update_root_pyproject(cwd / "pyproject.toml")
    print()
    update_marin_pyproject(cwd / "lib/marin/pyproject.toml")
    print()
    print("✓ All TOML files updated!")


if __name__ == "__main__":
    main()
