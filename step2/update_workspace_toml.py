#!/usr/bin/env -S uv run
# /// script
# dependencies = ["tomlkit", "click"]
# ///
"""
Update workspace TOML files for step 2.

Updates:
1. Root pyproject.toml: Add levanter to workspace members and sources
2. lib/marin/pyproject.toml:
   - Change levanter dependency from git URL to workspace
   - Optionally update dolma dependency SHA
"""

import sys
from pathlib import Path

import click
import tomlkit


def update_root_pyproject(path: Path = Path("pyproject.toml")) -> None:
    """Update root pyproject.toml to include levanter in workspace."""
    print(f"Updating {path}...")

    with open(path, encoding="utf-8") as f:
        doc = tomlkit.parse(f.read())

    # Add levanter to project dependencies if not present (experiments imports levanter directly)
    dependencies = doc.get("project", {}).get("dependencies", [])
    if "levanter" not in dependencies:
        dependencies.append("levanter")
        print("  ✓ Added levanter to root dependencies (experiments imports levanter)")
    else:
        print("  - levanter already in root dependencies")

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


def update_marin_pyproject(path: Path = Path("lib/marin/pyproject.toml"), dolma_sha: str | None = None) -> None:
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

    # Handle dolma dependency
    quality_dedup_deps = None
    if "project" in doc:
        project = doc["project"]
        if "optional-dependencies" in project:
            opt_deps = project["optional-dependencies"]
            if "quality_dedup_consolidate" in opt_deps:
                quality_dedup_deps = opt_deps["quality_dedup_consolidate"]

    if quality_dedup_deps is not None:
        dolma_updated = False
        for i, dep in enumerate(quality_dedup_deps):
            if isinstance(dep, str) and "dolma @" in dep:
                if dolma_sha:
                    # Update dolma SHA
                    new_dep = f"dolma @ git+https://github.com/marin-community/dolma@{dolma_sha}"
                    quality_dedup_deps[i] = new_dep
                    print(f"  ✓ Updated dolma SHA to {dolma_sha}")
                    dolma_updated = True
                else:
                    print(f"  - dolma dependency preserved as-is: {dep}")
                    dolma_updated = True
                break

        if not dolma_updated:
            print("  ⚠ dolma dependency not found in quality-dedup-consolidate")
    else:
        print("  - quality-dedup-consolidate section not found in pyproject.toml")

    # Write back
    content = tomlkit.dumps(doc)
    # Ensure single trailing newline
    content = content.rstrip() + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✓ Updated {path}")


@click.command()
@click.option(
    "--dolma-sha",
    "-D",
    help="dolma SHA to use (optional - if not provided, preserves existing)",
)
def main(dolma_sha: str | None):
    """Update workspace TOML files."""
    # Use current directory (assumes script is run from repo root)
    # Script is designed to be called from step2/main.sh which is already in repo root
    cwd = Path.cwd()

    print("Updating workspace TOML configuration...")
    print(f"Working directory: {cwd}")
    print()

    update_root_pyproject(cwd / "pyproject.toml")
    print()
    update_marin_pyproject(cwd / "lib/marin/pyproject.toml", dolma_sha=dolma_sha)
    print()
    print("✓ All TOML files updated!")


if __name__ == "__main__":
    main()
