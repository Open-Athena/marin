#!/usr/bin/env -S uv run
# /// script
# dependencies = ["tomlkit"]
# ///
"""Update workspace configuration for Haliax integration."""

import tomllib
from pathlib import Path

import tomlkit


def main():
    repo_root = Path(__file__).parent.parent.parent
    root_pyproject = repo_root / "pyproject.toml"
    marin_pyproject = repo_root / "lib/marin/pyproject.toml"
    levanter_pyproject = repo_root / "lib/levanter/pyproject.toml"

    print("Updating workspace configuration for Haliax...")

    # 1. Update root pyproject.toml
    with open(root_pyproject) as f:
        root_doc = tomlkit.load(f)

    # Add haliax to workspace members
    if "tool" not in root_doc:
        root_doc["tool"] = {}
    if "uv" not in root_doc["tool"]:
        root_doc["tool"]["uv"] = {}
    if "workspace" not in root_doc["tool"]["uv"]:
        root_doc["tool"]["uv"]["workspace"] = {}

    members = root_doc["tool"]["uv"]["workspace"].get("members", [])
    if "lib/haliax" not in members:
        members.append("lib/haliax")
        # Sort alphabetically
        members.sort()
        root_doc["tool"]["uv"]["workspace"]["members"] = members
        print("  ✓ Added lib/haliax to workspace members (alphabetized)")

    # Add haliax workspace source
    if "sources" not in root_doc["tool"]["uv"]:
        root_doc["tool"]["uv"]["sources"] = {}

    root_doc["tool"]["uv"]["sources"]["haliax"] = {"workspace": True}

    # Alphabetize workspace sources
    sources = root_doc["tool"]["uv"]["sources"]
    sorted_sources = dict(sorted(sources.items()))
    root_doc["tool"]["uv"]["sources"] = sorted_sources
    print("  ✓ Added haliax workspace source (alphabetized)")

    # Add haliax to root dependencies (experiments may import it)
    if "project" in root_doc and "dependencies" in root_doc["project"]:
        deps = root_doc["project"]["dependencies"]
        if "haliax" not in [d.split("[")[0].split(">=")[0].split("==")[0] for d in deps]:
            deps.append("haliax")
            # Sort alphabetically
            deps.sort()
            print("  ✓ Added haliax to root dependencies (alphabetized)")

    with open(root_pyproject, "w") as f:
        tomlkit.dump(root_doc, f)

    print(f"  ✓ Updated {root_pyproject.relative_to(repo_root)}")

    # 2. Update lib/marin/pyproject.toml - change git dependency to workspace
    with open(marin_pyproject) as f:
        marin_doc = tomlkit.load(f)

    # Remove git-based haliax dependency, workspace source will handle it
    if "dependencies" in marin_doc.get("project", {}):
        deps = marin_doc["project"]["dependencies"]
        # Modify in-place to preserve formatting
        for i, dep in enumerate(deps):
            if isinstance(dep, str) and dep.startswith("haliax"):
                # Extract version if present, remove git URL
                if ">=" in dep:
                    deps[i] = dep.split("@")[0].strip()
                else:
                    deps[i] = "haliax"
        print(f"  ✓ Updated {marin_pyproject.relative_to(repo_root)}")

    with open(marin_pyproject, "w") as f:
        tomlkit.dump(marin_doc, f)

    # 3. Update lib/levanter/pyproject.toml similarly
    with open(levanter_pyproject) as f:
        levanter_doc = tomlkit.load(f)

    if "dependencies" in levanter_doc.get("project", {}):
        deps = levanter_doc["project"]["dependencies"]
        # Modify in-place to preserve formatting
        for i, dep in enumerate(deps):
            if isinstance(dep, str) and dep.startswith("haliax"):
                # Extract version if present, remove git URL
                if ">=" in dep:
                    deps[i] = dep.split("@")[0].strip()
                else:
                    deps[i] = "haliax"
        print(f"  ✓ Updated {levanter_pyproject.relative_to(repo_root)}")

    with open(levanter_pyproject, "w") as f:
        tomlkit.dump(levanter_doc, f)

    print("Workspace configuration updated successfully")


if __name__ == "__main__":
    main()
