#!/usr/bin/env python3
"""Update ray_deps.py for workspace structure.

Changes PYTHONPATH from ["src", "experiments"] to workspace member paths.
"""

import sys
from pathlib import Path


def update_ray_deps():
    """Update ray_deps.py to use workspace member src directories."""
    ray_deps = Path("lib/marin/src/marin/run/ray_deps.py")

    if not ray_deps.exists():
        print(f"Error: {ray_deps} not found")
        sys.exit(1)

    content = ray_deps.read_text()

    # Find and replace the old PYTHONPATH definition
    old_paths = '    paths = ["src", "experiments"]'
    new_paths = '''    # Workspace member src directories + experiments directory
    paths = [
        "lib/marin/src",
        "lib/levanter/src",
        "experiments",
    ]'''

    if old_paths not in content:
        print(f"Warning: Expected pattern not found in {ray_deps}")
        print("File may have already been updated or has unexpected format")
        sys.exit(1)

    updated_content = content.replace(old_paths, new_paths)

    if updated_content == content:
        print(f"No changes made to {ray_deps}")
        sys.exit(1)

    ray_deps.write_text(updated_content)
    print(f"✓ Updated {ray_deps} for workspace structure")


if __name__ == "__main__":
    update_ray_deps()
