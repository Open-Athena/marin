#!/usr/bin/env -S uv run
# /// script
# dependencies = []
# ///
"""
Update file paths from src/marin to lib/marin/src/marin.

Updates all files using simple text replacement to preserve exact formatting.
"""

import re
from pathlib import Path


def update_file(file_path: Path, old_path: str, new_path: str):
    """Update paths in file using text replacement to preserve all formatting."""
    print(f"Updating {file_path}...")

    content = file_path.read_text()
    # Use negative lookbehind to avoid replacing src/marin that's already in lib/marin/src/marin
    # Match src/marin but NOT if preceded by lib/marin/
    pattern = r'(?<!lib/marin/)' + re.escape(old_path)
    updated_content = re.sub(pattern, new_path, content)

    if content != updated_content:
        file_path.write_text(updated_content)
        print(f"  ✓ Updated {file_path}")
    else:
        print(f"  - No changes needed in {file_path}")


def update_ray_deps():
    """Add --package marin to uv export command in ray_deps.py."""
    ray_deps_path = Path('lib/marin/src/marin/run/ray_deps.py')
    if not ray_deps_path.exists():
        print(f"  - {ray_deps_path} not found, skipping")
        return

    print(f"Updating {ray_deps_path}...")
    content = ray_deps_path.read_text()

    # Add --package marin to uv export command
    if '"uv",\n        "export",' in content and '--package' not in content:
        updated_content = content.replace(
            '"uv",\n        "export",',
            '"uv",\n        "export",\n        "--package",\n        "marin",'
        )
        ray_deps_path.write_text(updated_content)
        print(f"  ✓ Added --package marin to uv export command")
    else:
        print(f"  - Already updated or pattern not found")


def main():
    """Update all file paths."""
    old_path = 'src/marin'
    new_path = 'lib/marin/src/marin'

    print(f"Updating paths: {old_path} → {new_path}\n")

    # Update YAML files
    yaml_files = [
        Path('.github/workflows/build-docker-images.yaml'),
        Path('.github/workflows/update-leaderboard.yml'),
        Path('mkdocs.yml'),
    ]

    for yaml_file in yaml_files:
        if yaml_file.exists():
            update_file(yaml_file, old_path, new_path)

    # Update Makefile
    makefile = Path('Makefile')
    if makefile.exists():
        update_file(makefile, old_path, new_path)

    # Update markdown files in docs/
    docs_dir = Path('docs')
    if docs_dir.exists():
        for md_file in docs_dir.rglob('*.md'):
            update_file(md_file, old_path, new_path)

    # Update Python imports from src.marin to marin
    print("\nUpdating Python imports from 'from src.marin' to 'from marin'...")
    for directory in ['experiments', 'scripts']:
        dir_path = Path(directory)
        if dir_path.exists():
            for py_file in dir_path.rglob('*.py'):
                content = py_file.read_text()
                updated_content = content.replace('from src.marin', 'from marin')
                if content != updated_content:
                    py_file.write_text(updated_content)
                    print(f"  ✓ Updated imports in {py_file}")

    # Update ray_deps.py to add --package marin
    print()
    update_ray_deps()

    print("\n✓ All paths updated!")


if __name__ == '__main__':
    main()
