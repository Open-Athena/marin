#!/usr/bin/env python3
"""
Update CI/docs configs for workspace structure.

Updates:
- GHA workflows: Use workspace extra syntax (e.g., --extra=marin:cpu)
- ReadTheDocs: Install marin package from lib/marin
"""

import re
from pathlib import Path


def update_workflows():
    """Update GitHub Actions workflows to use workspace member installation."""
    print("Updating GitHub Actions workflows...")

    # Update unit-tests.yaml
    unit_tests = Path('.github/workflows/unit-tests.yaml')
    if unit_tests.exists():
        content = unit_tests.read_text()

        # Replace `uv sync --dev --extra=...` with two-step workspace sync
        # Step 1: uv sync (syncs workspace root)
        # Step 2: uv sync --package marin --extra cpu --group test (syncs member with extras/groups)
        # Match both --extra=cpu and --extra=marin:cpu
        updated = re.sub(
            r'uv sync --dev --extra=(cpu|marin:cpu)\b',
            'uv sync\n          uv sync --package marin --extra cpu --group test',
            content
        )

        # Also handle the case where we previously split into two lines
        updated = re.sub(
            r'uv sync --dev\n\s+uv pip install -e lib/marin\[cpu\]',
            'uv sync\n          uv sync --package marin --extra cpu --group test',
            updated
        )

        # Also handle current workspace pattern if it exists
        updated = re.sub(
            r'uv sync --package marin --group dev --extra cpu',
            'uv sync\n          uv sync --package marin --extra cpu --group test',
            updated
        )

        # Remove --extra=(cpu|marin:cpu) from uv run commands (not needed)
        updated = re.sub(
            r'uv run --extra=(cpu|marin:cpu) ',
            'uv run ',
            updated
        )

        if updated != content:
            unit_tests.write_text(updated)
            print(f"  ✓ Updated {unit_tests}")
        else:
            print(f"  - No changes needed in {unit_tests}")
    else:
        print(f"  ⚠ {unit_tests} not found")

    # Update docs.yaml
    docs_yaml = Path('.github/workflows/docs.yaml')
    if docs_yaml.exists():
        content = docs_yaml.read_text()

        # Replace `uv sync --dev` with workspace group syntax
        # `uv sync --package marin --group dev` correctly installs marin with dev group
        # Handle both single-line (run: uv sync) and multi-line (run: |\n  uv sync) formats
        updated = re.sub(
            r'run: uv sync --dev\b',
            'run: uv sync --package marin --group dev',
            content
        )
        # Also handle multiline format with pip install
        updated = re.sub(
            r'run: \|\n\s+uv sync --dev\n\s+uv pip install -e lib/marin',
            'run: uv sync --package marin --group dev',
            updated
        )

        if updated != content:
            docs_yaml.write_text(updated)
            print(f"  ✓ Updated {docs_yaml}")
        else:
            print(f"  - No changes needed in {docs_yaml}")
    else:
        print(f"  ⚠ {docs_yaml} not found")


def update_readthedocs():
    """Update ReadTheDocs config to install marin package."""
    print("Updating ReadTheDocs config...")

    rtd_config = Path('.readthedocs.yaml')
    if rtd_config.exists():
        content = rtd_config.read_text()

        # Add marin package installation if not present
        if 'pip install -e lib/marin' not in content:
            # Find the pre_install section and add the line
            updated = re.sub(
                r'(pre_install:.*?)(\n\nmkdocs:)',
                r'\1\n      - pip install -e lib/marin\2',
                content,
                flags=re.DOTALL
            )

            if updated != content:
                rtd_config.write_text(updated)
                print(f"  ✓ Updated {rtd_config}")
            else:
                print(f"  ⚠ Could not update {rtd_config} automatically")
        else:
            print(f"  - No changes needed in {rtd_config}")
    else:
        print(f"  ⚠ {rtd_config} not found")


def main():
    """Update CI and docs configurations."""
    print("\n" + "="*60)
    print("Updating CI and docs configs for workspace structure")
    print("="*60 + "\n")

    update_workflows()
    update_readthedocs()

    print("\n✓ CI/docs configs updated!")


if __name__ == '__main__':
    main()
