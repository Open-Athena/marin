#!/usr/bin/env python3
"""
Update CI/docs configs for workspace structure.

Updates:
- GHA workflows: Use workspace extra syntax (e.g., --extra=marin:cpu)
- ReadTheDocs: Install marin package from lib/marin
- mkdocs.yml: Update src/ paths to lib/marin/src/
- CodeQL workflow: Update src paths to lib/marin/src
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
    """Update ReadTheDocs config to use uv workspace structure."""
    print("Updating ReadTheDocs config...")

    rtd_config = Path('.readthedocs.yaml')
    if rtd_config.exists():
        content = rtd_config.read_text()

        # Replace jobs.pre_install structure with commands structure using uv
        # Old structure uses pip install -e lib/marin
        # New structure uses uv sync --package marin with build.commands
        if 'jobs:' in content and 'pre_install:' in content:
            # Replace entire build section with uv-based commands
            updated = re.sub(
                r'build:\n  os: "ubuntu-24\.04"\n  tools:\n    python: "[^"]+"\n  jobs:\n    pre_install:.*?(?=\n\n|\Z)',
                '''build:
  os: "ubuntu-24.04"
  tools:
    python: "3.11"
  commands:
    - pip install uv
    - uv sync --package marin
    - uv pip install mkdocs mkdocs-material mkdocstrings[python] markdown-include mkdocs-include-markdown-plugin
    - uv run mkdocs build --strict --site-dir $READTHEDOCS_OUTPUT/html''',
                content,
                flags=re.DOTALL
            )

            # Remove mkdocs section if present (redundant with commands)
            updated = re.sub(r'\n\nmkdocs:\n  configuration: mkdocs\.yml\n?', '', updated)

            # Ensure trailing newline (pre-commit requirement)
            if not updated.endswith('\n'):
                updated += '\n'

            if updated != content:
                rtd_config.write_text(updated)
                print(f"  ✓ Updated {rtd_config} to use uv workspace structure")
            else:
                print(f"  ⚠ Could not update {rtd_config} automatically")
        else:
            print(f"  - {rtd_config} already uses correct structure or needs manual update")
    else:
        print(f"  ⚠ {rtd_config} not found")


def update_mkdocs():
    """Update mkdocs.yml to use workspace paths."""
    print("Updating mkdocs.yml...")

    mkdocs_yml = Path('mkdocs.yml')
    if mkdocs_yml.exists():
        content = mkdocs_yml.read_text()

        # Update watch paths: src/ -> lib/marin/src/
        updated = re.sub(
            r'^(\s+)- src/$',
            r'\1- lib/marin/src/',
            content,
            flags=re.MULTILINE
        )

        # Update paths in mkdocstrings handler: [".", "src"] -> [".", "lib/marin/src"]
        updated = re.sub(
            r'paths: \["\."\, "src"\]',
            'paths: [".", "lib/marin/src"]',
            updated
        )

        if updated != content:
            mkdocs_yml.write_text(updated)
            print(f"  ✓ Updated {mkdocs_yml}")
        else:
            print(f"  - No changes needed in {mkdocs_yml}")
    else:
        print(f"  ⚠ {mkdocs_yml} not found")


def update_codeql():
    """Update CodeQL workflow to use workspace paths."""
    print("Updating CodeQL workflow...")

    codeql_yml = Path('.github/workflows/codeql.yml')
    if codeql_yml.exists():
        content = codeql_yml.read_text()

        # Update paths config: - src -> - lib/marin/src
        updated = re.sub(
            r'^(\s+)- src$',
            r'\1- lib/marin/src',
            content,
            flags=re.MULTILINE
        )

        if updated != content:
            codeql_yml.write_text(updated)
            print(f"  ✓ Updated {codeql_yml}")
        else:
            print(f"  - No changes needed in {codeql_yml}")
    else:
        print(f"  - {codeql_yml} not found (may not exist yet)")


def main():
    """Update CI and docs configurations."""
    print("\n" + "="*60)
    print("Updating CI and docs configs for workspace structure")
    print("="*60 + "\n")

    update_workflows()
    update_readthedocs()
    update_mkdocs()
    update_codeql()

    print("\n✓ CI/docs configs updated!")


if __name__ == '__main__':
    main()
