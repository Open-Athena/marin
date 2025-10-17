#!/usr/bin/env -S uv run
# /// script
# dependencies = ["tomlkit"]
# ///
"""
Transform pyproject.toml files for workspace step-1.

Creates lib/marin/pyproject.toml from root and replaces root with workspace config.
Root becomes a workspace that also packages experiments/ and depends on marin.
Tool configs (black, ruff, mypy, pytest) stay at root and are removed from lib/marin.
data_browser/ stays independent (not a workspace member).
"""

import sys
import tomlkit
from pathlib import Path


def validate_pyproject_structure(doc):
    """
    Validate that the pyproject.toml structure matches expectations.

    Exits with error if unexpected sections are found, requiring manual review.
    """
    # Expected top-level sections
    EXPECTED_TOP_LEVEL = {'build-system', 'project', 'dependency-groups', 'tool'}

    # Expected project subsections
    EXPECTED_PROJECT = {
        'name', 'version', 'requires-python', 'dependencies',
        'optional-dependencies', 'license', 'readme', 'description'
    }

    # Expected tool subsections
    EXPECTED_TOOL = {'black', 'ruff', 'mypy', 'pytest', 'hatch', 'uv'}

    # Tool configs that should be REMOVED from lib/marin (stay at root)
    TOOL_CONFIGS_TO_REMOVE = {'black', 'ruff', 'mypy', 'pytest'}

    # Tool configs that should be KEPT in lib/marin
    TOOL_CONFIGS_TO_KEEP = {'hatch', 'uv'}

    # Check top-level sections
    actual_top_level = set(doc.keys())
    unexpected_top = actual_top_level - EXPECTED_TOP_LEVEL
    if unexpected_top:
        print(f"❌ ERROR: Unexpected top-level sections in pyproject.toml: {unexpected_top}", file=sys.stderr)
        print(f"   Expected: {EXPECTED_TOP_LEVEL}", file=sys.stderr)
        print(f"   Found: {actual_top_level}", file=sys.stderr)
        print("   Manual review required before proceeding.", file=sys.stderr)
        sys.exit(1)

    # Check project subsections
    if 'project' in doc:
        actual_project = set(doc['project'].keys())
        unexpected_proj = actual_project - EXPECTED_PROJECT
        if unexpected_proj:
            print(f"❌ ERROR: Unexpected [project] subsections: {unexpected_proj}", file=sys.stderr)
            print(f"   Expected: {EXPECTED_PROJECT}", file=sys.stderr)
            print(f"   Found: {actual_project}", file=sys.stderr)
            print("   Manual review required before proceeding.", file=sys.stderr)
            sys.exit(1)

    # Check tool subsections
    if 'tool' in doc:
        actual_tool = set(doc['tool'].keys())
        unexpected_tool = actual_tool - EXPECTED_TOOL
        if unexpected_tool:
            print(f"❌ ERROR: Unexpected [tool] subsections: {unexpected_tool}", file=sys.stderr)
            print(f"   Expected: {EXPECTED_TOOL}", file=sys.stderr)
            print(f"   Found: {actual_tool}", file=sys.stderr)
            print("   Manual review required before proceeding.", file=sys.stderr)
            sys.exit(1)

    print("✓ pyproject.toml structure validation passed")
    return {
        'tool_configs_to_remove': TOOL_CONFIGS_TO_REMOVE,
        'tool_configs_to_keep': TOOL_CONFIGS_TO_KEEP,
    }


def transform_pyprojects():
    """Transform root pyproject.toml into workspace structure."""
    root_path = Path('pyproject.toml')
    lib_marin_path = Path('lib/marin/pyproject.toml')

    print("Reading root pyproject.toml...")
    with open(root_path, 'r') as f:
        root_doc = tomlkit.parse(f.read())

    # Validate structure before proceeding
    print("Validating pyproject.toml structure...")
    config = validate_pyproject_structure(root_doc)
    tool_configs_to_remove = config['tool_configs_to_remove']

    # ============================================================
    # 1. Create lib/marin/pyproject.toml (copy with modifications)
    # ============================================================
    print("Creating lib/marin/pyproject.toml...")
    lib_marin_doc = tomlkit.parse(tomlkit.dumps(root_doc))  # Deep copy

    # Remove readme (not needed for member)
    if 'project' in lib_marin_doc and 'readme' in lib_marin_doc['project']:
        del lib_marin_doc['project']['readme']

    # Update license path to point to root
    if 'project' in lib_marin_doc and 'license' in lib_marin_doc['project']:
        lib_marin_doc['project']['license'] = {'file': '../../LICENSE'}

    # Update packages to only include src/marin (remove experiments)
    if 'tool' in lib_marin_doc and 'hatch' in lib_marin_doc['tool']:
        hatch = lib_marin_doc['tool']['hatch']
        if 'build' in hatch and 'targets' in hatch['build']:
            targets = hatch['build']['targets']
            if 'wheel' in targets and 'packages' in targets['wheel']:
                targets['wheel']['packages'] = ['src/marin']
            if 'sdist' in targets and 'packages' in targets['sdist']:
                targets['sdist']['packages'] = ['src/marin']

    # Remove tool configs (they stay at root, shared by all members)
    if 'tool' in lib_marin_doc:
        for config in tool_configs_to_remove:
            if config in lib_marin_doc['tool']:
                del lib_marin_doc['tool'][config]

    # Ensure lib/marin directory exists
    lib_marin_path.parent.mkdir(parents=True, exist_ok=True)

    # Write lib/marin/pyproject.toml
    with open(lib_marin_path, 'w') as f:
        content = tomlkit.dumps(lib_marin_doc)
        # Ensure single trailing newline (tomlkit may add extra blank lines)
        content = content.rstrip('\n') + '\n'
        f.write(content)
    print(f"  ✓ Created {lib_marin_path}")

    # ============================================================
    # 2. Replace root pyproject.toml with workspace config
    # ============================================================
    print("Creating workspace root pyproject.toml...")
    workspace_doc = tomlkit.document()

    # Build system (keep from original)
    workspace_doc['build-system'] = root_doc.get('build-system', {})

    # Project section - root is a package that contains experiments/
    workspace_doc['project'] = tomlkit.table()
    workspace_doc['project']['name'] = 'marin-root'
    workspace_doc['project']['version'] = '0.1.0'
    workspace_doc['project']['description'] = 'Marin workspace root and experiments'
    workspace_doc['project']['license'] = root_doc.get('project', {}).get('license', {'file': 'LICENSE'})
    workspace_doc['project']['requires-python'] = root_doc.get('project', {}).get('requires-python', '>=3.11')
    # Root depends on marin (experiments code imports from marin)
    workspace_doc['project']['dependencies'] = ['marin']

    # Note: dependency-groups stay in lib/marin, not duplicated at root
    # Workflows should use `uv sync --group dev` which resolves to member groups

    # Copy/replace tool configs in original order to preserve ordering
    workspace_doc['tool'] = tomlkit.table()
    if 'tool' in root_doc:
        for config in root_doc['tool'].keys():
            if config == 'uv':
                # Replace with workspace-specific uv config
                workspace_doc['tool']['uv'] = tomlkit.table()
                workspace_doc['tool']['uv']['workspace'] = tomlkit.table()
                workspace_doc['tool']['uv']['workspace']['members'] = ['lib/marin']
                workspace_doc['tool']['uv']['sources'] = tomlkit.table()
                workspace_doc['tool']['uv']['sources']['marin'] = {'workspace': True}
            elif config == 'hatch':
                # Replace with workspace-specific hatch config
                workspace_doc['tool']['hatch'] = tomlkit.table()
                workspace_doc['tool']['hatch']['metadata'] = {'allow-direct-references': True}
                workspace_doc['tool']['hatch']['build'] = tomlkit.table()
                workspace_doc['tool']['hatch']['build']['targets'] = tomlkit.table()
                workspace_doc['tool']['hatch']['build']['targets']['wheel'] = {'packages': ['experiments']}
                workspace_doc['tool']['hatch']['build']['targets']['sdist'] = {'packages': ['experiments']}
            elif config in tool_configs_to_remove:
                # Keep as-is (black, ruff, mypy, pytest)
                workspace_doc['tool'][config] = root_doc['tool'][config]

    # Write root pyproject.toml
    with open(root_path, 'w') as f:
        content = tomlkit.dumps(workspace_doc)
        # Ensure single trailing newline (tomlkit may add extra blank lines)
        content = content.rstrip('\n') + '\n'
        f.write(content)
    print(f"  ✓ Created workspace root {root_path}")

    print("\n✓ Transformation complete!")
    print(f"  - lib/marin/pyproject.toml: Full deps/extras, no tool configs")
    print(f"  - pyproject.toml: Workspace root + experiments package, depends on marin")
    print(f"  - experiments/ stays at root (no move needed)")
    print(f"  - data_browser/ stays independent (not a workspace member)")


if __name__ == '__main__':
    transform_pyprojects()
