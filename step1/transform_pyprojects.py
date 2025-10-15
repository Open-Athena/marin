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

import tomlkit
from pathlib import Path


def transform_pyprojects():
    """Transform root pyproject.toml into workspace structure."""
    root_path = Path('pyproject.toml')
    lib_marin_path = Path('lib/marin/pyproject.toml')

    print("Reading root pyproject.toml...")
    with open(root_path, 'r') as f:
        root_doc = tomlkit.parse(f.read())

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
    tool_configs_to_remove = ['black', 'ruff', 'mypy', 'pytest']
    if 'tool' in lib_marin_doc:
        for config in tool_configs_to_remove:
            if config in lib_marin_doc['tool']:
                del lib_marin_doc['tool'][config]
        # Also remove pytest.ini_options if it exists
        if 'pytest' in lib_marin_doc['tool'] and 'ini_options' in lib_marin_doc['tool']['pytest']:
            del lib_marin_doc['tool']['pytest']

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

    # Workspace configuration
    workspace_doc['tool'] = tomlkit.table()
    workspace_doc['tool']['uv'] = tomlkit.table()
    workspace_doc['tool']['uv']['workspace'] = tomlkit.table()
    # Only lib/marin is a workspace member (experiments stays at root, data_browser independent)
    workspace_doc['tool']['uv']['workspace']['members'] = ['lib/marin']

    # uv sources - marin comes from workspace member
    workspace_doc['tool']['uv']['sources'] = tomlkit.table()
    workspace_doc['tool']['uv']['sources']['marin'] = {'workspace': True}

    # Hatch build config - package experiments/ (same as before)
    workspace_doc['tool']['hatch'] = tomlkit.table()
    workspace_doc['tool']['hatch']['metadata'] = {'allow-direct-references': True}
    workspace_doc['tool']['hatch']['build'] = tomlkit.table()
    workspace_doc['tool']['hatch']['build']['targets'] = tomlkit.table()
    workspace_doc['tool']['hatch']['build']['targets']['wheel'] = {'packages': ['experiments']}
    workspace_doc['tool']['hatch']['build']['targets']['sdist'] = {'packages': ['experiments']}

    # Copy tool configs (black, ruff, mypy, pytest) from original
    if 'tool' in root_doc:
        for config in tool_configs_to_remove:
            if config in root_doc['tool']:
                workspace_doc['tool'][config] = root_doc['tool'][config]
        # Also copy pytest.ini_options if present
        if 'pytest' in root_doc['tool']:
            workspace_doc['tool']['pytest'] = root_doc['tool']['pytest']

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
