#!/usr/bin/env python3
"""Update .pre-commit-config.yaml for monorepo structure.

Changes:
1. Exclude lib/levanter/ from insert-license hook (preserve Levanter license headers)
2. Configure black to run separately for lib/levanter/ with its own config
   (Black only uses ONE config per run, so we need separate hooks)
"""

from pathlib import Path
import sys
import re


def main():
    config_path = Path.cwd() / ".pre-commit-config.yaml"

    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        sys.exit(1)

    print("Updating .pre-commit-config.yaml...")

    content = config_path.read_text()

    modified = False

    # 1. Add exclude to insert-license hook to preserve Levanter license headers
    if not re.search(r'id: insert-license.*?exclude:', content, re.DOTALL):
        pattern = r'(      - id: insert-license[^\n]*\n        files: [^\n]+\n)'
        replacement = r'\1        exclude: ^lib/levanter/\n'
        new_content, count = re.subn(pattern, replacement, content)

        if count == 0:
            print("WARNING: Could not find insert-license hook")
        elif count > 1:
            print(f"WARNING: Found {count} insert-license matches, expected 1")
        else:
            content = new_content
            modified = True
            print("  ✓ Added exclude: ^lib/levanter/ to insert-license hook")
    else:
        print("  - insert-license already has exclude pattern")

    # 2. Configure black to run separately for lib/levanter/
    # Black only uses ONE config per run, so we need:
    # - First black hook: exclude lib/levanter/ (uses root config)
    # - Second black hook: only lib/levanter/ with --config (uses lib/levanter/pyproject.toml)

    # Check if we already have a separate levanter black hook
    if re.search(r'name: black \(levanter\)', content):
        print("  - black (levanter) hook already exists")
    else:
        # Add exclude to main black hook
        if not re.search(r'id: black.*?exclude:', content, re.DOTALL):
            pattern = r'(      - id: black\n)'
            replacement = r'\1        exclude: ^lib/levanter/\n'
            new_content, count = re.subn(pattern, replacement, content)

            if count == 0:
                print("WARNING: Could not find black hook")
            elif count > 1:
                print(f"WARNING: Found {count} black matches, expected 1")
            else:
                content = new_content
                modified = True
                print("  ✓ Added exclude: ^lib/levanter/ to black hook")

        # Add separate black hook for lib/levanter/ after the main black hook
        black_hook_pattern = r'(  - repo: https://github\.com/psf/black\n    rev: [^\n]+\n    hooks:\n      - id: black\n(?:        exclude: [^\n]+\n)?)'

        levanter_black_hook = r'''\1      - id: black
        name: black (levanter)
        files: ^lib/levanter/
        args: ["--config", "lib/levanter/pyproject.toml"]
'''

        new_content, count = re.subn(black_hook_pattern, levanter_black_hook, content)

        if count == 1:
            content = new_content
            modified = True
            print("  ✓ Added separate black hook for lib/levanter/")
        elif count == 0:
            print("WARNING: Could not add separate black hook (pattern not matched)")
        else:
            print(f"WARNING: Matched black hook pattern {count} times, expected 1")

    if modified:
        config_path.write_text(content)
    else:
        print("  - No changes needed")


if __name__ == "__main__":
    main()
