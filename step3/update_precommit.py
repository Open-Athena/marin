#!/usr/bin/env -S uv run
# /// script
# ///
"""Update infra/pre-commit.py to add Haliax config."""

from pathlib import Path


def main():
    repo_root = Path(__file__).parent.parent.parent
    precommit_file = repo_root / "infra/pre-commit.py"

    print("Updating infra/pre-commit.py for Haliax...")

    with open(precommit_file) as f:
        content = f.read()

    # Add Haliax constants after Levanter ones
    content = content.replace(
        "ROOT_DIR = pathlib.Path(__file__).parent.parent\n"
        "LEVANTER_LICENSE = ROOT_DIR / \"lib/levanter/etc/license_header.txt\"\n"
        "MARIN_LICENSE = ROOT_DIR / \"etc/license_header.txt\"\n"
        "LEVANTER_BLACK_CONFIG = ROOT_DIR / \"lib/levanter/pyproject.toml\"",
        "ROOT_DIR = pathlib.Path(__file__).parent.parent\n"
        "LEVANTER_LICENSE = ROOT_DIR / \"lib/levanter/etc/license_header.txt\"\n"
        "HALIAX_LICENSE = ROOT_DIR / \"lib/haliax/etc/license_header.txt\"\n"
        "MARIN_LICENSE = ROOT_DIR / \"etc/license_header.txt\"\n"
        "LEVANTER_BLACK_CONFIG = ROOT_DIR / \"lib/levanter/pyproject.toml\"\n"
        "HALIAX_BLACK_CONFIG = ROOT_DIR / \"lib/haliax/pyproject.toml\"",
    )

    # Add Haliax PrecommitConfig after Levanter's
    levanter_config = """    PrecommitConfig(
        patterns=["lib/levanter/**/*.py"],
        checks=[
            check_ruff,
            lambda files, fix: check_black(files, fix, config=LEVANTER_BLACK_CONFIG),
            lambda files, fix: check_license_headers(files, fix, LEVANTER_LICENSE),
            # check_mypy,
        ],
    ),"""

    haliax_config = """    PrecommitConfig(
        patterns=["lib/haliax/**/*.py"],
        checks=[
            check_ruff,
            lambda files, fix: check_black(files, fix, config=HALIAX_BLACK_CONFIG),
            lambda files, fix: check_license_headers(files, fix, HALIAX_LICENSE),
        ],
    ),"""

    content = content.replace(
        levanter_config,
        f"{levanter_config}\n{haliax_config}",
    )

    # Exclude lib/haliax from general Python config
    content = content.replace(
        'exclude_patterns=["lib/levanter/**"]',
        'exclude_patterns=["lib/levanter/**", "lib/haliax/**"]',
    )

    with open(precommit_file, "w") as f:
        f.write(content)

    print("  ✓ Updated infra/pre-commit.py")

    # Run pre-commit fix (use uv run to ensure dependencies available)
    import subprocess

    print("\nRunning uv run ./infra/pre-commit.py --all-files --fix...")
    result = subprocess.run(
        ["uv", "run", "./infra/pre-commit.py", "--all-files", "--fix"],
        cwd=repo_root,
        capture_output=False,
    )

    if result.returncode != 0:
        print("\n❌ Pre-commit checks failed!")
        print("This usually means you're running from the wrong branch.")
        print("Step 3 requires running from m/rw/lint (or a branch with passing pre-commit).")
        return result.returncode

    # Fix any duplicate licenses in tests/ (pre-existing Levanter licenses)
    test_file = repo_root / "tests/test_marin_chat_template.py"
    if test_file.exists():
        content = test_file.read_text()
        # Check for duplicate license headers (Marin + Levanter)
        if "# Copyright 2025 The Marin Authors" in content and "# Copyright 2025 The Levanter Authors" in content:
            print(f"\n  Fixing duplicate license in {test_file.relative_to(repo_root)}...")
            # Remove the Levanter license header
            lines = content.split("\n")
            new_lines = []
            skip_until_blank = False
            for i, line in enumerate(lines):
                if line.startswith("# Copyright 2025 The Levanter Authors"):
                    skip_until_blank = True
                    continue
                if skip_until_blank:
                    if line.strip() == "" or not line.startswith("#"):
                        skip_until_blank = False
                        if line.strip() == "":
                            continue  # Skip the blank line after removed license
                    else:
                        continue  # Skip Levanter license lines
                new_lines.append(line)
            test_file.write_text("\n".join(new_lines))

    print("Pre-commit config updated successfully")
    return 0


if __name__ == "__main__":
    exit(main())
