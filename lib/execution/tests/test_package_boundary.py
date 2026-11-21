# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""marin-execution must stay importable without marin-core or the ML stack."""

import subprocess
import sys

# The subprocess below does the checking. This import only lets the CI test selector
# (infra/ci/select_tests.py), which follows static imports, see that the test depends on
# every module: lazy imports all the others.
import marin.execution.lazy  # noqa: F401

# Top-level packages that are not dependencies of marin-execution but are installed
# beside it in the Marin workspace, so an accidental import would still succeed there.
FORBIDDEN_TOP_LEVEL = ("haliax", "jax", "levanter", "torch", "transformers", "zephyr")

IMPORT_EVERY_MODULE = f"""
import importlib, pkgutil, sys
import marin.execution as package
for info in pkgutil.iter_modules(package.__path__):
    importlib.import_module(f"marin.execution.{{info.name}}")
forbidden = {FORBIDDEN_TOP_LEVEL!r}
for name in sorted(sys.modules):
    top = name.split(".")[0]
    outside_package = name.startswith("marin.") and not name.startswith("marin.execution")
    if top in forbidden or outside_package:
        print(name)
"""


def test_importing_marin_execution_loads_nothing_outside_its_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_EVERY_MODULE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == []
