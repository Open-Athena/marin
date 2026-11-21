# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import time
from types import SimpleNamespace

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from rigging.pytest_config import hermetic_cluster_config as hermetic_cluster_config

import marin.execution.step_runner as step_runner_module


@pytest.fixture(autouse=True)
def fray_client(_configure_marin_prefix):
    """Run every test against a LocalClient.

    Depends on ``_configure_marin_prefix`` so it tears down first: shutting down the
    client joins every local-backend worker thread before that fixture removes the
    MARIN_PREFIX temp directory.
    """
    client = LocalClient()
    with set_current_client(client):
        yield client
    client.shutdown()


@pytest.fixture(autouse=True)
def _configure_marin_prefix():
    """Set MARIN_PREFIX to a temp directory for tests that rely on it."""
    if "MARIN_PREFIX" in os.environ:
        yield
        return
    with tempfile.TemporaryDirectory(prefix="marin_prefix") as temp_dir:
        os.environ["MARIN_PREFIX"] = temp_dir
        yield
        del os.environ["MARIN_PREFIX"]


@pytest.fixture(autouse=True)
def fake_step_runner_clock(monkeypatch):
    clock = SimpleNamespace(monotonic=time.monotonic, sleep=lambda _interval: None)
    monkeypatch.setattr(step_runner_module, "time", clock)
