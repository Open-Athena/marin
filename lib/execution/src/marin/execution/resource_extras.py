# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The uv extras a job needs for the accelerator its resources request."""

from fray.types import GpuConfig, ResourceConfig, TpuConfig


def extras_for_resources(resources: ResourceConfig) -> list[str]:
    """Return the uv extras for a resource device config."""
    device = resources.device
    if isinstance(device, TpuConfig):
        return ["tpu"]
    if isinstance(device, GpuConfig):
        return ["gpu"]
    return []


def dependency_groups_for_resources(
    resources: ResourceConfig,
    dependency_groups: list[str] | None,
) -> list[str]:
    """Return explicit dependency groups, or infer accelerator extras from resources."""
    if dependency_groups is not None:
        return dependency_groups
    return extras_for_resources(resources)
