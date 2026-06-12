"""Keep otgan.__version__ and the installed package metadata in sync."""

import importlib.metadata

import otgan


def test_version_in_sync_with_package_metadata():
    assert otgan.__version__ == importlib.metadata.version("otgan")
