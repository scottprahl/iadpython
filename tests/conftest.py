# pylint: disable=invalid-name

"""Shared pytest fixtures for the iadpython test suite."""

import os
import shutil

import pytest


def _find_mc_lost_binary():
    """Return path to mc_lost binary, or None if not found."""
    repo_binary = os.path.join(os.path.dirname(__file__), "..", "iad", "mc_lost")
    if os.path.isfile(repo_binary) and os.access(repo_binary, os.X_OK):
        return os.path.abspath(repo_binary)
    return shutil.which("mc_lost")


@pytest.fixture(scope="session")
def mc_lost_path():
    """Path to the mc_lost binary; skip the test if it cannot be found.

    Build with::

        cd iad && make mc_lost
    """
    path = _find_mc_lost_binary()
    if path is None:
        pytest.skip("mc_lost binary not found; build with: cd iad && make mc_lost")
    return path
