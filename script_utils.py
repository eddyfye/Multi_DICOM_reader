#!/usr/bin/env python3
"""Shared helpers for the repository's command-line utilities."""

from __future__ import annotations

import os  # Access environment variables for overrides
import platform  # Detect OS to choose tokenization and interpreter order
import shlex  # Split shell-style command strings safely
import shutil  # Probe PATH for available executables
from typing import Iterable, List  # Type hints for clarity


def split_command(command: str) -> List[str]:
    """Split a command string into components respecting the host shell rules."""
    # On Windows we avoid POSIX tokenization to respect quoted paths with spaces.
    return shlex.split(command, posix=platform.system() != "Windows")  # Use platform-aware parsing


def detect_python_command(env_var: str = "PYTHON_BIN") -> List[str]:
    """Return a Python interpreter command available on the current system."""
    overridden = os.environ.get(env_var)  # Check for explicit override
    if overridden:
        # Allow callers to force a specific interpreter via environment override.
        return split_command(overridden)  # Respect quoted arguments

    system = platform.system()  # Identify host OS
    if system == "Windows":
        # Prefer Windows launcher to respect per-user installations when present.
        candidates: Iterable[List[str]] = (["py", "-3"], ["py"], ["python"], ["python3"])
    else:
        # On POSIX systems the usual python3/python order is sufficient.
        candidates = (["python3"], ["python"], ["py", "-3"])

    for candidate in candidates:
        # Accept the first candidate whose executable exists on PATH.
        if shutil.which(candidate[0]):
            return list(candidate)  # Return command + optional args

    raise RuntimeError(
        "Unable to locate a usable Python interpreter. Set "
        f"{env_var} to the desired command."
    )
