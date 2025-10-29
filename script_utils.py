#!/usr/bin/env python3
"""Shared helpers for the repository's command-line utilities."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
from typing import Iterable, List


def split_command(command: str) -> List[str]:
    """Split a command string into components respecting the host shell rules."""
    return shlex.split(command, posix=platform.system() != "Windows")


def detect_python_command(env_var: str = "PYTHON_BIN") -> List[str]:
    """Return a Python interpreter command available on the current system."""
    overridden = os.environ.get(env_var)
    if overridden:
        return split_command(overridden)

    system = platform.system()
    if system == "Windows":
        candidates: Iterable[List[str]] = (["py", "-3"], ["py"], ["python"], ["python3"])
    else:
        candidates = (["python3"], ["python"], ["py", "-3"])

    for candidate in candidates:
        if shutil.which(candidate[0]):
            return list(candidate)

    raise RuntimeError(
        "Unable to locate a usable Python interpreter. Set "
        f"{env_var} to the desired command."
    )
