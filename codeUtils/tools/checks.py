#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   checks.py
@Time    :   2025/05/13 15:59:41
@Author  :   firstElfin 
@Version :   0.1.11.15
@Desc    :   检查相关的工具
'''

import re
import platform
from loguru import logger
from importlib import metadata


MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])


def parse_version(version="0.0.0") -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (Tuple[int, int, int]): Tuple of integers representing the numeric part of the version, i.e. (2, 0, 1)

    Note:
        Copy from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/checks.py
    """
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        logger.warning(f"failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0
    

def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str): Name to be used in warning message.
        hard (bool): If True, raise an AssertionError if the requirement is not met.
        verbose (bool): If True, print warning message if requirement is not met.
        msg (str): Extra message to display if verbose.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Examples:
        Check if current version is exactly 22.04
        >>> check_version(current="22.04", required="==22.04")

        Check if current version is greater than or equal to 22.04
        >>> check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        Check if current version is less than or equal to 22.04
        >>> check_version(current="22.04", required="<=22.04")

        Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        >>> check_version(current="21.10", required=">20.04,<22.04")
    """
    if not current:  # if current is '' or None
        logger.warning(f"invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(f"{current} package is required but not installed") from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()  # split '>=22.04' -> ('>=', '22.04')
        if not op:
            op = ">="  # assume >= if no op passed
        v = parse_version(version)  # '1.2.3' -> (1, 2, 3)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op == ">=" and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"{name}{required} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(warning)  # assert version requirements met
        if verbose:
            logger.warning(warning)
    return result
