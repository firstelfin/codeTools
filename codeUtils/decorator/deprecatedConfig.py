#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2026/02/22 15:31:55

import warnings
import hashlib
from functools import wraps
from loguru import logger

hash_set = set()

def deprecated(version: str, replacement: str = ""):
    """标记函数/类将在未来版本中弃用"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"{func.__name__} is deprecated since v{version}"
            if replacement:
                msg += f", use '{replacement}' instead"
            msg_hash = hashlib.md5(msg.encode()).hexdigest()
            if msg_hash not in hash_set:
                logger.warning(msg)
            hash_set.add(msg_hash)
            return func(*args, **kwargs)
        return wrapper
    return decorator
