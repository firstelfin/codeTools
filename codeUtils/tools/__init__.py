#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/08/18 14:51:23
@Author  :   firstelfin 
@Version :   0.0.6
@Desc    :   None
'''

from . import fontConfig
from . import cacheFile

def is_async_function(func):
    return func.__code__.co_flags & 0x80  # 检查CO_ASYNC标志


__all__ = ["is_async_function", "fontConfig", "cacheFile"]
