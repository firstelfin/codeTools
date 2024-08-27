#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/08/18 13:56:00
@Author  :   firstelfin 
@Version :   1.0
@Desc    :   None
'''

from .exec_time import log_time, inject_time, inject_attr
from .error_decorator import ErrorCheck

__all__ = ["log_time", "inject_time", "inject_attr", "ErrorCheck"]