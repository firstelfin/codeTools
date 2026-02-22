#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2024/08/18 14:51:23
@Author  :   firstelfin 
@Version :   0.0.6
@Desc    :   None
'''

import os
# from . import fontConfig
# from . import cacheFile

cpu_count = os.cpu_count()
CPU_KERNEL_NUM = 8 if cpu_count is None else cpu_count
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

from .fontConfig import font_download, valid_local_font, set_plt, colorstr
from .cocoTools import segmentation_to_polygons
from .cacheFile import CacheFile
from .loadFile import load_img
from .futureConf import FutureBar

def is_async_function(func):
    return func.__code__.co_flags & 0x80  # 检查CO_ASYNC标志


__all__ = [
    "is_async_function", "font_download", "valid_local_font", "set_plt", 
    "colorstr", "segmentation_to_polygons", "CacheFile", "load_img", "FutureBar"
]

__call__ = [
    font_download, valid_local_font, set_plt, colorstr, segmentation_to_polygons,
    CacheFile, load_img, FutureBar, CPU_KERNEL_NUM, IMG_EXTENSIONS
]
