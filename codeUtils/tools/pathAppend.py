#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   pathAppend.py
@Time    :   2024/12/09 15:46:23
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import sys
from pathlib import Path


def path_append(file, depth=1):
    FILE = Path(file).resolve()
    ROOT = FILE.parents[depth]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT
