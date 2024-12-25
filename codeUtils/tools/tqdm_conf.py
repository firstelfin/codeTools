#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   tqdm_conf.py
@Time    :   2024/12/25 17:09:56
@Author  :   firstElfin 
@Version :   0.1.9
@Desc    :   This file is used to configure the progress bar of tqdm.
'''

from codeUtils.tools.font_config import colorstr

BATCH_KEY = colorstr("yellow", "bold", "batch")
START_KEY = colorstr("yellow", "bold", "startIdx")
END_KEY = colorstr("yellow", "bold", "endIdx")
