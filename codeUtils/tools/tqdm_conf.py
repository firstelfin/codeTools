#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   tqdm_conf.py
@Time    :   2024/12/25 17:09:56
@Author  :   firstElfin 
@Version :   0.1.9
@Desc    :   This file is used to configure the progress bar of tqdm.
'''

import psutil
from codeUtils.tools.font_config import colorstr

BATCH_KEY = colorstr("yellow", "bold", "batch")
START_KEY = colorstr("yellow", "bold", "startIdx")
END_KEY = colorstr("yellow", "bold", "endIdx")
TPP = colorstr("yellow", "bold", "TPP")
FPP = colorstr("yellow", "bold", "FPP")  # 误报数量--pred
TPG = colorstr("yellow", "bold", "TPG")
TNG = colorstr("yellow", "bold", "FPG")  # 漏报数量--gt
GTN = colorstr("yellow", "bold", "GTN")  # 真实GT数量
PRN = colorstr("yellow", "bold", "PRN")  # 预测PRED数量

cpu_num = max(4, psutil.cpu_count(logical=False))
