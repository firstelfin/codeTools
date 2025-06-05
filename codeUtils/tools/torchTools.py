#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2024/06/12 17:05:01

import warnings
import torch
warnings.filterwarnings('ignore')

GB = 1024**3

def torch_empty_cache(thresh: str=4, defult_use: float = 1.1):
    """清除当前非激活显存
    :param thresh: 显存使用阈值, defaults to 4
    :type thresh: str, optional
    :param defult_use: 预留显存, defaults to 1.1
    :type defult_use: float, optional
    """
    used_memory = torch.cuda.memory_reserved() / GB
    if used_memory + defult_use >= thresh:
        torch.cuda.empty_cache()


def torch_set_memory_fraction(fraction: float = None, process_number: int = 0):
    """设置显存使用比例

    :param fraction: 显存使用比例, defaults to None
    :type fraction: float, optional
    :param process_number: 进程数量, defaults to 0
    :type process_number: int, optional
    """
    if fraction is None:
        fraction = 1 / process_number
    torch.cuda.set_per_process_memory_fraction(fraction)
