#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   tqdmCallback.py
@Time    :   2025/05/06 18:52:16
@Author  :   firstElfin 
@Version :   0.1.11.10
@Desc    :   tqdm 库相关的回调函数
'''

from concurrent.futures import Future
from tqdm import tqdm
from tqdm.std import tqdm as std_tqdm


class TqdmFutureCallback(object):
    """多线程、多进程任务进度条回调函数

    :param int timeout: future 超时时间, 默认20秒
    """

    def __init__(self, timeout: int = 20):
        self.future_error = list()
        self.timeout = timeout
    
    def __call__(self, future: Future, bar: std_tqdm, param_args, param_kwargs, *args, **kwargs):
        try:
            future.result(timeout=self.timeout)
        except Exception as e:
            self.future_error.append((param_args, param_kwargs, e))
        finally:
            bar.update(1)

