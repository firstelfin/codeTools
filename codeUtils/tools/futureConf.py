#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   futureConf.py
@Time    :   2025/05/06 19:06:03
@Author  :   firstElfin 
@Version :   0.1.11.10
@Desc    :   concurrent programming tools
'''

import os
from loguru import logger
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from codeUtils.tools.fontConfig import colorstr
from codeUtils.callback.tqdmCallback import TqdmFutureCallback


class FutureBar(object):
    """多进程、多线程并发执行任务, 并显示进度条. 进度条统一管理类, 异步对象错误收集重试自动化.

    :param int max_workers: 最大并发数, 默认为None, 自动根据CPU核数设置
    :param bool use_process: 是否使用多进程, 默认为False, 即使用多线程
    :param int timeout: 异步任务超时时间, 默认为20秒

    其他参数全部是
    """

    def __init__(
            self, max_workers=None, use_process=False, timeout=20,
            iterable=None, total=None, desc=None, colour="#CD8500",
            leave=True, file=None, ncols=None, mininterval=0.1, 
            maxinterval=10.0, miniters=None, ascii=None, disable=False, 
            unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.3, 
            bar_format=None, initial=0, position=None, postfix=None, 
            unit_divisor=1000, write_bytes=False, lock_args=None, nrows=None, 
            delay=0, gui=False, **kwargs
        ):
        self.max_workers = max_workers if isinstance(max_workers, int) else max(os.cpu_count() // 2, 6)
        self.use_process = use_process
        self.bar_callback = TqdmFutureCallback(timeout=timeout)
        new_desc = colorstr("bright_blue", "bold", desc) if isinstance(desc, str) else desc
        self.bar_kwargs = {
            "iterable": iterable, "total": total, "desc": new_desc, "colour": colour,
            "leave": leave, "file": file, "ncols": ncols, "mininterval": mininterval,
            "maxinterval": maxinterval, "miniters": miniters, "ascii": ascii, "disable": disable,
            "unit": unit, "unit_scale": unit_scale, "dynamic_ncols": dynamic_ncols, "smoothing": smoothing,
            "bar_format": bar_format, "initial": initial, "position": position, "postfix": postfix,
            "unit_divisor": unit_divisor, "write_bytes": write_bytes, "lock_args": lock_args, "nrows": nrows,
            "delay": delay, "gui": gui,
        }
        self.bar_kwargs.update(kwargs)
    
    def init_bar(self):
        self.bar = tqdm(**self.bar_kwargs)
        return self.bar

    def get_concurrent_executor(self):
        if self.use_process:
            return ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            return ThreadPoolExecutor(max_workers=self.max_workers)
    
    def retry_failed_tasks(self, exec_func):
        if self.bar_callback.future_error:
            logger.warning(f"There are {len(self.bar_callback.future_error)} errors in the concurrent tasks.")
        else:
            return None
        
        logger.info(f"Retrying {len(self.bar_callback.future_error)} tasks...")
        for param_args, param_kwargs, e in self.bar_callback.future_error:
            exec_func(*param_args, **param_kwargs)

    def __call__(self, exec_func, params, *args, **kwargs):
        """自定义多进程、多线程执行接口

        :param callable exec_func: 执行函数
        :param iterable params: 参数列表[可迭代对象], 每个元素包含一个参数元组(args, kwargs)
        """
        total = len(list(deepcopy(params))) if "total" not in kwargs else kwargs["total"]
        self.bar_kwargs.update({"total": total})
        self.bar = self.init_bar()
        with self.get_concurrent_executor() as executor:
            for param_args, param_kwargs in params:
                future = executor.submit(exec_func, *param_args, **param_kwargs)
                callback = partial(self.bar_callback, bar=self.bar, param_args=param_args, param_kwargs=param_kwargs)
                future.add_done_callback(callback)

        self.bar.close()
        self.retry_failed_tasks(exec_func=exec_func)


