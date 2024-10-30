#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   execTime.py
@Time    :   2024/08/18 11:45:20
@Author  :   firstelfin 
@Version :   1.0
@Desc    :   None
'''


import time
from loguru import logger
from ..tools import is_async_function

def log_time(prefix=""):
    """日志时间记录装饰器

    :param str prefix: 日志记录信息前缀, defaults to ""
    """

    def log_decorator(func):

        async def async_time_log(*args, **kwargs):
            log_start = time.perf_counter()
            res = await func(*args, **kwargs)
            log_end = time.perf_counter()
            logger.info(f"{func.__name__ if len(prefix) else prefix} execTime:\t{log_end-log_start:06.6f} second.")
            return res
        
        def sync_time_log(*args, **kwargs):
            log_start = time.perf_counter()
            res = func(*args, **kwargs)
            log_end = time.perf_counter()
            logger.info(f"{func.__name__ if len(prefix) else prefix} execTime:\t{log_end-log_start:06.6f} second.")
            return res
        
        return async_time_log if is_async_function(func) else sync_time_log
    
    return log_decorator


def inject_attr(inject_obj: str, inject_attr: str, search_obj: dict, value):
    """从`search_obj`搜索`inject_obj`, 将`inject_obj`的属性`inject_attr`设置为`value`

    :param str inject_obj: 待注入的对象名
    :param str inject_attr: 待注入对象的属性名
    :param dict search_obj: 搜索的域
    :param _type_ value: 设置的数值
    """
    if inject_obj in search_obj:
        if isinstance(search_obj[inject_obj], dict):
            search_obj[inject_obj].update({inject_attr: value})
        else:
            setattr(search_obj[inject_obj], inject_attr, value)
    else:
        logger.warning(f"Information injection failed, {inject_obj} not found in kwargs.")


def inject_time(obj_name: str, obj_attr: str):
    """记录异步/同步耗时到`obj_name`的`obj_attr`属性
    支持`dataclass`、`pydantic.BaseModel`数据属性修改, 同时支持修改`dict`的时间记录key: `obj_attr`.

    :param str obj_name: 要设置的对象变量名, 通过kwargs传参
    :param str obj_attr: 要设置的属性

    Example::

        ```
        @dataclass
        class Elfin:
            name:   str = "elfin"
            age:    int = 20
            time: float = 0

        @inject_time("elfin", "time")
        async def print_time(num, **kwargs):
            a = 0
            for _ in range(num):
                a += 1
            return a

        if __name__ == "__main__":
            import asyncio
            elfin = Elfin()
            asyncio.run(print_time(20, elfin=elfin))
            print(elfin)
        ```
    """

    def decorator_time(func):
        async def async_time_record(*args, **kwargs):
            start_time = time.perf_counter()
            res = await func(*args, **kwargs)
            end_time   = time.perf_counter()
            inject_attr(obj_name, obj_attr, kwargs, end_time - start_time)
            return res

        def sync_time_record(*args, **kwargs):
            start_time = time.perf_counter()
            res = func(*args, **kwargs)
            end_time   = time.perf_counter()
            inject_attr(obj_name, obj_attr, kwargs, end_time - start_time)
            return res
        
        return async_time_record if is_async_function(func) else sync_time_record
    
    return decorator_time
