#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   taskManage.py
@Time    :   2024/09/25 10:01:51
@Author  :   firstElfin 
@Version :   0.1.3
@Desc    :   全局算子调度管理模块
'''

import json
from loguru import logger
from pathlib import Path, PosixPath
from . import PARAM_REGISTRY
from ..decorator.registry import Registry


class TaskManager:
    """
    Example:
    ```
    >>> class AreasFilter():
    ...     def __init__(self) -> None:
    ...         super().__init__()
    ...         self.areas = IntParam(10000)  # 参数会被代理检查
    ...         self.area_thresh = 0.0001     # 参数不会被代理
    >>> tasks1 = [('add', 'AddTask'), ('sub', 'SubTask')]
    >>> taskManage1 = TaskManager(tasks=tasks1, operators={"AreasFilter": AreasFilter})
    >>> tasks2 = {'add: 'AddTask', 'sub': 'SubTask'}
    >>> taskManage2 = TaskManager(tasks=tasks2, operators={"AreasFilter": AreasFilter})
    ```
    """

    def __init__(self, operators: dict|Registry, tasks: list[tuple]|dict={}):
        """初始化 算子别名和算子类型 对象列表(或字典)

        :param list[tuple]|dict tasks: 元素tuple为(算子别名, 算子类型对象)
        :param dict|Registry operators: 算子类型对象列表或注册表
        :raises TypeError: tasks必须为list[tuple]或dict
        """
        if isinstance(tasks, list):
            self.tasks = {alias: task_type for alias, task_type in tasks}
        elif isinstance(tasks, dict):
            self.tasks = tasks
        else:
            raise TypeError("tasks must be a list[tuple] or a dict.")
        self.__tasks = []
        self.operators = operators

    def save_json_cfg(self, json_file:str|PosixPath):
        """生成计算图所有算子的json配置文件

        :param str | PosixPath json_file: json文件保存路径
        """

        # 缓存输入的任务配置
        finally_dict = {'tasks': self.tasks}

        # 遍历算子对象, 获取算子参数配置
        cfg_dict = {}
        for alias, task_type in self.tasks.items():
            operator = self.operators.get(task_type)()  # 实例化算子对象, 不要传参
            operator_param = dict()
            for key, value in operator.__dict__.items():
                if PARAM_REGISTRY.valid(value):
                    operator_param[key] = value.get()
            cfg_dict[alias] = operator_param
        finally_dict['operators'] = cfg_dict

        # 保存json配置文件
        with open(json_file, 'w+', encoding='utf-8') as f:
            json.dump(finally_dict, f, indent=4, ensure_ascii=False)
        
        # 打印日志
        for alias, operator_dict in finally_dict['operators'].items():
            logger.info(f"Operator {alias} config: {json.dumps(operator_dict, indent=4, ensure_ascii=False)}")
    
    def load_json_cfg(self, json_file:str|PosixPath):
        """从json配置文件中加载算子配置

        :param str | PosixPath json_file: json文件路径
        :param bool task_valid: 是否检查算子别名是否有效, 即配置文件的算子出现在tasks中
        """

        with open(json_file, 'r+', encoding='utf-8') as f:
            config_data = json.load(f)
            tasks_dict = config_data.get('tasks')
            config_dict = config_data.get('operators')
        
        self.tasks = tasks_dict
        # 从配置文件中加载算子配置
        for alias, task_type in self.tasks.items():
            operator = self.operators.get(task_type)()  # 实例化算子对象, 不要传参
            if alias not in config_dict:
                logger.warning(f"Operator {alias} config not found in json file.")
            else:
                operator_param = config_dict.get(alias)
                # 算子参数赋值
                for key, value in operator.__dict__.items():
                    if not PARAM_REGISTRY.valid(value):
                        continue
                    if key not in operator_param:
                        logger.warning(f"Param {key} not found in operator {alias} config.")
                    else:
                        operator.__dict__[key].set(operator_param.get(key))
                logger.info(f"Operator {alias} config loaded: {json.dumps(operator_param, indent=4, ensure_ascii=False)}")
            
            # 算子初始化后预处理载入!
            if hasattr(operator, 'init'):
                operator.init()
            
            self.__tasks.append(operator)

    def generate_graph(self):
        """生成计算图: 串行执行所有算子"""
        for operator in self.__tasks:
            yield operator
