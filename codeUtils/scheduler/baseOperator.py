#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   baseOperator.py
@Time    :   2024/09/26 14:56:47
@Author  :   firstElfin 
@Version :   0.1.4
@Desc    :   算子标准化基类, 并实现IOS算子作为子类参考
'''

from abc import ABCMeta, abstractmethod
from ..matchFactory.bboxMatch import ios_box
from ..decorator.registry import Registry
from .paramObject import FloatParam


OPERATORS_REGISTRY = Registry("operators")


@OPERATORS_REGISTRY
class BaseOperator(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self.class_name = self.__class__.__name__

    @abstractmethod
    def __call__(self, params: dict, res_item: dict={}) -> dict:
        # 第一步入参检查
        self.params_check(params)
        return res_item
    
    def params_check(self, params: dict) -> bool:
        for key in self.used_keys():
            assert key in params, f"KeyError: {self.class_name} operator did not receive the required parameter {key}."
        return True
    
    @classmethod
    @abstractmethod
    def used_keys(cls) -> list: ...    # 子类必须实现此方法, 返回本算子需要的keys列表

    @abstractmethod
    def inject_keys(self) -> list: ...  # 子类必须实现此方法, 返回本算子要注入的keys列表

    def init(self): ...    # 构造方法
        
    def deinit(self): ...  # 析构方法


@OPERATORS_REGISTRY
class IOSOperator(BaseOperator):
    """计算两个bbox的ios值, 并将结果注入到res_item中, 支持double模式返回两个框各自的交自比,
    支持bbox的坐标格式为: XYXY、XYWH(CXCYWH)

    :param BaseOperator: 基类
    :param threshold: 交并比阈值, 默认0.5
    :type threshold: FloatParam
    """

    def __init__(self):
        super().__init__()
        self.threshold = FloatParam(0.5)

    def __call__(self, params: dict, res_item: dict = {}) -> dict:
        res_item = super().__call__(params, res_item)
        box1, box2 = params["bboxes"]
        double = params.get("double", False)
        ios = ios_box(box1, box2, mode="xyxy", double=double)
        res_item["ios"] = ios if double else [ios]  # 格式对齐
        return res_item
    
    def used_keys(self) -> list:
        return ["bboxes"]
    
    def inject_keys(self) -> list:
        return ["ios"]
