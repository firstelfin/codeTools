#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   paramObject.py
@Time    :   2024/09/25 11:31:34
@Author  :   firstElfin 
@Version :   0.1.3
@Desc    :   定义常见类型的参数封装
'''

from abc import ABCMeta, abstractmethod
from ..decorator.registry import Registry


PARAM_REGISTRY = Registry("PARAM_CONFIG")


class ConfigBase(metaclass=ABCMeta):
    """
    所有参数类的基类
    """

    @abstractmethod
    def check(self): ...

    @abstractmethod
    def get(self): ...

    @abstractmethod
    def set(self, value): ...


# 定义整数、浮点数、字符串、布尔型、列表、元组、字典等常见类型的参数封装

@PARAM_REGISTRY
class IntParam(ConfigBase):
    """整数参数类"""
    def __init__(self, value: int):
        self.value = value
        self.check()

    def check(self):
        if not isinstance(self.value, int):
            raise TypeError("IntParamError: value must be an integer.")
    
    def get(self):
        return self.value
    
    def set(self, value: int):
        self.value = value
        self.check()


@PARAM_REGISTRY
class FloatParam(ConfigBase):
    """浮点数参数类"""
    def __init__(self, value: float):
        self.value = value
        self.check()

    def check(self):
        if not isinstance(self.value, float):
            raise TypeError("FloatParamError: value must be a float.")
    
    def get(self):
        return self.value
    
    def set(self, value: float):
        self.value = value
        self.check()


@PARAM_REGISTRY
class StrParam(ConfigBase):
    """字符串参数类"""
    def __init__(self, value: str):
        self.value = value
        self.check()

    def check(self):
        if not isinstance(self.value, str):
            raise TypeError("StrParamError: value must be a string.")
    
    def get(self):
        return self.value
    
    def set(self, value: str):
        self.value = value
        self.check()


@PARAM_REGISTRY
class BoolParam(ConfigBase):
    """布尔型参数类"""
    def __init__(self, value: bool):
        self.value = value
        self.check()

    def check(self):
        if not isinstance(self.value, bool):
            raise TypeError("BoolParamError: value must be a boolean.")
    
    def get(self):
        return self.value
    
    def set(self, value: bool):
        self.value = value
        self.check()


@PARAM_REGISTRY
class IntListParam(ConfigBase):
    """整数列表参数类"""
    def __init__(self, value: list[int]):
        self.value = [IntParam(item) for item in value]
        self.check()

    def check(self):
        if not isinstance(self.value, list):
            raise TypeError("IntListParamError: value must be a list.")
        for item in self.value:
            item.check()
    
    def get(self):
        return [item.get() for item in self.value]
    
    def set(self, value: list[int]):
        self.value = [IntParam(item) for item in value]
        self.check()


@PARAM_REGISTRY
class FloatListParam(ConfigBase):
    """浮点数列表参数类"""
    def __init__(self, value: list[float]):
        self.value = [FloatParam(item) for item in value]
        self.check()

    def check(self):
        if not isinstance(self.value, list):
            raise TypeError("FloatListParamError: value must be a list.")
        for item in self.value:
            item.check()
    
    def get(self):
        return [item.get() for item in self.value]
    
    def set(self, value: list[float]):
        self.value = [FloatParam(item) for item in value]
        self.check()
    

@PARAM_REGISTRY
class StrListParam(ConfigBase):
    """字符串列表参数类"""
    def __init__(self, value: list[str]):
        self.value = [StrParam(item) for item in value]
        self.check()

    def check(self):
        if not isinstance(self.value, list):
            raise TypeError("StrListParamError: value must be a list.")
        for item in self.value:
            item.check()
    
    def get(self):
        return [item.get() for item in self.value]
    
    def set(self, value: list[str]):
        self.value = [StrParam(item) for item in value]
        self.check()


@PARAM_REGISTRY
class BoolListParam(ConfigBase):
    """布尔型列表参数类"""
    def __init__(self, value: list[bool]):
        self.value = [BoolParam(item) for item in value]
        self.check()

    def check(self):
        if not isinstance(self.value, list):
            raise TypeError("BoolListParamError: value must be a list.")
        for item in self.value:
            item.check()
    
    def get(self):
        return [item.get() for item in self.value]
    
    def set(self, value: list[bool]):
        self.value = [BoolParam(item) for item in value]
        self.check()


@PARAM_REGISTRY
class IntTupleParam(ConfigBase):
    """整数元组参数类"""
    def __init__(self, value: tuple[int]):
        self.value = tuple(IntParam(item) for item in value)
        self.check()

    def check(self):
        if not isinstance(self.value, tuple):
            raise TypeError("IntTupleParamError: value must be a tuple.")
        for item in self.value:
            item.check()
    
    def get(self):
        return tuple(item.get() for item in self.value)
    
    def set(self, value: tuple[int]):
        self.value = tuple(IntParam(item) for item in value)
        self.check()


@PARAM_REGISTRY
class FloatTupleParam(ConfigBase):
    """浮点数元组参数类"""
    def __init__(self, value: tuple[float]):
        self.value = tuple(FloatParam(item) for item in value)
        self.check()

    def check(self):
        if not isinstance(self.value, tuple):
            raise TypeError("FloatTupleParamError: value must be a tuple.")
        for item in self.value:
            item.check()
    
    def get(self):
        return tuple(item.get() for item in self.value)
    
    def set(self, value: tuple[float]):
        self.value = tuple(FloatParam(item) for item in value)
        self.check()


@PARAM_REGISTRY
class StrTupleParam(ConfigBase):
    """字符串元组参数类"""
    def __init__(self, value: tuple[str]):
        self.value = tuple(StrParam(item) for item in value)
        self.check()

    def check(self):
        if not isinstance(self.value, tuple):
            raise TypeError("StrTupleParamError: value must be a tuple.")
        for item in self.value:
            item.check()
    
    def get(self):
        return tuple(item.get() for item in self.value)
    
    def set(self, value: tuple[str]):
        self.value = tuple(StrParam(item) for item in value)
        self.check()


@PARAM_REGISTRY
class BoolTupleParam(ConfigBase):
    """布尔型元组参数类"""
    def __init__(self, value: tuple[bool]):
        self.value = tuple(BoolParam(item) for item in value)
        self.check()

    def check(self):
        if not isinstance(self.value, tuple):
            raise TypeError("BoolTupleParamError: value must be a tuple.")
        for item in self.value:
            item.check()
    
    def get(self):
        return tuple(item.get() for item in self.value)
    
    def set(self, value: tuple[bool]):
        self.value = tuple(BoolParam(item) for item in value)
        self.check()


@PARAM_REGISTRY
class ListParam(ConfigBase):
    """自定义列表参数类基类"""
    def __init__(self, value: list, item_type: ConfigBase):
        self.value = [item_type(item) for item in value]
        self.item_type = item_type

    def check(self):
        if not isinstance(self.value, list):
            raise TypeError("ListParamError: value must be a list.")
        for item in self.value:
            item.check()

    def get(self):
        return [item.get() for item in self.value]

    def set(self, value: list):
        self.value = [self.item_type(item) for item in value]
        self.check()


@PARAM_REGISTRY
class TupleParam(ConfigBase):
    """自定义元组参数类基类"""
    def __init__(self, value: tuple, item_type: ConfigBase):
        self.value = tuple(item_type(item) for item in value)        
        self.item_type = item_type

    def check(self):
        if not isinstance(self.value, tuple):
            raise TypeError("TupleParamError: value must be a tuple.")
        for item in self.value:
            item.check()

    def get(self):
        return tuple(item.get() for item in self.value)

    def set(self, value: tuple):
        self.value = tuple(self.item_type(item) for item in value)
        self.check()


@PARAM_REGISTRY
class DictParam(ConfigBase):
    """自定义字典参数类基类"""
    def __init__(self, value: dict, key_type: ConfigBase, value_type: ConfigBase):
        self.value = {key_type(key): value_type(value) for key, value in value.items()}
        self.key_type = key_type
        self.value_type = value_type

    def check(self):
        if not isinstance(self.value, dict):
            raise TypeError("DictParamError: value must be a dict.")
        for key, value in self.value.items():
            key.check()
            value.check()

    def get(self):
        return {key.get(): value.get() for key, value in self.value.items()}

    def set(self, value: dict):
        self.value = {self.key_type(key): self.value_type(value) for key, value in value.items()}
        self.check()


if __name__ == '__main__':
    # 测试IntParam
    elfin = PARAM_REGISTRY.get('IntParam')(90)
    print(isinstance(elfin, IntParam))
    print(elfin.check())
