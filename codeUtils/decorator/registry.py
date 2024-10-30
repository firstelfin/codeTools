#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   registry.py
@Time    :   2024/09/25 15:10:35
@Author  :   firstElfin 
@Version :   0.1.3
@Desc    :   注册器
'''


class Registry(object):
    """注册器

    注册器是一个简单的类，用于管理类和函数的注册和获取。

    Example:

        ```
        >>> registry = Registry("my_registry")

        >>> @registry
        ... class MyClass:
        ...     pass
            
        >>> class MyOtherClass:
        ...     pass
        >>> registry.register(MyOtherClass)

        >>> @PARAM_REGISTRY
        ... def test():
        ...     print("test paramObject.py")
        >>> PARAM_REGISTRY.test()
        test paramObject.py
        >>> PARAM_REGISTRY["test"]()
        test paramObject.py
        ```
    """
    def __init__(self, name):
        self.name = name
        self._registry = {}

    def register(self, cls):
        """注册一个类或函数"""
        if cls.__name__ in self._registry:
            raise ValueError(f"{cls.__name__} is already registered in {self.name}")
        elif cls.__name__ in dir(self):
            raise ValueError(f"{cls.__name__} is already an attribute of {self.name}")
        self._registry[cls.__name__] = cls

    def get(self, key):
        """根据键获取注册的类或函数"""
        if key not in self._registry:
            raise KeyError(f"{key} is not registered in {self.name}")
        return self._registry[key]
    
    def valid(self, obj):
        for key in self._registry:
            if isinstance(obj, self._registry[key]):
                return True
        return False

    def list(self):
        """列出所有注册项"""
        return list(self._registry.keys())
    
    def __call__(self, cls):
        self.register(cls)
        return cls
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __getattr__(self, key):
        return self.get(key)
    
    def __iter__(self):
        return iter(self._registry.items())
    
    def items(self):
        return self.__iter__()


if __name__ == '__main__':
    TestRegistry = Registry("TestRegistry")

    @TestRegistry
    def test():
        print("test paramObject.py")

    TestRegistry.test()
    for key , value in TestRegistry.items():
        print(key, value)
    TestRegistry["test"]()
