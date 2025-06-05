#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   boxCallback.py
@Time    :   2025/03/05 10:59:12
@Author  :   firstElfin 
@Version :   0.1.10.5
@Desc    :   box相关的回调函数集合
'''

from loguru import logger


def box_filter_by_area(results: list[dict], area_threshold: dict, category_name: str = "class"):
    """通过bbox对象的面积过滤

    :param list[dict] results: 待过滤的bbox对象列表
    :param dict area_threshold: 面积阈值，格式为{类别名:面积阈值}
    :param str category_name: 类别名的字段名，默认为"class"
    :return _type_: list[dict]
    """
    
    class_name = category_name if category_name in results[0] else "label"
    res = [res_item for res_item in results if res_item['area'] >= area_threshold[res_item[class_name]]]
    return res 
