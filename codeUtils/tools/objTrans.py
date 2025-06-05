#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   obj_trans.py
@Time    :   2024/09/22 11:39:54
@Author  :   firstElfin 
@Version :   0.1.1
@Desc    :   python 对象转换
'''

import re
import ast

def str2list(input_str:str, seg: str=',', full: bool=True) -> list:
    """从字符串解析出列表, 列表的元素分隔符默认是逗号, 可以使用seg参数指定分隔符
    若字符串不全是列表(元祖)字符, 中括号(括号)标识符必须完整！

    :param str input_str: 输入字符串
    :param str seg: 分隔符
    :param bool full: 输入字符串是否不包含非列表(元祖)字符
    :return list: 解析列表

    Example:
    ----------
    
    ```python
    >>> str2list('[1,2,3,4,5]')
    [1, 2, 3, 4, 5]
    >>> str2list('[1 2 3 4 5]', seg=' ')
    [1, 2, 3, 4, 5]
    >>> str2list('(1,2,3,4,5)elfin', full=False)
    (1, 2, 3, 4, 5)
    ```
    """

    parser_str = input_str.replace(seg, ',')
    if full:
        s = parser_str
    else:
        try:
            s = re.findall('\[.*\]', parser_str)[0]
        except:
            s = re.findall('\(.*\)', parser_str)[0]
    s_list = ast.literal_eval(s)
    return s_list


if __name__ == '__main__':
    print(str2list('[1,2,3,4,5]'))
    print(str2list('[1 2 3 4 5]', seg=' '))
    print(str2list('(1,2,3,4,5)'))
    print(str2list('1 2 3 4 5 6', seg=' '))
    print(str2list('(1,2,3,4,5)elfin', full=False))

