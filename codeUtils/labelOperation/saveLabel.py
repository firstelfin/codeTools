#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   saveLabel.py
@Time    :   2024/12/12 14:14:21
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

from pathlib import Path


def save_yolo_label(label_path: str, label_list: list[str, list]):
    """支持自动化保存YOLO格式的标签文件

    :param str label_path: 文件路径
    :param list label_list: 文件内容, 格式为[[class_id, x_center, y_center, width, height],...]
    """
    with open(label_path, 'w+', encoding='utf-8') as f:
        for label in label_list:
            line_str = " ".join(str(i) for i in label) if isinstance(label, list) else str(label)
            if not line_str.endswith('\n'):
                line_str += '\n'
            f.write(line_str)