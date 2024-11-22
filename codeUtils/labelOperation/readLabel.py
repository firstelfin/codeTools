#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   readLabel.py
@Time    :   2024/12/09 15:34:02
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import json

def parser_json(json_file: str):
    """读取json文件, 返回一个字典"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except:
        return None
    
    return data


def read_yolo(label_file: str):
    """读取yolo格式的标签文件, 返回一个列表"""
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
    except:
        return None
    
    labels = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        label = line.split()
        box = map(float, label[1:])
        cls_id = int(label[0])
        labels.append([cls_id, *box])
    return labels

