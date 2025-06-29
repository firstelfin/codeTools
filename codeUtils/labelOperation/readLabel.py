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
from pathlib import Path
from bs4 import BeautifulSoup


def parser_json(json_file: str | Path):
    """读取json文件, 返回一个字典"""
    for _ in range(3):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if data is not None:
                return data
        except:
            continue
    return None


def read_yolo(label_file: str | Path):
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


def read_voc(label_file: str | Path, extra_keys: list = []) -> dict | None:
    """读取voc格式的标签文件, 返回一个json对象, 指定extra_keys可以读取标注实例中额外的键值对

    :param str | Path label_file: voc格式的标签文件路径
    :param list extra_keys: 额外需要添加到object元素字典中的键列表, 格式为["key1", "key2",...]
    :return dict: voc格式的标签文件内容, 组织格式是超文本标签组织格式的映射
    """
    
    # 读取xml文件
    try:
        with open(label_file, 'r') as f:
            xml_str = f.read()
    except:
        return None

    # 解析xml文件
    soup = BeautifulSoup(xml_str, 'xml')

    def safe_text(tag, default=""):
        return tag.text.strip() if tag else default
    
    def safe_find(tag, *keys):
        """安全地进行多级 find() TODO: 形参是否会改变实参"""
        for key in keys:
            if tag is None:
                return None
            tag = tag.find(key)
        return tag

    voc_dict = {
        'folder': safe_text(soup.find('folder')),
        'filename': safe_text(soup.find('filename')),
        'path': safe_text(soup.find('path')),
        'source': {"database": safe_text(safe_find(soup, 'source', 'database'), "Unknown")},
        'segmented': int(safe_text(soup.find('segmented'), "0")),
        'size': {
            'width': int(safe_text(safe_find(soup, "size", "width"), "0")),
            'height': int(safe_text(safe_find(soup, "size", "height"), "0")),
            'depth': int(safe_text(safe_find(soup, "size", "depth"), "3")),
        },
        'object': [
            {
                'name': safe_text(safe_find(obj, 'name')),
                'pose': safe_text(safe_find(obj, 'pose'), default="Unspecified"),
                'truncated': int(safe_text(safe_find(obj, 'truncated'), default="0")),
                'difficult': int(safe_text(safe_find(obj, 'difficult'), default="0")),
                'bndbox': {
                    'xmin': int(float(safe_text(safe_find(obj, 'bndbox', 'xmin'), default="0"))),
                    'ymin': int(float(safe_text(safe_find(obj, 'bndbox', 'ymin'), default="0"))),
                    'xmax': int(float(safe_text(safe_find(obj, 'bndbox', 'xmax'), default="0"))),
                    'ymax': int(float(safe_text(safe_find(obj, 'bndbox', 'ymax'), default="0")))
                },
                **{key: safe_text(safe_find(obj, key)) for key in extra_keys}
            } for obj in soup.find_all('object')
        ]
    }
    
    return voc_dict


def read_txt(txt_file: str):
    res = []
    with open(txt_file, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().replace("\n", "")
        if not line:
            continue
        res.append(line)
    return res

