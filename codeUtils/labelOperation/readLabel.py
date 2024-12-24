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
from bs4 import BeautifulSoup


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


def read_voc(label_file: str, extra_keys: list = None) -> dict:
    """读取voc格式的标签文件, 返回一个json对象, 指定extra_keys可以读取标注实例中额外的键值对

    :param str label_file: voc格式的标签文件路径
    :param list extra_keys: 额外需要添加到object元素字典中的键列表, 格式为["key1", "key2",...]
    :return dict: voc格式的标签文件内容, 组织格式是超文本标签组织格式的映射
    """
    
    # 读取xml文件
    try:
        with open(label_file, 'r') as f:
            xml_str = f.read()
    except:
        return None
    
    if extra_keys is None:
        extra_keys = []

    # 解析xml文件
    soup = BeautifulSoup(xml_str, 'xml')

    voc_dict = {
        'folder': soup.find('folder').text,
        'filename': soup.find('filename').text,
        'path': soup.find('path').text if soup.find('path') else "",
        'source': {"database": soup.find('source').find('database').text if soup.find('source') else "Unknown"},
        'segmented': int(soup.find('segmented').text) if soup.find('segmented') else 0,
        'size': {
            'width': int(soup.find('size').find('width').text),
            'height': int(soup.find('size').find('height').text),
            'depth': int(soup.find('size').find('depth').text)
        },
        'object': [
            {
                'name': obj.find('name').text,
                'pose': obj.find('pose').text if obj.find('pose') else "Unspecified",
                'truncated': int(obj.find('truncated').text) if obj.find('truncated') else 0,
                'difficult': int(obj.find('difficult').text) if obj.find('difficult') else 0,
                'bndbox': {
                    'xmin': int(obj.find('bndbox').find('xmin').text),
                    'ymin': int(obj.find('bndbox').find('ymin').text),
                    'xmax': int(obj.find('bndbox').find('xmax').text),
                    'ymax': int(obj.find('bndbox').find('ymax').text)
                },
                **{key: obj.find(key).text for key in extra_keys}
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

