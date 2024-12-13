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


def read_voc(label_file: str) -> dict:
    """读取voc格式的标签文件, 返回一个json对象

    :param str label_file: voc格式的标签文件路径
    :return dict: voc格式的标签文件内容, 组织格式是超文本标签组织格式的映射
    """
    
    # 读取xml文件
    try:
        with open(label_file, 'r') as f:
            xml_str = f.read()
    except:
        return None
    
    # 解析xml文件
    soup = BeautifulSoup(xml_str, 'lxml')

    voc_dict = {
        'folder': soup.find('folder').text,
        'filename': soup.find('filename').text,
        'source': {"database": soup.find('source').find('database').text},
        'segmented': int(soup.find('segmented').text),
        'size': {
            'width': int(soup.find('size').find('width').text),
            'height': int(soup.find('size').find('height').text),
            'depth': int(soup.find('size').find('depth').text)
        },
        'object': [
            {
                'name': obj.find('name').text,
                'pose': obj.find('pose').text,
                'truncated': int(obj.find('truncated').text),
                'difficult': int(obj.find('difficult').text),
                'bndbox': {
                    'xmin': int(obj.find('bndbox').find('xmin').text),
                    'ymin': int(obj.find('bndbox').find('ymin').text),
                    'xmax': int(obj.find('bndbox').find('xmax').text),
                    'ymax': int(obj.find('bndbox').find('ymax').text)
                }
            } for obj in soup.find_all('object')
        ]
    }
    
    return voc_dict
    

if __name__ == '__main__':
    voc_file = "/Users/elfin/project/codeTools/test/test/elfin/yoloLabelTest/voc/05_87728_金具-保护金具-防振锤-滑移-appress_2.xml"
    a = read_voc(voc_file)
    print(a)
