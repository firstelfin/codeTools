#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   labelme2yolo.py
@Time    :   2024/12/09 15:19:24
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   This script is used to convert labelme annotation to yolo format.
'''

import argparse
import click
from pathlib import Path, PosixPath
from codeUtils.labelOperation.readLabel import parser_json


def labelme_show():
    show_dict = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [
            {
                "label": "car",
                "points": [
                    [100.0, 100.0],
                    [200.0, 100.0],
                    [200.0, 200.0],
                    [100.0, 200.0]
                ],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            },
            {
                "label": "person",
                "points": [
                    [300.0, 300.0],
                    [654.0, 400.0]
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": "example.jpg",
        "imageData": None,
        "imageHeight": 300,
        "imageWidth": 400
    }
    print(show_dict)


def labelme2yolo(src_dir: PosixPath, dst_dir: PosixPath, classes: dict) -> None:
    """
    This function is used to convert labelme annotation to yolo format.

    :param PosixPath src_dir: labelme annotation directory.
    :param PosixPath dst_dir: yolo format save directory.
    :param dict classes: classes.txt file path or classes name dict: {class_name: class_id}.
    """
    
    if isinstance(classes, str):
        with open(classes, 'r+', encoding='utf-8') as f:
            cls_list = f.readlines()
        classes = {cls_txt.strip().split()[0]: i for i, cls_txt in enumerate(cls_list)}
    
    for json_file in Path(src_dir).rglob('*.json'):
        # 排出特殊文件
        if json_file.name.startswith('.'):
            continue

        # 读取labelme格式的json
        labelme_json = parser_json(json_file)

        # 标注转换
        labels_set = set()
        for shape in labelme_json['shapes']:
            label = classes.get(shape['label'], shape['label'])
            points = shape['points']
            img_h = labelme_json['imageHeight']
            img_w = labelme_json['imageWidth']
            x_list = [p[0] / img_w for p in points]
            y_list = [p[1] / img_h for p in points]
            
            if shape['shape_type'] == 'rectangle':
                x1, x2 = min(x_list), max(x_list)
                y1, y2 = min(y_list), max(y_list)
                w, h = x2 - x1, y2 - y1
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                labels_set.add(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            elif shape['shape_type'] == 'polygon':
                pair_points = [f"{x_list[i]:.6f} {y_list[i]:.6f}" for i in range(len(x_list))]
                polygon_points = " ".join(pair_points)
                labels_set.add(f"{label} {polygon_points}\n")
        
        # 保存yolo格式的txt文件
        txt_file = Path(dst_dir) / (json_file.stem + '.txt')
        labels = list(labels_set)
        with open(txt_file, 'w+', encoding='utf-8') as f:
            f.writelines(labels)


def labelme2voc(src_dir: PosixPath, dst_dir: PosixPath, classes: dict) -> None:
    pass


def labelme2coco(src_dir: PosixPath, dst_dir: PosixPath, classes: dict) -> None:
    pass


def labelme2industai(src_dir: PosixPath, dst_dir: PosixPath, classes: dict) -> None:
    pass

