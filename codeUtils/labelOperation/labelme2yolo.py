#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   labelme2yolo.py
@Time    :   2024/12/09 15:19:24
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   This script is used to convert labelme annotation to yolo format.
'''

import click
from pathlib import Path, PosixPath
from codeUtils.labelOperation.readLabel import parser_json

@click.command()
@click.option('--labelme_json_dir', type=click.Path(exists=True), help='labelme annotation directory.')
@click.option('--yolo_save_dir', type=click.Path(exists=True), help='yolo format save directory.')
@click.option('--cls_ids', type=click.Path(exists=True), help='class id mapping file.')
def labelme2yoloCli(labelme_json_dir: PosixPath, yolo_save_dir: PosixPath, cls_ids: dict) -> None:
    
    if not labelme_json_dir:
        raise ValueError('labelme_json_dir is not specified.')
    if not yolo_save_dir:
        raise ValueError('yolo_save_dir is not specified.')
    if not cls_ids:
        raise ValueError('cls_ids is not specified.')
    
    labelme2yolo(labelme_json_dir, yolo_save_dir, cls_ids)


def labelme2yolo(labelme_json_dir: PosixPath, yolo_save_dir: PosixPath, cls_ids: dict) -> None:
    """
    This function is used to convert labelme annotation to yolo format.

    :param PosixPath labelme_json_dir: labelme annotation directory.
    :param PosixPath yolo_save_dir: yolo format save directory.
    :param dict cls_ids: class id mapping.
    """
    
    if isinstance(cls_ids, str):
        with open(cls_ids, 'r+', encoding='utf-8') as f:
            cls_list = f.readlines()
        cls_ids = {cls_txt.strip().split()[0]: i for i, cls_txt in enumerate(cls_list)}
    
    for json_file in Path(labelme_json_dir).rglob('*.json'):
        # 排出特殊文件
        if json_file.name.startswith('.'):
            continue

        # 读取labelme格式的json
        labelme_json = parser_json(json_file)

        # 标注转换
        labels_set = set()
        for shape in labelme_json['shapes']:
            label = cls_ids.get(shape['label'], shape['label'])
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
        txt_file = Path(yolo_save_dir) / (json_file.stem + '.txt')
        labels = list(labels_set)
        with open(txt_file, 'w+', encoding='utf-8') as f:
            f.writelines(labels)