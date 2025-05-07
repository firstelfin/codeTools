#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   labelme2yolo.py
@Time    :   2024/12/09 15:19:24
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   This script is used to convert labelme annotation to yolo format.
'''

import math
import time
import psutil
import shutil
import cv2 as cv
from loguru import logger
from tqdm import tqdm
from pathlib import Path, PosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed
from codeUtils.labelOperation.readLabel import parser_json, read_txt
from codeUtils.labelOperation.saveLabel import save_voc_label, save_json
from codeUtils.labelOperation.converter import ToCOCO
from codeUtils.tools.fontConfig import colorstr
from codeUtils.tools.tqdmConf import BATCH_KEY, START_KEY, END_KEY


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


def labelme_to_yolo(json_file: str, dst_dir: str, classes: dict) -> str:
    if json_file.name.startswith('.'):
        return None

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

    tasks_num = len(list(Path(src_dir).rglob('*.json')))
    l2y_desc = colorstr("bright_blue", "bold", "labelme2yolo")
    cpu_num = max(4, psutil.cpu_count(logical=False) // 2)
    batch_size = 100
    epoch_num = math.ceil(tasks_num / batch_size)
    tqdm_tasks = Path(src_dir).rglob('*.json')

    with ThreadPoolExecutor(max_workers=cpu_num) as executor:
        
        with tqdm(total=tasks_num, desc=l2y_desc, dynamic_ncols=True, colour="#CD8500") as l2y_bar:
            for epoch in range(epoch_num):
                start_idx = epoch * batch_size
                end_idx = min(tasks_num, (epoch + 1) * batch_size)
                
                inner_tasks = []
                epoch_size = end_idx - start_idx

                for _ in range(start_idx, end_idx):
                    json_file = next(tqdm_tasks)
                    inner_tasks.append(executor.submit(labelme_to_yolo, json_file, dst_dir, classes))
                
                for ti, task in enumerate(as_completed(inner_tasks), start=1):
                    task.result()
                    l2y_bar.set_postfix({
                        BATCH_KEY: f"{ti}/{epoch_size}", 
                        START_KEY: start_idx, 
                        END_KEY: end_idx
                    })
                    l2y_bar.update()


def labelme_to_voc(json_file: str, dst_dir: str, extra_keys: list = None) -> bool:
    """labelme格式的json文件转为voc格式的xml文件

    :param str json_file: labelme格式的json文件路径
    :param str dst_dir: voc格式的xml文件保存文件夹路径
    :param list extra_keys: 额外的键值, defaults to None
    :return str: True / None
    """
    if Path(json_file).suffix != '.json':
        return None
    
    if extra_keys is None:
        extra_keys = []

    # 读取labelme格式的json
    labelme_json = parser_json(json_file)
    json_path = Path(json_file)
    seg_ins_idx = [i for i, shape in enumerate(labelme_json['shapes']) if shape["shape_type"] == "polygon"]
    have_segmented = int(len(seg_ins_idx) > 0)

    voc_dict = {
        'folder': str(json_path.parent),
        'filename': labelme_json['imagePath'],
        'path': labelme_json['imagePath'],
        'source': {"database": "Unknown"},
        'segmented': have_segmented,
        'size': {
            'width': labelme_json['imageWidth'],
            'height': labelme_json['imageHeight'],
            'depth':  3
        },
        'object': [
            {
                'name': obj.get("label", "Unknown"),
                'pose': "Unspecified",
                'truncated': 0,
                'difficult': 0,
                'bndbox': {
                    'xmin': obj['points'][0][0],
                    'ymin': obj['points'][0][1],
                    'xmax': obj['points'][1][0],
                    'ymax': obj['points'][1][1]
                },
                # "segmentation": [],
                **{key: obj.get(key, -1) for key in extra_keys}
            } for j, obj in enumerate(labelme_json["shapes"])
        ]
    }
    # 保存voc格式的xml文件
    xml_file = Path(dst_dir) / (json_path.stem + '.xml')
    save_voc_label(xml_file, voc_header=voc_dict, objects=voc_dict['object'], other_keys=extra_keys)
    return True


def labelme2voc(src_dir: PosixPath, dst_dir: PosixPath, extra_keys: list = None) -> None:
    """labelme格式的json文件批量转为voc格式的xml文件

    :param PosixPath src_dir: labelme格式的json文件路径
    :param PosixPath dst_dir: voc格式的xml文件保存文件夹路径
    """

    if extra_keys is None:
        extra_keys = []
    Path(dst_dir).mkdir(exist_ok=True, parents=True)
    l2v_desc = colorstr("bright_blue", "bold", "labelme2voc")
    cpu_num = max(4, psutil.cpu_count(logical=False) // 2)
    tasks_num = len(list(Path(src_dir).rglob('*.json')))
    batch_size = 100
    epoch_num = math.ceil(tasks_num / batch_size)
    tqdm_tasks = Path(src_dir).rglob('*.json')

    with ThreadPoolExecutor(max_workers=cpu_num) as executor:
        with tqdm(total=tasks_num, desc=l2v_desc, dynamic_ncols=True, colour="#CD8500") as l2v_bar:
            for epoch in range(epoch_num):
                start_idx = epoch * batch_size
                end_idx = min(tasks_num, (epoch + 1) * batch_size)
                
                inner_tasks = []
                epoch_size = end_idx - start_idx

                for _ in range(start_idx, end_idx):
                    json_file = next(tqdm_tasks)
                    inner_tasks.append(executor.submit(labelme_to_voc, json_file, dst_dir, extra_keys))
                
                for ti, task in enumerate(as_completed(inner_tasks), start=1):
                    task.result()
                    l2v_bar.set_postfix({
                        BATCH_KEY: f"{ti}/{epoch_size}", 
                        START_KEY: start_idx, 
                        END_KEY: end_idx
                    })
                    l2v_bar.update()


class Labelme2COCO(ToCOCO):

    def instance_prepare(self, lbl_file, img_id, split):
        return super().instance_prepare(lbl_file, img_id, split)


def labelme2coco(
        img_dir: PosixPath, dst_dir: PosixPath, classes: dict | str, 
        lbl_dir: PosixPath = None, img_idx: int = 0, ann_idx: int = 0, 
        use_link: bool = False, split: str = 'train', year: str = "", 
        class_start_index: int = 0
        ) -> None:
    lbl2coco = Labelme2COCO(
        img_dir, lbl_dir, dst_dir, classes, use_link=use_link, split=split, 
        img_idx=img_idx, ann_idx=ann_idx, year=year, class_start_index=class_start_index
    )
    lbl2coco()


def labelme2industai(src_dir: PosixPath, dst_dir: PosixPath, classes: dict) -> None:
    pass

