#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   saveLabel.py
@Time    :   2024/12/12 14:14:21
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import json
from pathlib import Path
from lxml import etree
from bs4 import BeautifulSoup

INS_sample = [
    {
        "name": "fzhenchui",
        "pose": "Unspecified",
        "truncated": 0,
        "difficult": 0,
        "bndbox": {
            "xmin": 163,
            "ymin": 288,
            "xmax": 387,
            "ymax": 339
        },
        "score": 0.9999999
    },
    {
        "name": "fzhenchui",
        "pose": "Unspecified",
        "truncated": 0,
        "difficult": 0,
        "bndbox": {
            "xmin": 390,
            "ymin": 267,
            "xmax": 623,
            "ymax": 307
        }
    }
]

VOC_HEADER = {
    "folder": "images",
    "filename": "000001.jpg",
    "path": "images/000001.jpg",
    "source": {
        "database": "Unknown"
    },
    "size": {
        "width": 100,
        "height": 200,
        "depth": 3
    },
    "segmented": 0
}


def save_json(json_file: str, data):
    with open(json_file, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_labelme_label(label_path: str, label_dict: dict):
    """保存labelme格式的标签文件

    :param label_path: labelme格式的标签文件路径
    :type label_path: str
    :param label_dict: labelme格式的标签文件内容
    :type label_dict: dict
    """
    save_json(label_path, label_dict)


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


def voc_generate(voc_header: dict = None, objects: list[dict] = None, other_keys: list = None):
    """生成 voc 格式的标注文件

    :param dict voc_header: voc格式的文件头信息(非实例信息), defaults to None
    :param list[dict] objects: 实例信息, defaults to None
    :param list other_keys: 自定义实例属性列表, defaults to None
    """

    if objects is None:
        objects = INS_sample
    if voc_header is None:
        voc_header = VOC_HEADER
    
    annotation = BeautifulSoup(features="xml")

    # 创建根元素 <annotation>
    annotation_tag = annotation.new_tag("annotation")
    annotation.append(annotation_tag)

    # 图片相关信息
    folder_tag = annotation.new_tag("folder")
    folder_tag.string = voc_header["folder"]
    annotation_tag.append(folder_tag)

    filename_tag = annotation.new_tag("filename")
    filename_tag.string = voc_header["filename"]
    annotation_tag.append(filename_tag)

    path_tag = annotation.new_tag("path")
    path_tag.string = voc_header["path"]
    annotation_tag.append(path_tag)

    source_tag = annotation.new_tag("source")
    annotation_tag.append(source_tag)

    database_tag = annotation.new_tag("database")
    database_tag.string = voc_header["source"]["database"]
    source_tag.append(database_tag)

    size_tag = annotation.new_tag("size")
    annotation_tag.append(size_tag)

    width_tag = annotation.new_tag("width")
    width_tag.string = str(voc_header["size"]["width"])
    size_tag.append(width_tag)

    height_tag = annotation.new_tag("height")
    height_tag.string = str(voc_header["size"]["height"])
    size_tag.append(height_tag)

    depth_tag = annotation.new_tag("depth")
    depth_tag.string = str(voc_header["size"]["depth"])  # 假设是 RGB 彩图
    size_tag.append(depth_tag)

    segmented_tag = annotation.new_tag("segmented")
    segmented_tag.string = str(voc_header["segmented"])
    annotation_tag.append(segmented_tag)

    # 物体相关信息
    for obj in objects:
        object_tag = annotation.new_tag("object")
        annotation_tag.append(object_tag)

        name_tag = annotation.new_tag("name")
        name_tag.string = obj["name"]
        object_tag.append(name_tag)

        pose_tag = annotation.new_tag("pose")
        pose_tag.string = "Unspecified"
        object_tag.append(pose_tag)

        truncated_tag = annotation.new_tag("truncated")
        truncated_tag.string = str(obj["truncated"])
        object_tag.append(truncated_tag)

        difficult_tag = annotation.new_tag("difficult")
        difficult_tag.string = str(obj["difficult"])
        object_tag.append(difficult_tag)

        bndbox_tag = annotation.new_tag("bndbox")
        object_tag.append(bndbox_tag)

        xmin_tag = annotation.new_tag("xmin")
        xmin_tag.string = str(obj["bndbox"]["xmin"])
        bndbox_tag.append(xmin_tag)

        ymin_tag = annotation.new_tag("ymin")
        ymin_tag.string = str(obj["bndbox"]["ymin"])
        bndbox_tag.append(ymin_tag)

        xmax_tag = annotation.new_tag("xmax")
        xmax_tag.string = str(obj["bndbox"]["xmax"])
        bndbox_tag.append(xmax_tag)

        ymax_tag = annotation.new_tag("ymax")
        ymax_tag.string = str(obj["bndbox"]["ymax"])
        bndbox_tag.append(ymax_tag)

        if other_keys is None:
            continue
        for key in other_keys:
            if key not in obj:
                continue
            key_tag = annotation.new_tag(key)
            key_tag.string = str(obj[key])
            object_tag.append(key_tag)

    return annotation


def save_voc_label(xml_file: str, voc_header: dict = None, objects: list[dict] = None, other_keys: list = None):
    """保存 voc 格式的标注文件

    :param str xml_file: 保存的文件路径
    :param dict voc_header: 文件基础信息, defaults to None
    :param list[dict] objects: 实例的列表, 元素是实例字典, defaults to None
    :param list other_keys: 自定义实例属性列表, defaults to None
    """
    if objects is None:
        if "object" in voc_header:
            objects = voc_header["object"]
        else:
            objects = []

    annotation = voc_generate(voc_header, objects, other_keys)
    lxml_tree = etree.ElementTree(etree.fromstring(str(annotation).encode("utf-8")))
    with open(xml_file, "wb") as f:
        lxml_tree.write(f, pretty_print=True, encoding="utf-8", xml_declaration=True)


def voc_show(voc_header: dict = None, objects: list[dict] = None, other_keys: list = None):
    annotation = voc_generate(voc_header, objects, other_keys)
    print(annotation.prettify(formatter="html"))


def yolo_show():
    yolo_list = [
        [0, 0.5, 0.5, 0.56, 0.8],
        [1, 0.25, 0.5, 0.23, 0.65],
    ]
    print("cls_id, x_center, y_center, width, height\n", yolo_list)
    return yolo_list

