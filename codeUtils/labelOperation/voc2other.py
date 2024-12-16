#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   voc2other.py
@Time    :   2024/12/12 22:10:58
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''
import lxml
import json
from lxml import etree
from tqdm import tqdm
from bs4 import BeautifulSoup
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from codeUtils.labelOperation.readLabel import read_voc, read_txt


INS_sample = [
    {
        "name": "fangzhenchui",
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
        "name": "fangzhenchui",
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


def voc_show(voc_header: dict = None, objects: list[dict] = None, other_keys: list = None):
    annotation = voc_generate(voc_header, objects, other_keys)
    print(annotation.prettify(formatter="html"))

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


def get_voc_names(voc_file: str):
    voc_dict = read_voc(voc_file)
    names = set()
    for obj in voc_dict["object"]:
        names.add(obj["name"])
    return names


def voc_gen_classes(src_dir: str, dst_file: str = None):
    """生成classes.txt文件

    :param str src_dir: xml标注目录路径, 文件夹下平铺所有XML文件
    :param str dst_file: classes.txt文件保存路径, defaults to None
    :return list: names列表
    """

    all_names = set()
    all_objects_list = []
    with ThreadPoolExecutor() as executor:
        for voc_file in Path(src_dir).rglob("*.xml"):
            all_objects_list.append(executor.submit(get_voc_names, str(voc_file)))
    
        for names in tqdm(as_completed(all_objects_list), total=len(all_objects_list), desc="GetAllNames"):
            name_list = names.result()
            all_names.update(name_list)

    all_names = list(all_names)
    if dst_file is None:
        dst_file = Path(src_dir) / "voc2YoloClasses.txt"
        
    with open(dst_file, "w+", encoding="utf-8") as f:
        for name in all_names:
            f.write(name + "\n")
    
    return all_names


def voc_to_yolo(src_file: str, dst_file: str = None, names: dict = None, img_valid: bool = False):
    """xml标注转yolo格式

    :param str src_file: xml文件路径
    :param str dst_file: yolo格式文件保存路径, defaults to None
    :param bool img_valid: 是否检查图片是否存在, defaults to False
    """
    if names is None:
        raise ValueError("names is None. Please provide a dict of class names(name: id).")
    if dst_file is None:
        dst_file = src_file.replace(".xml", ".txt")
    voc_dict = read_voc(src_file)
    if img_valid and not Path(voc_dict["path"]).exists():
        return None
    img_h, img_w = voc_dict["size"]["height"], voc_dict["size"]["width"]
    yolo_list = []
    for obj in voc_dict["object"]:
        name = obj["name"]
        if name not in names:
            raise ValueError(f"name {name} not in names.")
        x1, y1, x2, y2 = obj["bndbox"]["xmin"], obj["bndbox"]["ymin"], obj["bndbox"]["xmax"], obj["bndbox"]["ymax"]
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        nw, nh = w / img_w, h / img_h
        cls_id = names[name]
        yolo_list.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    
    with open(dst_file, "w+", encoding="utf-8") as f:
        f.writelines(yolo_list)
    return dst_file


def voc2yolo(src_dir: str, dst_dir: str = None, classes: list = None, img_valid: bool = False):
    """xml标注目录转换为yolo格式

    :param str src_dir: xml标注目录路径, 文件夹下平铺所有XML文件
    :param str dst_dir: yolo格式标注输出路径, defaults to None
    :param list classes: classes.txt文件路径, defaults to None
    :param bool img_valid: 是否对图片进行检查, defaults to False
    """
    if dst_dir is None:
        dst_dir = src_dir + "_yolo"
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)

    if classes is None:
        raise ValueError("classes is None. Please provide a list of class names.")
    if isinstance(classes, str) and Path(classes).exists():
        classes = read_txt(classes)
    if not isinstance(classes, list):
        raise ValueError("classes should be a list of class names.")

    # 生成类别映射表
    names = {name: i for i, name in enumerate(classes)}

    # xml文件转换为yolo格式, 文件已经包含了所有信息， 不需要额外
    all_results = []
    with ThreadPoolExecutor() as executor:
        for voc_file in Path(src_dir).rglob("*.xml"):
            all_results.append(
                executor.submit(
                    voc_to_yolo, 
                    str(voc_file), 
                    str(dst_dir / voc_file.name.replace(".xml", ".txt")), 
                    names,
                    img_valid
                )
            )

        for res in tqdm(as_completed(all_results), total=len(all_results), desc="Voc2Yolo"):
            res.result()

