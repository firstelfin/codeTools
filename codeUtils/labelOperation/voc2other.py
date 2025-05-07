#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   voc2other.py
@Time    :   2024/12/12 22:10:58
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import math
import psutil
import cv2 as cv
from tqdm import tqdm
from pathlib import Path, PosixPath
from loguru import logger
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from codeUtils.labelOperation.readLabel import read_voc, read_txt
from codeUtils.labelOperation.saveLabel import save_labelme_label
from codeUtils.tools.tqdmConf import BATCH_KEY, START_KEY, END_KEY
from codeUtils.tools.fontConfig import colorstr
from codeUtils.labelOperation.converter import ToCOCO


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
    if isinstance(classes, str) and not Path(classes).exists():
        raise FileNotFoundError(f"classes file {classes} not found.")
    if isinstance(classes, str) and Path(classes).exists():
        classes = read_txt(classes)
    if not isinstance(classes, list):
        raise ValueError("classes should be a list of class names.")

    # 生成类别映射表
    names = {name: i for i, name in enumerate(classes)}
    
    tasks_num = len(list(Path(src_dir).rglob("*.xml")))
    v2y_desc = colorstr("bright_blue", "bold", "voc2yolo")
    cpu_num = max(4, psutil.cpu_count(logical=False) // 2)
    batch_size = 100
    epoch_num = math.ceil(tasks_num / batch_size)
    tqdm_tasks = Path(src_dir).rglob("*.xml")

    # xml文件转换为yolo格式, 文件已经包含了所有信息
    with ThreadPoolExecutor(max_workers=cpu_num) as executor:
        with tqdm(total=tasks_num, desc=v2y_desc, dynamic_ncols=True, colour="#CD8500") as v2y_bar:
            for epoch in range(epoch_num):
                start_idx = epoch * batch_size
                end_idx = min(tasks_num, (epoch + 1) * batch_size)
                
                inner_tasks = []
                epoch_size = end_idx - start_idx

                for _ in range(start_idx, end_idx):
                    voc_file = next(tqdm_tasks)
                    inner_tasks.append(
                        executor.submit(
                            voc_to_yolo, str(voc_file),
                            str(dst_dir / voc_file.name.replace(".xml", ".txt")),
                            names,
                            img_valid
                        )
                    )
                
                for ti, task in enumerate(as_completed(inner_tasks), start=1):
                    task.result()
                    v2y_bar.set_postfix({
                        BATCH_KEY: f"{ti}/{epoch_size}", 
                        START_KEY: start_idx, 
                        END_KEY: end_idx
                    })
                    v2y_bar.update()


def voc_to_labelme(src_file: str, dst_file: str = None, img_valid: bool = False):
    """xml标注转labelme格式

    :param str src_file: xml文件路径
    :param str dst_file: labelme格式文件保存路径, defaults to None
    :param bool img_valid: 是否检查图片是否存在, defaults to False
    """
    voc_dict = read_voc(src_file)
    if voc_dict is None:
        logger.warning(f"xml file {src_file} not found.")
        return None
    
    labelme_dict = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": voc_dict["path"] if voc_dict["path"] else voc_dict["filename"],
        "imageData": None,
        "imageHeight": voc_dict["size"]["height"],
        "imageWidth": voc_dict["size"]["width"],
    }

    if img_valid and not Path(voc_dict["path"]).exists():
        return None
    if img_valid:
        src_img = cv.imread(str(voc_dict["path"]))
        height_valid = src_img.shape[0] == voc_dict["size"]["height"]
        width_valid = src_img.shape[1] == voc_dict["size"]["width"]
        if not (height_valid and width_valid):
            logger.warning(f"image size not match with xml file {src_file}.")
            return None
    
    labelme_dict["shapes"] = [
        {
            "label": obj["name"],
            "points": [
                [int(obj["bndbox"]["xmin"]), int(obj["bndbox"]["ymin"])],
                [int(obj["bndbox"]["xmax"]), int(obj["bndbox"]["ymax"])]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        } for obj in voc_dict["object"]
    ]
    
    Path(dst_file).parent.mkdir(exist_ok=True, parents=True)
    save_labelme_label(dst_file, labelme_dict)

    return True

def voc2labelme(src_dir: str, dst_dir: str = None, img_valid: bool = False):
    """src_dir下xml标注转换为labelme格式, 输出到dst_dir

    :param str src_dir: xml标注目录路径, 文件夹下平铺所有XML文件
    :param str dst_dir: labelme格式标注输出路径, defaults to None
    :param bool img_valid: 是否对图片进行检查, defaults to False
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)

    tasks_num = len(list(Path(src_dir).rglob("*.xml")))
    v2l_desc = colorstr("bright_blue", "bold", "voc2labelme")
    cpu_num = max(4, psutil.cpu_count(logical=False) // 2)
    batch_size = 100
    epoch_num = math.ceil(tasks_num / batch_size)
    tqdm_tasks = Path(src_dir).rglob("*.xml")

    # xml文件转换为yolo格式, 文件已经包含了所有信息
    with ThreadPoolExecutor(max_workers=cpu_num) as executor:
        with tqdm(total=tasks_num, desc=v2l_desc, dynamic_ncols=True, colour="#CD8500") as v2l_bar:
            for epoch in range(epoch_num):
                start_idx = epoch * batch_size
                end_idx = min(tasks_num, (epoch + 1) * batch_size)
                
                inner_tasks = []
                epoch_size = end_idx - start_idx

                for _ in range(start_idx, end_idx):
                    voc_file = next(tqdm_tasks)
                    inner_tasks.append(
                        executor.submit(
                            voc_to_labelme, str(voc_file),
                            str(dst_dir / f"{voc_file.stem}.json"),
                            img_valid
                        )
                    )
                
                for ti, task in enumerate(as_completed(inner_tasks), start=1):
                    task.result()
                    v2l_bar.set_postfix({
                        BATCH_KEY: f"{ti}/{epoch_size}", 
                        START_KEY: start_idx, 
                        END_KEY: end_idx
                    })
                    v2l_bar.update()


class VOC2COCO(ToCOCO):

    def load_items(self):
        return super().load_items(suffix=".xml")
    
    def instance_prepare(self, lbl_file, img_id, split):
        """从voc格式标注文件中解析出实例信息

        :param str lbl_file: voc格式标注文件路径
        :param int img_id: 图像id
        :param str split: coco数据集划分名称
        :return list: 实例列表
        """

        file_labels = None
        for _ in range(3):
            file_labels = read_voc(lbl_file)
            if file_labels is not None:
                break
            
        if file_labels is None:
            file_labels = {"object": []}

        for label in file_labels["object"]:
            if label["name"] not in self.name2idx:
                continue
            cls_id = self.name2idx[label["name"]]
            x_min = label["bndbox"]["xmin"]
            y_min = label["bndbox"]["ymin"]
            x_max = label["bndbox"]["xmax"]
            y_max = label["bndbox"]["ymax"]
            box_w, box_h = x_max - x_min, y_max - y_min
            ann_info = {
                "id": 0,
                "image_id": img_id,
                "category_id": cls_id+self.class_start_index,
                "bbox": [x_min, y_min, box_w, box_h],
                "iscrowd": 0,
                "area": box_w * box_h,
                "segmentation": [],  # TODO: 实现分割信息
            }
            self.coco_annotations[split].append(ann_info)
        return file_labels["object"]


def voc2coco(
        img_dir: PosixPath, dst_dir: PosixPath, classes: dict | str,
        lbl_dir: PosixPath = None, img_idx: int = 0, ann_idx: int = 0,
        use_link: bool = False, split: str = 'train', year: str = "",
        class_start_index: int = 0
        ) -> None:
    lbl2coco = VOC2COCO(img_dir, lbl_dir, dst_dir, classes, use_link=use_link, split=split, img_idx=img_idx, ann_idx=ann_idx, year=year, class_start_index=class_start_index)
    lbl2coco()
