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
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from codeUtils.labelOperation.readLabel import read_voc, read_txt
from codeUtils.tools.tqdm_conf import BATCH_KEY, START_KEY, END_KEY
from codeUtils.tools.font_config import colorstr


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

