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
from codeUtils.tools.font_config import colorstr
from codeUtils.tools.tqdm_conf import BATCH_KEY, START_KEY, END_KEY


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


class Labelme2COCO(object):
    """Convert labelme format to COCO format.
    
    Args:
        img_dir (str): labelme format img file directory.
        dst_dir (str): coco format file directory.
        lbl_dir (str): labelme format label file directory. 默认与img_dir相同.
        classes (str): yolo classes.txt file path.
        use_link (bool): whether to use symbolic link to save images.
        split (str): data split, 'train', 'val', 'test'.
        year (str): year of dataset.
        class_start_index (int): start index of class id. default is 0. lt 2.
    """

    def __init__(
            self, img_dir: str = None, lbl_dir: str = None, dst_dir: str = None, classes: str = None, 
            use_link: bool = False, split: str = 'train', 
            year: str = None, class_start_index: int = 0
        ):
        assert class_start_index < 2, 'class_start_index should be lt 2.'
        
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir) if lbl_dir is not None else self.img_dir
        self.dst_dir = Path(dst_dir)
        self.classes = classes
        self.use_link = use_link
        self.split = split
        self.year = year if year is not None else ""
        self.coco_images = dict()
        self.coco_annotations = dict()
        self.class_start_index = int(class_start_index)
    
    def load_classes(self):
        names = read_txt(self.classes) if isinstance(self.classes, str) else self.classes
        self.names = {i: name for i, name in enumerate(names)}
        self.name2idx = {name: i for i, name in self.names.items()}
        self.coco_images = {self.split: []}
        self.coco_annotations = {self.split: []}

    def load_items(self):
        for img_file in self.img_dir.iterdir():
            if img_file.suffix == ".json" or img_file.stem.startswith('.'):
                continue
            lbl_file = self.lbl_dir / f"{img_file.stem}.json"
            yield img_file, lbl_file, self.split

    def coco_prepare(self, img_file: Path, lbl_file: Path, split: str, img_id: int):
        """coco数据集整备

        :param Path img_file: 图片路径
        :param Path lbl_file: 标签路径
        :param str split: 数据集划分(不包含年份)
        :param int img_id: 图片id
        :raises ValueError: 标签格式错误
        """

        # 给COCO添加图片信息
        src_img = cv.imread(str(img_file))
        img_h, img_w = src_img.shape[:2]
        img_info = {
            'id': img_id,
            'file_name': str(self.dst_dir / f"{split}{self.year}" / img_file.name),
            'height': img_h,
            'width': img_w,
        }
        self.coco_images[split].append(img_info)

        # 给COCO添加标注信息
        file_labels = None
        for _ in range(3):
            file_labels = parser_json(lbl_file)
            if file_labels is not None:
                break
            
        if file_labels is None:
            file_labels = {"shapes": []}
        
        for label in file_labels["shapes"]:
            if label["label"] not in self.name2idx:
                continue
            cls_id = self.name2idx[label["label"]]  # 类别id
            [x_min, y_min], [x_max, y_max] = label["points"]
            box_w, box_h = x_max - x_min, y_max - y_min
            ann_info = {
                "id": 0,
                "image_id": img_id,
                "category_id": cls_id+self.class_start_index,
                "bbox": [x_min, y_min, box_w, box_h],
                "iscrowd": 0,
                "area": box_w * box_h,
                "segmentation": [],
            }
            self.coco_annotations[split].append(ann_info)
        
        # 保存图片
        img_link = self.dst_dir / f"{split}{self.year}" / img_file.name
        img_link.parent.mkdir(exist_ok=True, parents=True)
        if self.use_link:
            if not img_link.exists():
                img_link.symlink_to(img_file)
        else:
            shutil.copy(img_file, img_link)
        return len(file_labels)

    def anno_id_modify(self, start_index: int = 0):
        for split in self.coco_annotations:
            for ann in self.coco_annotations[split]:
                ann['id'] = start_index
                start_index += 1
    
    def save_coco_json(self, anno_file: Path, split: str):
        coco_info = {
            "description": f"COCO {self.year} Dataset",
            "version": "1.0",
            "year": self.year,
            "contributor": "firstelfin",
            "date_created": time.strftime("%Y/%m/%d", time.localtime()),
        }
        coco_images = self.coco_images[split]
        coco_annotations = self.coco_annotations[split]
        coco_dict = {
            "info": coco_info,
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [
                {
                    "id": i+self.class_start_index, 
                    "name": name, 
                    "supercategory": name
                } for i, name in self.names.items()
            ],
        }
        save_json(anno_file, coco_dict)
        logger.info(f"save {split} coco json file {anno_file} success.")

    def __call__(self, image_index: int = 0, anno_index: int = 0, *args, **kwds):
        
        res = []
        with ThreadPoolExecutor() as executor:
            for img_file, lbl_file, split in self.load_items():
                convert_res = executor.submit(self.coco_prepare, img_file, lbl_file, split, image_index)
                image_index += 1
                res.append(convert_res)

            # 等待所有任务完成
            labelme_bar = tqdm(as_completed(res), total=len(res), desc='yolo2coco')
            for convert_res in labelme_bar:
                convert_res.result()
                labelme_bar.set_postfix({'image_index': image_index})

        # 保存COCO格式数据集
        self.anno_id_modify(start_index=anno_index)
        anno_dir = self.dst_dir / 'annotations'
        anno_dir.mkdir(exist_ok=True, parents=True)

        for split in self.coco_annotations:
            anno_file = anno_dir / f"{split}{self.year}.json"
            self.save_coco_json(anno_file=anno_file, split=split)


def labelme2coco(
        img_dir: PosixPath, dst_dir: PosixPath, classes: dict | str, 
        lbl_dir: PosixPath = None, img_idx: int = 0, ann_idx: int = 0, 
        use_link: bool = False, split: str = 'train', year: str = "", 
        class_start_index: int = 0
        ) -> None:
    lbl2coco = Labelme2COCO(img_dir, lbl_dir, dst_dir, classes, use_link=use_link, split=split, year=year, class_start_index=class_start_index)
    lbl2coco(image_index=img_idx, anno_index=ann_idx)


def labelme2industai(src_dir: PosixPath, dst_dir: PosixPath, classes: dict) -> None:
    pass

