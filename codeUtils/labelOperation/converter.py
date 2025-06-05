#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   converter.py
@Time    :   2025/04/30 11:58:06
@Author  :   firstElfin 
@Version :   0.1.11.7
@Desc    :   数据转换基类
'''

import os
import time
import shutil
import cv2 as cv
from typing import Literal
from loguru import logger
from tqdm import tqdm
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from codeUtils.labelOperation.readLabel import read_txt, parser_json
from codeUtils.labelOperation.saveLabel import save_json, save_labelme_label
from codeUtils.tools.futureConf import FutureBar


# @dataclass
# class StandardData(object):
#     version: str = field(default='0.5.6')
#     flags: dict = field(default_factory=dict)
#     shapes: list = field(default_factory=list)
#     imagePath: str = field(default='')
#     imageData: bytes = field(default=None)
#     imageHeight: int = field(default=0)
#     imageWidth: int = field(default=0)

#     @staticmethod
#     def from_dict(data_dict):
#         return StandardData(
#             version=data_dict.get('version', '0.5.6'),
#             flags=data_dict.get('flags', {}),
#             shapes=data_dict.get('shapes', []),
#             imagePath=data_dict.get('imagePath', ''),
#             imageData=data_dict.get('imageData', None),
#             imageHeight=data_dict.get('imageHeight', 0),
#             imageWidth=data_dict.get('imageWidth', 0)
#         )
    
#     def to_dict(self):
#         return {
#             'version': self.version,
#             'flags': self.flags,
#             'shapes': self.shapes,
#             'imagePath': self.imagePath,
#             'imageData': self.imageData,
#             'imageHeight': self.imageHeight,
#             'imageWidth': self.imageWidth
#         }


# class BaseConverter(object):
#     """标注数据转换基类

#     定义数据转换的中间态结构, 输入输出根据不同格式设计接口

#     :param _type_ object: _description_
#     """

#     def __init__(self):
#         pass

#     @staticmethod
#     def from_labelme(args: dict):
#         return BaseConverter()

#     @staticmethod
#     def from_coco(args: dict):
#         return BaseConverter()

#     @staticmethod
#     def from_voc(args: dict):
#         return BaseConverter()
    
#     @staticmethod
#     def standardize(lbl_objs: list | dict) -> dict:
#         """标签数据实例标注数据标准化

#         :param list | dict lbl_objs: 标签数据实例
#         :return dict: 标准化后的标签数据实例(labelme的格式)
#         """
#         if isinstance(lbl_objs, list):
#             pass
#         elif isinstance(lbl_objs, dict) and 'shapes' in lbl_objs:
#             return lbl_objs
#         elif isinstance(lbl_objs, dict) and 'object' in lbl_objs:
#             return 
#         else:
#             raise TypeError("Invalid label data format. Only support yolo, voc, labelme, coco format.")

        
#     pass


# class NamesConverter(object):
#     pass


class ToCOCO(ABC):
    """Convert all format to COCO format.

    Args:
        img_dir (str|list): Pascal VOC format img file directory or list of directories.
        lbl_dir (str): Pascal VOC format label file directory or list of directories.
        dst_dir (str): coco format file directory.
        classes (str): yolo classes.txt file path.
        use_link (bool): whether to use symbolic link to save images.
        split (str): data split, 'train', 'val', 'test'.
        year (str): year of dataset.
        class_start_index (int): start index of class id. default is 0. lt 2.

    Methods:
        _post_init: 加载names文件(类yolo classes.txt文件), 并初始化name2idx、\
            coco_images、coco_annotations格式化对象.
        load_items: 加载图片和标签文件路径. 返回生成器, 每次生成(img_file, lbl_file, split).
        coco_prepare: 给COCO添加图片信息, 添加实例相关信息, 实例编码需要重新编码.
        anno_id_modify: 修改标注id.
        save_coco_json: 保存COCO格式数据集.
    """

    def __init__(
            self, img_dir: str = None, lbl_dir: str = None, dst_dir: str = None, classes: str = None, 
            use_link: bool = False, split: str = 'train', img_idx: int = 0, ann_idx: int = 0, 
            year: str = None, class_start_index: Literal[0, 1] = 0
        ):
        
        self.img_dir = [Path(p) for p in img_dir] if isinstance(img_dir, list) else Path(img_dir)
        if lbl_dir is None:
            self.lbl_dir = self.img_dir
        elif isinstance(lbl_dir, list):
            self.lbl_dir = [Path(p) for p in lbl_dir]
        else:
            self.lbl_dir = Path(lbl_dir)
        self.dst_dir = Path(dst_dir)
        self.classes = classes
        self.use_link = use_link
        self.split = split
        self.start_img_idx = img_idx
        self.img_idx = img_idx
        self.ann_idx = ann_idx
        self.year = year if year is not None else ""
        self.names = dict()
        self.name2idx = dict()
        self.coco_images = dict()
        self.coco_annotations = dict()
        self.class_start_index = int(class_start_index)
        self._post_init()

    def _post_init(self):
        """加载names文件, 并生成name2idx、coco_images、coco_annotations格式化对象"""

        names = read_txt(self.classes) if isinstance(self.classes, str) else self.classes
        self.names = {i: name for i, name in enumerate(names)}
        self.name2idx = {name: i for i, name in self.names.items()}
        if isinstance(self.split, str):
            self.coco_images = {self.split: []}
            self.coco_annotations = {self.split: []}
        elif isinstance(self.split, list):
            for split in self.split:
                self.coco_images[split] = []
                self.coco_annotations[split] = []
        else:
            raise ValueError("Invalid split type.")

    def load_items(self, suffix: str = ".json"):
        """加载并发处理函数的参数, 参数分两部分回传, 分别为args和kwargs.
        
        load_items函数返回禁止使用生成器!

        :param str suffix: 文件后缀, defaults to ".json", 默认处理labelme格式的标签
        """

        res = []
        if isinstance(self.img_dir, PosixPath):
            for img_file in self.img_dir.iterdir():
                if img_file.suffix == suffix or img_file.stem.startswith('.'):
                    continue
                lbl_file = self.lbl_dir / f"{img_file.stem}{suffix}"
                self.img_idx += 1
                res.append(([img_file, lbl_file, self.split, self.img_idx], {}))
        else:
            for img_dir in self.img_dir:
                for img_file in img_dir.iterdir():
                    if img_file.suffix == suffix or img_file.stem.startswith('.'):
                        continue
                    lbl_file = self.lbl_dir / img_dir.name / f"{img_file.stem}{suffix}"
                    self.img_idx += 1
                    res.append(([img_file, lbl_file, self.split, self.img_idx], {}))
        return res

    @abstractmethod
    def instance_prepare(self, lbl_file: Path, img_id: int, split: str) -> list:
        """COCO实例整备

        :param Path lbl_file: 标签文件路径
        :param int img_id: 图片id
        :param str split: 数据集划分(不包含年份)
        :return list: 实例列表, 实例组织格式无要求
        
        ## 标准coco中转格式, Example:

        ```
        standard_instance = {
            "id": 0,
            "image_id": img_id,
            "category_id": cls_id+self.class_start_index,
            "bbox": [x_min, y_min, box_w, box_h],
            "iscrowd": 0,
            "area": box_w * box_h,
            "segmentation": [],
        }
        ```
        """
        file_labels = None
        for _ in range(3):
            file_labels = parser_json(lbl_file)
            if file_labels is not None:
                break
            
        if file_labels is None:
            file_labels = {"shapes": []}
        
        for label in file_labels["shapes"]:  # 
            if label["label"] not in self.name2idx:
                continue
            cls_id = self.name2idx[label["label"]]  # 类别id  TODO: 归一化操作
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
        return file_labels["shapes"]

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
        file_labels = self.instance_prepare(lbl_file, img_id, split)
        
        # 保存图片
        img_link = self.dst_dir / f"{split}{self.year}" / img_file.name
        img_link.parent.mkdir(exist_ok=True, parents=True)
        if self.use_link:
            if not img_link.exists():
                img_link.symlink_to(img_file)
        else:
            shutil.copy(img_file, img_link)
        return len(file_labels)

    def anno_id_modify(self):
        """修改标注id

        :param int start_index: 起始索引, defaults to 0
        """
        for split in self.coco_annotations:
            for ann in self.coco_annotations[split]:
                self.ann_idx += 1
                ann['id'] = self.ann_idx

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
                    "id": i+self.class_start_index,  # 类别id, 起始索引由class_start_index指定
                    "name": name, 
                    "supercategory": name
                } for i, name in self.names.items()
            ],
        }
        save_json(anno_file, coco_dict)
        logger.info(f"save {split} coco json file {anno_file} success. {self.img_idx} images, {self.ann_idx} annotations.")

    def __call__(self, *args, **kwargs):

        all_async_items = self.load_items()
        cpu_num = max(os.cpu_count() // 2, 6)
        exec_bar = FutureBar(max_workers=cpu_num, timeout=20, desc=self.__class__.__name__)
        exec_bar(self.coco_prepare, all_async_items, total=self.img_idx-self.start_img_idx)

        # 保存COCO格式数据集
        self.anno_id_modify()
        anno_dir = self.dst_dir / 'annotations'
        anno_dir.mkdir(exist_ok=True, parents=True)

        for split in self.coco_annotations:
            anno_file = anno_dir / f"{split}{self.year}.json"
            self.save_coco_json(anno_file=anno_file, split=split)


class COCOToAll(ABC):

    def __init__(self, img_dir: str, lbl_file: str, dst_dir: str):
        self.img_dir = Path(img_dir)
        self.lbl_file = Path(lbl_file)
        self.dst_dir = Path(dst_dir)
        self.dst_dir.mkdir(exist_ok=True, parents=True)
        self.coco_names = None
        self.coco_instances = dict()
        self._post_init()
        super().__init__()

    def _post_init(self):
        # 加载COCO数据集
        self.coco_dict = parser_json(self.lbl_file)
        if self.coco_dict is None:
            raise ValueError("Invalid COCO format.")
        self.coco_names = {cat['id']: cat['name'] for cat in self.coco_dict['categories']}
        coco_init_bar = tqdm(self.coco_dict['images'], desc='COCO init', colour='#CD8500')
        for img_info in coco_init_bar:
            self.coco_instances[img_info['file_name']] = {
                'imagePath': img_info['file_name'],
                'imageHeight': img_info['height'],
                'imageWidth': img_info['width'],
                'shapes': []
            }
            for obj in self.coco_dict['annotations']:
                if obj['image_id'] != img_info['id']:
                    continue
                seg_points_num = sum([sum(obj['segmentation'][i]) for i in range(len(obj['segmentation']))]) // 2
                is_rectangle = seg_points_num <= 2
                points = []
                if not is_rectangle:
                    for i in range(len(obj['segmentation'])):
                        seg_i = [
                            [obj['segmentation'][i][j], obj['segmentation'][i][j+1]]
                            for j in range(0, len(obj['segmentation'][i]), 2)
                        ]
                        if seg_i[0] != seg_i[-1]:
                            seg_i.append(seg_i[0])
                        points += seg_i
                else:
                    points = [
                        [obj['bbox'][0], obj['bbox'][1]],
                        [obj['bbox'][0]+obj['bbox'][2], obj['bbox'][1]+obj['bbox'][3]],
                    ]
                self.coco_instances[img_info['file_name']]['shapes'].append({
                    "label": self.coco_names[obj['category_id']],
                    "points": points,
                    "group_id": None,
                    "shape_type": "rectangle" if is_rectangle else "polygon",
                    "flags": {}
                })

    @abstractmethod
    def save_label(self, img_path: str, labelme_dict: dict):
        save_label_file = self.dst_dir / f"{Path(img_path).stem}.json"
        save_labelme_label(save_label_file, labelme_dict)

    def __call__(self, *args, **kwargs):
        cpu_num = max(os.cpu_count() // 2, 6)
        error_list = []
        with ThreadPoolExecutor(max_workers=cpu_num) as executor:
            
            res = []
            for img_path, labelme_dict in self.coco_instances.items():
                res.append(executor.submit(self.save_label, img_path, labelme_dict, kwargs.get("extra_keys", [])))
            
            call_bar = tqdm(
                as_completed(res),
                total=len(res),
                colour='#CD8500',
                desc=self.__class__.__name__
            )
            for cb in call_bar:
                try:
                    cb.result(timeout=60)
                except Exception as e:
                    error_list.append((img_path, e))
                call_bar.set_postfix({'errorNum': len(error_list)})

            if error_list:
                logger.error(f"Retrying {len(error_list)} failed tasks.")
                new_res = [
                    executor.submit(self.save_label, fail.img_path, fail.labelme_dict, fail.extra_keys) for fail in error_list
                ]
                for fail_res in as_completed(new_res):
                    try:
                        fail_res.result(timeout=60)
                    except Exception as e:
                        logger.error(f"Failed to save {img_path}: {e}")




