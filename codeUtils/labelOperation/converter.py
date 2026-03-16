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
import math
import shutil
import cv2 as cv
from enum import Enum
from typing import Literal, Callable, Optional, Any, List
from loguru import logger
from tqdm import tqdm
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from codeUtils.labelOperation.readLabel import read_txt, read_yolo, read_voc, read_json, read_yaml
from codeUtils.labelOperation.saveLabel import save_json, save_labelme_label, save_yolo_label, save_voc_label
from codeUtils.tools import load_img, FutureBar, segmentation_to_polygons, CPU_KERNEL_NUM, IMG_EXTENSIONS


def load_names_dict(names: str | dict) -> dict:
    """读取类别名称文件"""

    if isinstance(names, str):
        names_suffix = names.split(".")[-1].lower()
        if names_suffix == "txt":
            names = {i: v for i, v in enumerate(read_txt(names))}
        elif names_suffix in ["yml", "yaml"]:
            names = read_yaml(names)['names']
            if isinstance(names, list):
                names = {i: v for i, v in enumerate(names)}
        else:
            raise ValueError("names must be a str of txt or yml file.")
    elif isinstance(names, dict) and not names:
        return names

    assert isinstance(names, dict), "names must be a non-empty dict or a str."
    return names
    

@dataclass(slots=True)
class ShapeInstance(object):
    """实例数据结构
    
    Attributes:
        label (str|int): 实例标签
        points (list[list[int|float]]): 实例坐标点
        shape_type (str): 实例类型, 如"polygon", "rectangle", "rotation"
        flags (dict): 实例标注属性
        score (float): 实例分数(置信度分数)
    
    rotation:  # 使用标准的DOTA格式
        points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  # 顺时针方向矩形四个点的坐标
        shape_type: "rotation"
    """

    label: str | int
    points: list[list[int|float]] = field(default_factory=list)
    shape_type: Literal["polygon", "rectangle", "rotation"] = field(default="rectangle")
    flags: dict = field(default_factory=dict)
    score: float = field(default=1.0)


@dataclass(slots=True)
class LabelmeData(object):
    """标准化转换中间态数据结构, 以labelme格式为准

    Attributes:
        shapes (list[ShapeInstance]): 标注实例列表
        imagePath (str): 图片路径
        imageHeight (int): 图片高度
        imageWidth (int): 图片宽度
    """

    shapes: list[ShapeInstance] = field(default_factory=list)
    imagePath: str = field(default='')
    imageHeight: int = field(default=0)
    imageWidth: int = field(default=0)


class DetConverter(ABC):
    """检测、分割数据转换基类

    定义LabelmeData转为其他数据类的接口, 并提供转换的统一入口.

    :param list[Path]|Path lbl_dir: 源标签地址列表
    :param Path dst_dir: 保存地址
    :param list[Path]|Path img_dir: 源图片地址列表
    :param str split: 子集的划分标识
    :param int start_img_idx: 图片开始的索引
    :param int start_ann_idx: 标签开始的索引
    :param str year: 年份
    :param int class_start_index: 类别开始的索引
    :param list classes: 类别列表
    :param dict names: 索引和类别的map字典
    :param float min_area: 实例的最小面积参数
    
    处理流程
    ============
    
    ```mermaid
    graph TD
        A[接收 LabelmeData 对象] --> B{判断对象类型}
        B -- 标注文件独立 --> C[直接输出新的标签文件]
        B -- 标注文件耦合 --> D[转换为中间状态并缓存]
        D --> E[等待统一保存]
        F[调用 save_all 方法] --> G[遍历缓存的耦合对象]
        G --> H[逐个保存为目标格式文件]
    ```

    标签读取
    ============

    - 支持读取labelme格式的标注文件
    - 支持读取yolo格式的标注文件
    - 支持读取voc格式的标注文件
    - 支持读取coco格式的标注文件

    转标签:

    - labelme, yolo, voc, COCO

    读标签:

    - labelme, yolo, voc, COCO
    
    """

    def __init__(self, lbl_dir: list, dst_dir: Path, img_dir: list = [], split: str = 'train',
                 start_img_idx: int = 0, start_ann_idx: int = 0, year: str = '',
                 class_start_index: Literal[0, 1] = 0, classes: list = [], names: dict = {},
                 min_area: float = 1.0, ignore_img: bool = True, use_link: bool = False, **kwargs):
        """初始化转换器"""
        if isinstance(lbl_dir, (str, Path)):
            lbl_dir = [lbl_dir]
        if isinstance(img_dir, (str, Path)):
            img_dir = [img_dir]
        self.lbl_dir = [Path(ld) for ld in lbl_dir]
        self.img_dir = [Path(imd) for imd in img_dir]
        assert len(self.img_dir) == len(self.lbl_dir), \
            f"The number of images[{len(self.img_dir)}] and labels[{len(self.lbl_dir)}] is inconsistent."

        self.dst_dir = dst_dir
        self.split = split
        self.img_idx_start = start_img_idx
        self.ann_idx_start = start_ann_idx
        self.year = year
        self.class_start_index = class_start_index
        self.classes = classes
        self.names = names if names else {i: self.classes[i] for i in range(len(self.classes))}
        self.name2id = {name: i+self.class_start_index for i, name in self.names.items()}
        self.min_area = min_area
        self.ignore_img = ignore_img
        self.use_link = use_link
        self.suffix = ".json"
        self.gather = dict()  # 耦合数据缓存
        self.read_lbl_func = self.from_labelme  # 子类需要按照需求改变
        self.save_lbl_func = self.to_labelme  # 子类需要按照需求改变
        self.dst_dir.mkdir(exist_ok=True, parents=True)

    def __call__(self, *args, **kwargs):
        return self.convert(*args, **kwargs)

    # 角度整备
    def direction_prepare(
            self, points: list[list[int|float]],
            radiance: bool = True,
            mode: Literal["semantic_refer", "long_edge_refer", "opencv_refer"] = "semantic_refer",
        ) -> dict:
        u"""从标准DOTA格式的四点坐标中获取顺时针方向的角度.
            - DOTA, YOLO都直接支持原始四个点的坐标, 无需转换, 无需角度计算

        :param list[list[int | float]] points: 四个坐标点
        :param bool radiance: 是否返回弧度制的角度, defaults to True
        :param str mode: 角度计算模式, defaults to "semantic_refer"

        参数使用说明
        ==============

            - opencv_refer: 是否角度在第四象限内, 默认为False, 角度取值范围是(0, $\pi$/2]
            - long_edge_refer: 是否使用长边作为参考, 默认为False, 角度取值范围是[0, $\pi$)
            - semantic_refer: 是否使用语义方向(point1 -> point2)作为参考, 默认为False, 角度取值范围是[0, 2$\pi$)
        
        :return dict: 返回旋转框的robndbox
        """

        robndbox = []
        direction = 0.0

        [x1, y1], [x2, y2], [x3, y3], [x4, y4] = points
        cx = (x1 + x3 + x2 + x4) / 4.0  # 矩形中心点x
        cy = (y1 + y3 + y2 + y4) / 4.0  # 矩形中心点y
        pri_vec = [x2 - x1, y2 - y1]  # 矩形左上角到右上角的向量
        sec_vec = [x3 - x2, y3 - y2]  # 矩形右上角到右下角的向量
        width = (pri_vec[0] ** 2 + pri_vec[1] ** 2) ** 0.5  # 矩形宽度
        height = (sec_vec[0] ** 2 + sec_vec[1] ** 2) ** 0.5  # 矩形高度
        pri_angle = (math.atan2(pri_vec[1], pri_vec[0]) + math.pi) % math.pi  # 宽与水平向右的夹角
        sec_angle = (math.atan2(sec_vec[1], sec_vec[0]) + math.pi) % math.pi  # 高与水平向右的夹角

        if mode == "semantic_refer":  # 矩形左上角到右上角的角度
            direction = (math.atan2(-pri_vec[1], pri_vec[0]) + 2 * math.pi) % (2 * math.pi)
            robndbox = [cx, cy, width, height, direction]
        elif mode == "long_edge_refer":  # 长边与水平向右的夹角
            direction = pri_angle if width >= height else sec_angle
            robndbox = [cx, cy, width, height, direction]
        elif mode == "opencv_refer":  # 限制第四象限 (opencv、YOLO)
            direction = pri_angle if pri_angle <= (math.pi / 2) else sec_angle  # 取最小角度, 非负
            if direction == sec_angle:
                width, height = height, width  # 指定角度方向为宽所在的边
            robndbox = [cx, cy, width, height, direction]
        else:
            raise ValueError(f"Invalid mode: {mode}.")


        robndbox_dict = {
            "robndbox": robndbox,
            "direction": direction,
        }
        return robndbox_dict

    # 转换标签

    def pre_validate(
            self,
            lbmd: LabelmeData | None = None,
            lbl_file: str | Path | None = None, 
            img_file: str | Path | None = None,
            save_dir: str | Path | None = None, 
            **kwargs,
        ) -> tuple[LabelmeData, Path, Path, Path]:
        """数据转换前的入参检查.

        :param LabelmeData | None lbmd: 标准标注数据对象, defaults to None, Optional
        :param str | Path | None lbl_file: 标注文件路径, defaults to None
        :param str | Path | None img_file: 图像文件路径, defaults to None, Optional
        :param str | Path | None save_dir: 保存目录, defaults to None
        :return tuple[LabelmeData, Path, Path, Path]: 标注文件的LabelmeData对象, 标注文件路径, 图像文件路径, 保存目录路径
        """
        if lbl_file is None:
            raise ValueError("lbl_file is None. Expected a label file path.")
        if save_dir is None:
            raise ValueError("save_dir is None. Expected a save directory path.")
        save_dir = Path(save_dir)
        lbl_file = Path(lbl_file) if lbl_file is not None else Path("")
        img_file = Path(img_file) if img_file is not None else Path("")
        
        # lbmd是None, 读取标签文件
        if lbmd is None:
            lbmd = self.read_lbl_func(lbl_file, img_file, **kwargs)[0]
        if lbmd is not None and not isinstance(lbmd, LabelmeData):
            raise TypeError(f"Invalid type of lbmd. Expected LabelmeData, but got {type(lbmd)}.")
        return lbmd, lbl_file, img_file, save_dir
    
    def to_labelme(
            self, lbmd: LabelmeData | None = None, 
            lbl_file: str | Path | None = None, 
            img_file: str | Path | None = None,
            save_dir: str | Path | None = None
        ) -> None:
        """转为labelme格式的标准输出接口.

        :param LabelmeData | None lbmd: 标准的LabelmeData对象, defaults to None
        :param str | Path | None lbl_file: 源标注文件, defaults to None
        :param str | Path | None img_file: 源图像文件, defaults to None
        :param str | Path | None save_dir: 保存目录, defaults to None
        :raises ValueError: 输入参数错误
        :raises FileExistsError: 图像文件不存在
        """

        lbmd, lbl_file, img_file, save_dir = self.pre_validate(lbmd, lbl_file, img_file, save_dir)
        
        labelme_update = {
            "version": "4.5.6",
            "flags": {},
            "imageData": None,
        }
        if lbmd is not None and lbmd.imagePath:
            save_file_path = save_dir / f"{Path(lbmd.imagePath).stem}.json"
        else:
            save_file_path = save_dir / f"{lbl_file.stem}.json"

        # 基于LabelmeData对象转换
        if isinstance(lbmd, LabelmeData):
            lbl_dict = asdict(lbmd)
            lbl_dict.update(labelme_update)
            save_labelme_label(save_file_path, lbl_dict)
        else:
            # lbmd是None, 保存空实例文件, 图像路径与图像size需要根据实际情况填写
            src_img = load_img(img_file)
            if img_file is None:
                raise ValueError(f"img_file is None. Expected a image file path. got lbl_file: {lbl_file}.")
            elif src_img is None:
                raise FileExistsError(f"Failed to load image file {img_file}.")
            img_h, img_w = src_img.shape[:2]
            labelme_update.update({"shapes": [], "imagePath": Path(img_file).name, "imageHeight": img_h, "imageWidth": img_w})
            save_labelme_label(save_file_path, labelme_update)

    def to_yolo(
            self, lbmd: LabelmeData | None = None, 
            lbl_file: str | Path | None = None, 
            img_file: str | Path | None = None,
            save_dir: str | Path | None = None,
        ) -> None:
        """转为yolo格式的标准输出接口.

        :param LabelmeData | None lbmd: 标准的LabelmeData对象, defaults to None
        :param str | Path | None lbl_file: 源标注文件, defaults to None
        :param str | Path | None img_file: 源图像文件, defaults to None
        :param str | Path | None save_dir: 保存目录, defaults to None
        :raises ValueError: 输入参数错误
        """
        
        lbmd, lbl_file, img_file, save_dir = self.pre_validate(lbmd, lbl_file, img_file, save_dir)
        if lbmd is not None and lbmd.imagePath:
            save_file_path = str(save_dir / f"{Path(lbmd.imagePath).stem}.txt")
        else:
            save_file_path = str(save_dir / f"{lbl_file.stem}.txt")
        
        # 基于LabelmeData对象转换
        lbl_list = []
        if isinstance(lbmd, LabelmeData):
            img_h, img_w = lbmd.imageHeight, lbmd.imageWidth
            for shape in lbmd.shapes:  # shape: ShapeInstance
                if shape.shape_type == "polygon":
                    lbl_list.append(self.name2id[shape.label])
                    for i in range(len(shape.points) // 2):
                        lbl_list.extend([shape.points[i][0] / img_w, shape.points[i][1] / img_h])
                    lbl_list.append(shape.score)
                elif shape.shape_type == "rectangle":
                    x1, y1 = shape.points[0][0] / img_w, shape.points[0][1] / img_h
                    x2, y2 = shape.points[1][0] / img_w, shape.points[1][1] / img_h
                    w, h = x2 - x1, y2 - y1
                    x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2
                    lbl_list.append([self.name2id[shape.label], x_c, y_c, w, h, shape.score])
                else:
                    raise ValueError(f"Unsupported shape type: {shape.shape_type}")
        save_yolo_label(save_file_path, lbl_list)

    def to_voc(
            self, lbmd: LabelmeData | None = None, 
            lbl_file: str | Path | None = None, 
            img_file: str | Path | None = None,
            save_dir: str | Path | None = None,
        ) -> None:
        """LabelmeData对象转为voc格式的标准输出接口.

        :param LabelmeData | None lbmd: 标准的LabelmeData对象, defaults to None
        :param str | Path | None lbl_file: 源标注文件, defaults to None
        :param str | Path | None img_file: 源图像文件, defaults to None
        :param str | Path | None save_dir: 保存目录, defaults to None
        """

        lbmd, lbl_file, img_file, save_dir = self.pre_validate(lbmd, lbl_file, img_file, save_dir)
        if lbmd is not None and lbmd.imagePath:
            save_file_path = str(save_dir / f"{Path(lbmd.imagePath).stem}.xml")
        else:
            save_file_path = str(save_dir / f"{lbl_file.stem}.xml")
        
        # 基于LabelmeData对象转换
        voc_dict = {
            "folder": img_file.parent.name if isinstance(img_file, Path) else "",
            "filename": img_file.name if isinstance(img_file, Path) else "",
            "path": str(img_file) if isinstance(img_file, Path) else "",
            "source": {"database": "Unknown"},
            "size": {"width": 0, "height": 0, "depth": 0},
            "segmented": 0,
            "object": []
        }
        if isinstance(lbmd, LabelmeData):
            voc_dict["size"] = {"width": lbmd.imageWidth, "height": lbmd.imageHeight, "depth": 3}
            for shape in lbmd.shapes:  # shape: ShapeInstance
                if shape.shape_type == "polygon":
                    x_list = [p[0] for p in shape.points]
                    y_list = [p[1] for p in shape.points]
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    obj_dict = {
                        "name": shape.label,
                        "pose": "Unspecified",
                        "truncated": 0,
                        "difficult": 0,
                        "bndbox": {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max},
                        # TODO: 增加分割表示 RLX格式, 添加score字段到保存内
                        "score": shape.score,
                    }
                    voc_dict["object"].append(obj_dict)
                elif shape.shape_type == "rectangle":
                    obj_dict = {
                        "name": shape.label,
                        "pose": "Unspecified",
                        "truncated": 0,
                        "difficult": 0,
                        "bndbox": {
                            "xmin": shape.points[0][0],
                            "ymin": shape.points[0][1],
                            "xmax": shape.points[1][0],
                            "ymax": shape.points[1][1],
                        },
                        "confidence": shape.score,
                    }
                    voc_dict["object"].append(obj_dict)
                else:
                    raise ValueError(f"Unsupported shape type: {shape.shape_type}")
        else:
            # lbmd是None, 保存空实例文件, 图像路径与图像size需要根据实际情况填写
            src_img = load_img(img_file)
            if img_file is None:
                raise ValueError(f"img_file is None. Expected a image file path. got lbl_file: {lbl_file}.")
            elif src_img is None:
                raise FileExistsError(f"Failed to load image file {img_file}.")
            img_h, img_w = src_img.shape[:2]
            voc_dict["size"] = {"width": img_w, "height": img_h, "depth": 3}
        
        save_voc_label(save_file_path, voc_dict)

    def to_coco(
            self, lbmd: LabelmeData | None = None, 
            lbl_file: str | Path | None = None, 
            img_file: str | Path | None = None,
            save_dir: str | Path | None = None,
            **kwargs,
        ) -> None:
        """将标准标注对象LabelmeData对象转换为COCO的数据标注. COCO转换时需要对数据进行聚合, 所以此类转换统一归档后处理.

        :param LabelmeData lbmd: 标准的LabelmeData对象, defaults to None, optional
        :param str|Path|None lbl_file: 标注文件路径, defaults to None, optional
        :param str|Path|None img_file: 图像文件路径, defaults to None, optional
        :param str|Path|None save_dir: 保存目录, defaults to None, optional
        :param split str: COCO数据集的子集类型, 常见为['train', 'test', 'val'], 默认为train
        """
        lbmd, lbl_file, img_file, save_dir = self.pre_validate(lbmd, lbl_file, img_file, save_dir)
        split = kwargs.get("split", "train")
        if split not in self.gather:
            self.gather[split] = []
        self.gather[split].append({
            "lbmd": lbmd,
            "lbl_file": lbl_file,
            "img_file": img_file,
            "save_dir": save_dir,
        })

    # 读取标签文件为LabelmeData对象
    
    @classmethod
    def from_labelme(cls, lbl_file: str | Path, img_file: str | Path = "", **kwargs) -> List[LabelmeData]:
        """从labelme标注文件读取数据, 并返回LabelmeData对象.(kwargs未使用)
        已经支持:
        1. 矩形标注, 转为points字段[2, 1, 2]
        2. 多边形标注, 转为points字段[n, 1, 2]

        :param str|Path lbl_file: labelme格式的标注文件路径
        :param str|Path img_file: 图像文件路径
        :return List[LabelmeData]: 标准化的LabelmeData对象
        """
        res = LabelmeData(imagePath=str(img_file), imageHeight=0, imageWidth=0, shapes=[])
        # 读取labelme格式的标注文件
        lbl_dict = read_json(lbl_file)
        if lbl_dict is None:
            if not Path(lbl_file).exists():
                src_img = load_img(img_file)
                if src_img is None:
                    raise FileExistsError(f"Failed to load image file {img_file}.")
                img_h, img_w = src_img.shape[:2]
                res.imageHeight = img_h
                res.imageWidth = img_w
            else:
                raise FileExistsError(f"Failed to load label file {lbl_file}.")
        else:
            res.shapes = [
                ShapeInstance(
                    label=shape["label"], 
                    points=shape["points"], 
                    shape_type=shape["shape_type"], 
                    flags=shape.get("flags", {}), 
                    score=shape.get("score", 1.0)
                ) for shape in lbl_dict.get("shapes", [])
            ]
            res.imageHeight = lbl_dict.get("imageHeight", 0)
            res.imageWidth = lbl_dict.get("imageWidth", 0)
        return [res]

    def from_yolo(self, lbl_file: str | Path, img_file: str | Path, **kwargs) -> List[LabelmeData]:
        """从yolo标注文件读取数据, 并返回LabelmeData对象. (kwargs未使用)
        已经支持:
        1. conf字段, LabelmeData记录为score字段
        2. 多边形标注, 转为points字段[n, 1, 2]
        3. 矩形标注, 转为points字段[2, 1, 2]

        :param lbl_file: yolo格式的标注文件路径
        :type lbl_file: str | Path
        :param img_file: 图像文件路径
        :type img_file: str | Path
        :return: 标准化的LabelmeData对象
        :rtype: List[LabelmeData]
        """
        
        # 初始化LabelmeData对象
        res = LabelmeData(imagePath=str(img_file), imageHeight=0, imageWidth=0, shapes=[])
        src_img = load_img(img_file)
        if src_img is None:
            raise FileExistsError(f"Failed to load image file {img_file}.")
        img_h, img_w = src_img.shape[:2]
        res.imageHeight = img_h
        res.imageWidth = img_w

        # 读取yolo格式的标注文件
        lbl_list = read_yolo(lbl_file)
        if lbl_list is None:
            lbl_list = []
        
        # 转换为LabelmeData对象
        for label, *points in lbl_list:
            label_name = self.names.get(label, str(label))
            if len(points) % 2 != 0:
                conf = points[-1]
                points = points[:-1]
            else:
                conf = 1.0
            if len(points) != 4:
                points = [[points[2*i]*img_w, points[2*i+1]*img_h] for i in range(len(points)//2)]
                shape_type = "polygon"
            else:
                cx = points[0] * img_w
                cy = points[1] * img_h
                w = points[2] * img_w
                h = points[3] * img_h
                points = [[cx-w/2, cy-h/2], [cx+w/2, cy+h/2]]
                shape_type = "rectangle"
            res.shapes.append(
                ShapeInstance(label=label_name, points=points, shape_type=shape_type, flags={}, score=conf)
            )
        return [res]
    
    def from_voc(self, lbl_file: str | Path, img_file: str | Path = "", **kwargs) -> List[LabelmeData]:
        """从voc标注文件读取数据, 并返回LabelmeData对象. (kwargs使用extra_keys参数)
        已经支持:
        1. 矩形标注, 转为points字段[2, 1, 2]
        2. 多边形标注, 转为points字段[n, 1, 2](暂未支持)

        :param lbl_file: voc格式的标注文件路径
        :type lbl_file: str | Path
        :param img_file: 图像文件路径, lbl_file加载不到图像尺寸时, 图像就必须要能够加载到
        :type img_file: str | Path
        :param list extra_keys: xml中对象需要自定义的字段
        :return: 标准化的LabelmeData对象
        :rtype: List[LabelmeData]
        """
        res = LabelmeData(imagePath=str(img_file), imageHeight=0, imageWidth=0, shapes=[])
        # 读取voc格式的标注文件
        voc_dict = read_voc(label_file=lbl_file, extra_keys=kwargs.get("extra_keys", []))
        if voc_dict is None:
            src_img = load_img(img_file)
            if src_img is None:
                raise FileExistsError(f"Failed to load image file {img_file}.")
            img_h, img_w = src_img.shape[:2]
            res.imageHeight = img_h
            res.imageWidth = img_w
            return [res]
        # 确保图像size不为0
        if voc_dict["size"]["width"] == 0 or voc_dict["size"]["height"] == 0:
            src_img = load_img(img_file)
            if src_img is None:
                raise FileExistsError(f"Failed to load image file {img_file}.")
            img_h, img_w = src_img.shape[:2]
            voc_dict["size"]["width"] = img_w
            voc_dict["size"]["height"] = img_h
        res.imageHeight = voc_dict["size"]["height"]
        res.imageWidth = voc_dict["size"]["width"]

        use_seg = kwargs.get("use_seg", False) and voc_dict.get("segmented")  # TODO: 支持Seg转换
        # 转换为LabelmeData对象
        if use_seg:
            raise NotImplementedError("Seg conversion is not supported yet.")
        else:
            for ins_obj in voc_dict.get("object", []):
                label_name: str = ins_obj.get("name")
                score: float = float(ins_obj.get("conf", 1.0))
                bbox: dict = ins_obj["bndbox"]
                xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
                x1, y1 = max(min(xmin, xmax), 0), max(min(ymin, ymax), 0)
                x2, y2 = min(max(xmin, xmax), res.imageWidth), min(max(ymin, ymax), res.imageHeight)
                res.shapes.append(
                    ShapeInstance(label=label_name, points=[[x1, y1], [x2, y2]], shape_type="rectangle", flags={}, score=score)
                )
        return [res]
    
    def from_coco(self, lbl_file: str | Path, img_file: str | Path = '', **kwargs) -> List[LabelmeData]:
        """coco数据预处理时, 会直接将对象处理为LabelmeData对象, 所以不需要再次处理.(kwargs未使用)
        :param str|Path lbl_file: coco标注文件(json)
        :param str|Path img_file: 图像的存储文件夹地址

        已经支持:
        1. 矩形标注, 转为points字段[2, 1, 2]
        2. 多边形标注, 转为points字段[n, 1, 2]. 未完全适应适应多边形内部空洞.
        """
        coco_dict = read_json(lbl_file)
        img_file = Path(img_file)
        if coco_dict is None:
            return []
        id_to_name = {img_item['id']: (img_item['file_name'], img_item['height'], img_item['width']) for img_item in coco_dict['images']}
        categories = {cate_item['id']: cate_item['name'] for cate_item in coco_dict['categories']}
        self.names = categories
        self.name2id = {v: k for k, v in categories.items()}
        res = list()
        for img_id, (img_name, img_height, img_width) in id_to_name.items():
            img_labelme_data = LabelmeData(imagePath=str(img_file / Path(img_name).name), imageHeight=img_height, imageWidth=img_width, shapes=[])
            for ann in coco_dict['annotations']:
                if ann['image_id'] != img_id:
                    continue
                is_seg = len(ann['segmentation']) > 0
                if len(ann['segmentation']) > 0:
                    points = segmentation_to_polygons(ann, min_area=self.min_area)[0]
                else:
                    x1, y1, w, h = ann['bbox']
                    points = [[x1, y1], [x1 + w, y1 + h]]
                img_labelme_data.shapes.append(ShapeInstance(
                    label=categories[ann['category_id']],
                    points=points,
                    shape_type='polygon' if is_seg else 'rectangle'
                ))
            res.append(img_labelme_data)
        return res

    def coco_gather(self) -> None:
        """COCO聚合的标签保存"""
        for split, data_list in self.gather.items():
            save_split_path = self.dst_dir / "annotations" / f"{split}{self.year}.json"
            save_image_dir = self.dst_dir / f"{split}{self.year}"
            save_image_dir.mkdir(exist_ok=True, parents=True)
            save_split_path.parent.mkdir(exist_ok=True, parents=True)
            coco_info = {
                "description": f"COCO {self.year} Dataset",
                "version": "1.0",
                "year": self.year,
                "contributor": "firstelfin",
                "date_created": time.strftime("%Y/%m/%d", time.localtime()),
            }
            save_coco_dict = {
                "info": coco_info,
                "images": list(),
                "annotations": list(),
                "categories": [
                    {
                        "id": i+self.class_start_index,  # 类别id, 起始索引由class_start_index指定
                        "name": name,
                        "supercategory": name
                    } for i, name in self.names.items()
                ],
            }
            for labelme_data in data_list:  # type labelme_data: LabelmeData
                img_info = {
                    'id': self.img_idx_start,
                    'file_name': f"{split}{self.year}/" + labelme_data["img_file"].name,
                    'height': labelme_data["lbmd"].imageHeight,
                    'width': labelme_data["lbmd"].imageWidth,
                }
                self.img_idx_start += 1
                save_coco_dict['images'].append(img_info)
                save_img_path = self.dst_dir / img_info['file_name']
                if self.use_link:
                    save_img_path.link_to(labelme_data["img_file"])
                else:
                    shutil.copy2(labelme_data["img_file"], save_img_path)

                for shape in labelme_data["lbmd"].shapes:
                    x_list = [min(max(0, p[0]), labelme_data["lbmd"].imageWidth) for p in shape.points]
                    y_list = [min(max(0, p[1]), labelme_data["lbmd"].imageHeight) for p in shape.points]
                    x_min, x_max = min(x_list), max(x_list)
                    y_min, y_max = min(y_list), max(y_list)
                    box_w, box_h = x_max - x_min, y_max - y_min
                    ann_info = {
                        "id": self.ann_idx_start,
                        "image_id": self.img_idx_start-1,
                        "category_id": self.name2id[shape.label],
                        "bbox": [x_min, y_min, box_w, box_h],
                        "iscrowd": 0,
                        "area": box_h * box_w,
                        "segmentation": [],
                    }
                    self.ann_idx_start += 1
                    save_coco_dict['annotations'].append(ann_info)

            save_json(save_split_path, save_coco_dict)
        logger.info(f"COCO {self.year} Dataset Gather Finished.")
        logger.info(f"Images: {self.img_idx_start}, Annotations: {self.ann_idx_start}.")

    # 加载数据集信息
    @abstractmethod
    def load_datasets(self, *args, **kwargs):
        """加载数据集, 要求标签和图像文件夹对齐, 即数量要相同
        通过status参数控制以图像为准, 还是以标签为准, 加载数据集.

        :yield tuple: 标签, 图像路径
        """

        status = kwargs.get("status", "images")
        all_status = ["images", "labels"]
        assert status in all_status, f"status must be in {all_status}, got {status}."
        if status == "labels":
            for lbl_dir, img_dir in zip(self.lbl_dir, self.img_dir):
                for lbl_file in lbl_dir.iterdir():
                    # 排除图像文件和系统隐藏文件
                    if lbl_file.suffix != self.suffix or lbl_file.stem.startswith('.'):
                        continue
                    img_files = [img_file for img_file in img_dir.rglob(f"{lbl_file.stem}.*") if img_file.suffix.lower() in IMG_EXTENSIONS]
                    img_file = img_files[0] if len(img_files) > 0 else ''
                    if not img_file and not self.ignore_img:
                        logger.warning(f"Failed to find image file for label file {lbl_file}.")
                        continue
                    yield lbl_file, img_file
        else:
            for lbl_dir, img_dir in zip(self.lbl_dir, self.img_dir):
                for img_file in img_dir.iterdir():

                    # 排除标签文件和系统隐藏文件
                    if img_file.suffix == self.suffix or img_file.stem.startswith('.'):
                        continue
                    lbl_file = lbl_dir / f"{img_file.stem}{self.suffix}"
                    yield lbl_file, img_file

    @abstractmethod
    def convert(self, *args, **kwargs):
        """标准转换接口, 转换过程使用FutureBar进行封装, 需要指定self.read_lbl_func和self.save_lbl_func.

        其中:
            - self.read_lbl_func: 读取标签文件函数, 输入为(lbl_file, img_file), 输出为List[LabelmeData]对象
            - self.save_lbl_func: 保存标签文件函数, 输入为(labelme_data, lbl_file, img_file, save_dir), 输出为None
        
        kwargs参数可以控制自定义的读取参数和保存参数. 数据集聚合型标签文件默认加载标签时, 是将所有标签分文件加载为LabelmeData对象,
        然后再进行转换. 转换为聚合型标签文件时, 会将LabelmeData对象聚合到self.gather, 并保存为一个文件. 从self.gather保存到文
        件需要自定义, 子类先调用super().convert()即可.
        """
        # 加载源数据集, 需要指定数据集的from_xxx函数, 转换为标准的LabelmeData对象
        all_lbl_img_files = list(self.load_datasets(*args, **kwargs))
        params = [([lbl_file, img_file], kwargs) for lbl_file, img_file in all_lbl_img_files]
        loadding_bar = FutureBar(
            max_workers=CPU_KERNEL_NUM,
            use_process=False,
            timeout=kwargs.get("timeout", 20),
            desc="Loadding Data"
        )
        all_labelme_data = loadding_bar(self.read_lbl_func, params, total=len(params))

        # 转换数据集
        convert_params = [([labelme_data, lbl_file, img_file, self.dst_dir], {}) for idx, (lbl_file, img_file) in enumerate(all_lbl_img_files)
                          for labelme_data in all_labelme_data[idx]]
        convert_bar = FutureBar(
            max_workers=CPU_KERNEL_NUM,
            use_process=False,
            timeout=kwargs.get("timeout", 20),
            desc="Converting Data"
        )
        all_labelme_data = convert_bar(self.save_lbl_func, convert_params, total=len(convert_params))


# Labelme数据集转其他数据集
class DetLabelmeConverter(DetConverter):

    def __init__(
            self, lbl_dir: List[Path | str] | Path | str, dst_dir: Path,
            img_dir: List[Path | str] | Path | str = [], split: str = 'train',
            start_img_idx: int = 0, start_ann_idx: int = 0, year: str = '',
            class_start_index: Literal[0, 1] = 0, classes: list = [], names: str|dict = {},
            min_area: float = 1.0, ignore_img: bool = True, **kwargs):
        names = load_names_dict(names)
        if isinstance(lbl_dir, (Path, str)):
            lbl_dir = [lbl_dir]
        lbl_dir = [Path(p) for p in lbl_dir]
        if isinstance(img_dir, (Path, str)):
            img_dir = [img_dir]
        img_dir = [Path(p) for p in img_dir]
        if not img_dir:
            img_dir = lbl_dir.copy()
        super().__init__(lbl_dir=lbl_dir, dst_dir=dst_dir, img_dir=img_dir, split=split,
            start_img_idx=start_img_idx, start_ann_idx=start_ann_idx, year=year,
            class_start_index=class_start_index, classes=classes, names=names,
            min_area=min_area, ignore_img=ignore_img, **kwargs)
        self.suffix = ".json"
        self.read_lbl_func = self.from_labelme

    def load_datasets(self, *args, **kwargs):
        return super().load_datasets(*args, status="labels", **kwargs)

    def convert(self, *args, **kwargs):
        return super().convert(*args, **kwargs)


def labelme2yolo(lbl_dir: List[Path | str] | Path | str, dst_dir: str | Path,
                 names: str|dict, img_dir: List[Path | str] | Path | str = []) -> None:
    """labelme格式的标签转换为yolo格式

    :param List[Path|str]| Path| str lbl_dir: labelme标签源数据地址
    :param str | Path dst_dir: yolo标签保存地址
    :param str|dict names: 索引到类别名称的映射
    :param List[Path|str]| Path| str img_dir: 图像源数据地址, 默认为空, 则使用labelme标签源数据地址

    Example:

    ```python
    >>> from codeUtils.labelOperation import labelme2yolo
    >>> labelme2yolo(
    ...     lbl_dir=Path("/Users/elfindan/datasets/coco128/jsons"),
    ...     dst_dir=Path("/Users/elfindan/datasets/coco128/yolo2"),
    ...     names={
    ...         0: "person",
    ...         1: "bicycle",
    ...         2: "car",
    ...         78: "hair drier",
    ...         79: "toothbrush",
    ...     }
    ... )
    ```
    """
    assert names, "names must be provided. not empty."
    det_converter = DetLabelmeConverter(lbl_dir=lbl_dir, dst_dir=Path(dst_dir), names=names, img_dir=img_dir)
    det_converter.save_lbl_func = det_converter.to_yolo
    det_converter.convert()


def labelme2voc(lbl_dir: List[Path | str] | Path | str, dst_dir: str | Path,
                img_dir: List[Path | str] | Path | str = []) -> None:
    """labelme格式的标签转换为PasCal VOC格式标签

    :param List[Path|str]| Path| str lbl_dir: labelme标签源数据地址
    :param str | Path dst_dir: VOC标签保存地址
    :param List[Path|str]| Path| str img_dir: 图像源数据地址, defaults to [], 默认使用labelme标签源数据地址

    Example:

    ```python
    >>> from codeUtils.labelOperation import labelme2voc
    >>> labelme2voc(
    ...     lbl_dir=Path("/Users/elfindan/datasets/coco128/jsons"),
    ...     dst_dir=Path("/Users/elfindan/datasets/coco128/xmls2")
    ... )
    ```
    """

    det_converter = DetLabelmeConverter(lbl_dir=lbl_dir, dst_dir=Path(dst_dir), img_dir=img_dir)
    det_converter.save_lbl_func = det_converter.to_voc
    det_converter.convert()


def labelme2coco(lbl_dir: List[Path | str] | Path | str, dst_dir: str | Path, 
                 img_dir: List[Path | str] | Path | str = [], names: dict | str = dict(),
                 use_link: bool = False, split: str = 'train', img_idx: int = 0, ann_idx: int = 0,
                 year: str = "", class_start_index: Literal[0, 1] = 0) -> None:
    """labelme格式的标签转换为COCO格式标签

    :param List[Path|str]| Path| str lbl_dir: labelme标签源数据地址
    :param str | Path dst_dir: COCO标签保存地址
    :param List[Path|str]| Path| str img_dir: 图像源数据地址, defaults to []
    :param dict | str names: 索引到类别名称的映射, defaults to dict()
    :param bool use_link: 是否使用软链接, defaults to False
    :param str split: 数据集划分, defaults to 'train'
    :param int img_idx: 图像索引起始值, defaults to 0
    :param int ann_idx: 标注索引起始值, defaults to 0
    :param str year: 数据集年份, defaults to ""
    :param Literal[0, 1] class_start_index: 类别索引起始值, defaults to 0

    Example:

    ```python
    >>> from codeUtils.labelOperation import labelme2coco
    >>> labelme2coco(
    ...     lbl_dir=Path("datasets/coco128/jsons"),
    ...     img_dir=Path("datasets/coco128/images"),
    ...     dst_dir=Path("datasets/coco128/coco3"),
    ...     names={
    ...         0: "person",
    ...         1: "bicycle",
    ...         2: "car",
    ...         79: "toothbrush",
    ...     }
    ... )
    ```
    """
    assert names, "names must be provided. not empty."
    det_converter = DetLabelmeConverter(
        lbl_dir=lbl_dir, dst_dir=Path(dst_dir), img_dir=img_dir, use_link=use_link, split=split, names=names,
        start_img_idx=img_idx, start_ann_idx=ann_idx, year=year, class_start_index=class_start_index)
    assert len(det_converter.img_dir) == len(det_converter.lbl_dir), \
        "The number of image and label directories must be the same."
    det_converter.save_lbl_func = det_converter.to_coco
    det_converter.ignore_img = False
    det_converter.convert()
    det_converter.coco_gather()

# YOLO数据集转其他数据集

class DetYOLOConverter(DetConverter):

    def __init__(
            self, src_dir: Path, dst_dir: Path, split: str = 'train',
            start_img_idx: int = 0, start_ann_idx: int = 0, year: str = '',
            class_start_index: Literal[0, 1] = 0, classes: list = [], names: str|dict = {},
            min_area: float = 1.0, ignore_img: bool = True, **kwargs):
        assert names, "names must be provided. not empty."
        img_dir = [Path(ld) for ld in (src_dir / "images").iterdir() if ld.is_dir() and not ld.stem.startswith(".")]
        lbl_dir = [src_dir / "labels" / ld.name for ld in img_dir]
        names = load_names_dict(names)
        super().__init__(lbl_dir=lbl_dir, dst_dir=dst_dir, img_dir=img_dir, split=split,
            start_img_idx=start_img_idx, start_ann_idx=start_ann_idx, year=year,
            class_start_index=class_start_index, classes=classes, names=names,
            min_area=min_area, ignore_img=ignore_img, **kwargs)
        self.suffix = ".txt"
        self.read_lbl_func = self.from_yolo

    def load_datasets(self, *args, **kwargs):
        return super().load_datasets(*args, **kwargs)

    def convert(self, *args, **kwargs):
        return super().convert(*args, **kwargs)


def yolo2labelme(src_dir: Path, dst_dir: Path, names: str|dict) -> None:
    """YOLO标签转换为labelme标签, 完成COCO128数据集(yolo格式标签)的转换测试

    images内部所有子集都将执行转换, 没有对应的标签也将执行转换.

    :param Path src_dir: YOLO数据集的根目录, 内包含images图像文件夹和labels标签文件夹
    :param Path dst_dir: 数据集的保存地址
    :param str|dict names: 索引到类别名称的映射

    Example:

    ```python
    >>> from codeUtils.labelOperation import yolo2labelme
    >>> yolo2labelme(
    ...     src_dir=Path("datasets/coco128/"),
    ...     dst_dir=Path("datasets/coco128/jsons"),
    ...     names={
    ...         0: "person",
    ...         1: "bicycle",
    ...         2: "car",
    ...         3: "motorcycle",
    ...         4: "airplane",
    ...         79: "toothbrush",
    ...     }
    ... )
    ```
    """
    det_converter = DetYOLOConverter(src_dir=Path(src_dir), dst_dir=dst_dir, names=names)
    det_converter.save_lbl_func = det_converter.to_labelme
    det_converter.convert(status="images")  # 以图像为基准


def yolo2voc(src_dir: Path, dst_dir: Path, names: str|dict) -> None:
    """YOLO标签转换为Pascal VOC格式标签, 完成COCO128数据集(yolo格式标签)的转换测试

    images内部所有子集都将执行转换, 没有对应的标签也将执行转换.

    :param Path src_dir: YOLO数据集的根目录, 内包含images图像文件夹和labels标签文件夹
    :param Path dst_dir: 数据集的保存地址
    :param dict str|names: 索引到类别名称的映射

    Example:

    ```python
    >>> from codeUtils.labelOperation import yolo2voc
    >>> yolo2voc(
    ...     src_dir=Path("datasets/coco128/"),
    ...     dst_dir=Path("datasets/coco128/xmls"),
    ...     names={
    ...         0: "person",
    ...         1: "bicycle",
    ...         2: "car",
    ...         3: "motorcycle",
    ...         4: "airplane",
    ...         79: "toothbrush",
    ...     }
    ... )
    ```
    """
    det_converter = DetYOLOConverter(src_dir=Path(src_dir), dst_dir=dst_dir, names=names)
    det_converter.save_lbl_func = det_converter.to_voc
    det_converter.convert(status="images")  # 以图像为基准


def yolo2coco(src_dir: Path, dst_dir: Path, names: str|dict, use_link: bool = False,
              split: str = 'train', img_idx: int = 0, ann_idx: int = 0,
              year: str = "", class_start_index: Literal[0, 1] = 0) -> None:
    """YOLO标签转换为COCO格式标签, 完成COCO128数据集(yolo格式标签)的转换测试

    images内部所有子集都将执行转换, 没有对应的标签也将执行转换.

    :param Path src_dir: YOLO数据集的根目录, 内包含images图像文件夹和labels标签文件夹
    :param Path dst_dir: COCO数据集的保存根目录地址
    :param str|dict names: 索引到类别名称的映射
    :param bool use_link: 图像是否使用软连接, defaults to False
    :param str split: 数据子集的名称, defaults to 'train'
    :param int img_idx: 图像id起始索引, defaults to 0
    :param int ann_idx: 标注id起始索引, defaults to 0
    :param str year: 数据集年份, defaults to ""
    :param Literal[0, 1] class_start_index: 类别id起始索引, defaults to 0

    Example:

    ```python
    >>> from codeUtils.labelOperation import yolo2coco
    >>> yolo2coco(
    ...     src_dir=Path("datasets/coco128/"),
    ...     dst_dir=Path("datasets/coco128/coco"),
    ...     names={
    ...         0: "person",
    ...         1: "bicycle",
    ...         2: "car",
    ...         3: "motorcycle",
    ...         4: "airplane",
    ...         79: "toothbrush",
    ...     }
    ... )
    ```

    """
    det_converter = DetYOLOConverter(
        src_dir=Path(src_dir), dst_dir=dst_dir, names=names, use_link=use_link, split=split,
        start_img_idx=img_idx, start_ann_idx=ann_idx, year=year, class_start_index=class_start_index)
    det_converter.save_lbl_func = det_converter.to_coco
    det_converter.ignore_img = False
    det_converter.convert(status="images")  # 以图像为基准
    det_converter.coco_gather()


# Pascal VOC数据集转其他数据集
class DetVocConverter(DetConverter):

    def __init__(
            self, lbl_dir: List[Path | str] | Path | str, dst_dir: Path,
            img_dir: List[Path | str] | Path | str = [], split: str = 'train',
            start_img_idx: int = 0, start_ann_idx: int = 0, year: str = '',
            class_start_index: Literal[0, 1] = 0, classes: list = [], names: str|dict = {},
            min_area: float = 1.0, ignore_img: bool = True, **kwargs):
        names = load_names_dict(names)
        if isinstance(lbl_dir, (Path, str)):
            lbl_dir = [lbl_dir]
        lbl_dir = [Path(p) for p in lbl_dir]
        if isinstance(img_dir, (Path, str)):
            img_dir = [img_dir]
        img_dir = [Path(p) for p in img_dir]
        if not img_dir:
            img_dir = lbl_dir.copy()
        super().__init__(lbl_dir=lbl_dir, dst_dir=dst_dir, img_dir=img_dir, split=split,
            start_img_idx=start_img_idx, start_ann_idx=start_ann_idx, year=year,
            class_start_index=class_start_index, classes=classes, names=names,
            min_area=min_area, ignore_img=ignore_img, **kwargs)
        self.suffix = ".xml"
        self.read_lbl_func = self.from_voc

    def load_datasets(self, *args, **kwargs):
        return super().load_datasets(*args, status="labels", **kwargs)

    def convert(self, *args, **kwargs):
        return super().convert(*args, **kwargs)


def voc2labelme(lbl_dir: List[Path | str] | Path | str, dst_dir: str | Path,
                img_dir: List[Path | str] | Path | str = []) -> None:
    """Pascal VOC格式标签转换为labelme格式标签

    :param List[Path|str]| Path| str lbl_dir: 源标签文件目录
    :param str | Path dst_dir: labelme格式标签保存目录
    :param List[Path|str]| Path| str img_dir: 源图像文件目录, 默认为空

    Example:

    ```python
    >>> from codeUtils.labelOperation import voc2labelme
    >>> voc2labelme(
    ...     lbl_dir=Path("/Users/elfindan/datasets/coco128/xmls"),
    ...     dst_dir=Path("/Users/elfindan/datasets/coco128/jsons3"),
    ...     img_dir=Path("/Users/elfindan/datasets/coco128/images")
    ... )
    ```
    """
    
    det_converter = DetVocConverter(lbl_dir=lbl_dir, dst_dir=Path(dst_dir), img_dir=img_dir)
    det_converter.save_lbl_func = det_converter.to_labelme
    det_converter.convert()


def voc2yolo(lbl_dir: List[Path | str] | Path | str, dst_dir: str | Path,
             names: str|dict, img_dir: List[Path | str] | Path | str = []) -> None:
    """Pascal VOC格式标签转换为YOLO格式标签

    :param List[Path|str]| Path| str lbl_dir: 源标签文件目录
    :param str | Path dst_dir: YOLO格式标签保存目录
    :param str|dict names: 索引到类别名称的映射
    :param List[Path|str]| Path| str img_dir: 源图像文件目录, defaults to []

    Example:

    ```python
    >>> from codeUtils.labelOperation import voc2yolo
    >>> voc2yolo(
    ...     lbl_dir=Path("datasets/coco128/xmls"),
    ...     dst_dir=Path("datasets/coco128/yolo3"),
    ...     img_dir=Path("datasets/coco128/images"),
    ...     names={
    ...         0: "person",
    ...         1: "bicycle",
    ...         2: "car",
    ...         3: "motorcycle",
    ...         79: "toothbrush",
    ...     }
    ... )
    ```
    """
    assert names, "names must be provided. not empty."
    det_converter = DetVocConverter(lbl_dir=lbl_dir, dst_dir=Path(dst_dir),
                                    names=names, img_dir=img_dir)
    det_converter.save_lbl_func = det_converter.to_yolo
    det_converter.convert()


def voc2coco(lbl_dir: List[Path | str] | Path | str, dst_dir: str | Path,
             img_dir: List[Path | str] | Path | str = [], names: str|dict = dict(),
             use_link: bool = False, split: str = 'train', img_idx: int = 0, ann_idx: int = 0,
             year: str = "", class_start_index: Literal[0, 1] = 0) -> None:
    """Pascal VOC格式标签转换为COCO格式标签

    :param List[Path|str]| Path| str lbl_dir: 源标签文件目录
    :param str | Path dst_dir: COCO格式标签保存目录
    :param List[Path|str]| Path| str img_dir: 源图像文件目录, 默认为空
    :param str|dict names: 索引到类别名称的映射, 默认为空
    :param bool use_link: 图像是否使用软连接, 默认为False
    :param str split: 数据子集的名称, 默认为'train'
    :param int img_idx: 图像id起始索引, 默认为0
    :param int ann_idx: 标注id起始索引, 默认为0
    :param str year: 数据集年份, 默认为空
    :param Literal[0, 1] class_start_index: 类别id起始索引, 默认为0
    
    Example:

    ```python
    >>> from codeUtils.labelOperation import voc2coco
    >>> voc2coco(
    ...     lbl_dir=Path("datasets/coco128/xmls"),
    ...     dst_dir=Path("datasets/coco128/coco3"),
    ...     img_dir=Path("datasets/coco128/images"),
    ...     names={
    ...         0: "person",
    ...         1: "bicycle",
    ...         2: "car",
    ...         3: "motorcycle",
    ...         79: "toothbrush",
    ...     }
    ... )
    ```
    """
    assert names, "names must be provided. not empty."
    det_converter = DetVocConverter(
        lbl_dir=lbl_dir, dst_dir=Path(dst_dir), img_dir=img_dir, use_link=use_link, split=split, names=names,
        start_img_idx=img_idx, start_ann_idx=ann_idx, year=year, class_start_index=class_start_index)
    assert len(det_converter.img_dir) == len(det_converter.lbl_dir), \
        "The number of image and label directories must be the same."
    det_converter.save_lbl_func = det_converter.to_coco
    det_converter.ignore_img = False
    det_converter.convert()
    det_converter.coco_gather()


class DetCocoConverter(DetConverter):
    """Coco数据集转其他格式转换器

    :param List[Path|str]| Path| str lbl_dir: coco标签文件, 支持指定json所在文件夹, 也支持指定json文件
    :param Path dst_dir: 目标数据集保存根目录
    """

    def __init__(self, lbl_dir: List[Path|str]| Path| str, dst_dir: Path | str, **kwargs):
        self._lbl_dir = self.get_lbl_files(lbl_dir)
        self._img_dir = [p.parents[1] / p.stem for p in self._lbl_dir]
        super().__init__(lbl_dir=self._lbl_dir, img_dir=self._img_dir, dst_dir=Path(dst_dir), **kwargs)
        self.suffix = ".json"
        self.read_lbl_func = self.from_coco
    
    def get_lbl_files(self, lbl_dir: List[Path|str]| Path| str) -> List[Path]:
        """获取指定的lbl_dir下所有的coco json文件

        :param List[Path|str]| Path| str lbl_dir: coco json文件所在文件夹或coco标注文件
        :return List[Path]: 所有coco json文件路径
        """
        lbl_paths = [Path(lbl_dir)] if isinstance(lbl_dir, (Path, str)) else [Path(p) for p in lbl_dir]
        all_lbl_files = []
        for p in lbl_paths:
            if p.is_dir():
                all_sub_files = list(p.rglob("*.json"))
                all_lbl_files.extend(all_sub_files)
            elif p.is_file():
                all_lbl_files.append(p)
            else:
                raise ValueError(f"{p} is not a valid path.")
        for p in all_lbl_files:
            assert p.suffix == ".json", f"{p} is not a json file."
            assert p.parent.name == "annotations", f"{p} is not in annotations file."
        return all_lbl_files
    
    def load_datasets(self, *args, **kwargs):
        """这里加载数据集以图像文件夹为准, 因为不同的子集一般不会融合到一起, 
        所以一般同步保留转换后的标签为多个子集
        """
        for coco_json_file, coco_img_dir in zip(self.lbl_dir, self.img_dir):
            yield coco_json_file, coco_img_dir
    
    def convert(self, *args, **kwargs):
        for coco_json_file, coco_img_dir in zip(self._lbl_dir, self._img_dir):
            self.img_dir = [coco_img_dir]
            self.lbl_dir = [coco_json_file]
            split_name = coco_json_file.stem
            self.dst_dir = self.dst_dir / split_name
            self.dst_dir.mkdir(exist_ok=True, parents=True)
            super().convert(*args, **kwargs)
            self.dst_dir = self.dst_dir.parent  # 重置根目录


def coco2yolo(lbl_dir: List[Path|str]| Path| str, dst_dir: Path | str):
    """COCO数据集转换为YOLO格式

    :param List[Path|str]| Path| str lbl_dir: coco json文件所在文件夹或coco标注文件
    :param Path dst_dir: 目标数据集保存根目录

    Example:

    ```python
    >>> from codeUtils.labelOperation import coco2yolo
    >>> coco2yolo(
    ...     lbl_dir=["/Users/elfindan/datasets/coco128/coco/annotations/train2026.json"],
    ...     dst_dir=Path("/Users/elfindan/datasets/coco128/yolo4"),
    ... )
    ```
    """
    det_converter = DetCocoConverter(lbl_dir=lbl_dir, dst_dir=dst_dir)
    det_converter.save_lbl_func = det_converter.to_yolo
    det_converter.convert()


def coco2labelme(lbl_dir: List[Path|str]| Path| str, dst_dir: Path | str):
    """COCO数据集转换为PasCAL VOC格式

    :param List[Path|str]| Path| str lbl_dir: coco json文件所在文件夹或coco标注文件
    :param Path dst_dir: 目标数据集保存根目录

    Example:

    ```python
    >>> from codeUtils.labelOperation import coco2labelme
    >>> coco2labelme(
    ...     lbl_dir=["datasets/coco128/coco/annotations"],
    ...     dst_dir=Path("datasets/coco128/jsons4"),
    ... )
    ```
    """
    det_converter = DetCocoConverter(lbl_dir=lbl_dir, dst_dir=dst_dir)
    det_converter.save_lbl_func = det_converter.to_labelme
    det_converter.convert()


def coco2voc(lbl_dir: List[Path|str]| Path| str, dst_dir: Path | str):
    """COCO数据集转换为VOC格式

    :param List[Path|str]| Path| str lbl_dir: coco json文件所在文件夹或coco标注文件
    :param Path dst_dir: 目标数据集保存根目录

    Example:

    ```python
    >>> from codeUtils.labelOperation import coco2voc
    >>> coco2voc(
    ...     lbl_dir=["datasets/coco128/coco/annotations"],
    ...     dst_dir=Path("datasets/coco128/xmls4"),
    ... )
    """
    det_converter = DetCocoConverter(lbl_dir=lbl_dir, dst_dir=dst_dir)
    det_converter.save_lbl_func = det_converter.to_voc
    det_converter.convert()


# class ToCOCO(ABC):
#     """Convert all format to COCO format.

#     Args:
#         img_dir (str|list): Pascal VOC format img file directory or list of directories.
#         lbl_dir (str|list): Pascal VOC format label file directory or list of directories.
#         dst_dir (str): coco format file directory, default is './COCODatasets'
#         classes (str|list): yolo classes.txt file path or list of class names.
#         use_link (bool): whether to use symbolic link to save images.
#         split (str): data split, 'train', 'val', 'test'.
#         year (str): year of dataset.
#         class_start_index (int): start index of class id. default is 0. lt 2.

#     Methods:
#         _post_init: 加载names文件(类yolo classes.txt文件), 并初始化name2idx、\
#             coco_images、coco_annotations格式化对象.
#         load_items: 加载图片和标签文件路径. 返回生成器, 每次生成(img_file, lbl_file, split).
#         coco_prepare: 给COCO添加图片信息, 添加实例相关信息, 实例编码需要重新编码.
#         anno_id_modify: 修改标注id.
#         save_coco_json: 保存COCO格式数据集.
#     """

#     def __init__(
#             self, img_dir: str | list = '', lbl_dir: str | list = '', dst_dir: str = './COCODatasets',
#             classes: str | list = [], use_link: bool = False, split: str | list = 'train', img_idx: int = 0,
#             ann_idx: int = 0, year: str = "", class_start_index: Literal[0, 1] = 0
#         ):
        
#         self.img_dir = [Path(p) for p in img_dir] if isinstance(img_dir, list) else Path(img_dir)
#         if lbl_dir == '':
#             self.lbl_dir = self.img_dir
#         elif isinstance(lbl_dir, list):
#             self.lbl_dir = [Path(p) for p in lbl_dir]
#         else:
#             self.lbl_dir = Path(lbl_dir)
#         self.dst_dir = Path(dst_dir)
#         self.classes = classes
#         self.use_link = use_link
#         self.split = split
#         self.start_img_idx = img_idx
#         self.img_idx = img_idx
#         self.ann_idx = ann_idx
#         self.year = year
#         self.names = dict()
#         self.name2idx = dict()
#         self.coco_images = dict()
#         self.coco_annotations = dict()
#         self.class_start_index = int(class_start_index)
#         self._post_init()

#     def _post_init(self):
#         """加载names文件, 并生成name2idx、coco_images、coco_annotations格式化对象"""

#         names = read_txt(self.classes) if isinstance(self.classes, str) else self.classes
#         self.names = {i: name for i, name in enumerate(names)}
#         self.name2idx = {name: i for i, name in self.names.items()}
#         if isinstance(self.split, str):
#             self.coco_images = {self.split: []}
#             self.coco_annotations = {self.split: []}
#         elif isinstance(self.split, list):
#             for split in self.split:
#                 self.coco_images[split] = []
#                 self.coco_annotations[split] = []
#         else:
#             raise ValueError("Invalid split type.")

#     def load_items(self, suffix: str = ".json"):
#         """加载并发处理函数的参数, 参数分两部分回传, 分别为args和kwargs.
        
#         load_items函数返回禁止使用生成器!

#         :param str suffix: 文件后缀, defaults to ".json", 默认处理labelme格式的标签
#         """

#         res = []
#         if isinstance(self.img_dir, PosixPath):
#             for img_file in self.img_dir.iterdir():
#                 if img_file.suffix == suffix or img_file.stem.startswith('.'):
#                     continue
#                 assert isinstance(self.lbl_dir, Path), "lbl_dir must be a Path object."
#                 lbl_file = self.lbl_dir / f"{img_file.stem}{suffix}"
#                 self.img_idx += 1
#                 res.append(([img_file, lbl_file, self.split, self.img_idx], {}))
#         else:
#             assert isinstance(self.img_dir, list), "img_dir must be a list of Path objects."
#             assert isinstance(self.lbl_dir, list), "lbl_dir must be a list of Path objects."
#             for i, img_dir in enumerate(self.img_dir):
#                 for img_file in img_dir.iterdir():
#                     # 排除json文件和系统隐藏文件
#                     if img_file.suffix == suffix or img_file.stem.startswith('.'):
#                         continue
#                     lbl_file = self.lbl_dir[i] / f"{img_file.stem}{suffix}"
#                     self.img_idx += 1
#                     res.append(([img_file, lbl_file, self.split, self.img_idx], {}))
#         return res

#     @abstractmethod
#     def instance_prepare(self, lbl_file: Path, img_id: int, split: str) -> list:
#         """COCO实例整备

#         :param Path lbl_file: 标签文件路径
#         :param int img_id: 图片id
#         :param str split: 数据集划分(不包含年份)
#         :return list: 实例列表, 实例组织格式无要求
        
#         ## 标准coco中转格式, Example:

#         ```
#         standard_instance = {
#             "id": 0,
#             "image_id": img_id,
#             "category_id": cls_id+self.class_start_index,
#             "bbox": [x_min, y_min, box_w, box_h],
#             "iscrowd": 0,
#             "area": box_w * box_h,
#             "segmentation": [],
#         }
#         ```
#         """
#         file_labels = None
#         for _ in range(3):
#             file_labels = read_json(lbl_file)
#             if file_labels is not None:
#                 break
            
#         if file_labels is None:
#             file_labels = {"shapes": []}
        
#         for label in file_labels["shapes"]:  # 
#             if label["label"] not in self.name2idx:
#                 continue
#             cls_id = self.name2idx[label["label"]]  # 类别id  TODO: 归一化操作
#             [x_min, y_min], [x_max, y_max] = label["points"]
#             box_w, box_h = x_max - x_min, y_max - y_min
#             ann_info = {
#                 "id": 0,
#                 "image_id": img_id,
#                 "category_id": cls_id+self.class_start_index,
#                 "bbox": [x_min, y_min, box_w, box_h],
#                 "iscrowd": 0,
#                 "area": box_w * box_h,
#                 "segmentation": [],
#             }
#             self.coco_annotations[split].append(ann_info)
#         return file_labels["shapes"]

#     def coco_prepare(self, img_file: Path, lbl_file: Path, split: str, img_id: int):
#         """coco数据集整备

#         :param Path img_file: 图片路径
#         :param Path lbl_file: 标签路径
#         :param str split: 数据集划分(不包含年份)
#         :param int img_id: 图片id
#         :raises ValueError: 标签格式错误
#         """

#         # 给COCO添加图片信息
#         src_img = cv.imread(str(img_file))
#         img_h, img_w = src_img.shape[:2]
#         img_info = {
#             'id': img_id,
#             'file_name': str(self.dst_dir / f"{split}{self.year}" / img_file.name),
#             'height': img_h,
#             'width': img_w,
#         }
#         self.coco_images[split].append(img_info)

#         # 给COCO添加标注信息
#         file_labels = self.instance_prepare(lbl_file, img_id, split)
        
#         # 保存图片
#         img_link = self.dst_dir / f"{split}{self.year}" / img_file.name
#         img_link.parent.mkdir(exist_ok=True, parents=True)
#         if self.use_link:
#             if not img_link.exists():
#                 img_link.symlink_to(img_file)
#         else:
#             shutil.copy(img_file, img_link)
#         return len(file_labels)

#     def anno_id_modify(self):
#         """修改标注id

#         :param int start_index: 起始索引, defaults to 0
#         """
#         for split in self.coco_annotations:
#             for ann in self.coco_annotations[split]:
#                 self.ann_idx += 1
#                 ann['id'] = self.ann_idx

#     def save_coco_json(self, anno_file: Path, split: str):
#         coco_info = {
#             "description": f"COCO {self.year} Dataset",
#             "version": "1.0",
#             "year": self.year,
#             "contributor": "firstelfin",
#             "date_created": time.strftime("%Y/%m/%d", time.localtime()),
#         }
#         coco_images = self.coco_images[split]
#         coco_annotations = self.coco_annotations[split]
#         coco_dict = {
#             "info": coco_info,
#             "images": coco_images,
#             "annotations": coco_annotations,
#             "categories": [
#                 {
#                     "id": i+self.class_start_index,  # 类别id, 起始索引由class_start_index指定
#                     "name": name, 
#                     "supercategory": name
#                 } for i, name in self.names.items()
#             ],
#         }
#         save_json(anno_file, coco_dict)
#         logger.info(f"save {split} coco json file {anno_file} success. {self.img_idx} images, {self.ann_idx} annotations.")

#     def __call__(self, *args, **kwargs):

#         all_async_items = self.load_items()
#         cpu_kernel_num = os.cpu_count() or 6
#         cpu_num = max(cpu_kernel_num // 2, 6)
#         exec_bar = FutureBar(max_workers=cpu_num, timeout=20, desc=self.__class__.__name__)
#         exec_bar(self.coco_prepare, all_async_items, total=self.img_idx-self.start_img_idx)

#         # 保存COCO格式数据集
#         self.anno_id_modify()
#         anno_dir = self.dst_dir / 'annotations'
#         anno_dir.mkdir(exist_ok=True, parents=True)

#         for split in self.coco_annotations:
#             anno_file = anno_dir / f"{split}{self.year}.json"
#             self.save_coco_json(anno_file=anno_file, split=split)


# class COCOToAll(ABC):

#     def __init__(self, img_dir: str, lbl_file: str, dst_dir: str):
#         self.img_dir = Path(img_dir)
#         self.lbl_file = Path(lbl_file)
#         self.dst_dir = Path(dst_dir)
#         self.dst_dir.mkdir(exist_ok=True, parents=True)
#         self.coco_names = None
#         self.coco_instances = dict()
#         self._post_init()
#         super().__init__()

#     def _post_init(self):
#         # 加载COCO数据集
#         self.coco_dict = read_json(self.lbl_file)
#         if self.coco_dict is None:
#             raise ValueError("Invalid COCO format.")
#         self.coco_names = {cat['id']: cat['name'] for cat in self.coco_dict['categories']}
#         coco_init_bar = tqdm(self.coco_dict['images'], desc='COCO init', colour='#CD8500')
#         for img_info in coco_init_bar:
#             self.coco_instances[img_info['file_name']] = {
#                 'imagePath': img_info['file_name'],
#                 'imageHeight': img_info['height'],
#                 'imageWidth': img_info['width'],
#                 'shapes': []
#             }
#             for obj in self.coco_dict['annotations']:
#                 if obj['image_id'] != img_info['id']:
#                     continue
#                 seg_points_num = sum([sum(obj['segmentation'][i]) for i in range(len(obj['segmentation']))]) // 2
#                 is_rectangle = seg_points_num <= 2
#                 points = []
#                 if not is_rectangle:
#                     for i in range(len(obj['segmentation'])):
#                         seg_i = [
#                             [obj['segmentation'][i][j], obj['segmentation'][i][j+1]]
#                             for j in range(0, len(obj['segmentation'][i]), 2)
#                         ]
#                         if seg_i[0] != seg_i[-1]:
#                             seg_i.append(seg_i[0])
#                         points += seg_i
#                 else:
#                     points = [
#                         [obj['bbox'][0], obj['bbox'][1]],
#                         [obj['bbox'][0]+obj['bbox'][2], obj['bbox'][1]+obj['bbox'][3]],
#                     ]
#                 self.coco_instances[img_info['file_name']]['shapes'].append({
#                     "label": self.coco_names[obj['category_id']],
#                     "points": points,
#                     "group_id": None,
#                     "shape_type": "rectangle" if is_rectangle else "polygon",
#                     "flags": {}
#                 })

#     @abstractmethod
#     def save_label(self, img_path: str, labelme_dict: dict, **kwargs):
#         save_label_file = self.dst_dir / f"{Path(img_path).stem}.json"
#         save_labelme_label(save_label_file, labelme_dict)

#     def __call__(self, *args, **kwargs):
#         cpu_num = max((os.cpu_count() or 8) // 2, 6)
#         params = [([img_path, labelme_dict], kwargs) for img_path, labelme_dict in self.coco_instances.items()]
#         exec_bar = FutureBar(
#             max_workers=cpu_num, 
#             use_process=kwargs.get("use_process", False), 
#             timeout=kwargs.get("timeout", 20), 
#             desc=self.__class__.__name__
#         )
#         exec_bar(self.save_label, params, total=len(params))

