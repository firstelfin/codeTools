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
from enum import Enum
from typing import Literal, Callable, Optional, Any
from loguru import logger
from tqdm import tqdm
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from codeUtils.labelOperation.readLabel import read_txt, read_yolo, read_voc, parser_json
from codeUtils.labelOperation.saveLabel import save_json, save_labelme_label
from codeUtils.tools.futureConf import FutureBar
from codeUtils.tools.loadFile import load_img


@dataclass(slots=True)
class ShapeInstance(object):
    """实例数据结构
    
    Attributes:
        label (str|int): 实例标签
        points (list[list[int|float]]): 实例坐标点
        shape_type (str): 实例类型, 如"polygon", "rectangle"
        flags (dict): 实例标注属性
        score (float): 实例分数
    """

    label: str | int
    points: list[list[int|float]] = field(default_factory=list)
    shape_type: Literal["polygon", "rectangle"] = field(default="rectangle")
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


class DectConverter(object):
    """检测、分割数据转换基类

    定义LabelmeData转为其他数据类的接口, 并提供转换的统一入口.
    
    处理流程:
    
    ```mermaid
    graph TD
        A[接收 LabelmeData 对象] --> B{判断对象类型}
        B -- 标注文件独立 --> C[直接输出新的标签文件]
        B -- 标注文件耦合 --> D[转换为中间状态并缓存]
        D --> E[等待统一保存]
        F[调用 save_all 方法] --> G[遍历缓存的耦合对象]
        G --> H[逐个保存为目标格式文件]
    ```

    标签读取:

    - 支持读取labelme格式的标注文件
    - 支持读取yolo格式的标注文件
    - 支持读取voc格式的标注文件
    - 支持读取coco格式的标注文件
    
    """

    def __init__(self, src_dir: list, dst_dir: str, split: str = 'train', img_idx: int = 0, ann_idx: int = 0,
                 year: str = '', class_start_index: Literal[0, 1] = 0, classes: list = [], names: dict = {}, **kwargs):
        """初始化转换器"""
        if isinstance(src_dir, str):
            src_dir = [src_dir]
        self.src_dir = src_dir
        self.classes = classes
        self.names = names if names else {i: self.classes[i] for i in range(len(self.classes))}

    def __call__(self, *args, **kwargs):
        """转换入口"""

    def pre_validate(
            self, 
            lbl_file: str | Path | None, 
            save_dir: str | Path | None, 
            img_file: str | Path | None = None
        ) -> tuple[Path, Path, Path]:
        if lbl_file is None:
            raise ValueError("lbl_file is None. Expected a label file path.")
        if save_dir is None:
            raise ValueError("save_dir is None. Expected a save directory path.")
        if img_file is None:
            res_img = Path("")
        else:
            res_img = Path(img_file)
        return Path(lbl_file), Path(save_dir), res_img
    
    def to_labelme(
            self, lbmd: LabelmeData | None = None, 
            lbl_file: str | Path | None = None, 
            img_file: str | Path | None = None, 
            read_func: Callable[[str | Path, str | Path, Any], LabelmeData] | None = None, 
            save_dir: str | Path | None = None
        ) -> None:
        """转为labelme格式的标准输出接口.

        :param lbmd: 标准的LabelmeData对象, defaults to None
        :type lbmd: LabelmeData | None, optional
        :param lbl_file: 源标注文件, defaults to None
        :type lbl_file: str | None, optional
        :param img_file: 源图像文件, defaults to None
        :type img_file: str | None, optional
        :param read_func: 读取标签的函数, defaults to None
        :type read_func: Callable | None, optional
        :param save_dir: 保存目录, defaults to None
        :type save_dir: str | None, optional
        """

        lbl_file, save_dir, img_file = self.pre_validate(lbl_file, save_dir, img_file)
        
        labelme_update = {
            "version": "4.5.6",
            "flags": {},
            "imageData": None,
        }
        save_file_path = save_dir / f"{lbl_file.stem}.json"

        # lbmd是None, 读取标签文件
        if lbmd is None:
            if read_func is None:
                raise TypeError("read_func is None. Expected a callable function to read label file.")
            lbmd = read_func(lbl_file, img_file)

        # 基于LabelmeData对象转换
        if lbmd is not None and isinstance(lbmd, LabelmeData):
            lbl_dict = asdict(lbmd)
            lbl_dict.update(labelme_update)
            save_labelme_label(save_file_path, lbl_dict)
        elif lbmd is not None:
            raise TypeError(f"Invalid type of lbmd. Expected LabelmeData, but got {type(lbmd)}.")
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
            read_func: Callable[[str | Path | LabelmeData, str | Path], LabelmeData] | None = None, 
            save_dir: str | Path | None = None,
        ) -> None:
        """转为yolo格式的标准输出接口.

        :param lbmd: 标准的LabelmeData对象, defaults to None
        :type lbmd: LabelmeData | None, optional
        :param lbl_file: 源标注文件, defaults to None
        :type lbl_file: str | Path | None
        :param img_file: 源图像文件, defaults to None
        :type img_file: str | Path | None, optional
        :param read_func: 读取标签的函数, defaults to None
        :type read_func: Callable[[str | Path], LabelmeData] | None, optional
        :param save_dir: 保存目录, defaults to None
        :type save_dir: str | Path | None, optional
        """
        
        lbl_file, save_dir, img_file = self.pre_validate(lbl_file, save_dir, img_file)

    def to_voc(
            self, lbmd: LabelmeData | None = None, 
            lbl_file: str | Path | None = None, 
            img_file: str | Path | None = None, 
            read_func: Callable[[str | Path | LabelmeData, str | Path], LabelmeData] | None = None, 
            save_dir: str | Path | None = None,
        ) -> None:
        pass

    def to_coco(
            self, lbmd: LabelmeData | None = None, 
            lbl_file: str | Path | None = None, 
            img_file: str | Path | None = None, 
            read_func: Callable[[str | Path | LabelmeData, str | Path], LabelmeData] | None = None, 
            save_dir: str | Path | None = None,
        ) -> None:
        pass


    @classmethod
    def from_labelme(cls, lbl_file: str | Path, img_file: str | Path = "", **kwargs) -> LabelmeData:
        """从labelme标注文件读取数据, 并返回LabelmeData对象.
        已经支持:
        1. 矩形标注, 转为points字段[2, 1, 2]
        2. 多边形标注, 转为points字段[n, 1, 2]

        :param lbl_file: labelme格式的标注文件路径
        :type lbl_file: str | Path
        :param img_file: 图像文件路径
        :type img_file: str | Path
        :return: 标准化的LabelmeData对象
        :rtype: LabelmeData
        """
        res = LabelmeData(imagePath=str(img_file), imageHeight=0, imageWidth=0, shapes=[])
        # 读取labelme格式的标注文件
        lbl_dict = parser_json(lbl_file)
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
        return res

    def from_yolo(self, lbl_file: str | Path, img_file: str | Path, **kwargs) -> LabelmeData:
        """从yolo标注文件读取数据, 并返回LabelmeData对象. 
        已经支持:
        1. conf字段, LabelmeData记录为score字段
        2. 多边形标注, 转为points字段[n, 1, 2]
        3. 矩形标注, 转为points字段[2, 1, 2]

        :param lbl_file: yolo格式的标注文件路径
        :type lbl_file: str | Path
        :param img_file: 图像文件路径
        :type img_file: str | Path
        :return: 标准化的LabelmeData对象
        :rtype: LabelmeData
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
        return res
    
    def from_voc(self, lbl_file: str | Path, img_file: str | Path = "", **kwargs) -> LabelmeData:
        """从voc标注文件读取数据, 并返回LabelmeData对象.
        已经支持:
        1. 矩形标注, 转为points字段[2, 1, 2]
        2. 多边形标注, 转为points字段[n, 1, 2](暂未支持)

        :param lbl_file: voc格式的标注文件路径
        :type lbl_file: str | Path
        :param img_file: 图像文件路径, lbl_file加载不到图像尺寸时, 图像就必须要能够加载到
        :type img_file: str | Path
        :return: 标准化的LabelmeData对象
        :rtype: LabelmeData
        """
        res = LabelmeData(imagePath=str(img_file), imageHeight=0, imageWidth=0, shapes=[])
        # 读取voc格式的标注文件
        voc_dict = read_voc(label_file=lbl_file, extra_keys=kwargs.get("extra_keys", []))
        if voc_dict is None:
            return res
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
        return res
    
    def from_coco(self, lbl_data: LabelmeData, img_file: str | Path = '', **kwargs) -> LabelmeData:
        """coco数据预处理时, 会直接将对象处理为LabelmeData对象, 所以不需要再次处理.
        已经支持:
        1. 矩形标注, 转为points字段[2, 1, 2]
        2. 多边形标注, 转为points字段[n, 1, 2]
        """
        return lbl_data


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
    def save_label(self, img_path: str, labelme_dict: dict, **kwargs):
        save_label_file = self.dst_dir / f"{Path(img_path).stem}.json"
        save_labelme_label(save_label_file, labelme_dict)

    def __call__(self, *args, **kwargs):
        cpu_num = max((os.cpu_count() or 8) // 2, 6)
        params = [([img_path, labelme_dict], kwargs) for img_path, labelme_dict in self.coco_instances.items()]
        exec_bar = FutureBar(
            max_workers=cpu_num, 
            use_process=kwargs.get("use_process", False), 
            timeout=kwargs.get("timeout", 20), 
            desc=self.__class__.__name__
        )
        exec_bar(self.save_label, params, total=len(params))

