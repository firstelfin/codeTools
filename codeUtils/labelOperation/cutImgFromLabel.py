#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cutImgFromLabel.py
@Time    :   2024/12/23 14:01:48
@Author  :   firstElfin 
@Version :   0.1.8
@Desc    :   This script is used to cut images from label files.
'''

import math
import psutil
import random
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from numpy import ndarray
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from codeUtils.tools.font_config import colorstr
from codeUtils.tools.tqdm_conf import BATCH_KEY, START_KEY, END_KEY
from codeUtils.matchFactory.bboxMatch import box_valid, ios_box, rel_box
from codeUtils.labelOperation.readLabel import read_voc, read_yolo
from codeUtils.labelOperation.saveLabel import save_voc_label, save_yolo_label, save_json


class CutImgFromLabel:
    """
    This class is used to cut images from label files. 裁剪图片区域记录在dst_dir目录下. 
    从什么格式的标注文件裁切就会生成对应格式的标注文件.

    Attributes:
        src_dir: The directory of source images.
        dst_dir: The directory of destination images.
        pattern: The pattern of label files.

    ## Methods:
        __call__: Cut images from label files.
        read_voc_label: Read voc label file.
        read_yolo_label: Read yolo label file.
        save_voc_label: Save voc label file.
        save_yolo_label: Save yolo label file.
        padding_cut_bbox: Expand the bbox to ensure it is in the image and scale it to ~target_size.
        ios_valid: Calculate the ios between the box and the bbox_list.
        add_cut_out: 剪除部分区域, 黑色填充


    ## Examples:
        >>> cutImgFromLabel = CutImgFromLabel(src_dir="src_dir", dst_dir="dst_dir", pattern="voc")
        >>> cutImgFromLabel(img_dir_name="images", lbl_dir_name="labels")

    ## cli:

    ```shell
    elfin cutImg test/test/elfin/yoloLabelTest test/test/elfin/yoloLabelTestCut --pattern voc --img_dir_name images--lbl_dir_name voc
    ```

    """

    PATTERNS = ("voc", "yolo")

    def __init__(
            self, 
            src_dir: str, 
            dst_dir: str, 
            pattern: str, 
            target_size: int = None, 
            sample_rate: dict = None, 
            use_mask: bool = False, 
            use_shape: bool = False, 
            ios_thrash: float = 0.7):
        assert pattern.lower() in self.PATTERNS, f"The pattern should be {self.PATTERNS}, but got {pattern}."
        self.pattern = pattern.lower()
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.ios_thrash = ios_thrash
        self.use_mask = use_mask
        self.target_size = int(target_size) if target_size is not None else 640
        self.sample_rate = dict() if sample_rate is None else sample_rate
        self.name2cut_box = dict()
        self.cpu_num = max(4, psutil.cpu_count(logical=False) // 2)
        self.batch_size = 100
        self.use_shape = use_shape

    def read_voc_label(self, label_file: str, img_shape: tuple = None):
        label_file = Path(label_file).with_suffix(".xml")
        xml_dict = read_voc(label_file)
        xml_shape = (xml_dict["size"]["height"], xml_dict["size"]["width"])
        if isinstance(img_shape, list):
            img_shape = tuple(img_shape)
        if img_shape is None:
            img_shape = xml_shape
        else:
            assert img_shape[:2] == xml_shape, \
            f"The image shape {img_shape} is not equal to the xml shape {xml_shape}."
        
        res = [
            [
                obj['name'], 
                obj['bndbox']['xmin'], 
                obj['bndbox']['ymin'],
                obj['bndbox']['xmax'], 
                obj['bndbox']['ymax']
            ] for obj in xml_dict["object"]
        ]
        return res

    def read_yolo_label(self, label_file: str, img_shape: tuple = None):
        label_file = Path(label_file).with_suffix(".txt")
        label_list = read_yolo(label_file)
        if img_shape is None:
            img_shape = (1, 1)

        res = []
        for label in label_list:
            cx, cy, w, h = label[1:5]
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            x1, y1, x2, y2 = int(x1 * img_shape[1]), int(y1 * img_shape[0]), int(x2 * img_shape[1]), int(y2 * img_shape[0])
            res.append([label[0], x1, y1, x2, y2])
        return res

    def save_voc_label(self, label_file: str, bbox_list: list, box: list, sub_dir_name: tuple = None):
        """保存voc格式的label文件

        :param str label_file: 标注文件保存路径, 默认会修正文件后缀为.xml
        :param list bbox_list: 标注实例的正外接矩形, list of [class_name or label_id, x1, y1, x2, y2]
        :param list box: 图片的边界框, list of [x1, y1, x2, y2]
        :param tuple sub_dir_name: 子目录名称, 如("images", "labels"), defaults to None
        """
        if sub_dir_name is None:
            sub_dir_name = ("images", "labels")
        label_file = Path(label_file).with_suffix(".xml")
        
        new_box = [[s_box[0], *rel_box(box, s_box[1:5])] for s_box in bbox_list]
        
        voc_header = {
            'folder': sub_dir_name[0],
            'filename': Path(label_file).stem + ".jpg",
            'path': str(self.dst_dir / sub_dir_name[0] / f"{Path(label_file).stem + '.jpg'}"),
            'source': {'database': "Unknown"},
            'segmented': "0",
            'size': {
                'width': box[2] - box[0],
                'height': box[3] - box[1],
                'depth': 3
            }
        }
        voc_object = [
            {
                'name': obj[0],
                'pose': "Unspecified",
                'truncated': "0",
                'difficult': "0",
                'bndbox': {
                    'xmin': obj[1],
                    'ymin': obj[2],
                    'xmax': obj[3],
                    'ymax': obj[4]
                }
            } for obj in new_box
        ]
        save_voc_label(label_file, voc_header, voc_object)

    def save_yolo_label(self, label_file: str, bbox_list: list, box: list, sub_dir_name: tuple = None):
        """保存yolo格式的label文件

        :param str label_file: 标注文件保存路径, 默认会修正文件后缀为.txt
        :param list bbox_list: 标注实例的正外接矩形, list of [class_name or label_id, x1, y1, x2, y2]
        :param list box: 图片的边界框, list of [x1, y1, x2, y2]
        :param tuple sub_dir_name: 子目录名称, 如("images", "labels"), defaults to None
        """
        label_file = Path(label_file).with_suffix(".txt")
        img_h, img_w = box[3] - box[1], box[2] - box[0]
        new_bbox = []
        for bbox in bbox_list:
            rel_bbox = rel_box(box, bbox[1:5])
            x1, y1, x2, y2 = rel_bbox
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            new_bbox.append([bbox[0], cx / img_w, cy / img_h, w / img_w, h / img_h])
        
        save_yolo_label(label_file, new_bbox)

    @staticmethod
    def padding_cut_bbox(box: list[int], img_shape: tuple, target_size: int):
        """扩充裁剪区域, 保证bbox在图片内, 并按比例缩放到~target_size大小

        :param list[int] box: 标注实例的正外接矩形, list of [x1, y1, x2, y2]
        :param tuple img_shape: 原始图片大小
        :param int target_size: 目标尺寸
        """

        x1, y1, x2, y2 = box
        img_h, img_w = img_shape
        # 扩充bbox
        h, w = y2 - y1, x2 - x1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        target_thresh = target_size - 30

        w = target_size if w < target_thresh else int(w * 1.2)
        h = target_size if h < target_thresh else int(h * 1.2)
        # 重置bbox的高宽
        max_size = max(w, h)
        size_thresh = int(max_size * random.randrange(80, 90) / 100)
        w = w if w > size_thresh else size_thresh
        h = h if h > size_thresh else size_thresh
        half_w, half_h = w // 2, h // 2
        # 中心点偏移
        if cx < half_w:
            cx = half_w
        elif cx > img_w - half_w:
            cx = img_w - half_w
        if cy < half_h:
            cy = half_h
        elif cy > img_h - half_h:
            cy = img_h - half_h

        a1 = max(0, cx - half_w)
        b1 = max(0, cy - half_h)
        a2 = min(img_w, a1 + w)
        b2 = min(img_h, b1 + h)
        return [a1, b1, a2, b2]

    def ios_valid(self, box: list[int], bbox_list: list) -> list[float]:
        """计算box与bbox_list中所有bbox的ios指标

        :param list[int] box: 查询bbox
        :param list bbox_list: 候选bbox列表
        :return: 候选bbox与查询bbox的ios指标列表
        :rtype: list[float]
        """
        ios_list = [ios_box(bbox[1:5], box, mode="xyxy") for bbox in bbox_list]
        return ios_list

    def add_cut_out(self, img: ndarray, bbox_list: list, img_box=None) -> ndarray:
        """给图片添加黑色的mask

        :param ndarray img: 裁切图片
        :param list bbox_list: 候选bbox列表
        :param list img_box: 原始图片的bbox, 默认为None
        :return ndarray: 返回mask后的图片
        """
        if img_box is None:
            raise ValueError("img_box should not be None, please provide the original image bbox.")
        dst_img = deepcopy(img)
        box_anchor = img_box if len(img_box) == 4 else img_box[1:5]
        for bbox in bbox_list:
            search_box = bbox if len(bbox) == 4 else bbox[1:5]
            new_box = rel_box(box_anchor, search_box)
            dst_img[new_box[1]:new_box[3], new_box[0]:new_box[2]] = 0
        return dst_img

    def cut(self, img_file: str, label_file: str, sub_dir_name: tuple = None):
        if sub_dir_name is None:
            sub_dir_name = ("images", "labels")
        img_dir = self.dst_dir / sub_dir_name[0]
        lbl_dir = self.dst_dir / sub_dir_name[1]
        img_dir.mkdir(exist_ok=True, parents=True)
        lbl_dir.mkdir(exist_ok=True, parents=True)

        src_img = cv.imread(str(img_file))
        img_h, img_w = src_img.shape[:2]

        # 裁切一定是根据bbox裁切, bbox列表的表示是: [[cls_id, x1, y1, x2, y2]]
        bbox_list = getattr(self, f"read_{self.pattern}_label")(label_file, (img_h, img_w) if self.use_shape else None)
        selected_status = [False] * len(bbox_list)
        save_label_func = getattr(self, f"save_{self.pattern}_label")

        for i, bbox in enumerate(bbox_list):
            if selected_status[i]:
                continue
            
            if not box_valid(bbox[1:5]):
                raise ValueError(f"Invalid bbox: {bbox}")
            
            x1, y1, x2, y2 = self.padding_cut_bbox(bbox[1:5], (img_h, img_w), self.target_size)
            crop_img = src_img[y1:y2, x1:x2]
            
            # 开始处理图片标签
            # 1. 验证标签是否被包含
            dst_label_path = lbl_dir / f"{img_file.stem}_cut_{i}.txt"
            self.name2cut_box[img_file.stem].append({"bbox": [x1, y1, x2, y2], "name": dst_label_path.stem})
            ios_list = self.ios_valid([x1, y1, x2, y2], bbox_list)
            if not self.use_mask:
                for j, ios in enumerate(ios_list):
                    if ios > self.ios_thrash:
                        selected_status[j] = True
                save_label_bbox = [bbox_list[j] for j, v in enumerate(ios_list) if v > self.ios_thrash]
                save_label_func(dst_label_path, save_label_bbox, [x1, y1, x2, y2], sub_dir_name)  
            else:
                # cut mask
                for j, ios in enumerate(ios_list):
                    if ios >= 0.95:
                        selected_status[j] = True
                mask_label_bbox = [bbox_list[j] for j, v in enumerate(ios_list) if v > 0 and v < 0.95]
                crop_img = self.add_cut_out(crop_img, mask_label_bbox, [x1, y1, x2, y2])
                save_label_func(
                    dst_label_path, [bbox_list[j] for j, v in enumerate(ios_list) if v >= 0.95],
                    [x1, y1, x2, y2], sub_dir_name
                )
            
            # 保存图片
            dst_img_path = img_dir / f"{img_file.stem}_cut_{i}.jpg"
            cv.imwrite(str(dst_img_path), crop_img, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    def __call__(self, img_dir_name: str = None, lbl_dir_name: str = None):
        """裁剪流程

        :param str img_dir_name: 图片目录名称, defaults to 'images'
        :param str lbl_dir_name: 标签目录名称, defaults to 'labels'
        """

        if img_dir_name is None:
            img_dir_name = "images"
        if lbl_dir_name is None:
            lbl_dir_name = "labels"

        # 并发循环src_dir下所有文件
        all_tasks = []

        image_dir = self.src_dir / img_dir_name
        label_dir = self.src_dir / lbl_dir_name
        tasks_num = len(list(image_dir.iterdir()))
        # 向上取整
        epoch_num = math.ceil(tasks_num / self.batch_size)

        with ThreadPoolExecutor(max_workers=self.cpu_num) as executor:
            tqdm_tasks = image_dir.iterdir()  # 使用生成器双循环迭代
            epoch_desc = colorstr("bright_blue", "bold", "epochProgress")

            with tqdm(total=tasks_num, desc=epoch_desc, position=0, dynamic_ncols=True, colour="#CD8500") as epoch_bar:
                for epoch in range(epoch_num):
                    start_idx = epoch * self.batch_size
                    end_idx = min(tasks_num, (epoch + 1) * self.batch_size)
                    
                    inner_tasks = []
                    epoch_size = end_idx - start_idx
                    for _ in range(start_idx, end_idx):
                        img_file = next(tqdm_tasks)
                        lbl_file = label_dir / f"{img_file.stem}.txt"
                        self.name2cut_box[img_file.stem] = []  # 记录每个图片的裁剪bbox
                        inner_tasks.append(
                            executor.submit(self.cut, img_file, lbl_file, (img_dir_name, lbl_dir_name))
                        )
                    # 更新进度条
                    for ti, task in enumerate(as_completed(inner_tasks), start=1):
                        task.result()
                        epoch_bar.set_postfix({
                            BATCH_KEY: f"{ti}/{epoch_size}",
                            START_KEY: start_idx,
                            END_KEY: end_idx
                        })
                    
                        epoch_bar.update()
        
        # 开始保存json文件
        json_file = self.dst_dir / "name2cut_box.json"
        save_json(json_file, self.name2cut_box)

