#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2024/12/26 10:51:00
# @desc: This file is the base file of infer.

import os
import sys
import math
import warnings
import cv2 as cv
import numpy as np
from copy import deepcopy
from typing import Literal
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
warnings.filterwarnings('ignore')
from tqdm import tqdm
from loguru import logger
from prettytable import PrettyTable
from concurrent.futures import ThreadPoolExecutor, as_completed

from codeUtils.tools.fontConfig import colorstr
from codeUtils.tools.torchTools import torch_empty_cache
from codeUtils.tools.futureConf import FutureBar
from codeUtils.tools.tqdmConf import cpu_num, BATCH_KEY, START_KEY, END_KEY, TPP, FPP, TPG, TNG, GTN, PRN
from codeUtils.__base__ import strPath
from codeUtils.labelOperation.readLabel import read_voc, read_yolo, read_txt, parser_json
from codeUtils.labelOperation.saveLabel import save_labelme_label
from codeUtils.matrix.confusionMatrix import ConfusionMatrix
from codeUtils.matrix.distMatrix import DetMetrics
from codeUtils.matchFactory.bboxMatch import abs_box, iou_np_boxes
from codeUtils.inference.boxMatch import yolo_match
from codeUtils.decorator.registry import Registry


SliceRegistry = Registry("SliceRegistry")
CombineRegistry = Registry("CombineRegistry")
InferRegistry = Registry("InferRegistry")
StatisticRegistry = Registry("StatisticRegistry")


def get_exp_dir(dst_dir: str, project: str = 'inference') -> PosixPath:
    """获取实验结果保存目录, 文件夹不存在则创建

    :param dst_dir: 实验根目录
    :type dst_dir: str
    :param project: 实验名称, 默认为'inference'
    :type project: str
    :return: _description_
    :rtype: PosixPath
    """
    res_dir = Path(dst_dir) / project
    if not res_dir.exists():
        res_dir.mkdir(exist_ok=True, parents=True)
        return res_dir
    elif not list(res_dir.iterdir()):
        return res_dir
    i = 1
    while (Path(dst_dir) / f'{project}{i}').exists():
        # 目录已存在, 判断时候为空文件夹
        if not (Path(dst_dir) / f'{project}{i}').iterdir():
            break
        i += 1
    (Path(dst_dir) / f'{project}{i}').mkdir(exist_ok=True, parents=True)
    return Path(dst_dir) / f'{project}{i}'


def path_list_valid(path_dir):
    if isinstance(path_dir, (str, PosixPath)):
        datasets = [path_dir]
    else:
        datasets = path_dir
    datasets = [Path(dataset) for dataset in datasets]
    return datasets


@InferRegistry
class DetectBase(object):
    """模型推理基类, 指定模型, 设备, 推理参数等, 并提供预测接口

    :param object: _description_
    :type object: _type_
    """

    def __init__(self, model: str, device: str, conf: list, nms_iou: float, *args, **kwargs):
        import torch
        self.model_path = model
        self.conf = conf if isinstance(conf, list) else [conf]
        self.model = self.init_model(model)
        self.infer_conf = min(conf) if isinstance(conf, list) else conf
        self.nms_iou = nms_iou
        self.device = torch.device(device)

    def init_model(self, model_path: str = None):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装 ultralytics 库, CLI: `pip install -U ultralytics`")
        model_path = 'yolo11l.yaml' if model_path is None else model_path
        model = YOLO(model_path, verbose=False)
        model_classes_length = len(model.names)
        if len(self.conf) < model_classes_length:
            self.conf *= model_classes_length
        return model
    
    @staticmethod
    def split(src_box) -> list:
        """根据原始图片的bbox, 自定义图片裁切方案, 返回裁切后子图的bbox列表

        :param src_box: 待裁切图片区域的bbox
        :type src_box: list, ins. [x1, y1, x2, y2]
        :return: list of [a1, b1, a2, b2]
        :rtype: list[list[int]]
        """
        return [src_box]
    
    @staticmethod
    def merge(results: list) -> list:
        """将多个子图的预测结果合并为最终结果

        :param results: list of ins. {'box': [x1, y1, x2, y2], 'label': 'xxx', 'confidence': 0.9, 'segment': [...]}
        :type results: list
        :return: list of ins. {'box': [x1, y1, x2, y2], 'label': 'xxx', 'confidence': 0.9, 'segment': [...]}
        :rtype: list
        """
        return results
    
    def infer(self, src_img: np.ndarray, windows: list[list], **kwargs) -> list:
        """深度学习模型推理接口, 返回预测结果相对原始图像的坐标

        ### 重构注意事项
            - 返回结构体格式：参考 InferBase.save_infer_result.__doc__
            - verbose全部设置为False, 避免多余的打印
            - 打印信息统一使用logger进行打印, 不要在细节处设置logger配置

        :param imgs: 待推理图片
        :type imgs: numpy.ndarray
        :param windows: 窗口box列表, 如: [x1, y1, x2, y2]
        :type windows: list[list]
        """
        
        all_sub_preds = []
        all_sub_imgs = [src_img[box[1]:box[3], box[0]:box[2]] for box in windows]
        empty_cache = kwargs.get('empty_cache', None)
        if empty_cache is None:
            results = self.model(all_sub_imgs, verbose=False, conf=self.infer_conf, iou=self.nms_iou)
        elif isinstance(empty_cache, int):
            results = []
            for sub_img in all_sub_imgs:
                results.append(self.model(sub_img, verbose=False, conf=self.infer_conf, iou=self.nms_iou)[0])
                torch_empty_cache(empty_cache)

        for i, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            cls_ids = result.boxes.cls.cpu().numpy().tolist()
            scores = result.boxes.conf.cpu().numpy().tolist()
            sub_preds = [{
                'name': result.names[int(obj[0])],
                'code': int(obj[0]),
                'box': abs_box(windows[i], obj[2]), # 坐标还原
                'confidence': obj[1],
            } for obj in zip(cls_ids, scores, boxes) if  obj[1] >= self.conf[int(obj[0])]]
            
            # 坐标还原
            all_sub_preds.extend(sub_preds)
        return all_sub_preds
    
    def __call__(self, src_img: np.ndarray, **kwargs) -> list:
        img_shape = src_img.shape[:2]
        src_box = [0, 0, img_shape[1], img_shape[0]]
        all_windows = self.split(src_box)
        all_sub_preds = self.infer(src_img, all_windows, **kwargs)
        results = self.merge(all_sub_preds)
        return results


@InferRegistry
class PredictBase(DetectBase):
    """模型推理的基类, 推理结果保存到dst_dir / expName 文件夹下, 文件存在则添加编号, 编码从1开始
    推理基类包含以下功能：
        - 图片[视频]资源加载：返回一个生成器对象, 元素是推理接口的输入数据
        - 图片[视频]拆分：根据输入数据, 调用split方法, 得到子图的bbox列表, 并返回子图列表
        - 子图推理: 调用infer方法, 得到子图的预测结果, 并返回预测结果列表
        - 结果合并: 调用merge方法, 将多个子图的预测结果合并为最终结果
        - 结果保存: 将预测结果以labelme格式保存到实验目录下, 并同步图片[视频]资源(软连接)到实验目录下

    ## 标准result输出格式

        ```python
        labelme_dict = {
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
                }
            ],
            "imagePath": "example.jpg",
            "imageData": None,
            "imageHeight": 300,
            "imageWidth": 400
        }
        ```

    ## 标准infer接口返回格式
        ```python
        result = [{
            'name': 'car',
            'code': 1,
            'box': [100, 100, 200, 200],
            'confidence': 0.9,
            'segment': [[100, 100], [200, 100], [200, 200], [100, 200]]
        }]
        ```
    
    ## 自定义功能模块
        1. load_entity : 根据不同类型实体加载识别数据源
        2. load_datasets : 加载指定数据集的图片, 返回一个生成器对象, 第一个元素是items数量
        3. split : 根据原始图片的bbox, 自定义图片裁切方案, 返回裁切后子图的bbox列表
        4. infer : 调用深度学习模型推理接口, 返回预测结果相对原始图像的坐标
        5. merge : 将多个子图的预测结果合并为最终结果
        6. save_infer_result : 保存推理结果为labelme格式标注文件
    
    默认实现是基于ultralytics的yolo模型, 若需要使用其他模型, 请重写infer方法.

    
    ## Example
    ```python
    data_name = 'srcLabelTest-test'
    infer = PredictBase(
        src_dir=f'/data1/2024_datasets/xxx/{data_name}/images',
        dst_dir='/data1/2025_datasets/',
        project=data_name
    )
    infer(
        verbose=True, pattern='yolo', 
        model_path='path/to/model.pt'
    )
    ```
    """

    def __init__(
            self, src_dir, dst_dir, project: str = 'inference', 
            model: str = None, device: str = 'cuda:0', conf: list = None, nms_iou: float = None):
        super().__init__(model, device, conf, nms_iou)
        self.src_dir = self.src_valid(src_dir)
        self.project = project
        self.dst_dir = get_exp_dir(dst_dir, project)

    @classmethod
    def get_defult_dict(cls, img_path: str = "", img_shape: tuple = (1080, 1920), version: str = '4.5.6') -> dict:
        return {
            "version": version,
            "flags": {},
            "shapes": [],
            "imagePath": str(img_path),
            "imageData": None,
            "imageHeight": img_shape[0],
            "imageWidth": img_shape[1]
        }

    @staticmethod
    def src_valid(src_dir):
        if isinstance(src_dir, (str, PosixPath)):
            all_img_dirs = [src_dir]
        else:
            all_img_dirs = src_dir
        result = [Path(img_dir) for img_dir in all_img_dirs]
        return result

    def load_datasets(self):
        """生成所有图片的路径生成器对象, 并设置第一个输入是items数量
        
        ## 数据集默认存储结构
        ```shell
        $ tree -d -L 2
        .
        ├── srcLabelTrain-train1
        │   ├── images  <--- 需要指定的目录
        │   └── labels  # 标签目录[xmls, labels]
        └── srcLabelTrain-train2
            ├── images
            └── labels
        ```

        :return: 一个生成器对象, iter of img_file
        :rtype: generator # 生成器对象, 第一个元素是items数量
        """
        
        all_img_dirs = self.src_dir
        total_num = 0
        for img_dir in all_img_dirs:
            if not img_dir.exists():
                raise ValueError(f"图片目录{img_dir}不存在")
            for file in img_dir.iterdir():
                if not file.is_file():
                    continue
                total_num += 1
        yield total_num

        for sub_dir in all_img_dirs:
            for file in sub_dir.iterdir():
                if not file.is_file():
                    continue
                yield file

    def load_entity(self, entity_file: str):
        """根据不同识别实体加载识别数据源

        :param entity_file: 实体对象, 如：'img_path', 'test.mp4' ... 不同格式需要自己适配.
        :type entity_file: str
        :return: _description_
        :rtype: _type_
        """
        # 使用opencv加载图片
        src_img = cv.imread(str(entity_file))
        return src_img

    @classmethod
    def save_infer_result(cls, lbl_path: str, lbl_dict: dict, results: list):
        """保存推理结果为labelme格式标注文件

        ### result的标准格式定义
            ```python
            result = [{
                'name': 'car',
                'code': 1,
                'box': [100, 100, 200, 200],
                'confidence': 0.9,
                'segment': [[100, 100], [200, 100], [200, 200], [100, 200]]
            }]
            ```

        :param lbl_path: 文件保存地址
        :type lbl_path: str
        :param lbl_dict: 文件头对象
        :type lbl_dict: dict
        :param results: shapes对象源数据
        :type results: list
        """

        lbl_dict['shapes'] = [{
            'label': obj['name'],
            'points': obj['segment'] if obj.get('segment', None) else [obj['box'][:2], obj['box'][2:4]],
            'group_id': None,
            'shape_type': 'polygon' if obj.get('segment', None) else 'rectangle',
            'flags': {},
            'code': obj.get('code', -1)
        } for obj in results]
        save_labelme_label(lbl_path, lbl_dict)

    def execute(self, img_file: str, **kwargs) -> dict: 
        """根据图片和标签文件路径, 调用split, infer, merge方法, 完成单张图片的推理, 并返回预测结果

        :param img_file: 图片文件路径
        :type img_file: str
        :param lbl_file: 标注文件路径
        :type lbl_file: str
        :return: 参考 InferBase.__doc__
        :rtype: dict
        """
        
        img_file = Path(img_file)
        src_entity = self.load_entity(img_file)
        img_shape = src_entity.shape[:2]
        labelme_img_path = img_file.name  # 保存labelme格式的图片路径
        sub_dir_save = self.dst_dir / img_file.parents[1].name  # 保存子图的路径
        sub_dir_save.mkdir(exist_ok=True, parents=True)
        labelme_lbl_path = sub_dir_save / (img_file.stem + '.json')  # 保存labelme格式的标签路径
        lbl_dict = self.get_defult_dict(img_path=labelme_img_path, img_shape=img_shape)
        src_box = [0, 0, src_entity.shape[1], src_entity.shape[0]]

        all_windows = self.split(src_box)
        all_sub_preds = self.infer(src_entity, all_windows, **kwargs)

        results = self.merge(all_sub_preds)
        # 保存labelme格式的图片和标签
        self.save_infer_result(labelme_lbl_path, lbl_dict, results)

        # 保存软连接
        if not (sub_dir_save / img_file.name).exists():
            (sub_dir_save / img_file.name).symlink_to(img_file)
        
        return results

    def __call__(self, *args, **kwargs):

        # 获取所有实验对象
        all_src_imgs = self.load_datasets()
        tasks_num = next(all_src_imgs)  # 读取数据集数量, 默认第一个返回值为items数量
        instance_batch_size = getattr(self, 'batch_size', 4)
        batch_size = kwargs.get('batch_size', instance_batch_size)
        epoch_num = math.ceil(tasks_num / batch_size)
        bar_desc = colorstr("bright_blue", "bold", "Inference")

        # for img_file in all_src_imgs:  # 调试代码段
        #     self.execute(img_file, **kwargs)

        with ThreadPoolExecutor(max_workers=cpu_num) as executor:
            # 开始处理子数据集
            with tqdm(total=tasks_num, desc=bar_desc, unit='img', leave=True, position=0, dynamic_ncols=True) as epoch_bar:
                for epoch in range(epoch_num):
                    start_idx = epoch * batch_size
                    end_idx = min(tasks_num, (epoch + 1) * batch_size)
                    
                    inner_tasks = []
                    epoch_size = end_idx - start_idx
                    for _ in range(start_idx, end_idx):
                        img_file = next(all_src_imgs)
                        inner_tasks.append(executor.submit(self.execute, img_file, **kwargs))
                    
                    # 更新进度条
                    for ti, task in enumerate(as_completed(inner_tasks), start=1):
                        reulst = task.result()
                        # TODO: 处理进度条打印内容
                        epoch_bar.set_postfix({
                            BATCH_KEY: f"{ti}/{epoch_size}", 
                            START_KEY: start_idx, 
                            END_KEY: end_idx
                        })
                    
                        epoch_bar.update()


class StatisticSimple(object):
    """统计基类

    :param Literal[".txt", ".json", ".xml"] pred_suffix: 预测文件的后缀类型, defaults to ".json"
    :param Literal[".txt", ".json", ".xml"] gt_suffix: 标注文件的后缀类型, defaults to ".json"
    :param dict suffix_load_func: 各类标签加载的方法, defaults to None
    """

    def __init__(
            self, pred_suffix: Literal[".txt", ".json", ".xml"] = ".json",
            gt_suffix: Literal[".txt", ".json", ".xml"] = ".json", classes: str = None,
            chinese: str | bool = False, suffix_load_func: dict = None, **kwargs
        ):
        """
        :param Literal[.txt, .json, .xm] pred_suffix: 预测文件的后缀类型, defaults to ".json"
        :param Literal[.txt, .json, .xm] gt_suffix: 标注文件的后缀类型, defaults to ".json"
        :param dict suffix_load_func: 各类标签加载的方法, defaults to None
        :param str|list|dict classes: 类别文件路径(支持list, dict), defaults to None
        """
        self.pred_suffix = pred_suffix
        self.gt_suffix = gt_suffix
        self.suffix_load_func = {
            ".txt": read_yolo,
            ".json": parser_json,
            ".xml": read_voc
        }
        if suffix_load_func:
            self.suffix_load_func.update(suffix_load_func)
        self.classes = self.get_classes(classes)
        self.background = True  # 标记是否有背景类
        self.is_yolo_lbl = gt_suffix == ".txt" or pred_suffix == ".txt"
        if chinese:
            ConfusionMatrix.set_plt(font_path=chinese if isinstance(chinese, str) else None)
    
    @classmethod
    def get_classes(cls, class_file: str) -> list:
        """获取类别列表

        :param class_file: 类别文件路径
        :type class_file: str
        """
        if isinstance(class_file, (str, PosixPath)):
            classes = read_txt(class_file)
        elif isinstance(class_file, list):
            classes = class_file
        elif isinstance(class_file, dict):  # 传入是names时, 取name列表得到classes
            classes = [class_file[i] for i in sorted(class_file.keys())]
        else:
            raise ValueError(f"不支持的类别文件类型{type(class_file)}")
        classes.append('background')
        
        return classes

    def load_lbl_data(self, lbl_file: str, suffix: str) -> dict:
        """加载单个预测结果文件, 返回预测结果[dict]

        :param lbl_file: 预测结果文件路径
        :type lbl_file: str
        :param suffix: 标签文件后缀名
        :type suffix: str
        :return: 预测结果[dict]
        """
        lbl_data = self.suffix_load_func[suffix](lbl_file)
        return lbl_data

    @classmethod
    def labelme2match(cls, pred_entities: dict, use_conf: bool = False, **kwargs) -> list:
        """将labelme格式的对象转换为匹配格式, 匹配格式由自定义匹配模块定义

        :param pred_entities: labelme格式的预测结果对象
        :type pred_entities: dict
        :param use_conf: 是否使用置信度, defaults to False
        :type use_conf: bool, optional
        """
        pred_boxes = []
        for shape in pred_entities['shapes']:
            box_cls = shape['label']  # yolo模型注入的类别编号
            x1, y1, x2, y2 = shape['points'][0][0], shape['points'][0][1], shape['points'][1][0], shape['points'][1][1]
            box = [x1, y1, x2, y2]
            if use_conf:
                conf = min(shape.get('conf', 1.0), shape.get("score", 1.0))
                pred_boxes.append([box_cls, conf,  *box])
            else:
                pred_boxes.append([box_cls, *box])
        
        return pred_boxes
    
    def yolo2match(self, gt_entities: list, use_conf: bool = False, **kwargs) -> list:
        """将标注文件加载内容转为匹配格式, 匹配格式由自定义匹配模块定义

        :param gt_entities: 标注文件内容
        :type gt_entities: list
        :param use_conf: 是否使用置信度, defaults to False
        :type use_conf: bool, optional

        Note: 带conf的实例格式是: [class_id, x1, y1, x2, y2, conf]
        """
        imgh, imgw = kwargs.get('img_shape', (1, 1))
        gt_boxes = []
        for entity in gt_entities:
            x, y, w, h = entity[1:5]
            x1 = max(0, (x - w / 2) * imgw)
            y1 = max(0, (y - h / 2) * imgh)
            x2 = min(imgw, (x + w / 2) * imgw)
            y2 = min(imgh, (y + h / 2) * imgh)
            label = self.classes[entity[0]+1] if self.background else self.classes[entity[0]]
            if use_conf:
                conf = entity[5] if len(entity) == 6 else 1.0
                gt_boxes.append([label, conf, x1, y1, x2, y2])
            else:
                gt_boxes.append([label, x1, y1, x2, y2])
        return gt_boxes
    
    @classmethod
    def voc2match(cls, gt_entities: dict, use_conf: bool = False, **kwargs) -> list:
        """将voc格式的标注文件加载内容转为匹配格式, 匹配格式由自定义匹配模块定义

        :param gt_entities: voc格式的标注文件内容
        :type gt_entities: dict
        :param use_conf: 是否使用置信度, defaults to False
        :type use_conf: bool, optional
        """
        result = []
        for obj in gt_entities["object"]:
            label = obj["name"]
            x1, y1, x2, y2 = obj["bndbox"]["xmin"], obj["bndbox"]["ymin"], obj["bndbox"]["xmax"], obj["bndbox"]["ymax"]
            if use_conf:
                conf = obj.get('confidence', 1.0)
                result.append([label, conf, x1, y1, x2, y2])
            else:
                result.append([label, x1, y1, x2, y2])
        return result
    
    def middle2match(self, entities: list | dict, suffix: str = None, use_conf: bool = False, **kwargs) -> list:
        """从labelme|yolo|voc格式的标注文件加载内容转为匹配格式, 匹配格式由自定义匹配模块定义

        :param list entities: 各种格式的标签直接加载的对象
        :param str suffix: 标签文件后缀名
        :param bool use_conf: 是否使用置信度
        :return list: 示例列表
        """
        if entities is None:
            return []
        if suffix is None or suffix == ".json":
            return self.labelme2match(entities, use_conf=use_conf, **kwargs)
        elif suffix == ".txt":
            return self.yolo2match(entities, use_conf=use_conf, **kwargs)
        elif suffix == ".xml":
            return self.voc2match(entities, use_conf=use_conf, **kwargs)
        else:
            raise ValueError(f"不支持的标签文件后缀名{suffix}")

    def get_image_shape(self, pred_entities, gt_entities, **kwargs) -> tuple:
        if not self.is_yolo_lbl:
            img_shape = (1, 1)
        if self.pred_suffix == ".json" and pred_entities is not None:
            img_shape = (pred_entities['imageHeight'], pred_entities['imageWidth'])
        elif self.pred_suffix == ".xml" and pred_entities is not None:
            img_shape = (pred_entities["size"]["height"], pred_entities["size"]["width"])
        elif self.gt_suffix == ".json" and gt_entities is not None:
            img_shape = (gt_entities['imageHeight'], gt_entities['imageWidth'])
        elif self.gt_suffix == ".xml" and gt_entities is not None:
            img_shape = (gt_entities["size"]["height"], gt_entities["size"]["width"])
        else:
            img_shape = (1, 1)
        return img_shape


class StatisticConfusion(StatisticSimple):
    """加载 PredictBase 推理结果 和 标签文件, 统计各类别的数量, 并保存到统计文件中
    推理结果文件夹和标注文件夹需要对应, 文件夹名称可以不一样.

    注: 全流程默认使用ultralytics的yolo模型, 若需要使用其他模型, 请重写以下方法:\n
        - gt2match : gt格式的对象转换为匹配格式, 匹配格式由自定义匹配模块定义
        - load_lbl_data : 读取标签文件, 返回标签对象, 目前只支持txt, json, xml格式
        - xxx2match: 各种数据格式转换为匹配格式, 匹配格式由自定义匹配模块定义
        - get_image_shape: 读取图片的shape, 用于YOLO标注box百分比坐标还原
        - match: 自定义匹配模块, 输入为gt和pred, 返回匹配结果

    Args:
        src_gt (list[str]): 标签文件路径, 支持多个数据子集，需要指定到数据子集的标签文件存放路径
        src_pred (list[str]): 推理结果文件路径, 支持多个数据子集，需要指定到数据子集的推理结果文件存放路径
        dst_dir (str): 预测结果保存目录
        project (str, optional): 实验名称, defaults to 'inference'.
        use_ios (bool, optional): 是否使用IOS计算, defaults to True
        classes (str, optional): 类别文件路径, defaults to 'classes.txt'.
        chinese (bool, optional): 是否使用中文类别, 可以指定中文字体文件的路径, defaults to True
        gt_suffix (str, optional): 标签文件后缀名, defaults to '.txt'.
        pred_suffix (str, optional): 推理结果文件后缀名, defaults to '.json'.
    
    注意: gt_suffix 和 pred_suffix 可以任选txt, json, xml格式, 但是若都使用YOLO格式, 则算法失效(不能得到像素坐标).
        
    Example:
        ```python
        >>> statistic = StatisticBase(
        ...     src_gt=['/data1/2024_datasets/infenrence/srcLabelTest-test/labels'],  # 标签文件路径
        ...     src_pred=[infer.dst_dir],  # 预测文件路径
        ...     dst_dir='/data1/2025_datasets/',  # 实验存放的根目录
        ...     project='statistic',  # 实验名称, 程序会默认追加 'Statistic'
        ...     use_ios=True,  # 是否使用IOS计算匹配程度
        ...     classes='/data1/classes.txt'  # 类别文件路径, YOLO格式
        ... )
        >>> statistic(
        ...     rendering=True,  # 是否渲染统计结果
        ...     ios_thresh=0.5,
        ...     iou_thresh=0.5
        ... )
        ```
    """

    def __init__(
            self, src_gt: list[str], src_pred: list[str], dst_dir: str, 
            project: str = 'inference', use_ios: bool = True, 
            classes: str = 'classes.txt', chinese: bool = True, 
            gt_suffix: Literal[".txt", ".json", ".xml"] = '.txt', 
            pred_suffix: Literal[".txt", ".json", ".xml"] = '.json', 
            use_fpfn: bool = False, **kwargs
        ):
        """初始化统计类

        :param src_gt: 标签文件路径
        :type src_gt: list[str]
        :param src_pred: 推理结果文件路径
        :type src_pred: list[str]
        :param dst_dir: 预测结果保存目录
        :type dst_dir: str
        :param project: 实验名称, defaults to 'inference'
        :type project: str, optional
        :param use_ios: 是否使用IOS计算, defaults to True
        :type use_ios: bool, optional
        :param classes: 类别文件路径, defaults to 'classes.txt'
        :type classes: str, optional
        :param chinese: 是否使用中文类别, 可以指定中文字体文件的路径, defaults to True
        :type chinese: bool, optional
        :param gt_suffix: 标签文件后缀名, defaults to '.txt'
        :type gt_suffix: Literal[".txt", ".json", ".xml"], optional
        :param pred_suffix: 推理结果文件后缀名, defaults to '.json'
        :type pred_suffix: Literal[".txt", ".json", ".xml"], optional
        :param use_fpfn: 是否使用FP, FN保存为子数据集, defaults to False
        :type use_fpfn: bool, optional
        """
        super().__init__(
            gt_suffix=gt_suffix, pred_suffix=pred_suffix, 
            suffix_load_func=kwargs.get('suffix_load_func', None),
            classes=classes, chinese=chinese, **kwargs
        )
        assert gt_suffix in self.suffix_load_func, f"不支持的标签文件后缀名{gt_suffix}"
        assert pred_suffix in self.suffix_load_func, f"不支持的预测文件后缀名{pred_suffix}"
        self.src_gt = path_list_valid(src_gt)
        self.src_pred = path_list_valid(src_pred)
        self.dst_dir = get_exp_dir(dst_dir, project + "_Statistic")
        self.project = project + "_Statistic"
        self.use_ios = use_ios
        self.background = False
        
        # 初始化统计的matrix
        self.matrix = ConfusionMatrix(len(self.classes), self.classes)
        self.use_fpfn = use_fpfn
    
    def load_datasets(self):
        """从预测文件加载数据集, 返回一个生成器对象, 第一个元素是items数量

        :yield: 预测文件, 标签文件元组
        """

        # 统计所有的预测结果文件, 没有预测预测文件也会保存
        datasets = self.src_pred
        
        # 统计预测文件数量
        total_num = 0
        for sub_datasets in datasets:
            if not sub_datasets.exists():
                raise ValueError(f"预测结果目录{sub_datasets}不存在")
            for file in sub_datasets.iterdir():
                if file.suffix != self.pred_suffix:
                    continue
                total_num += 1
        yield total_num

        for i, sub_datasets in enumerate(datasets):
            for file in sub_datasets.iterdir():
                if file.suffix != self.pred_suffix:
                    continue
                lbl_file = Path(self.src_gt[i]) / (file.stem + self.gt_suffix)
                yield file, lbl_file

    def update(self, update_dict: dict):
        """更新统计矩阵

        :param update_dict: 更新字典, 格式为{类别: 预测向量}, 如: {"background": [1,0,0,1,0,1]}
        :type update_dict: dict
        """
        for key, value in update_dict.items():
            key_index = self.classes.index(key)
            self.matrix.matrix[:, key_index] += value

    def save_fpfn_pipeline(self, match_object: dict, gt_file: str, img_shape: tuple):
        """保存FP, FN实例为子数据集, 标签使用labelme格式
        :param match_object: 匹配结果
        :type match_object: dict
        :param gt_file: 标注文件路径
        :type gt_file: str
        :param img_shape: 图片shape
        :type img_shape: tuple

        note: match_object['boxesStatus']中保存了TPG, TPP, FP, FN四种实例列表, \
            实例的保存格式是: [label, x1, y1, x2, y2]
        """

        fp_list = match_object['boxesStatus']['fp']
        fn_list = match_object['boxesStatus']['fn']
        if len(fp_list) == 0 and len(fn_list) == 0:
            return None
        
        tpg_list = match_object['boxesStatus']['tpg']
        labelme_dict = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": f"{gt_file.stem}.jpg",
            "imageData": None,
            "imageHeight": img_shape[0],
            "imageWidth": img_shape[1],
        }

        def _add_instance(instance_list: list[list], add_suffix: str = ""):
            for instance in instance_list:
                label, x1, y1, x2, y2 = instance
                labelme_dict['shapes'].append({
                    "label": f"{label}-{add_suffix}",
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {"predStatus": add_suffix}
                })
        
        _add_instance(tpg_list)
        _add_instance(fp_list, add_suffix='fp')
        _add_instance(fn_list, add_suffix='fn')

        save_file_path = self.dst_dir / "fpfn_datasets" / f"{gt_file.stem}.json"
        save_file_path.parent.mkdir(exist_ok=True, parents=True)
        save_labelme_label(save_file_path, labelme_dict)

    def match(self, pred_file: str, gt_file: str, **kwargs):
        pred_entities = self.load_lbl_data(pred_file, self.pred_suffix)
        gt_entities = self.load_lbl_data(gt_file, self.gt_suffix)
        img_shape = self.get_image_shape(pred_entities, gt_entities, **kwargs)
        # 匹配预测结果和标签文件
        pred_boxes = self.middle2match(pred_entities, suffix=self.pred_suffix, img_shape=img_shape)
        gt_boxes = self.middle2match(gt_entities, suffix=self.gt_suffix, img_shape=img_shape)
        # 计算IoU
        ios_thresh = kwargs.get('ios_thresh', 0.5)
        iou_thresh = kwargs.get('iou_thresh', 0.5)
        match_object = yolo_match(
            pred_boxes, gt_boxes, 
            use_ios=self.use_ios, mode="xyxy", 
            iou_thresh=iou_thresh, ios_thresh=ios_thresh,
            classes=self.classes
        )
        
        # 更新统计实验数据
        self.update(match_object['updateItems'])

        # 保存GT和误报实例为数据子集, 标签使用labelme格式
        if self.use_fpfn:
            self.save_fpfn_pipeline(match_object, gt_file, img_shape)

        return True

    def __call__(self, iou_thresh: float = 0.5, ios_thresh: float = 0.5, rendering: bool = False, *args, **kwargs):
        """统计接口

        :param float iou_thresh: 交并比阈值, defaults to 0.5
        :param float ios_thresh: 交自比阈值, defaults to 0.5
        :param bool rendering: 是否渲染, defaults to False
        """
        # TODO: 增加字体渲染size、token渲染size信息判别
        
        entities_generator = self.load_datasets()
        total_num = next(entities_generator)
        bar_desc = colorstr("bright_blue", "bold", "Statistic")
        instance_batch_size = getattr(self, 'batch_size', 4)
        batch_size = kwargs.get('batch_size', instance_batch_size)
        epoch_num = math.ceil(total_num / batch_size)

        with ThreadPoolExecutor(max_workers=cpu_num) as executor:
            with tqdm(total=total_num, desc=bar_desc, unit='img', leave=True, position=0, dynamic_ncols=True, colour="#CD8500") as epoch_bar:
                for epoch in range(epoch_num):
                    start_idx = epoch * batch_size
                    end_idx = min(total_num, (epoch + 1) * batch_size)
                    
                    inner_tasks = []
                    epoch_size = end_idx - start_idx
                    for _ in range(start_idx, end_idx):
                        pred_file, lbl_file = next(entities_generator)
                        inner_tasks.append(
                            executor.submit(
                                self.match, pred_file, lbl_file, 
                                ios_thresh=ios_thresh, iou_thresh=iou_thresh, rendering=rendering
                            )
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

        # 保存统计结果
        self.matrix.save_figure(self.dst_dir / f"{self.dst_dir.name}_confusion_matrix.png")
        self.matrix.save_figure(self.dst_dir / f"{self.dst_dir.name}_confusion_matrix_normalized.png", mode='normalize')
        self.matrix.save_xlsx(self.dst_dir / f"{self.dst_dir.name}_confusion_matrix.xlsx")


class StatisticBase(StatisticConfusion):
    pass

@StatisticRegistry
class StatisticMatrix(StatisticSimple):
    """预测分布指标统计
    
    :param list[str] pred_dir: 预测结果目录, defaults to None
    :param list[str] gt_dir: 标注文件目录, defaults to None
    :param str dst_dir: 结果保存目录, defaults to "."
    :param str project: 实验名称, defaults to "detection"
    :param bool plot: 是否绘制统计图, defaults to False
    :param list | dict | str names: 类别名称, defaults to {}
    :param str pred_suffix: 预测结果文件后缀名, defaults to ".json"
    :param str gt_suffix: 标注文件后缀名, defaults to ".json"
    tp (np.ndarray): True positive array.
    conf (np.ndarray): Confidence array.
    pred_cls (np.ndarray): Predicted class indices array.
    target_cls (np.ndarray): Target class indices array.
    on_plot (callable, optional): Function to call after plots are generated.

    """

    def __init__(
            self, pred_dir: list[str] = None, gt_dir: list[str] = None, 
            dst_dir=Path("."), project: str = "detection", plot=False,
            classes: list | dict | str = {}, pred_suffix=".json", gt_suffix=".json",
            verbose=False, chinese=False, **kwargs
        ):
        super().__init__(
            pred_suffix=pred_suffix, gt_suffix=gt_suffix,
            classes=classes, chinese=chinese, **kwargs
        )
        assert pred_dir is not None and gt_dir is not None, "预测结果目录和标注文件目录不能为空"
        self.pred_dir = pred_dir if isinstance(pred_dir, list) else [pred_dir]
        self.gt_dir = gt_dir if isinstance(gt_dir, list) else [gt_dir]
        self.pred_dir = [Path(p) for p in self.pred_dir]
        self.gt_dir = [Path(p) for p in self.gt_dir]
        self.dst_dir = dst_dir if isinstance(dst_dir, PosixPath) else Path(dst_dir)
        self.save_dir = get_exp_dir(self.dst_dir, project + "_Matrix")
        self.names = self.load_names(classes)
        self.nc = len(self.names)
        self.pred_suffix = pred_suffix
        self.gt_suffix = gt_suffix
        self.verbose = verbose
        self.seen = 0
        self.table = None
        
        self.matrix = DetMetrics(save_dir=self.save_dir, plot=plot, names=self.names, **kwargs)
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.iou_vector = np.linspace(0.5, 0.95, 10)  # IoU阈值向量
        self.num_iou = self.iou_vector.size  # IoU阈值数量
    
    @staticmethod
    def load_names(names: list | dict | str):
        if isinstance(names, (str, PosixPath)):
            names_obj = read_txt(names)
        else:
            names_obj = names
        
        if isinstance(names_obj, list):
            names_dict = {i: v for i, v in enumerate(names_obj)}
        elif isinstance(names_obj, dict):
            names_dict = names_obj
        else:
            raise ValueError(f"不支持的names数据类型{type(names_obj)}")
        return names_dict
        
    def load_items(self):
        pred_num = 0
        for pred_dir in self.pred_dir:
            for pred_file in pred_dir.iterdir():
                if pred_file.suffix != self.pred_suffix:
                    continue
                pred_num += 1
        yield pred_num

        num = 0
        for pred_dir, gt_dir in zip(self.pred_dir, self.gt_dir):
            for pred_file in pred_dir.iterdir():
                if pred_file.suffix != self.pred_suffix:
                    continue
                gt_file = gt_dir / (pred_file.stem + self.gt_suffix)
                num += 1
                yield (pred_file, gt_file, num), dict()

    def match_cls_and_iou(self, pred_cls, gt_cls, iou_matrix):
        """根据IoU矩阵匹配预测结果和标签文件

        :param pred_cls: 预测框的预测类别数组
        :param gt_cls: 标注框的标签数组
        :param iou_matrix: IoU矩阵, 元素为[pred_idx, gt_idx, iou]
        """

        # 针对单张图片的预测与GT构建一个不同IOU阈值下的匹配记录矩阵
        correct = np.zeros((pred_cls.shape[0], self.iou_vector.shape[0])).astype(bool)
        # 记录预测与GT之间的类别匹配情况
        correct_cls = gt_cls[:, None] == pred_cls
        iou = iou_matrix * correct_cls

        for i, threshold in enumerate(self.iou_vector):
            matches = np.nonzero(iou > threshold)
            matches = np.array(matches).T  # 匹配上的索引, list of [pred_idx, gt_idx]
            if matches.shape[0]:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]  # 按照匹配的iou数值重排匹配数组
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]     # 去除重复的gt_idx
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]     # 去除重复的pred_idx
            correct[matches[:, 1].astype(int), i] = True                              # 标记每个预测对象在self.iou_vector[i]下是否有匹配
        
        return correct

    def match(self, pred_file: str, gt_file: str, num: int, **kwargs):
        
        pred_entities = self.load_lbl_data(pred_file, self.pred_suffix)
        gt_entities = self.load_lbl_data(gt_file, self.gt_suffix)
        img_shape = self.get_image_shape(pred_entities, gt_entities, **kwargs)
        # 匹配预测结果和标签文件
        ## pred_boxes的元素为: [label, conf, x1, y1, x2, y2]
        pred_boxes = self.middle2match(pred_entities, suffix=self.pred_suffix, img_shape=img_shape, use_conf=True)
        gt_boxes = self.middle2match(gt_entities, suffix=self.gt_suffix, img_shape=img_shape)
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 5))
        pred_boxes = np.array(pred_boxes) if pred_boxes else np.zeros((0, 6))

        stat = dict(
            conf=pred_boxes[:, 1].astype(float),  # 预测对象默认格式是: [label, conf, x1, y1, x2, y2]
            tp=np.zeros((pred_boxes.shape[0], self.num_iou), dtype=bool),
            pred_cls=pred_boxes[:, 0],
            target_cls=gt_boxes[:, 0],
            target_img=np.unique(gt_boxes[:, 0])
        )
        self.seen += 1

        # 没有预测时或着没有GT时, 立即整备self.stats, 并结束
        if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
            for k in self.stats.keys():
                self.stats[k].append(stat[k])
            return None

        # 计算iou匹配矩阵
        iou_matrix = iou_np_boxes(gt_boxes[:, 1:], pred_boxes[:, 2:])
        iou_cls_iou_vector = self.match_cls_and_iou(pred_boxes[:, 0], gt_boxes[:, 0], iou_matrix)
        stat['tp'] = iou_cls_iou_vector

        for k in self.stats.keys():
            self.stats[k].append(stat[k])

        return None
    
    @staticmethod
    def bincount(x, minlength=None):
        if x.shape[0] and isinstance(x[0], np.str_):
            res = np.unique(x, return_counts=True)[1]
        else:
            res = np.bincount(x.astype(int), minlength=minlength)
        return res
    
    def get_stats(self):
        """返回分布指标统计结果"""
        stats = {k: np.concatenate(v) for k, v in self.stats.items()}
        self.nt_per_class = self.bincount(stats['target_cls'], minlength=self.nc)
        self.nt_per_image = self.bincount(stats['target_img'], minlength=self.nc)
        stats.pop('target_img', None)
        if len(stats) and stats["tp"].any():
            self.matrix.process(**stats)
        return self.matrix.results_dict
    
    @staticmethod
    def number_format(x):
        return f"{x:11.4g}"
    
    def print_results(self):
        """Prints training/validation set metrics per class."""
        table = PrettyTable()
        table.field_names = ["class", "images", "objects", *self.matrix.keys]
        table.add_row(["all", self.seen, self.nt_per_class.sum(), 
                       *[self.number_format(x) for x in self.matrix.mean_results()]])

        if self.nt_per_class.sum() == 0:
            logger.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        self.name2id = {v: k for k, v in self.names.items()}
        if self.verbose and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.matrix.ap_class_index):
                table.add_row([
                    self.names[self.name2id[c]], 
                    self.nt_per_image[self.name2id[c]], 
                    self.nt_per_class[self.name2id[c]], 
                    *[self.number_format(x) for x in self.matrix.class_result(i)]
                ])
        print(table)
        self.table = table
    
    def __call__(self, *args, **kwargs):
        entities_generator = self.load_items()
        total_num = next(entities_generator)

        # 匹配预测结果和标签文件, 结果保存到self.match_matrix
        future_bar = FutureBar(total=total_num, desc="StatisticMatrix")
        future_bar(self.match, entities_generator, total=total_num)

        # 合并self.stats, 并计算指标
        self.get_stats()
        self.print_results()

        return self.matrix




    pass


class SlidingWindowBase(object):

    """滑窗的基类, 用于计算滑窗产生的子窗口坐标.

    Methods:
        match_coord: 根据横纵坐标x1,y1,x2,y2分割点, 计算分割的窗口坐标
        match_coord_by_split: 根据横纵坐标分割点, 计算分割的窗口坐标

    """

    @staticmethod
    def match_coord_by_split(width_cut_point: list, height_cut_point: list):
        """根据横纵坐标分割点, 计算分割的窗口坐标

        :param width_cut_point: 横坐标分割点列表
        :type width_cut_point: list | np.ndarray
        :param height_cut_point: 纵坐标分割点列表
        :type height_cut_point: list | np.ndarray
        """
        assert len(width_cut_point) > 1 and len(height_cut_point) > 1, "坐标分割点数量必须大于1"
        width_cut_matrix = np.array(width_cut_point)
        height_cut_matrix = np.array(height_cut_point)
        width_cut_matrix = width_cut_matrix.reshape(1, -1)
        height_cut_matrix = height_cut_matrix.reshape(-1, 1)
        width_cut_matrix = width_cut_matrix.repeat(height_cut_matrix.shape[0], axis=0)
        height_cut_matrix = height_cut_matrix.repeat(width_cut_matrix.shape[1], axis=1)
        windows = np.zeros((width_cut_matrix.shape[0] - 1, height_cut_matrix.shape[1] - 1, 4), dtype=np.int64)
        windows[:, :, 0] = width_cut_matrix[:-1, :-1]
        windows[:, :, 1] = height_cut_matrix[:-1, :-1]
        windows[:, :, 2] = width_cut_matrix[1:, 1:]
        windows[:, :, 3] = height_cut_matrix[1:, 1:]
        return windows
    
    @staticmethod
    def match_coord(x1: list, x2: list, y1: list, y2: list):
        """根据横纵坐标分割点, 计算分割的窗口坐标

        :param x1: box的左上角x坐标
        :type x1: list
        :param x2: box的右下角x坐标
        :type x2: list
        :param y1: box的左上角y坐标
        :type y1: list
        :param y2: box的右下角y坐标
        :type y2: list
        :return: 窗口坐标
        :rtype: np.ndarray
        """
        left = np.array(x1)
        right = np.array(x2)
        top = np.array(y1)
        bottom = np.array(y2)

        left_top_x = left.reshape(1, -1)
        left_top_y = top.reshape(-1, 1)
        right_bottom_x = right.reshape(1, -1)
        right_bottom_y = bottom.reshape(-1, 1)

        left_top_x = left_top_x.repeat(left_top_y.shape[0], axis=0)
        left_top_y = left_top_y.repeat(left_top_x.shape[1], axis=1)
        right_bottom_x = right_bottom_x.repeat(right_bottom_y.shape[0], axis=0)
        right_bottom_y = right_bottom_y.repeat(right_bottom_x.shape[1], axis=1)

        windows = np.zeros((left_top_x.shape[0], left_top_x.shape[1], 4), dtype=np.int64)
        windows[:, :, 0] = left_top_x
        windows[:, :, 1] = left_top_y
        windows[:, :, 2] = right_bottom_x
        windows[:, :, 3] = right_bottom_y

        return windows


@SliceRegistry
class BoostSlidingWindow(SlidingWindowBase):
    """步进滑窗: 滑动窗口类, 用于分割图片

    Example:

        ```python
        >>> from pprint import pprint
        >>> bsw = BoostSlidingWindow(640, overlap=0.2)
        >>> windows = bsw((1080, 1920))
        >>> pprint(windows)
        ```

    Methods:
        - __init__: 初始化滑动窗口类, window_size: 窗口大小, overlap: 窗口重叠率
        - calculate_cut_point: 根据图像的边长, 计算 X [Y] 的 X1,X2 [Y1,Y2] 分割点
        - __call__: 调用滑动窗口类, 计算窗口坐标, 返回窗口坐标列表, box对象是一个(x1, y1, x2, y2)元组
    
    Note:
        - 窗口坐标格式为(x1, y1, x2, y2)元组, 其中(x1, y1)为左上角坐标, (x2, y2)为右下角坐标
        - 窗口的高宽与图像的高宽一样, 计算为: width = x2 - x1, height = y2 - y1
        - 窗口的坐标范围为: x1 <= x <= x2, y1 <= y <= y2
        - 约定图像或者box的边框坐标为(x1, y1, x2, y2)格式, x1, y1都能取到, x2, y2都取不到, 契合python的切片索引
    """

    def __init__(self, window_size, overlap=0.2, **kwargs):
        super().__init__()
        assert 0 < overlap < 1, "overlap should be in (0, 1)"
        self.window_size = window_size
        self.overlap = overlap
        self.stride = int(window_size * (1 - overlap))

    def calculate_cut_point(self, size: int):
        """计算分割点
        - img
            - img_w --> img.shape[1]
            - img_h --> img.shape[0]
        - bbox
            - img_w --> bbox[:, 2] - bbox[:, 0] + 1
            - img_h --> bbox[:, 3] - bbox[:, 1] + 1

        :param size: 图片边长
        :type size: int
        """

        left_size = [i for i in range(0, size, self.stride) if i + self.window_size <= size]
        right_size = [i for i in range(self.window_size, size+1, self.stride) if i <= size]
        remainder_size = (size - right_size[-1]) if right_size else size
        if remainder_size > 10:
            right_size.append(size)
            left_size.append(max(0, (size - self.window_size)))
        return left_size, right_size
    
    def __call__(self, img_size: tuple[int], *args, **kwargs) -> list:
        left, right = self.calculate_cut_point(img_size[1])
        top, bottom = self.calculate_cut_point(img_size[0])
        windows = self.match_coord(left, right, top, bottom)
        windows = windows.reshape(-1, 4)
        windows = [tuple(box) for box in windows.tolist()]
        return windows


@CombineRegistry
class MergeSlidingBase(object):
    """预测合并基类

    原理：
        -> 1. *_merge: 搜索需要合并的对象, 接收标准预测格式list[dict], TODO: 适配numpy数据格式;
              返回合并的映射字典, 格式为{原box索引: 待合并的box索引}
        -> 2. combine: 根据映射字典和预测对象更新预测对象

    :param not_combine_ids:不需要进行合并的类别编码列表, 默认为空
    :type not_combine_ids: list, optional, default=False
    """

    def __init__(self, not_combine_ids: list = None, *args, **kwargs):
        self.not_combine_ids = not_combine_ids if not_combine_ids else []


    @staticmethod
    def combine(pred_boxes: list[dict], merge_dict: dict, **kwargs):
        """合并预测对象

        :param pred_boxes: 预测对象列表
        :type pred_boxes: list[dict]
        :param merge_dict: 合并映射字典
        :type merge_dict: dict
        """
        results = []
        for src_idx, combine_idx_list in merge_dict.items():
            target_box = pred_boxes[src_idx]
            if len(combine_idx_list):
                combine_boxes = np.array(
                    [[pred_boxes[i]['box'] for i in combine_idx_list] + [target_box['box']]]
                ).reshape(-1, 2)
                combine_box = combine_boxes.min(axis=0).tolist() + combine_boxes.max(axis=0).tolist()
                results.append({
                    "name": target_box['name'],
                    "code": target_box.get('code', -1),
                    "box": combine_box,
                    "confidence": target_box.get('confidence', 0.4),
                })
            else:
                results.append(target_box)
        return results

    def sahi_merge(self, pred_boxes: list[dict], mode: str = None, threshold: float = 0.5, greedy: bool = True, **kwargs):
        """sahi合并
        参考: https://github.com/obss/sahi/blob/main/sahi/postprocess/combine.py

        box object of pred_boxes:
        ```python
        >>> box = {
        ...     "name": "person",
        ...     "code": 72,
        ...     "box": [x1, y1, x2, y2],
        ...     "confidence": 0.89,
        ...     "segment": [x1, y1, x2, y2, ..., xn, yn],
        ... }
        ```

        greedy:
            - True: 贪婪合并, 递归合并所有满足阈值的box, 直到集和内的box和其他box都不能通过匹配阈值
            - False: 非贪婪合并, 循环所有对象box, box与剩余box的iou大于等于threshold时, 合并为一个box
        
        :param pred_boxes: DetectBase预测对象列表
        :type pred_boxes: list[dict]
        :param mode: 合并模式, 可选值: "iou", "ios"
        :type mode: str
        :param threshold: 合并阈值, 合并两个box的iou或ios大于等于threshold时, 合并为一个box
        :type threshold: float
        :param greedy: 是否贪婪合并, 即是否合并所有满足阈值的box, 贪婪时输出的框比较多, 非贪婪时会递归合并
        :type greedy: bool, optional, default=True
        """

        keep_to_merge_list = {}
        hit_dict = {}
        if mode is None:
            mode = "ios"
        mode = mode.lower()
        assert mode in ["iou", "ios"], "mode should be in ['iou', 'ios']"
        assert 0 <= threshold <= 1, "threshold should be in [0, 1]"

        pred_object = np.array([
            [*box['box'], box['confidence']] for box in pred_boxes
        ], dtype=np.float32)
        total_num = len(pred_object)
        if not total_num:
            return keep_to_merge_list
        
        scores = pred_object[:, 4]
        areas = (pred_object[:, 2]-pred_object[:, 0]) * (pred_object[:, 3]-pred_object[:, 1])
        order = scores.argsort()[::-1]  # Sort in descending order of confidence
        src_order = deepcopy(order) if not greedy else None
        
        search_idx = 0
        while len(order) > 0:
            if search_idx == total_num:
                break

            idx = order[-1] if greedy else search_idx  # greedy=False时，idx是int
            order = order[:-1] if greedy else src_order[src_order!= idx]
            search_idx += 1
            if len(order) == 0:
                # 没有剩余box就退出循环
                keep_to_merge_list[idx.tolist() if greedy else idx] = []
                break
            
            # 根据索引选择剩余的box对象
            rx1 = pred_object[order, 0]
            ry1 = pred_object[order, 1]
            rx2 = pred_object[order, 2]
            ry2 = pred_object[order, 3]

            ix1 = np.maximum(pred_object[idx, 0], rx1)
            iy1 = np.maximum(pred_object[idx, 1], ry1)
            ix2 = np.minimum(pred_object[idx, 2], rx2)
            iy2 = np.minimum(pred_object[idx, 3], ry2)

            # 计算交集面积
            w = ix2 - ix1
            h = iy2 - iy1
            w = np.maximum(0.0, w)
            h = np.maximum(0.0, h)
            inter = w * h

            rem_areas = areas[order]

            # 根据模式选择匹配方式
            if mode == "iou":
                ious = inter / (areas[idx] + rem_areas - inter)
            elif mode == "ios":
                smaller = np.minimum(areas[idx], rem_areas)
                ious = inter / smaller

            # 选择满足阈值的box
            mask = ious < threshold
            matched_box_idx = order[(mask == False).nonzero()[0]]
            
            
            if greedy:
                # 贪婪模式, 递归合并
                unmatched_box_idx = order[(mask == True).nonzero()[0]]
                order = unmatched_box_idx[scores[unmatched_box_idx].argsort()]
                keep_to_merge_list[idx.tolist()] = matched_box_idx.tolist()
            elif idx not in hit_dict:
                # 非贪婪模式
                keep_to_merge_list[idx] = []
                for matched_box_ind in matched_box_idx.tolist():
                    if matched_box_ind not in hit_dict:
                        keep_to_merge_list[idx].append(matched_box_ind)
                        hit_dict[matched_box_ind] = idx
            else:
                # 非贪婪模式, 合并
                keep = hit_dict[idx]
                for matched_box_ind in matched_box_idx.tolist():
                    if matched_box_ind not in keep_to_merge_list and matched_box_ind not in hit_dict:
                        keep_to_merge_list[keep].append(matched_box_ind)
                        hit_dict[matched_box_ind] = keep

        return keep_to_merge_list

    def merge(self, pred_boxes: list[dict], **kwargs) -> list[dict]:
        """合并预测对象

        kwargs:
            - mode: 合并模式, 可选值: "iou", "ios"
            - threshold: 合并阈值, 合并两个box的iou或ios大于等于threshold时, 合并为一个box
            - greedy: 是否贪婪合并, 即是否合并所有满足阈值的box, 贪婪时输出的框比较多, 非贪婪时会递归合并

        :param pred_boxes: 预测对象列表
        :type pred_boxes: list[dict]
        :return: 合并后的预测对象列表
        :rtype: list[dict]
        """
        # 按类别存储预测到字典
        pred_dict = {}
        for box in pred_boxes:
            code = int(box.get('code', -1))
            if code not in pred_dict:
                pred_dict[code] = []
            pred_dict[code].append(box)
        
        results = []
        for class_id, boxes_list in pred_dict.items():
            if class_id in self.not_combine_ids:
                results.extend(boxes_list)
                continue
            keep_to_merge_list = self.sahi_merge(boxes_list, **kwargs)
            new_pred_boxes = self.combine(boxes_list, keep_to_merge_list, **kwargs)
            results.extend(new_pred_boxes)

        # keep_to_merge_list = self.sahi_merge(pred_boxes, **kwargs)
        # new_pred_boxes = self.combine(pred_boxes, keep_to_merge_list, **kwargs)
        return results


@SliceRegistry
class PresetAdaptiveWindow(object):
    """根据预设窗口数量生成裁切窗口, 窗口数量小于等于window_num

    step1: 先聚类, 并将类内的box融合为一个框;\n
    step2: 根据window_size, max_scale_ratio, 合并窗口, 满足合并后的窗口size不大于max_scale_ratio倍的窗口大小;\n
    step3: 窗口边界调整, 以满足window_size和window_num的要求;\n

    Args:
        window_num (int): 预设窗口数量
        same_ratio (bool, optional): 是否保持窗口比例, defaults to False
          - True: 高宽保持一样的放缩倍率, 使得高宽比和window_size保持一致
          - False: 类别窗口较大时, 不改变box
        keep_size (bool, optional): 是否保持窗口大小(低于基础窗口大小时生效), defaults to False
          - True: 保持窗口比例, 即窗口的高宽比不变, 聚类后box是多少, 窗口就返回多少
          - False: 根据window_size修改边框

    Example:

        ```python
        >>> from pprint import pprint
        >>> paw = PresetAdaptiveWindow(window_num=10, same_ratio=False, keep_size=False)
        ... windows = paw(
        ...     bboxes=[
        ...         [100, 100, 200, 200], [98, 120, 230, 168],
        ...         [150, 150, 250, 250], [0, 100, 198, 348],
        ...         # [789, 123, 987, 321]
        ...     ],
        ...     window_size=(640, 640),
        ...     image_shape=(1080, 1920, 3)
        ... )
        ... print("paw:")
        ... pprint(windows)
        ```
    """

    def __init__(self, window_num: int, same_ratio: bool = False, keep_size: bool = False, *args, **kwargs):
        self.KMeans = self.check_import()
        self.keep_size = keep_size
        self.window_num = window_num
        assert window_num > 0, "window_num should be greater than 0"
        self.same_ratio = same_ratio
        super().__init__()
    
    def get_centers(self, bboxes: list[list]):
        """根据bboxes获取中心点坐标

        :param list[list] bboxes: 目标实例边框列表
        """

        centers = []
        for box in bboxes:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append([cx, cy])
        
        centers = np.array(centers)
        return centers
    
    def get_cluster_window(self, k, cluster_labels: list, bboxes: list[list], image_shape=None, *args, **kwargs):
        
        windows = []
        src_size_dict = dict() if image_shape is None else {"img_h": image_shape[0], "img_w": image_shape[1]}
        
        for i in range(k):
            # 获取当前聚类中的所有box
            cluster_boxes = [bboxes[j] for j in range(len(bboxes)) if cluster_labels[j] == i]
            
            if not cluster_boxes:
                continue
                
            # 计算覆盖整个聚类的最小矩形
            min_x1 = min(box[0] for box in cluster_boxes)
            min_y1 = min(box[1] for box in cluster_boxes)
            max_x2 = max(box[2] for box in cluster_boxes)
            max_y2 = max(box[3] for box in cluster_boxes)

            win_x1 = int(max(0, min_x1))
            win_y1 = int(max(0, min_y1))
            win_x2 = int(min(src_size_dict.get("img_w", max_x2), max_x2))
            win_y2 = int(min(src_size_dict.get("img_h", max_y2), max_y2))
            
            windows.append((win_x1, win_y1, win_x2, win_y2))
        return windows
    
    def fuse_windows(self, windows: list, window_size: tuple[int] = (640, 640), max_scale_ratio: float = 1.2, *args, **kwargs):
        """融合距离足够近的窗口, 并放缩窗口大小以满足设计比例

        :param list windows: 聚类算法生成的窗口列表
        :param tuple[int] window_size: 裁切窗口的大小
        :param float max_scale_ratio: 最大放缩比例, defaults to 1.2
        """

        max_w, max_h = window_size[0] * max_scale_ratio, window_size[1] * max_scale_ratio

        delete_index = []
        for i in range(len(windows)-1):
            for j in range(i+1, len(windows)):
                new_x1 = min(windows[i][0], windows[j][0])
                new_y1 = min(windows[i][1], windows[j][1])
                new_x2 = max(windows[i][2], windows[j][2])
                new_y2 = max(windows[i][3], windows[j][3])
                new_w = new_x2 - new_x1
                new_h = new_y2 - new_y1
                if new_w <= max_w and new_h <= max_h:
                    # 窗口距离足够近, 融合窗口
                    windows[j] = (new_x1, new_y1, new_x2, new_y2)
                    delete_index.append(i)

        windows = [windows[i] for i in range(len(windows)) if i not in delete_index]
        return windows

    def modify_windows(self, windows: list, window_size: tuple[int], same_ratio: bool, keep_size: bool, image_shape=None, *args, **kwargs):
        """调整窗口

        :param list windows: _description_
        :param tuple[int] window_size: _description_
        :param bool same_ratio: _description_
        :param bool keep_size: _description_
        :param _type_ image_shape: _description_, defaults to None
        :return _type_: _description_
        """
        if not windows or keep_size:
            return windows
        base_w, base_h = window_size
        for i in range(len(windows)):
            min_x1, min_y1, max_x2, max_y2 = windows[i]
            
            # 计算当前区域的宽高
            region_w = max_x2 - min_x1
            region_h = max_y2 - min_y1
            
            # 计算窗口缩放比例，保持宽高比
            scale_w = max(1.0, region_w / base_w)
            scale_h = max(1.0, region_h / base_h)
            scale = max(scale_w, scale_h)
            
            # 计算缩放后的窗口大小
            scaled_w = int(base_w * scale if same_ratio else  base_w * scale_w)
            scaled_h = int(base_h * scale if same_ratio else  base_h * scale_h)
            
            # 计算窗口中心点(使用聚类中心)
            # center_x, center_y = k_means.cluster_centers_[i]
            center_x, center_y = int(min_x1 + region_w / 2), int(min_y1 + region_h / 2)
            
            # 生成窗口坐标
            win_x1 = max(0, center_x - scaled_w // 2)
            win_y1 = max(0, center_y - scaled_h // 2)
            win_x2 = win_x1 + scaled_w
            win_y2 = win_y1 + scaled_h
            
            # 如果提供了图像尺寸，确保窗口不超出图像边界
            if image_shape is not None:
                h, w = image_shape[:2]
                win_x1 = max(0, win_x1)
                win_y1 = max(0, win_y1)
                win_x2 = min(w, win_x2)
                win_y2 = min(h, win_y2)
            windows[i] = (win_x1, win_y1, win_x2, win_y2)
        return windows

    def check_import(self):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("Please install sklearn: [pip install scikit-learn] to use PresetAdaptiveWindow")
        return KMeans
    
    def __call__(self, bboxes: list[list], window_size: tuple[int], image_shape=None, *args, **kwargs):
        same_ratio = self.same_ratio if "same_ratio" not in kwargs else kwargs["same_ratio"]
        keep_size = self.keep_size if "keep_size" not in kwargs else kwargs["keep_size"]

        if not bboxes:
            return []
        centers = self.get_centers(bboxes)
        centers = np.array(centers)

        # 使用K-means聚类将目标分组
        k = min(self.window_num, len(bboxes))  # 实际使用的窗口数不超过目标数
        if k == 0:
            return []

        k_means = self.KMeans(n_clusters=k, random_state=0).fit(centers)
        labels = k_means.labels_

        # 为每个聚类计算覆盖所有目标的最小窗口, 并合并较小的类外接矩形框
        windows = self.get_cluster_window(k, labels, bboxes, image_shape)
        windows = self.fuse_windows(windows, window_size, max_scale_ratio=1.2)

        # 调整窗口大小
        windows = self.modify_windows(windows, window_size, same_ratio, keep_size, image_shape)

        return windows




if __name__ == '__main__':
    from pprint import pprint
    bsw = BoostSlidingWindow(640, overlap=0.2)
    windows = bsw((1080, 1920))
    pprint(windows)
