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
from abc import ABC, abstractmethod
from pathlib import Path, PosixPath
warnings.filterwarnings('ignore')
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from codeUtils.tools.font_config import colorstr
from codeUtils.tools.tqdm_conf import cpu_num, BATCH_KEY, START_KEY, END_KEY, TPP, FPP, TPG, TNG, GTN, PRN
from codeUtils.__base__ import strPath
from codeUtils.labelOperation.readLabel import read_voc, read_yolo, read_txt, parser_json
from codeUtils.labelOperation.saveLabel import save_labelme_label
from codeUtils.matrix.confusionMatrix import ConfusionMatrix
from codeUtils.matchFactory.bboxMatch import abs_box
from codeUtils.inference.boxMatch import yolo_match


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


class InferBase(object):
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
    infer = InferBase(
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

    def __init__(self, src_dir, dst_dir, project: str = 'inference'):
        self.src_dir = Path(src_dir)
        self.project = project
        self.dst_dir = get_exp_dir(dst_dir, project)
        self.model = None
    
    def get_defult_dict(self, img_path: str = "", img_shape: tuple = (1080, 1920), version: str = '4.5.6') -> dict:
        return {
            "version": version,
            "flags": {},
            "shapes": [],
            "imagePath": str(img_path),
            "imageData": None,
            "imageHeight": img_shape[0],
            "imageWidth": img_shape[1]
        }

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
        
        if isinstance(self.src_dir, (str, PosixPath)):
            all_img_dirs = [self.src_dir]
        else:
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
        src_img = cv.imread(entity_file)
        return src_img

    @staticmethod
    def split(src_box) -> list:
        """根据原始图片的bbox, 自定义图片裁切方案, 返回裁切后子图的bbox列表

        :param src_box: 待裁切图片区域的bbox
        :type src_box: list, ins. [x1, y1, x2, y2]
        :return: list of [a1, b1, a2, b2]
        :rtype: list[list[int]]
        """
        return [src_box]
    
    def model_init(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请安装 ultralytics 库")
        
        self.model = YOLO(model_path)

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
        
        if self.model is None:
            self.model_init(kwargs.get('model_path', 'yolo11l.yaml'))

        all_sub_preds = []
        all_sub_imgs = [src_img[box[1]:box[3], box[0]:box[2]] for box in windows]
        results = self.model(all_sub_imgs, verbose=False)

        for i, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy().tolist()
            cls_ids = result.boxes.cls.cpu().numpy().tolist()
            scores = result.boxes.conf.cpu().numpy().tolist()
            sub_preds = [{
                'name': result.names[int(obj[0])],
                'code': int(obj[0]),
                'box': abs_box(windows[i], obj[2]), # 坐标还原
                'confidence': obj[1],
            } for obj in zip(cls_ids, scores, boxes)]
            
            # 坐标还原
            all_sub_preds.extend(sub_preds)
        return all_sub_preds

    @staticmethod
    def merge(results: list) -> list:
        """将多个子图的预测结果合并为最终结果

        :param results: list of ins. {'box': [x1, y1, x2, y2], 'label': 'xxx', 'confidence': 0.9, 'segment': [...]}
        :type results: list
        :return: list of ins. {'box': [x1, y1, x2, y2], 'label': 'xxx', 'confidence': 0.9, 'segment': [...]}
        :rtype: list
        """
        return results

    def save_infer_result(self, lbl_path: str, lbl_dict: dict, results: list):
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
        
        src_entity = self.load_entity(img_file)
        img_shape = src_entity.shape[:2]
        labelme_img_path = img_file.name  # 保存labelme格式的图片路径
        labelme_lbl_path = self.dst_dir / (img_file.stem + '.json')  # 保存labelme格式的标签路径
        lbl_dict = self.get_defult_dict(img_path=labelme_img_path, img_shape=img_shape)
        src_box = [0, 0, src_entity.shape[1], src_entity.shape[0]]

        all_windows = self.split(src_box)
        all_sub_preds = self.infer(src_entity, all_windows, **kwargs)

        results = self.merge(all_sub_preds)
        # 保存labelme格式的图片和标签
        self.save_infer_result(labelme_lbl_path, lbl_dict, results)

        # 保存软连接
        if not (self.dst_dir / img_file.name).exists():
            (self.dst_dir / img_file.name).symlink_to(img_file)
        
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


class StatisticBase(object):
    """加载 InferBase 推理结果 和 标签文件, 统计各类别的数量, 并保存到统计文件中
    推理结果文件夹和标注文件夹需要对应, 文件夹名称可以不一样.

    注: 全流程默认使用ultralytics的yolo模型, 若需要使用其他模型, 请重写以下方法:
        - gt2match : gt格式的对象转换为匹配格式, 匹配格式由自定义匹配模块定义
        - get_gt_suffix : 获取标签文件后缀名
        - read_label : 读取标签文件, 返回标签对象
        - match: 自定义匹配模块, 输入为gt和pred, 返回匹配结果

    Args:
        src_gt (list[str]): 标签文件路径, 支持多个数据子集，需要指定到数据子集的标签文件存放路径
        src_pred (list[str]): 推理结果文件路径, 支持多个数据子集，需要指定到数据子集的推理结果文件存放路径
        dst_dir (str): 预测结果保存目录
        project (str, optional): 实验名称, defaults to 'inference'.
        use_ios (bool, optional): 是否使用IOS计算, defaults to True
        classes (str, optional): 类别文件路径, defaults to 'classes.txt'.
    
    """

    def __init__(self, src_gt: list[str], src_pred: list[str], dst_dir: str, project: str = 'inference', use_ios: bool = True, classes: str = 'classes.txt'):
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
        """
        self.src_gt = src_gt
        self.src_pred = src_pred
        self.dst_dir = get_exp_dir(dst_dir, project + "statistic")
        self.project = project + "statistic"
        self.use_ios = use_ios
        self.backgroud = False
        self.classes = self.get_classes(classes)
        # 初始化统计的matrix
        self.matrix = ConfusionMatrix(len(self.classes), self.classes, chinese=True)
    
    def get_classes(self, class_file: str) -> list:
        """获取类别列表

        :param class_file: 类别文件路径
        :type class_file: str
        """
        classes = read_txt(class_file)
        classes.append('background')
        self.backgroud = True  # 标记是否有背景类
        return classes

    @staticmethod
    def read_label(label_file: str):
        yolo_label = read_yolo(label_file)
        # 转换yolo标注为xyxy格式
        return yolo_label

    @staticmethod
    def get_gt_suffix():
        return '.txt'

    def load_datasets(self):
        """从预测文件加载数据集, 返回一个生成器对象, 第一个元素是items数量

        :raises ValueError: _description_
        :yield: _description_
        :rtype: _type_
        """

        # 统计所有的预测结果文件, 没有预测预测文件也会保存
        if isinstance(self.src_pred, (str, PosixPath)):
            datasets = [self.src_pred]
        else:
            datasets = self.src_pred

        if isinstance(self.src_gt, (str, PosixPath)):
            gt_datasets = [self.src_gt]
        else:
            gt_datasets = self.src_gt
        
        # 统计预测文件数量
        total_num = 0
        for sub_datasets in datasets:
            if not sub_datasets.exists():
                raise ValueError(f"预测结果目录{sub_datasets}不存在")
            for file in sub_datasets.iterdir():
                if file.suffix != '.json':
                    continue
                total_num += 1
        yield total_num

        for i, sub_datasets in enumerate(datasets):
            for file in sub_datasets.iterdir():
                if file.suffix != '.json':
                    continue
                lbl_file = Path(self.src_gt[i]) / (file.stem + self.get_gt_suffix())
                yield file, lbl_file

    def load_pred(self, pred_file: str) -> dict:
        """加载单个预测结果文件, 返回预测结果[dict]

        :param pred_file: 预测结果文件路径
        :type pred_file: str
        """
        labelme_data = parser_json(pred_file)
        return labelme_data
    
    def load_gt(self, gt_file: str):
        gt_entities = self.read_label(gt_file)
        return gt_entities

    def pred2match(self, pred_entities: dict) -> list:
        """将labelme格式的对象转换为匹配格式, 匹配格式由自定义匹配模块定义

        :param pred_entities: labelme格式的预测结果对象
        :type pred_entities: dict
        """
        pred_boxes = []
        for shape in pred_entities['shapes']:
            box_cls = shape['code']  # yolo模型注入的类别编号
            x1, y1, x2, y2 = shape['points'][0][0], shape['points'][0][1], shape['points'][1][0], shape['points'][1][1]
            box = [x1, y1, x2, y2]
            pred_boxes.append([box_cls, *box])
            pass
        return pred_boxes
    
    def gt2match(self, gt_entities: list, **kwargs) -> list:
        """将标注文件加载内容转为匹配格式, 匹配格式由自定义匹配模块定义

        :param gt_entities: 标注文件内容
        :type gt_entities: list
        """
        imgh, imgw = kwargs.get('img_shape', (0, 0))
        if imgh == 0 or imgw == 0:
            raise ValueError("img_shape参数不能为空")
        gt_boxes = []
        for entity in gt_entities:
            x, y, w, h = entity[1:5]
            x1 = int(max(0, (x - w / 2) * imgw))
            y1 = int(max(0, (y - h / 2) * imgh))
            x2 = int(min(imgw, (x + w / 2) * imgw))
            y2 = int(min(imgh, (y + h / 2) * imgh))
            gt_boxes.append([entity[0], x1, y1, x2, y2])
        return gt_boxes

    def update(self, update_dict: dict):
        """更新统计矩阵

        :param update_dict: 更新字典, 格式为{类别: 预测向量}, 如: {"backgroud": [1,0,0,1,0,1]}
        :type update_dict: dict
        """
        for key, value in update_dict.items():
            key_index = self.classes.index(key)
            self.matrix.matrix[:, key_index] += value

    def match(self, pred_file: str, gt_file: str, **kwargs):
        labelme_dict = self.load_pred(pred_file)
        gt_entities = self.load_gt(gt_file)
        img_shape = (labelme_dict['imageHeight'], labelme_dict['imageWidth'])
        # 匹配预测结果和标签文件
        pred_boxes = self.pred2match(labelme_dict)
        gt_boxes = self.gt2match(gt_entities, img_shape=img_shape)
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

        # 保存渲染结果
        if kwargs.get('rendering', False):
            pass

        return True

    def __call__(self, *args, **kwargs):
        
        entities_generator = self.load_datasets()
        total_num = next(entities_generator)
        bar_desc = colorstr("bright_blue", "bold", "Statistic")
        rendering = kwargs.get('rendering', False)
        ios_thresh = kwargs.get('ios_thresh', 0.5)
        iou_thresh = kwargs.get('iou_thresh', 0.5)
        instance_batch_size = getattr(self, 'batch_size', 4)
        batch_size = kwargs.get('batch_size', instance_batch_size)
        epoch_num = math.ceil(total_num / batch_size)

        with ThreadPoolExecutor(max_workers=cpu_num) as executor:
            with tqdm(total=total_num, desc=bar_desc, unit='img', leave=True, position=0, dynamic_ncols=True) as epoch_bar:
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
                        # TODO: 处理进度条打印内容
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


if __name__ == '__main__':
    cpu_num = 1
    data_name = 'srcLabelTest-附属设施-标志牌-色标牌-正常-test1'
    infer = InferBase(
        src_dir=f'/data1/2024_datasets/UAV/无人机原始数据--归档数据/ancillaryFacilitiesCut/{data_name}/images',
        dst_dir='/data1/2025_datasets/',
        project=data_name
    )
    infer(
        verbose=True, pattern='yolo', 
        model_path='/data1/2024_project/UVA/uavApplication/uav_defect_identify/Pytorch/detect-signIndentify-GPU-uav.pt'
    )
    statistic = StatisticBase(
        src_gt=[f'/data1/2024_datasets/UAV/无人机原始数据--归档数据/ancillaryFacilitiesCut/{data_name}/labels'],
        src_pred=[infer.dst_dir],
        dst_dir=f'/data1/2025_datasets/',
        project=infer.dst_dir.name,
        use_ios=False,
        classes='/data1/2024_datasets/UAV/无人机原始数据--归档数据/ancillaryFacilities/classes.txt'
    )
    statistic(
        rendering=True,
        ios_thresh=0.5,
        iou_thresh=0.5
    )
