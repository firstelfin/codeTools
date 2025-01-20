#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2025/01/14 10:50:15

import warnings
from copy import deepcopy
warnings.filterwarnings('ignore')

from codeUtils.inference.base import DetectBase
from codeUtils.inference.base import SliceRegistry, CombineRegistry, InferRegistry


@InferRegistry
class YoloDetectInfer(DetectBase):
    """YOLO检测推理接口

    Example:
        >>> from codeUtils.inference.yoloInfer import YoloDetectInfer
        >>> yolo_infer = YoloDetectInfer(
        ...     model=model_path, 
        ...     device='cuda:0', 
        ...     conf=[0.1, 0.2, 0.3], 
        ...     nms_iou=0.65, 
        ...     window_size=640, 
        ...     overlap=0.5, 
        ...     slice='BoostSlidingWindow', 
        ...     combine='MergeSlidingBase'
        ... )
    """

    def __init__(self, model: str, device: str, conf: list, nms_iou: float, *args, **kwargs):
        """初始化yolo模型推理类

        :param model: 模型路径
        :type model: str
        :param device: 推理使用的设备
        :type device: str
        :param conf: 置信度阈值[列表]
        :type conf: list
        :param nms_iou: NMS阈值
        :type nms_iou: float
        :param window_size: 滑窗大小
        :type window_size: int
        :param overlap: 滑窗重叠率
        :type overlap: float
        :param slice: 滑窗模式
        :type slice: list
        :param combine: 合并模式
        :type combine: list
        """
        super(YoloDetectInfer, self).__init__(model, device, conf, nms_iou, *args, **kwargs)
        self.slice_mode = kwargs.get('slice', [])  # type: list
        self.combine_mode = kwargs.get('combine', [])  # type: list
        self.kwargs = kwargs
        self.init_slice_combine()

    def init_slice_combine(self):
        """初始化滑窗与合并模式: 将滑窗和合并func注册到全局列表中, 方便重复使用"""
        m_list = self.slice_mode if isinstance(self.slice_mode, list) else [self.slice_mode]
        slice_class = [SliceRegistry[m] for m in m_list]
        self.slice_obj = [sc(**self.kwargs) for sc in slice_class]

        com_list = self.combine_mode if isinstance(self.combine_mode, list) else [self.combine_mode]
        combine_class = [CombineRegistry[m] for m in com_list]
        self.combine_obj = [cc(**self.kwargs) for cc in combine_class]

    def split(self, src_box) -> list[tuple]:
        # 使用步进滑窗, 并合并原始图片的边界框
        all_windows = set()
        all_windows.add(tuple(src_box))
        a1, b1, a2, b2 = src_box
        img_size = (b2 - b1, a2 - a1)
        for m in self.slice_obj:
            slice_boxes = m(img_size)
            for w_box in slice_boxes:
                all_windows.add(tuple(w_box))
        res_windows = list(all_windows)
        return res_windows
    
    def merge(self, pred_boxes):
        inner_boxes = deepcopy(pred_boxes)
        for m in self.combine_obj:
            inner_boxes = m.merge(inner_boxes)
        return inner_boxes


