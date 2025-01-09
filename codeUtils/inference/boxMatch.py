#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2024/12/30 09:59:37

import os
import sys
import warnings
import numpy as np
from pathlib import Path
warnings.filterwarnings('ignore')

from codeUtils.matchFactory.bboxMatch import iou_box, ios_box, xywh2xyxy


def yolo_match(pred_boxes, gt_boxes, iou_thresh=0.5, ios_thresh=0.5, use_ios=False, mode="xyxy", classes: list=None):
    """yolo格式的预测框和真值框的匹配, 返回匹配结果

    ### Keyword
        - pred_boxes、gt_boxes: 若每个元素为 [类别, x, y, w, h] , 必须使用mode="xywh"

    :param pred_boxes: 预测框列表
    :type pred_boxes: lists
    :param gt_boxes: gt框列表
    :type gt_boxes: list
    :param iou_thresh: IOU阈值, defaults to 0.5
    :type iou_thresh: float, optional
    :param ios_thresh: IOS阈值, defaults to 0.5
    :type ios_thresh: float, optional
    :param use_ios: 是否使用IOS匹配, defaults to False
    :type use_ios: bool, optional
    :param mode: 预测框和真值框的格式, "xyxy"表示(x1,y1,x2,y2), defaults to "xyxy"
    :type mode: str, optional
    :return: tpg, tpp, fp, fn
    :rtype: dict
    """

    # 格式转换
    if mode.lower() == "xywh":
        box1_list = [[box[0], *xywh2xyxy(box[1:5])] for box in pred_boxes]
        box2_list = [[box[0], *xywh2xyxy(box[1:5])] for box in gt_boxes]
    else:
        box1_list = pred_boxes
        box2_list = gt_boxes

    # 选择使用的匹配函数和阈值
    if use_ios:
        iou_func = ios_box
        thresh = ios_thresh
    else:
        iou_func = iou_box
        thresh = iou_thresh
    
    # 遍历预测框和真值框，计算IOU，并更新匹配状态

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    cls_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.bool)
    for i, pbox in enumerate(box1_list):
        for j, gbox in enumerate(box2_list):
            iou = iou_func(gbox[1:], pbox[1:])
            iou_matrix[i, j] = iou
            cls_matrix[i, j] = pbox[0] == gbox[0]
    iou_status_matrix = iou_matrix > thresh

    # 计算pred2gt_matrix
    pred2gt_matrix = iou_status_matrix * cls_matrix

    # 排除较小的目标
    # TODO：过滤GT小目标
    # TODO：过滤pred小目标

    gt_status = np.any(pred2gt_matrix, axis=0)
    pred_status = np.any(pred2gt_matrix, axis=1)
    # 获取每一列最大值的索引和值
    update_items = dict()
    if classes is not None:
        for i, box in enumerate(gt_boxes):
            box_cls = classes[box[0]]
            if box_cls not in update_items:
                update_items[box_cls] = [0] * len(classes)
            if pred2gt_matrix.shape[0] and pred2gt_matrix[:, i].max():          # cls 和 box都匹配上的对象
                update_items[box_cls][box[0]] += 1
                continue
            if iou_matrix.shape[0] and iou_status_matrix[:, i].max():       # cbox匹配上，匹配不上cls的对象
                pred_index = pred_boxes[iou_matrix[:, i].argmax()][0]
                update_items[box_cls][pred_index] += 1
            else:
                update_items[box_cls][-1] += 1      # 未匹配上的对象, 预测为backgroud
        # 开始统计backgroud的数量
        update_items['background'] = [0] * len(classes)
        for i, box in enumerate(pred_boxes):
            if iou_matrix.shape[1] and (pred2gt_matrix[i, :].max() or iou_status_matrix[i, :].max()):
                continue
            # if pred2gt_matrix[i, :].max():
            #     continue
            # if iou_status_matrix[i, :].max():
            #     continue
            update_items['background'][box[0]] += 1

    # 输出预测框和真值框的匹配情况
    match_object = {
        "boxesStatus": {
            "tpg": [gt_boxes[i] for i, status in enumerate(gt_status) if status],         # GT命中的框
            "tpp": [pred_boxes[i] for i, status in enumerate(pred_status) if status],     # pred命中的框
            "fp": [pred_boxes[i] for i, status in enumerate(pred_status) if not status],  # pred没有命中的框
            "fn": [gt_boxes[i] for i, status in enumerate(gt_status) if not status],      # GT没有命中的框
        },
        "updateItems": update_items,  # 按列更新
    }

    return match_object
            
