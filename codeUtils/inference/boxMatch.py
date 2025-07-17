#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2024/12/30 09:59:37

import os
import sys
import warnings
import numpy as np
from copy import deepcopy
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
    cls_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=bool)
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

    # 输出预测框和真值框的匹配情况
    match_object = {
        "boxesStatus": {
            "tpg": [gt_boxes[i] for i, status in enumerate(gt_status) if status],         # GT命中的框
            "tpp": [pred_boxes[i] for i, status in enumerate(pred_status) if status],     # pred命中的框
            "fp": [pred_boxes[i] for i, status in enumerate(pred_status) if not status],  # pred没有命中的框
            "fn": [gt_boxes[i] for i, status in enumerate(gt_status) if not status],      # GT没有命中的框
        },
        "updateItemsRecall": {},  # 记录召回, 按列更新
        "updateItemsPrecision": {},  # 记录精度, 按行更新
    }
    if classes is None:
        return match_object

    # 获取每一列最大值的索引和值
    _classes = deepcopy(classes)
    if _classes[-1] != 'background':
        _classes.append('background')
    update_items_recall = {class_name: [0]*len(_classes) for class_name in _classes}
    update_items_precision = {class_name: [0]*len(_classes) for class_name in _classes}
    # Note: 记录fn、tpg数据；fn也即将instance预测为background, tpg是gt中和预测完美匹配的实例
    for i, box in enumerate(gt_boxes):
        cls_index = box[0] if isinstance(box[0], int) else _classes.index(box[0])  # gt类别索引
        box_cls = _classes[cls_index]  # gt类别名称
        # 判别是否漏报
        if gt_status[i]:  # 非漏报场景: tpg
            update_items_recall[box_cls][cls_index] += 1
        elif iou_matrix.shape[0] and iou_status_matrix[:, i].max():  # 误报场景: fp for gt
            # 选择最佳iou匹配
            pred_index = int(iou_status_matrix[:, i].argmax())
            pred_box = pred_boxes[pred_index]
            pred_cls_index = pred_box[0] if isinstance(pred_box[0], int) else _classes.index(pred_box[0])
            update_items_recall[box_cls][pred_cls_index] += 1
        else:  # 漏报场景: fn
            update_items_recall[box_cls][-1] += 1

    # Note: 记录fp数据; 
    # fp有两种情况, 1. background预测为目标实例, 2. 目前类别预测其他类别, 且IOU大于阈值; 
    for j, box in enumerate(pred_boxes):
        cls_index = box[0] if isinstance(box[0], int) else _classes.index(box[0])  # pred类别索引
        box_cls = _classes[cls_index]  # pred类别名称
        if pred_status[j]:  # 预测框命中
            update_items_precision[box_cls][cls_index] += 1
            continue
        
        # 判断是否和gt box通过匹配预值
        if iou_matrix.shape[1] and iou_status_matrix[j, :].max():  # 类别没有匹配, 但是IOU大于阈值
            # 获取iou_status_matrix[j, :]为True的索引
            gt_index = int(iou_status_matrix[j, :].argmax())
            gt_box = gt_boxes[gt_index]
            gt_cls_index = gt_box[0] if isinstance(gt_box[0], int) else _classes.index(gt_box[0])
            # gt_cls = _classes[gt_cls_index]
            update_items_precision[box_cls][gt_cls_index] += 1
        else:  # 和GT关于IOU没有匹配上
            update_items_precision[box_cls][-1] += 1
    
    match_object["updateItemsRecall"] = update_items_recall
    match_object["updateItemsPrecision"] = update_items_precision
    return match_object
            
