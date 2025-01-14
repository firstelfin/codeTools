#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   bboxMatch.py
@Time    :   2024/09/10 21:05:25
@Author  :   firstElfin 
@Version :   0.0.6
@Desc    :   bbox相关的匹配算子
'''

def box_valid(box: list) -> bool:
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return False
    return True


def inter_box(box1: list, box2: list) -> tuple[list, float]:
    """_summary_

    :param list box1: 边框左上右下角坐标
    :param list box2: 边框左上右下角坐标
    :return tuple[list, float]: 交集的坐标和面积
    """
    # 计算交集区域的坐标
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if not box_valid([x_left, y_top, x_right, y_bottom]):
        area = 0
    else:
        area = (x_right - x_left) * (y_bottom - y_top)
    return [x_left, y_top, x_right, y_bottom], area


def xywh2xyxy(bbox: list) -> list:
    """xywh(cx, cy, w, h)矩形框标注转换为xyxy模式"""
    a, b, w, h = bbox
    w_shift = w // 2
    h_shift = h // 2
    a1, b1, a2, b2 = int(a-w_shift), int(b-h_shift), int(a+w_shift), int(b+h_shift)
    return [a1, b1, a2, b2]


def ios_box(box1: list, box2: list, mode: str="xywh", double: bool=False):
    """交自比

    :param list box1: 预测bbox
    :param list box2: 匹配的查询bbox
    :param str mode: bbox的组成模式, 'xywh'表示框中心和宽高, defaults to 'xywh', options: ['xywh', 'xyxy']
    :param bool double: 是否双边计算, defaults to False
    """
    if mode not in ["xywh", "xyxy"]:
        raise Exception("modeError: IOP_box的mode参数不在可选范围内.")

    if mode == "xywh":
        bbox1 = xywh2xyxy(box1)
        bbox2 = xywh2xyxy(box2)
    else:
        bbox1 = box1
        bbox2 = box2
    
    # 求bbox的交集
    if not box_valid(bbox1) or not box_valid(bbox2):
        raise Exception(f"bboxError: 边框的坐标不符合要求, bbox1={bbox1}, bbox2={bbox2}.")
    
    _, inter_area = inter_box(bbox1, bbox2)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    ios = inter_area / bbox1_area
    if double:
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        ios_2 = inter_area / bbox2_area
        return ios, ios_2
    return ios


def iou_box(box1: list, box2: list) -> float:
    """计算两个边框的交并比

    :param list box1: x1, y1, x2, y2分别是左上和右下角坐标
    :param list box2: x1, y1, x2, y2分别是左上和右下角坐标
    :raises Exception: box坐标不符合要求
    :return float: 交并比数值
    """
    if not box_valid(box1) or not box_valid(box2):
        raise Exception(f"bboxError: 边框的坐标不符合要求, bbox1={box1}, bbox2={box2}.")
    _, inter_area = inter_box(box1, box2)

    # 计算两个矩形框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集区域的面积
    union_area = box1_area + box2_area - inter_area

    # 计算交并比（IoU）
    iou = inter_area / union_area

    return iou


def rel_box(box1: list, box2: list, trunc: bool=True) -> list:
    """生成box2相对于box1的相对坐标框

    :param list box1: 边框左上右下角坐标
    :param list box2: 边框左上右下角坐标
    :param bool trunc: 是否截断超出边界的坐标, defaults to True
    :return list: 相对坐标框
    """
    
    x1, y1, x2, y2 = box1
    x1_r, y1_r, x2_r, y2_r = box2
    x1_r_n = int(x1_r - x1)
    y1_r_n = int(y1_r - y1)
    x2_r_n = int(x2_r - x1)
    y2_r_n = int(y2_r - y1)

    # 截断超出边界的坐标
    if trunc:
        x1_r_n = max(0, x1_r_n)
        y1_r_n = max(0, y1_r_n)
        x2_r_n = min(int(x2-x1), x2_r_n)
        y2_r_n = min(int(y2-y1), y2_r_n)
    coord = [x1_r_n, y1_r_n, x2_r_n, y2_r_n]
    if not box_valid(coord):
        raise Exception(f"bboxError: 边框的坐标不符合要求, coord={coord}.")
    return coord


def abs_box(box1: list, box2: list) -> list:
    """box2还原相对坐标到原始坐标系

    :param list box1: 原始图片box1的坐标
    :param list box2: box2相对于box1的相对坐标框
    :return list: box1相对于原图的绝对坐标框
    """
    x1, y1, _, _ = box1
    x1_r, y1_r, x2_r, y2_r = box2
    x1_a = int(x1 + x1_r)
    y1_a = int(y1 + y1_r)
    x2_a = int(x1 + x2_r)
    y2_a = int(y1 + y2_r)
    coord = [x1_a, y1_a, x2_a, y2_a]
    if not box_valid(coord):
        raise Exception(f"bboxError: 边框的坐标不符合要求, coord={coord}.")
    return coord