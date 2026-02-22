#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2026/02/20 21:56:16

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from pycocotools import mask as mask_utils
import cv2
from typing import List, Dict


def segmentation_to_polygons(
        annotation: Dict,
        min_area: float = 0.0) -> List[List[List[float]]]:
    """
    将 COCO annotation 中的 segmentation 转换为多边形列表
    
    参数:
        annotation: COCO 格式的 annotation 字典
        min_area: 最小保留面积（过滤太小的多边形）
    
    返回:
        polygons: 多边形列表 [[x1, y1, x2, y2, ...], ...]
    """
    segmentation = annotation['segmentation']
    
    # ========== 情况 1: 已经是多边形格式 ==========
    if isinstance(segmentation, list):
        filtered_polygons = []
        max_area = 0
        for poly in segmentation:
            if len(poly) < 6:
                continue
            poly_arr = np.array(poly).reshape(-1, 2)
            area = cv2.contourArea(poly_arr)
            if min_area > 0 and area < min_area:
                continue
            if area > max_area:
                filtered_polygons.insert(0, poly_arr.tolist())
            else:
                filtered_polygons.append(poly_arr.tolist())
        return filtered_polygons
    
    # ========== 情况 2: RLE 格式 ==========
    elif isinstance(segmentation, dict):
        # 准备 RLE 数据（处理 Python 3 的 bytes 问题）
        rle = segmentation.copy()
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        
        # 解码为二值掩码
        binary_mask = mask_utils.decode(rle).astype(np.uint8)  # type: ignore[arg-type]
        
        # 使用 OpenCV 提取轮廓. TODO: 多个多边形适配
        # 提取所有轮廓
        contours, hierarchy = cv2.findContours(
            binary_mask, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        hierarchy = hierarchy[0]  # 去掉第一维
        external_contours = []  # 外部轮廓
        hole_contours = []      # 内部孔洞

        for cnt, hier in zip(contours, hierarchy):
            if hier[-1] == -1:
                # 没有父级 = 外部轮廓
                external_contours.append(cnt)
            else:
                # 有父级 = 内部孔洞
                hole_contours.append(cnt)
        contours = external_contours + hole_contours
        
        # 将轮廓转换为 COCO 多边形格式
        polygons = []
        for contour in contours:
            # 过滤太小的轮廓
            if min_area > 0:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
            
            # 轮廓点数为偶数才能构成有效多边形
            if len(contour) < 3:
                continue
            
            # 展平为 [x1, y1, x2, y2, ...] 格式
            polygon = contour.reshape(-1, 2).tolist()
            polygons.append(polygon)
        
        return polygons
    
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

