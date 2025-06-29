#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2025/06/29 11:11:47

import warnings
import cv2 as cv
warnings.filterwarnings('ignore')


def load_img(img_path):
    """加载图像文件

    :param img_path: 图像文件路径
    :type img_path: str
    :return: 图像数据
    :rtype: numpy.ndarray
    """

    if img_path is None:
        return None
    
    for _ in range(3):
        img = cv.imread(str(img_path))
        if img is not None:
            return img
    return None
    
