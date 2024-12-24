#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   symmetry.py
@Time    :   2024/12/23 10:16:30
@Author  :   firstElfin 
@Version :   0.1.8
@Desc    :   图像对称变化
'''


class FlipLR(object):

    def __init__(self):
        pass

    def __call__(self, img, label: dict = None) -> tuple:
        print("FlipLR")
