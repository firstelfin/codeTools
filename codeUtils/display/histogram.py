#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   histogram.py
@Time    :   2024/09/24 14:48:22
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

from matplotlib import pyplot as plt
import numpy as np


class HistDisplay:

    class_name = 'HistDisplay'

    def __init__(self, data):
        self.data = data

    def subplots_display(
            self, 
            subplots=(1, 1), figsize=(12, 8), bins=[10], 
            title=['Histogram'], xlabel=['x'], ylabel=['y']):
        if subplots[0] * subplots[1] == 1:
            self.single_display(figsize=figsize, title=title[0], xlabel=xlabel[0], ylabel=ylabel[0])
        
        fig_size = max(max(subplots), 3) * 4
        figsize_tuple = figsize if figsize != (12, 8) else (fig_size, int(fig_size * 0.67))
        fig, axs = plt.subplots(subplots[0], subplots[1], figsize=figsize_tuple)  # 创建子图
        for i in range(m):
            for j in range(n):
                idx = i * n + j
                axs[i, j].hist(data[idx], bins=bins[idx], alpha=0.7, color='blue')
                axs[i, j].set_title(title[idx])
                axs[i, j].set_xlabel(xlabel[idx])
                axs[i, j].set_ylabel(ylabel[idx])
        plt.tight_layout()  # 自动调整子图间距
        plt.show()  # 显示图形
        pass


if __name__ == '__main__':
    m, n = 3, 4  # 2 行 3 列
    data = [np.random.randn(100) for _ in range(m * n)]  # 生成随机数据
    hd = HistDisplay(data)
    hd.subplots_display(
        subplots=(m, n), 
        bins=[10, 20, 30, 40] * 3, 
        title=[f'Histogram{k+1}' for k in range(m * n)], 
        xlabel=[f"x{k+1}" for k in range(m * n)], 
        ylabel=[f"y{k+1}" for k in range(m * n)]
    )