#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   histogram.py
@Time    :   2024/09/24 14:48:22
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


class HistDisplay:

    class_name = 'HistDisplay'

    def __init__(self, data, save_path: str = "", chinese: bool = False, chinese_path: str = None):
        self.data = data
        self.save_path = Path(save_path)
        if chinese:
            from ..matrix.confusionMatrix import ConfusionMatrix
            ConfusionMatrix.set_plt(font_path=chinese_path if chinese_path else "codeUtils/SourceHanSerifSC-Regular.otf")

    def subplots_display(
            self, 
            subplots=(1, 1), figsize=(12, 8), bins=[10], 
            title=['Histogram'], xlabel=['x'], ylabel=['y'],
            save_name: str="subplots_display.jpeg"):
        if subplots[0] * subplots[1] == 1:
            self.single_display(figsize=figsize, title=title[0], xlabel=xlabel[0], ylabel=ylabel[0])
        
        fig_size = max(max(subplots), 3) * 4
        figsize_tuple = figsize if figsize != (12, 8) else (fig_size, int(fig_size * 0.67))
        fig, axs = plt.subplots(subplots[0], subplots[1], figsize=figsize_tuple)  # 创建子图
        m, n = subplots
        for i in range(m):
            for j in range(n):
                idx = i * n + j
                if idx >= len(self.data):
                    # 删除超过界限的子图
                    fig.delaxes(axs[i, j])
                    continue
                axs[i, j].hist(self.data[idx], bins=bins[idx], alpha=0.7, color='blue')
                axs[i, j].set_title(title[idx])
                axs[i, j].set_xlabel(xlabel[idx])
                axs[i, j].set_ylabel(ylabel[idx])
        plt.tight_layout()  # 自动调整子图间距
        plt.show()  # 显示图形
        fig.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')  # 保存图形


if __name__ == '__main__':
    m1, n1 = 3, 4  # 2 行 3 列
    data = [np.random.randn(100) for _ in range(m1 * n1-1)]  # 生成随机数据
    hd = HistDisplay(data)
    hd.subplots_display(
        subplots=(m1, n1), 
        bins=[10, 20, 30] * 3 + [20, 30], 
        title=[f'Histogram{k+1}' for k in range(m1 * n1 -1)], 
        xlabel=[f"x{k+1}" for k in range(m1 * n1 -1)], 
        ylabel=[f"y{k+1}" for k in range(m1 * n1 -1)]
    )