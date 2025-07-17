#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   numpyTools.py
@Time    :   2025/07/16 12:06:04
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from pathlib import Path
from .fontConfig import set_plt


def array2picture(
        data, category, title_name: str = "Confusion Matrix", 
        dst_path: str = '', mode=None, chinese: bool | str = False,
        x_label: str = "GT", y_label: str = "PREDICT",
        char_width_px: int = 8, char_height_px: int = 12,
        cell_padding: int = 12, label_padding: int = 18, 
        cmap: str = "viridis", change_last_axis_label= [None, None], **kwargs
    ):
    """将np.ndarray数据转换为图片并保存

    :param ndarray data: matrix数据, 可以是numpy二维数组, 支持float, int, str类型
    :param list category: 标签列表, 目前仅支持1-D列表
    :param str title_name: 图片标题, 默认为"Confusion Matrix"
    :param str dst_path: 图片保存路径, 默认为当前路径下保存为'Confusion Matrix.png', 后缀强制使用png
    :param str mode: 数值显示模式, 默认为None, 即自动判断
    :param bool chinese: 是否使用中文标签, 默认为False
    :param str x_label: 横坐标标签, 默认为"GT"
    :param str y_label: 纵坐标标签, 默认为"PREDICT"
    :param int char_width_px: 字符宽度（像素）
    :param int char_height_px: 字符高度（像素）
    :param int cell_padding: 单元格间距（像素）
    :param int label_padding: 标签间距（像素）
    :param str cmap: 颜色映射, 默认为"viridis"
    :param list change_last_axis_label: 是否修改最后一个标签, 默认为[None, None], 第一个元素为x轴标签, 第二个元素为y轴标签
    """

    if chinese:
        if isinstance(chinese, str):
            set_plt(font_path=chinese)
        elif isinstance(chinese, bool):
            set_plt()
        else:
            raise TypeError("chinese must be bool or str")

    # 0. 设置默认参数
    char_width_px = char_width_px
    grid_font_size = char_height_px
    cell_padding = cell_padding
    label_padding = label_padding
    dst_file_path = f"{title_name}.png" if not dst_path else Path(dst_path).with_suffix(".png")

    # 1. matrix数据计算最长长度
    fmt = "d" if mode is None else mode
    str_data = np.vectorize(lambda x: format(x, fmt))(data)
    max_str_len = max(max(len(s) for s in str_data.ravel()), 7)

    # 2. 计算单元格尺寸（像素）
    cell_width_px = max_str_len * char_width_px + cell_padding * 2
    cell_height_px = grid_font_size * 2 + cell_padding * 2

    # 3. 计算标签空间
    # 横纵坐标标签
    x_labels = deepcopy(category[:data.shape[0]])
    y_labels = deepcopy(category[:data.shape[1]])
    if change_last_axis_label[0] is not None:
        x_labels[-1] = change_last_axis_label[0]
    if change_last_axis_label[1] is not None:
        y_labels[-1] = change_last_axis_label[1]
    # 横纵坐标标签最大长度
    x_label_length = max(len(s) for s in x_labels)                         
    y_label_length = max(len(s) for s in y_labels)
    # 横纵坐标空间占比大小
    x_label_space = x_label_length * char_width_px + label_padding
    y_label_space = y_label_length * char_width_px + label_padding

    # 4. 计算标题空间
    title_fontsize = math.ceil(grid_font_size * 1.4)
    title_height_px = title_fontsize * 1.5
    x_axis_label_fontsize = math.ceil(grid_font_size * 1.1)
    y_axis_label_fontsize = math.ceil(grid_font_size * 1.1)

    # 5. 计算颜色条宽度
    colorbar_width_px = cell_width_px * 0.5

    # 6. 计算图像总像素尺寸（含标签、标题、颜色条）
    rows, cols = data.shape
    width_px = cols * cell_width_px + y_label_space + colorbar_width_px + y_axis_label_fontsize
    height_px = rows * cell_height_px + x_label_space + title_height_px + x_axis_label_fontsize
    # 7. 设置 DPI 和 图像尺寸（英寸）
    dpi = 100
    imgw = width_px / dpi
    imgh = height_px / dpi

    # 创建 figure（不使用 constrained_layout）
    fig, ax = plt.subplots(figsize=(imgw, imgh), dpi=dpi)

    # 设置颜色条位置（归一化坐标）
    cbar_left = 0.93
    cbar_bottom = (x_label_space + x_axis_label_fontsize)/ height_px
    cbar_width = 0.025
    cbar_height = (rows * cell_height_px - x_axis_label_fontsize) / height_px

    # 添加颜色条
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    # 绘制 heatmap
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap=cmap, xticklabels=y_labels, yticklabels=x_labels, ax=ax, 
        linewidths=1, annot_kws={"size": grid_font_size, "weight": "bold"}, cbar=True, cbar_ax=cbar_ax
    )

    # 添加标题
    ax.set_title(str(title_name), fontsize=title_fontsize)
    ax.set_ylabel(y_label, fontsize=y_axis_label_fontsize)
    ax.set_xlabel(x_label, fontsize=x_axis_label_fontsize)

    # 调整热图区域边距（为颜色条预留空间）
    plt.subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')  # 右对齐防止标签被截断

    # 使用 tight_layout 并指定热图区域范围（避免扩展到颜色条）
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # 保存图片
    plt.savefig(dst_file_path, dpi=dpi, bbox_inches='tight', format="png")
