#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   confusion_matrix.py
@Time    :   2024/08/26 17:38:44
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import math
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path
from ..tools.font_config import set_plt, FONT_PATH


class ConfusionMatrix:
    r"""Confusion Matrix 绘制工具, 此对象接收预测值和真实值, 并根据预测值和真实值计算混淆矩阵, 并绘制混淆矩阵. 混淆矩阵记录
    GT类数据预测为PRED类数据的数量, 并可视化展示. 建议xtick尽量使用长度差异小的字符串.


    Args:
        num_classes (int): 类别数量
        category (list[str], optional): 类别名称列表. Defaults to None.
        title (str, optional): 图标题. Defaults to "Confusion Matrix".
        cmap (str, optional): 颜色映射. Defaults to "YlGnBu".
        chinese (bool, optional): 是否使用简体中文. Defaults to False.


    Attributes:
        num_classes (int): 类别数量
        matrix (np.ndarray): 混淆矩阵
        category (list[str]): 类别名称列表
        title (str): 图标题
        cmap (str): 颜色映射
        figure (matplotlib.figure.Figure): 绘图对象
        ax (matplotlib.axes.Axes): 绘图轴
        figure_status (bool): 绘图状态
    

    Examples:
        ```python
        >>> from codeUtils.matrix.confusion_matrix import ConfusionMatrix
        >>> cm = ConfusionMatrix(num_classes=3, category=["cat1", "cat2", "cat3"])
        >>> cm.add_matrix_item(i=0, j=1, value=1)
        >>> cm.add_matrix_item(i=1, j=2, value=2)
        >>> cm.add_matrix_item(i=0, j=2, value=3)
        >>> cm.get_fig()
        >>> cm.show_figure()
        >>> cm.save_figure("confusion_matrix.png")  # 保存混淆矩阵图片
        >>> cm.save_xlsx("confusion_matrix.xlsx")   # 保存混淆矩阵和Recall-Precision到Excel文件
        >>> cm.simplified_chinese_help()            # 简体中文支持帮助文档
        ```
    
    """

    class_name = "ConfusionMatrix"

    def __init__(self, num_classes, category: list[str] = None, title: str = "Confusion Matrix", cmap: str = "YlGnBu", chinese: bool = False, dpi: int = 100):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        self.category = category
        self.title = title
        self.cmap = cmap
        self.dpi = dpi
        self.figure_status = False
        self.matrix_num_len = None
        if chinese:
            self.set_plt()
    
    def xlabel_size(self):
        len_list = [len(v) for v in self.category]
        max_len = math.ceil(max(len_list) // self.matrix_num_len) + 1
        return max_len

    @classmethod
    def set_plt(cls, font_path = "codeUtils/SourceHanSerifSC-Regular.otf"):
        if not Path(font_path).exists():
            assert Path(FONT_PATH).exists(), f"字体文件 {font_path} 不存在, 默认字体文件 {FONT_PATH} 也不存在, 请检查!"
            font_path = FONT_PATH
        set_plt(font_path=font_path)
        

    def add_matrix_item(self, i: int, j: int, value: int):
        self.matrix[i][j] += value

    def add_matrix_items(self, matrix: np.ndarray):
        self.matrix += matrix

    def get_matrix_num_len(self):
        if self.matrix_num_len is None:
            self.matrix_num_len = len(str(np.max(self.matrix)))
        return self.matrix_num_len

    def get_cell_size(self):
        matrix_num_len = self.get_matrix_num_len()
        return int(self.dpi * 0.7) // matrix_num_len
    
    def get_xticks_size(self):
        matrix_num_len = self.get_matrix_num_len()
        return self.dpi // matrix_num_len
    
    def get_fig(self):
        """获取混淆矩阵图片"""
        cell_size = self.get_cell_size()
        xticks_size = self.get_xticks_size()

        self.imgh = self.xlabel_size() + len(self.category)
        self.imgw = self.xlabel_size() + len(self.category)
        self.imgh = int(0.8 * self.imgw)
        self.figure, self.ax = plt.subplots(figsize=(self.imgw, self.imgh), dpi=self.dpi)
        
        sns.heatmap(
            self.matrix, annot=True, fmt="d", cmap=self.cmap, 
            xticklabels=self.category, yticklabels=self.category,
            ax=self.ax, linewidths=1, annot_kws={"size": cell_size, "weight": "bold"}
        )
        plt.title(self.title, fontsize=int(xticks_size*1.3))
        plt.ylabel("GT", fontsize=xticks_size)
        plt.xlabel("PREDICT", fontsize=xticks_size)
        plt.yticks(rotation=0, fontsize=xticks_size)
        plt.xticks(rotation=90, fontsize=xticks_size)
        plt.tight_layout()
        self.figure = plt.gcf()
        self.figure_status = True
    
    def show_figure(self, screen_size: int = 1920):
        """展示混淆矩阵图片

        :param screen_size: 单前使用屏幕的分辨率, defaults to 1920
        :type screen_size: int, optional
        :return: None
        :rtype: None
        """
        if not self.figure_status:
            self.get_fig()
        # 图片太大时，显示可能会和屏幕分辨率产生冲突，因此需要缩放
        if self.imgw * self.dpi> screen_size:
            logger.warning("图片宽度过大, 可能导致显示出错, 请使用save_figure保存图片并查看.")
            return None
        plt.show()

    def save_figure(self, path: str):
        if not self.figure_status:
            self.get_fig()
        self.figure.savefig(path, format="png")

    @classmethod
    def simplified_chinese_help(cls):
        r"""简体中文帮助文档
        
        下载:
        ---------

            ```
            # step1: 下载 Source Han Serif SC 字体
            wget https://github.com/adobe-fonts/source-han-serif/releases/download/2.003R/09_SourceHanSerifSC.zip
            # step2: 解压
            unzip 09_SourceHanSerifSC.zip
            # step3: 复制到指定目录
            sudo cp -r 09_SourceHanSerifSC/SimplifiedChinese /usr/share/fonts/

            # step4: 刷新字体缓存
            sudo fc-cache -fv
            # step5:验证字体是否安装成功
            fc-list | grep "Source Han Serif SC"
            ```

        """
        print(cls.simplified_chinese_help.__doc__)

    def save_xlsx(self, path: str):
        """保存混淆矩阵和Recall-Precision到Excel文件

        :param path: 文件路径
        :type path: str
        """
        df = pd.DataFrame(self.matrix, columns=self.category, index=self.category)
        gt_num = self.matrix.sum(axis=1)
        pred_num = self.matrix.sum(axis=0)
        # 对角线元素
        recall = self.matrix.diagonal() / (gt_num  + 1e-5)
        precision = self.matrix.diagonal() / (pred_num  + 1e-5)
        rp = np.stack([gt_num, recall, pred_num, precision], axis=1)
        total_recall = np.sum(self.matrix.diagonal()) / (gt_num.sum() + 1e-5)
        total_precision = np.sum(self.matrix.diagonal()) / (pred_num.sum() + 1e-5)
        rp = np.vstack([rp, [gt_num.sum(), total_recall, pred_num.sum(), total_precision]])
        df_rp = pd.DataFrame(rp, columns=["GtNum", "Recall", "PredNum", "Precision"], index=self.category+["Total"])
        
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, sheet_name="Confusion Matrix")
            df_rp.to_excel(writer, sheet_name="Recall-Precision")
