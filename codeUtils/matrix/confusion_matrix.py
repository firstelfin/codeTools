#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   confusion_matrix.py
@Time    :   2024/08/26 17:38:44
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from ..tools.font_config import set_plt, FONT_PATH


class ConfusionMatrix:
    r"""Confusion Matrix 绘制工具, 此对象接收预测值和真实值, 并根据预测值和真实值计算混淆矩阵, 并绘制混淆矩阵. 混淆矩阵记录
    GT类数据预测为PRED类数据的数量, 并可视化展示.


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
        >>> cm.save_figure("confusion_matrix.png")
        ```
    
    """

    class_name = "ConfusionMatrix"

    def __init__(self, num_classes, category: list[str] = None, title: str = "Confusion Matrix", cmap: str = "YlGnBu", chinese: bool = False):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        self.category = category
        self.title = title
        self.cmap = cmap
        self.figure, self.ax = plt.subplots()
        self.figure_status = False
        if chinese:
            self.set_plt()

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
    
    def get_fig(self):
        sns.heatmap(
            self.matrix, annot=True, fmt="d", cmap=self.cmap, 
            xticklabels=self.category, yticklabels=self.category,
            # ax=self.ax
        )
        plt.title(self.title)
        plt.ylabel("GT")
        plt.xlabel("PREDICT")
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        plt.tight_layout()
        self.figure = plt.gcf()
        self.figure_status = True
    
    def show_figure(self):
        if not self.figure_status:
            self.get_fig()
        plt.show()

    def save_figure(self, path: str):
        if not self.figure_status:
            self.get_fig()
        self.figure.savefig(path, format="png")
    
    def _deinit_figure(self):
        plt.close(self.figure)

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
