#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   confusionMatrix.py
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
from matplotlib.font_manager import FontProperties
from loguru import logger
from pathlib import Path
from codeUtils.tools.fontConfig import set_plt
from codeUtils.tools.numpyTools import array2picture


class ConfusionMatrix:
    r"""Confusion Matrix 绘制工具, 此对象接收预测值和真实值, 并根据预测值和真实值计算混淆矩阵, 并绘制混淆矩阵. 混淆矩阵记录
    GT类数据预测为PRED类数据的数量, 并可视化展示. 建议xtick尽量使用长度差异小的字符串.
    
    ### Note: 
        1. set_plt 用于设置matplotlib字体, 全局生效. 字体文件会自动下载, 并缓存到 /home/usename/.config/elfin/fonts/
        2. matrix 横纵坐标设置为yolo的样式, 降低理解成本
        3. 类别名称列表category可以为空, 则自动生成0-num_classes-1的索引列表, category长度应等于num_classes, 最后一个是backgroud

    Args:
        num_classes (int): 类别数量
        category (list[str], optional): 类别名称列表. Defaults to None.
        title (str, optional): 图标题. Defaults to "Confusion Matrix".
        cmap (str, optional): 颜色映射. Defaults to "YlGnBu".
        chinese (bool, optional): 是否使用简体中文. Defaults to False.
        exclude_zero (bool, optional): 是否排除0值. Defaults to True.


    Attributes:
        num_classes (int): 类别数量
        matrix (np.ndarray): 混淆矩阵
        category (list[str]): 类别名称列表
        title (str): 图标题
        cmap (str): 颜色映射
        figure (matplotlib.figure.Figure): 绘图对象
        ax (matplotlib.axes.Axes): 绘图轴
    

    Examples:
        ```python
        >>> from codeUtils.matrix.confusionMatrix import ConfusionMatrix
        >>> cm = ConfusionMatrix(num_classes=3, category=["cat1", "cat2", "cat3"])
        >>> cm.add_matrix_item(i=0, j=1, value=1)
        >>> cm.add_matrix_item(i=1, j=2, value=2)
        >>> cm.add_matrix_item(i=0, j=2, value=3)
        >>> cm.save_figure(dst_dir="xxx/xxx")      # 保存混淆矩阵图片
        >>> cm.save_xlsx("confusionMatrix.xlsx")   # 保存混淆矩阵和Recall-Precision到Excel文件
        >>> cm.simplified_chinese_help()           # 简体中文支持帮助文档
        ```
    
    """

    class_name = "ConfusionMatrix"

    def __init__(
            self, num_classes, category: list[str] = None, cmap: str = "YlGnBu", 
            chinese: bool | str = False, exclude_zero=True):
        self.num_classes = num_classes
        self.matrix_recall = np.zeros((num_classes, num_classes), dtype=np.int32)
        self.matrix_precision = np.zeros((num_classes, num_classes), dtype=np.int32)
        self.normal_matrix_recall = None
        self.normal_matrix_precision = None
        self.category = category
        self.cmap = cmap
        if chinese:
            self.set_plt(font_path=chinese if isinstance(chinese, str) else None)
        self.exclude_zero = exclude_zero

    @classmethod
    def set_plt(cls, font_path = None):
        set_plt(font_path=font_path)

    def add_matrix_item(self, i: int, j: int, value: int, recall: bool = True):
        if recall:
            self.matrix_recall[i][j] += value
        else:
            self.matrix_precision[i][j] += value

    def add_matrix_items(self, matrix: np.ndarray, recall: bool = True):
        if recall:
            self.matrix_recall += matrix
        else:
            self.matrix_precision += matrix

    def save_figure(self, dst_dir: str | Path):
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        self.get_normalize_matrix()

        # 绘制 self.matrix_recall
        array2picture(
            data=self.matrix_recall, category=self.category, title_name="Confusion Matrix Recall Num",
            dst_path=dst_dir / "confusionMatrix_recall.png", mode="d", cmap=self.cmap,
            change_last_axis_label=["FN", None]
        )
        # 绘制 self.matrix_precision
        array2picture(
            data=self.matrix_precision, category=self.category, title_name="Confusion Matrix Precision Num",
            dst_path=dst_dir / "confusionMatrix_precision.png", mode="d", cmap=self.cmap,
            change_last_axis_label=[None, "FP"]
        )
        # 绘制 self.normal_matrix_recall
        array2picture(
            data=self.normal_matrix_recall, category=self.category, title_name="Confusion Matrix Recall Rate",
            dst_path=dst_dir / "confusionMatrix_recall_rate.png", mode=".2f", cmap=self.cmap,
            change_last_axis_label=["FN", None]
        )
        # 绘制 self.normal_matrix_precision
        array2picture(
            data=self.normal_matrix_precision, category=self.category, title_name="Confusion Matrix Precision Rate",
            dst_path=dst_dir / "confusionMatrix_precision_rate.png", mode=".2f", cmap=self.cmap,
            change_last_axis_label=[None, "FP"]
        )

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
        eps = 1e-5
        self.get_normalize_matrix()  # 计算归一化的混淆矩阵

        pd_matrix_recall = pd.DataFrame(self.matrix_recall, columns=self.category, index=self.category)
        pd_matrix_precision = pd.DataFrame(self.matrix_precision, columns=self.category, index=self.category)
        pd_normal_matrix_recall = pd.DataFrame(self.normal_matrix_recall, columns=self.category, index=self.category)
        pd_normal_matrix_precision = pd.DataFrame(self.normal_matrix_precision, columns=self.category, index=self.category)

        gt_num = self.matrix_recall.sum(axis=0)
        pred_num = self.matrix_precision.sum(axis=1)
        # 对角线元素
        recall = self.matrix_recall.diagonal() / (gt_num + eps)
        precision = self.matrix_precision.diagonal() / (pred_num + eps)
        rp = np.stack([gt_num, recall, pred_num, precision], axis=1)
        # 整体召回精度计算需要排除backgroud, 默认索引为-1
        gt_total_num = gt_num[:-1].sum()
        pred_total_num = pred_num[:-1].sum()
        
        total_recall = (np.sum(recall[:-1]) + eps) / (gt_total_num + eps)
        total_precision = (np.sum(precision[:-1]) + eps) / (pred_total_num + eps)
        rp = np.vstack([rp, [gt_total_num, total_recall, pred_total_num, total_precision]])

        # 预测计数为0, GT计数为0的类别, 在exclude_zero模式下排除
        if self.exclude_zero:
            index_array = np.bitwise_or(gt_num[:-1] != 0, pred_num[:-1] != 0)
        else:
            index_array = np.ones_like(recall[:-1], dtype=bool)
        
        mr = np.mean(recall[:-1][index_array])
        mp = np.mean(precision[:-1][index_array])
        rp = np.vstack([rp, ["-", mr, "-", mp]])

        df_rp = pd.DataFrame(rp, columns=["GtNum", "Recall", "PredNum", "Precision"], index=self.category+["Total", "Mean"])
        
        for col in df_rp.columns:
            # 尝试转换为 float 或 int，无法转换的保持原样
            is_numeric = pd.to_numeric(df_rp[col], errors='coerce').notna()
            df_rp[col][is_numeric] = pd.to_numeric(df_rp[col][is_numeric], errors='coerce')
        
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            # 设置表头和内容样式
            start_row, start_col = (2, 2)
            workbook = writer.book
            # ✅ 设置默认字体大小（模拟 Excel 默认）
            DEFAULT_FONT_SIZE = 10
            workbook.formats[0].set_font_size(DEFAULT_FONT_SIZE)
            # 设置表头和内容样式
            format_header = workbook.add_format({
                'bold': True,
                'align': 'center',
                'font_size': DEFAULT_FONT_SIZE + 1,  # 假设正文是 10，这里加粗字体 11
                'top': 2,         # 上边框
                'bottom': 1,      # 下边框
            })
            format_bottom = workbook.add_format({
                'font_size': DEFAULT_FONT_SIZE,
                'align': 'center',      # 居中对齐
                'bottom': 2,      # 只在底部下方画线
            })
            format_content = workbook.add_format({
                'font_size': DEFAULT_FONT_SIZE,
                'align': 'center',      # 居中对齐
            })
            format_index = workbook.add_format({
                'font_size': DEFAULT_FONT_SIZE,
                'align': 'left'
            })
            format_index_bottom = workbook.add_format({
                'font_size': DEFAULT_FONT_SIZE,
                'align': 'left',
                'bottom': 2
            })

            # 写入混淆矩阵到xlsx文件
            pd_matrix_recall.to_excel(writer, sheet_name="Confusion Matrix Recall Num")
            pd_matrix_precision.to_excel(writer, sheet_name="Confusion Matrix Precision Num")
            pd_normal_matrix_recall.to_excel(writer, sheet_name="Confusion Matrix Recall Rate")
            pd_normal_matrix_precision.to_excel(writer, sheet_name="Confusion Matrix Precision Rate")

            # 创建一个空的sheet
            worksheet_rp = workbook.add_worksheet("Recall-Precision")
            
            # 写入索引（index）
            last_row = 0
            for row_num, data in enumerate(df_rp.index):
                worksheet_rp.write(row_num + start_row + 1, start_col, data, format_index)
                last_row = row_num
            # 写入列名（columns）
            worksheet_rp.write(start_row , start_col, None, format_header)
            for col_num, data in enumerate(df_rp.columns):
                worksheet_rp.write(start_row , start_col + 1 + col_num, data, format_header)
            # 写入数据
            for row_num, row_data in enumerate(df_rp.values):
                for col_num, data in enumerate(row_data):
                    worksheet_rp.write(row_num + start_row + 1, start_col + 1 + col_num, data, format_content)
            # 设置表格底部的线
            worksheet_rp.write(last_row + start_row + 1, start_col, df_rp.index[-1], format_index_bottom)
            for col_num, data in enumerate(df_rp.values[last_row]):
                worksheet_rp.write(last_row + start_row + 1, col_num + start_col + 1, data, format_bottom)
            # 自动调整列宽
            for idx, col in enumerate(df_rp):  # Iterate through data to auto fit
                series = df_rp[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
                worksheet_rp.set_column(idx + 1 + start_row, idx + 1 + start_row, max_len)  # set column width

    def get_normalize_matrix(self):
        """获取归一化的混淆矩阵"""
        if self.normal_matrix_recall is None:
            self.normal_matrix_recall = np.zeros_like(self.matrix_recall)
            gt_num = self.matrix_recall.sum(axis=0, keepdims=True).clip(min=1)
            self.normal_matrix_recall = 100 *self.matrix_recall / gt_num

        if self.normal_matrix_precision is None:
            self.normal_matrix_precision = np.zeros_like(self.matrix_precision)
            pred_num = self.matrix_precision.sum(axis=1, keepdims=True).clip(min=1)
            self.normal_matrix_precision = 100 * self.matrix_precision / pred_num
