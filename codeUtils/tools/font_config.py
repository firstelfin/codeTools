#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2024/08/27 10:11:49

import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams
from matplotlib import font_manager

FONT_PATH = "/usr/share/fonts/SimplifiedChinese/SourceHanSerifSC-Regular.otf"


def set_plt(font_path: str=FONT_PATH):
    font_prop = font_manager.FontProperties(fname=font_path)
    # 获取字体名称
    font_name = font_prop.get_name()
    font_manager.fontManager.addfont(font_path)
    # 更新 rcParams 设置
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [font_name]  # 替换为实际字体名称
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
