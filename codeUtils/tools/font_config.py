#!/usr/bin/env python3
# encoding: utf-8
# @author: firstelfin
# @time: 2024/08/27 10:11:49

import httpx
import shutil
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from matplotlib import rcParams
from matplotlib import font_manager


def font_download():
    name = "Arial.Unicode.ttf"
    font_path = Path.home() / f'.config/elfin/fonts/{name}'
    if font_path.exists():
        print(f"字体文件 {font_path} 已存在！")
    else:
        # 开始下载字体文件
        print(f"开始下载字体文件 {name} 到 {font_path} ...")
        font_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{name}"
        # 发起 GET 请求并下载文件
        resume_header = {}
        temp_filename = font_path.parent / f"temp_{font_path.name}"
        if temp_filename.exists():
            resume_header['Range'] = f"bytes={temp_filename.stat().st_size}-"  # 获取已下载部分的字节数

        # 使用 httpx 发送同步请求
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(url, headers=resume_header)

            if response.status_code == 200 or response.status_code == 206:  # 206 是断点续传成功的状态码
                with open(temp_filename, 'ab') as f:  # 以追加模式打开文件
                    for chunk in response.iter_bytes(chunk_size=8192):  # 每次读取 8KB 数据
                        f.write(chunk)
                shutil.move(temp_filename, font_path)  # 重命名文件
                print(f"文件下载完成，保存为: {font_path}")
            else:
                print(f"下载失败, HTTP 请求状态码: {response.status_code}")

    pass


def valid_local_font(font_path: str = None):
    """加载本地字体文件，并设置 matplotlib 全局字体

    :param str font_path: 字体文件路径, defaults to None
    """
    chinese_font = ["SimHei", "Arial.Unicode", "PingFang"]
    if font_path is not None and Path(font_path).exists():
        return font_path
    else:
        for name in chinese_font:
            temp_path = Path.home() / f'.config/elfin/fonts/{name}.ttf'
            if temp_path.exists():
                return str(temp_path)
        print(f"字体文件 {font_path} 不存在, 也未发现中文字体文件. 下载请调用命令 'elfin font download' 下载默认字体文件！🏇")
        return None



def set_plt(font_path: str = None):
    if font_path is None or not Path(font_path).exists():
        font_dir = Path.home() / ".config/elfin/fonts/"
        if not font_dir.exists():
            raise FileNotFoundError(f"字体文件目录 {font_path if font_path else font_dir} 不存在！")
        font_path = valid_local_font(font_path)
        if font_path is None:
            raise FileNotFoundError(f"未找到有效的中文字体文件！")
    font_prop = font_manager.FontProperties(fname=font_path)
    # 获取字体名称
    font_name = font_prop.get_name()
    font_manager.fontManager.addfont(font_path)
    # 更新 rcParams 设置
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [font_name]  # 替换为实际字体名称
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print(f"字体设置成功, 字体文件: {Path(font_path).name}")


if __name__ == '__main__':
    # font_download()
    set_plt()
