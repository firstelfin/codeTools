#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   imgShow.py
@Time    :   2024/11/01 17:24:23
@Author  :   firstElfin 
@Version :   0.1.6
@Desc    :   None
'''

import shutil
import cv2 as cv
import numpy as np
from pathlib import Path, PosixPath
from PIL import Image, ImageDraw, ImageOps


def cv_show(
        img: str|np.ndarray, contours: list = [], 
        target_size: tuple = (1080, 1920), color: tuple = (0, 255, 0),
        return_monitoring = False
        ):
    
    if isinstance(img, np.ndarray):
        src_img = img
    else:
        src_img = cv.imread(img)
    img_h, img_w = src_img.shape[:2]
    
    # 绘制多边形，不填充
    if contours:
        cv.drawContours(src_img, contours, -1, color, 2)
    
    if img_h > target_size[0] or img_w > target_size[1]:
        show_img = cv.resize(src_img, target_size[::-1])
    else:
        show_img = src_img
    cv.imshow('img', show_img)
    res = cv.waitKey(0)
    cv.destroyAllWindows()
    if return_monitoring:
        return res
    return None


def pil_show(
        img: str|np.ndarray, contours: list = [], 
        target_size: tuple = (1920, 1080), color: tuple = (0, 255, 0)
        ):
    if isinstance(img, np.ndarray):
        src_img = Image.fromarray(img)
    else:
        src_img = Image.open(img)
        src_img = ImageOps.exif_transpose(src_img)
    img_h, img_w = src_img.size
    
    # 绘制多边形，不填充
    if contours:
        draw = ImageDraw.Draw(src_img)
        for contour in contours:
            draw.polygon(contour, outline=color, fill=None)
    
    if img_h > target_size[0] or img_w > target_size[1]:
        show_img = src_img.resize(target_size)
    else:
        show_img = src_img
    show_img.show()


def select_label_by_img(img_path: str|PosixPath, lbl_dir: str|PosixPath, save_dir=str|PosixPath, suffix=".txt"):
    """根据图片选择图片和对应的标签，并复制到指定目录

    :param str | PosixPath img_path: 图片路径
    :param str | PosixPath lbl_dir: 标签保存路径
    :param _type_ save_dir: 保存路径, defaults to str | PosixPath
    :param str suffix: 标签文件的后缀, defaults to ".txt"
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(img_path, save_dir / Path(img_path).name)
    lbl_file = Path(lbl_dir) / Path(img_path).with_suffix(suffix).name
    if lbl_file.exists():
        shutil.copy(lbl_file, save_dir / lbl_file.name)


def select_img(images: list[str|PosixPath], show_func="cv_show", sync_func=None, sync_kwargs=None):
    """根据反馈对图片进行操作，包括：
    - 按空格或d键切换到下一张图片
    - 按enter或回车键切换到下一张图片, 并调用同步函数
    - 按a键切换到上一张图片
    - 按ESC键退出

    :param list[str | PosixPath] images: 要处理的图片列表
    :param str show_func: 图片展示函数, defaults to "cv_show"
    :param _type_ sync_func: save操作时的同步处理函数, defaults to None
    :param _type_ sync_kwargs: 同步函数的参数, defaults to None
    """
    img_idx = 0
    while True:
        img_path = images[img_idx]
        if show_func == "cv_show":
            res = cv_show(str(img_path), return_monitoring=True)
            if res == 27:  # ESC
                break
            elif res == 32 or res == ord("d"):  # space or d
                img_idx = (img_idx + 1) % len(images)
            elif res == ord("\n") or res == ord("\r"):  # \n
                img_idx = (img_idx + 1) % len(images)
                if sync_func:
                    sync_func(img_path, **sync_kwargs)
            elif res == ord("a"):  # a
                img_idx = (img_idx - 1) % len(images)

if __name__ == '__main__':
    # all_images = list(Path("...").iterdir())
    # select_img(
    #     all_images,
    #     show_func="cv_show",
    #     sync_func=None,
    #     # sync_func=select_label_by_img,
    #     sync_kwargs={
    #         "lbl_dir": Path("..."), 
    #         "save_dir": Path("...")
    #         }
    #     )
    pass
