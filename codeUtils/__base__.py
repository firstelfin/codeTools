#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   __base__.py
@Time    :   2024/12/10 17:25:47
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import sys
import argparse
from pathlib import PosixPath


PYTHON_MAJOR = sys.version_info[0]
PYTHON_MINOR = sys.version_info[1]
PYTHON_VERSION = f'{PYTHON_MAJOR}.{PYTHON_MINOR}'

if PYTHON_VERSION >= '3.11':
    strPath = str | PosixPath
else:
    from typing import Union
    strPath = Union[str, PosixPath]

def labelme2yolo_set_args(labelme2yolo_parser):
    labelme2yolo_config = labelme2yolo_parser.add_parser('labelme2yolo', help='🔁. labelme to yolo format')
    labelme2yolo_config.add_argument('src_dir', type=str, help='labelme annotation directory.')
    labelme2yolo_config.add_argument('dst_dir', type=str, help='yolo format save directory.')
    labelme2yolo_config.add_argument('classes', type=str, help='class id mapping file.')


def yolo_label_exclude_set_args(yolo_label_exclude_parser):
    yolo_label_exclude_config = yolo_label_exclude_parser.add_parser('yoloLabelExclude', help='exclude some labels from yolo label')
    yolo_label_exclude_config.add_argument('include_classes', nargs='+', type=int, help='include classes id list, i.e.: 1 2 4.')
    yolo_label_exclude_config.add_argument('data_yaml', type=str, help='data yaml file path.')
    yolo_label_exclude_config.add_argument('--dst_dir', default=None, type=str, help='total datasets save directory.')
    yolo_label_exclude_config.add_argument('--cp_img', action='store_true', help='copy image to save directory.')


def voc2yolo_set_args(voc2yolo_parser):
    voc2yolo_config = voc2yolo_parser.add_parser('voc2yolo', help='🔁. voc to yolo format')
    voc2yolo_config.add_argument('src_dir', type=str, help='voc annotation directory.')
    voc2yolo_config.add_argument('dst_dir', type=str, help='yolo format save directory.')
    voc2yolo_config.add_argument('classes', type=str, help='classes.txt file path.')
    voc2yolo_config.add_argument('--img_valid', action='store_true', help='copy image to save directory.')


def voc_gen_classes_set_args(voc2classes_parser):
    voc2classes_config = voc2classes_parser.add_parser(
        'voc2yoloClasses', help='generate classes.txt file from voc annotation',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    voc2classes_config.add_argument('src_dir', type=str, help='voc annotation directory.')
    voc2classes_config.add_argument('--dst_file', type=str, default=None, help='classes.txt file path.')


def font_download_set_args(font_download_parser):
    font_download_config = font_download_parser.add_parser('font', help='download font file')
    font_download_config.add_argument('--download', action='store_true', help='download font file.')


def cut_img_set_args(cut_img_parser):
    cut_img_config = cut_img_parser.add_parser('cutImg', help='✂️ . cut image by label.')
    cut_img_config.add_argument('src_dir', type=str, help='datasets directory.')
    cut_img_config.add_argument('dst_dir', type=str, help='cutImage datasets save directory.')
    cut_img_config.add_argument('--size', type=int, default=640, help='cut image size.')
    cut_img_config.add_argument('--pattern', type=str, default='yolo', help='cut image pattern. options: yolo, voc, etc.')
    cut_img_config.add_argument('--img_dir_name', type=str, default='images', help='image directory name.')
    cut_img_config.add_argument('--lbl_dir_name', type=str, default='labels', help='label directory name.')

# 开始设置命令行工具
def set_args():
    labelOperation = argparse.ArgumentParser(
        description='Label conversion tool',
        epilog='Enjoy the program! 😄',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    labelOperation.add_argument('--mode', help='Subcommand to run')
    sub_command_parser = labelOperation.add_subparsers(dest="mode", title="subcommands")
    labelme2yolo_set_args(sub_command_parser)
    voc2yolo_set_args(sub_command_parser)
    yolo_label_exclude_set_args(sub_command_parser)
    voc_gen_classes_set_args(sub_command_parser)
    font_download_set_args(sub_command_parser)
    cut_img_set_args(sub_command_parser)
    args = labelOperation.parse_args()
    return args
