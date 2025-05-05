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


def to_coco_set_args(to_coco_config):
    to_coco_config.add_argument('img_dir', type=str, help='source images directory.')
    to_coco_config.add_argument('--lbl_dir', type=str, default=None, help='source annotations directory.')
    to_coco_config.add_argument('dst_dir', type=str, help='coco format save directory.')
    to_coco_config.add_argument('classes', type=str, help='yolo classes.txt format file.')
    to_coco_config.add_argument('--use_link', type=bool, default=False, help='use symlink to save images. default: False.')
    to_coco_config.add_argument('--split', type=str, default='train', help='split name. default: train.')
    to_coco_config.add_argument('--year', type=str, default="", help='dataset year. default: "".')
    to_coco_config.add_argument('--class_start_index', type=int, default=0, help='class start index. default: 0.')
    to_coco_config.add_argument('--img_idx', type=int, default=0, help='image start index. default: 0.')
    to_coco_config.add_argument('--ann_idx', type=int, default=0, help='annotation start index. default: 0.')


def labelme2coco_set_args(labelme2coco_parser):
    labelme2coco_config = labelme2coco_parser.add_parser('labelme2coco', help='🔁. labelme to coco format')
    to_coco_set_args(labelme2coco_config)


def from_coco_set_args(from_coco_config):
    from_coco_config.add_argument('img_dir', type=str, help='coco images directory.')
    from_coco_config.add_argument('lbl_file', type=str, help='coco annotations file.')
    from_coco_config.add_argument('dst_dir', type=str, help='save directory.')


def coco2labelme_set_args(coco2labelme_parser):
    coco2labelme_config = coco2labelme_parser.add_parser('coco2labelme', help='🔁. coco to labelme format')
    from_coco_set_args(coco2labelme_config)

def coco2voc_set_args(coco2voc_parser):
    coco2voc_config = coco2voc_parser.add_parser('coco2voc', help='🔁. coco to voc format')
    from_coco_set_args(coco2voc_config)
    coco2voc_config.add_argument('--extra_keys', nargs="*", type=list, default=[], help='extra keys to add to voc xml file.')


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


def voc2coco_set_args(voc2coco_parser):
    voc2coco_config = voc2coco_parser.add_parser('voc2coco', help='🔁. voc to coco format')
    to_coco_set_args(voc2coco_config)


def voc2labelme_set_args(voc2labelme_parser):
    voc2labelme_config = voc2labelme_parser.add_parser('voc2labelme', help='🔁. voc to labelme format')
    voc2labelme_config.add_argument('src_dir', type=str, help='voc annotation directory.')
    voc2labelme_config.add_argument('dst_dir', type=str, help='labelme format save directory.')


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

def yolo2coco_set_args(yolo2coco_parser):
    yolo2coco_config = yolo2coco_parser.add_parser('yolo2coco', help='🔁. yolo to coco format')
    yolo2coco_config.add_argument('--src_dir', type=str, help='yolo format directory. subdir should be "images" and "labels".')
    yolo2coco_config.add_argument('dst_dir', type=str, help='coco format save directory.')
    yolo2coco_config.add_argument('--classes', type=str, help='classes.txt file path. use with --src_dir.')
    yolo2coco_config.add_argument('--data', type=str, help='data yaml file path. Using this parameter does not '\
                                  'require specifying src_dir and classes.')
    yolo2coco_config.add_argument('--use_link', type=bool, default=False, help='use symlink to save images. default: False.')
    yolo2coco_config.add_argument('--split', type=str, default='train', help='split name. default: train.')
    yolo2coco_config.add_argument('--class_start_index', type=int, default=0, help='class start index. default: 0.')
    yolo2coco_config.add_argument('--image_index', type=int, default=0, help='image start index. default: 0.')
    yolo2coco_config.add_argument('--anno_index', type=int, default=0, help='annotation start index. default: 0.')


def labelme2voc_set_args(labelme2voc_parser):
    labelme2voc_config = labelme2voc_parser.add_parser('labelme2voc', help='🔁. labelme to voc format')
    labelme2voc_config.add_argument('src_dir', type=str, help='labelme annotation directory.')
    labelme2voc_config.add_argument('dst_dir', type=str, help='voc format save directory.')
    labelme2voc_config.add_argument('--extra_keys', nargs="*", type=list, default=[], help='extra keys to add to voc xml file.')

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
    labelme2voc_set_args(sub_command_parser)
    labelme2coco_set_args(sub_command_parser)
    voc2yolo_set_args(sub_command_parser)
    voc2coco_set_args(sub_command_parser)
    voc2labelme_set_args(sub_command_parser)
    voc_gen_classes_set_args(sub_command_parser)
    coco2labelme_set_args(sub_command_parser)
    coco2voc_set_args(sub_command_parser)
    yolo2coco_set_args(sub_command_parser)
    yolo_label_exclude_set_args(sub_command_parser)
    font_download_set_args(sub_command_parser)
    cut_img_set_args(sub_command_parser)
    args = labelOperation.parse_args()
    return args
