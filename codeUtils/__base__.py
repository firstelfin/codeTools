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
from pathlib import Path


PYTHON_MAJOR = sys.version_info[0]
PYTHON_MINOR = sys.version_info[1]
PYTHON_VERSION = f'{PYTHON_MAJOR}.{PYTHON_MINOR}'

if PYTHON_VERSION >= '3.11':
    strPath = str | Path
else:
    from typing import Union
    strPath = Union[str, Path]


def to_coco_set_args(to_coco_config):
    to_coco_config.add_argument('--dst_dir', '-d', required=True, type=Path, help='coco format save directory.')
    to_coco_config.add_argument('--names', '-n', required=True, type=str, help='class id mapping file. classes.txt、xxx.yaml')
    to_coco_config.add_argument('--use_link', '-u', type=bool, default=False, help='use symlink to save images. default: False.')
    to_coco_config.add_argument('--split', '-s', type=str, default='train', help='split name. default: train.')
    to_coco_config.add_argument('--year', '-y', type=str, default="", help='dataset year. default: "".')
    to_coco_config.add_argument('--class_start_index', '-c', type=int, default=0, help='class start index. default: 0.')
    to_coco_config.add_argument('--img_idx', '-x', type=int, default=0, help='image start index. default: 0.')
    to_coco_config.add_argument('--ann_idx', '-a', type=int, default=0, help='annotation start index. default: 0.')


def labelme2yolo_set_args(labelme2yolo_parser):
    examples = """
    Examples:
    \u2714 %(prog)s path/to/jsons1 path/to/jsons2 -d path/to/yolo_format -n path/to/classes.txt
    \u2714 %(prog)s path/to/jsons1 path/to/jsons2 -d path/to/yolo_format -n path/to/classes.yaml
    \u2714 %(prog)s path/to/jsons1 -d path/to/yolo_format -n path/to/classes.txt -i path/to/images1
    \u2714 %(prog)s path/to/jsons1 path/to/jsons2 -d path/to/yolo_format -n path/to/classes.txt -i path/to/images1 path/to/images2
    """.strip()
    labelme2yolo_config = labelme2yolo_parser.add_parser(
        'labelme2yolo',
        help='🔁. labelme to yolo format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples)
    labelme2yolo_config.add_argument('lbl_dir', nargs='+', type=Path, help='labelme annotation directory.')
    labelme2yolo_config.add_argument('--dst_dir', '-d', required=True, type=str, help='yolo format save directory.')
    labelme2yolo_config.add_argument('--names', '-n', required=True, type=str, help='class id mapping file. classes.txt、xxx.yaml')
    labelme2yolo_config.add_argument('--img_dir', '-i', nargs='*', type=Path, default=[], help='images directory. default: [].')


def labelme2voc_set_args(labelme2voc_parser):
    examples = """
    Examples:
    \u2714 %(prog)s path/to/jsons1 path/to/jsons2 -d path/to/voc_format
    \u2714 %(prog)s path/to/jsons1 path/to/jsons2 -d path/to/voc_format
    \u2714 %(prog)s path/to/jsons1 -d path/to/voc_format -i path/to/images1
    \u2714 %(prog)s path/to/jsons1 path/to/jsons2 -d path/to/voc_format -i path/to/images1 path/to/images2
    """.strip()
    labelme2voc_config = labelme2voc_parser.add_parser(
        'labelme2voc',
        help='🔁. labelme to voc format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    labelme2voc_config.add_argument('lbl_dir', nargs='+', type=Path, help='labelme annotation directory.')
    labelme2voc_config.add_argument('--dst_dir', '-d', required=True, type=str, help='voc format save directory.')
    labelme2voc_config.add_argument('--img_dir', '-i', nargs='*', type=Path, default=[], help='images directory. default: [].')


def labelme2coco_set_args(labelme2coco_parser):
    examples = """
    Examples:
    \u2714 %(prog)s path/to/jsons1 path/to/jsons2 -i path/to/images1 path/to/images2 -d path/to/coco_format -n path/to/classes.txt
    \u2714 %(prog)s path/to/jsons1 -i path/to/images1 -d path/to/coco_format -n path/to/classes.yaml
    """.strip()
    labelme2coco_config = labelme2coco_parser.add_parser(
        'labelme2coco',
        help='🔁. labelme to coco format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    labelme2coco_config.add_argument('lbl_dir', nargs='+', type=Path, help='labelme annotation directory.')
    labelme2coco_config.add_argument('--img_dir', '-i', required=True, nargs='+',
                                type=Path, help='images directory. Imd_dir and lbl_dir correspond one-to-one.')
    to_coco_set_args(labelme2coco_config)


def yolo2labelme_set_args(yolo2labelme_parser):
    examples = """
    Examples:
    \u2714 %(prog)s path/to/yolo -d path/to/labelme_format -n path/to/classes.txt
    \u2714 %(prog)s path/to/yolo -d path/to/labelme_format -n path/to/coco128.yaml
    """.strip()
    labelme2coco_config = yolo2labelme_parser.add_parser(
        'yolo2labelme',
        help='🔁. yolo to labelme format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    labelme2coco_config.add_argument('src_dir', type=Path, help='yolo format directory. subdir should be "images" and "labels".')
    labelme2coco_config.add_argument('--dst_dir', '-d', required=True, type=Path, help='labelme format save directory.')
    labelme2coco_config.add_argument('--names', '-n', required=True, type=str, help='class id mapping file. classes.txt、xxx.yaml')


def yolo2voc_set_args(yolo2voc_parser):
    examples = """
    Examples:
    \u2714 %(prog)s path/to/yolo -d path/to/voc_format -n path/to/classes.txt
    \u2714 %(prog)s path/to/yolo -d path/to/voc_format -n path/to/coco128.yaml
    """.strip()
    yolo2voc_config = yolo2voc_parser.add_parser(
        'yolo2voc',
        help='🔁. yolo to voc format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    yolo2voc_config.add_argument('src_dir', type=Path, help='yolo format directory. subdir should be "images" and "labels".')
    yolo2voc_config.add_argument('--dst_dir', '-d', required=True, type=Path, help='voc format save directory.')
    yolo2voc_config.add_argument('--names', '-n', required=True, type=str, help='class id mapping file. classes.txt、xxx.yaml')


def yolo2coco_set_args(yolo2coco_parser):
    examples = """
    Examples:
    \u2714 %(prog)s path/to/yolo_root -d path/to/coco_format -n path/to/classes.txt
    \u2714 %(prog)s path/to/yolo_root -d path/to/coco_format -n path/to/classes.yaml
    """.strip()
    yolo2coco_config = yolo2coco_parser.add_parser(
        'yolo2coco',
        help='🔁. yolo to coco format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    yolo2coco_config.add_argument('src_dir', type=Path, help='yolo format directory. subdir should be "images" and "labels".')
    to_coco_set_args(yolo2coco_config)


def voc2labelme_set_args(voc2labelme_parser):
    examples = """
    Examples:
    \u2714 %(prog)s path/to/voc -d path/to/labelme_format
    \u2714 %(prog)s path/to/voc -d path/to/labelme_format -i path/to/images
    """.strip()
    voc2labelme_config = voc2labelme_parser.add_parser(
        'voc2labelme',
        help='🔁. voc to labelme format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    voc2labelme_config.add_argument('lbl_dir', nargs='+', type=Path, help='voc annotation directory.')
    voc2labelme_config.add_argument('--dst_dir', '-d', required=True, type=str, help='labelme format save directory.')
    voc2labelme_config.add_argument('--img_dir', '-i', nargs='*', type=Path, default=[], help='images directory. default: [].')
  

def voc2yolo_set_args(voc2yolo_parser):
    examples = """
    Examples:
    \u2714 %(prog)s path/to/voc -d path/to/yolo_format -n path/to/classes.txt
    \u2714 %(prog)s path/to/voc -d path/to/yolo_format -n path/to/coco128.yaml
    \u2714 %(prog)s path/to/voc -d path/to/yolo_format -n path/to/coco128.yaml -i path/to/images
    \u2714 %(prog)s path/to/voc1 path/to/voc2 -d path/to/yolo_format -n path/to/coco128.yaml -i path/to/images1 path/to/images2
    """.strip()
    voc2yolo_config = voc2yolo_parser.add_parser(
        'voc2yolo',
        help='🔁. voc to yolo format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    voc2yolo_config.add_argument('lbl_dir', nargs='+', type=Path, help='voc annotation directory.')
    voc2yolo_config.add_argument('--dst_dir', '-d', required=True, type=str, help='yolo format save directory.')
    voc2yolo_config.add_argument('--names', '-n', required=True, type=str, help='class id mapping file. classes.txt、xxx.yaml')
    voc2yolo_config.add_argument('--img_dir', '-i', nargs='*', type=Path, default=[], help='images directory. default: [].')


def voc2coco_set_args(voc2coco_parser):
    examples = """
    Examples:
    \u2714 %(prog)s path/to/xmls1 path/to/xmls2 -i path/to/images1 path/to/images2 -d path/to/coco_format -n path/to/classes.txt
    \u2714 %(prog)s path/to/jxmls1 -i path/to/images1 -d path/to/coco_format -n path/to/classes.yaml
    """.strip()
    labelme2coco_config = voc2coco_parser.add_parser(
        'voc2coco',
        help='🔁. voc to coco format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples
    )
    labelme2coco_config.add_argument('lbl_dir', nargs='+', type=Path, help='voc annotation directory.')
    labelme2coco_config.add_argument('--img_dir', '-i', required=True, nargs='+',
                                type=Path, help='images directory. Imd_dir and lbl_dir correspond one-to-one.')
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
    labelme2voc_set_args(sub_command_parser)
    labelme2coco_set_args(sub_command_parser)
    yolo2labelme_set_args(sub_command_parser)
    yolo2voc_set_args(sub_command_parser)
    yolo2coco_set_args(sub_command_parser)

    voc2yolo_set_args(sub_command_parser)
    voc2coco_set_args(sub_command_parser)
    voc2labelme_set_args(sub_command_parser)
    voc_gen_classes_set_args(sub_command_parser)
    coco2labelme_set_args(sub_command_parser)
    coco2voc_set_args(sub_command_parser)
    
    yolo_label_exclude_set_args(sub_command_parser)
    font_download_set_args(sub_command_parser)
    cut_img_set_args(sub_command_parser)
    args = labelOperation.parse_args()
    return args
