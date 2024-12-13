#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   __base__.py
@Time    :   2024/12/10 17:25:47
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import argparse


def labelme2yolo_set_args(labelme2yolo_parser):
    labelme2yolo_config = labelme2yolo_parser.add_parser('labelme2yolo', help='labelme to yolo format')
    labelme2yolo_config.add_argument('src_dir', type=str, help='labelme annotation directory.')
    labelme2yolo_config.add_argument('dst_dir', type=str, help='yolo format save directory.')
    labelme2yolo_config.add_argument('classes', type=str, help='class id mapping file.')


def yolo_label_exclude_set_args(yolo_label_exclude_parser):
    yolo_label_exclude_config = yolo_label_exclude_parser.add_parser('yoloLabelExclude', help='exclude some labels from yolo label')
    yolo_label_exclude_config.add_argument('include_classes', nargs='+', type=int, help='yolo label directory.')
    yolo_label_exclude_config.add_argument('data_yaml', type=str, help='data yaml file path.')
    yolo_label_exclude_config.add_argument('--save_dir', default=None, type=str, help='total datasets save directory.')
    yolo_label_exclude_config.add_argument('--cp_img', action='store_true', help='copy image to save directory.')


def voc2yolo_set_args(voc2yolo_parser):
    voc2yolo_config = voc2yolo_parser.add_parser('voc2yolo', help='voc to yolo format')
    voc2yolo_config.add_argument('src_dir', type=str, help='voc annotation directory.')
    voc2yolo_config.add_argument('dst_dir', type=str, help='yolo format save directory.')
    voc2yolo_config.add_argument('names2id', type=str, help='json file: class name to id mapping.')
    voc2yolo_config.add_argument('--img_valid', action='store_true', help='copy image to save directory.')


def voc_gen_names2id_set_args(voc2names2id_parser):
    voc2names2id_config = voc2names2id_parser.add_parser('vocGenNames2Id', help='generate class name to id mapping')
    voc2names2id_config.add_argument('src_dir', type=str, help='voc annotation directory.')
    voc2names2id_config.add_argument('--dst_file', type=str, default=None, help='json file: class name to id mapping.')

# 开始设置命令行工具

labelOperation = argparse.ArgumentParser(
    description='Label conversion tool',
    epilog='Enjoy the program! :)',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
labelOperation.add_argument('--mode', help='Subcommand to run')
sub_command_parser = labelOperation.add_subparsers(dest="mode", title="labelme2yolo")
labelme2yolo_set_args(sub_command_parser)
yolo_label_exclude_set_args(sub_command_parser)
voc2yolo_set_args(sub_command_parser)
voc_gen_names2id_set_args(sub_command_parser)
args = labelOperation.parse_args()
