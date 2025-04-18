#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   labelOperationCli.py
@Time    :   2024/12/10 15:55:01
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   标签转换工具的命令行接口
'''

from codeUtils.__base__ import set_args
from codeUtils.labelOperation.labelme2other import labelme2yolo, labelme2voc
from codeUtils.labelOperation.yoloLabelExclude import YoloLabelExclude
from codeUtils.labelOperation.voc2other import voc2yolo, voc_gen_classes
from codeUtils.labelOperation.cutImgFromLabel import CutImgFromLabel
from codeUtils.labelOperation.yolo2other import yolo2coco
from codeUtils.tools.font_config import font_download


def elfin():
    print("Welcome to elfin's label operation tool!")
    args = set_args()
    if args.mode == "labelme2yolo":
        src_dir, dst_dir, classes = args.src_dir, args.dst_dir, args.classes
        labelme2yolo(src_dir, dst_dir, classes)
    elif args.mode == "yoloLabelExclude":
        include_classes, data_yaml = args.include_classes, args.data_yaml
        yle = YoloLabelExclude(include_classes=include_classes, data_yaml=data_yaml)
        yle(dst_dir=args.dst_dir, cp_img=args.cp_img)
    elif args.mode == "voc2yolo":
        src_dir, dst_dir, classes, img_valid = args.src_dir, args.dst_dir, args.classes, args.img_valid
        voc2yolo(src_dir, dst_dir, classes, img_valid)
    elif args.mode == "voc2yoloClasses":
        src_dir, dst_file = args.src_dir, args.dst_file
        voc_gen_classes(src_dir, dst_file)
    elif args.mode == "font":
        if args.download:
            font_download()
    elif args.mode == "cutImg":
        cut_op = CutImgFromLabel(args.src_dir, args.dst_dir, pattern=args.pattern, target_size=args.size)
        cut_op(img_dir_name=args.img_dir_name, lbl_dir_name=args.lbl_dir_name)
    elif args.mode == "yolo2coco":
        yolo2coco(
            src_dir=args.src_dir, dst_dir=args.dst_dir, classes=args.classes, data=args.data, 
            use_link=args.use_link, split=args.split, class_start_index=args.class_start_index, 
            image_index=args.image_index, anno_index=args.anno_index
        )
    elif args.mode == "labelme2voc":
        labelme2voc(args.src_dir, args.dst_dir, args.extra_keys)
    else:
        print("Invalid subcommand")