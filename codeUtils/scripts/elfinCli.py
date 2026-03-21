#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   labelOperationCli.py
@Time    :   2024/12/10 15:55:01
@Author  :   firstElfin 
@Version :   1.8
@Desc    :   标签转换工具的命令行接口
'''

from codeUtils.__base__ import set_args
from codeUtils.labelOperation import labelme2voc, labelme2coco, labelme2yolo
from codeUtils.labelOperation import voc2yolo, voc2labelme, voc2coco
from codeUtils.labelOperation import yolo2coco, yolo2labelme, yolo2voc
from codeUtils.labelOperation import coco2labelme, coco2voc, coco2yolo
from codeUtils.labelOperation import statitic_gen_names
from codeUtils.labelOperation.yoloLabelExclude import YoloLabelExclude
from codeUtils.labelOperation.cutImgFromLabel import CutImgFromLabel
from codeUtils.tools.fontConfig import font_download


def elfin():
    print("Welcome to elfin's label operation tool!")
    args = set_args()
    if args.mode == "labelme2yolo":
        labelme2yolo(args.lbl_dir, args.dst_dir, args.names, args.img_dir)
    elif args.mode == "labelme2voc":
        labelme2voc(args.lbl_dir, args.dst_dir, args.img_dir)
    elif args.mode == "labelme2coco":
        labelme2coco(
            img_dir=args.img_dir, dst_dir=args.dst_dir, names=args.names, 
            lbl_dir=args.lbl_dir, img_idx=args.img_idx, ann_idx=args.ann_idx, 
            use_link=args.use_link, split=args.split, year=args.year, 
            class_start_index=args.class_start_index
        )
    elif args.mode == "yolo2labelme":
        yolo2labelme(args.src_dir, args.dst_dir, args.names)
    elif args.mode == "yolo2voc":
        yolo2voc(args.src_dir, args.dst_dir, args.names)
    elif args.mode == "yolo2coco":
        yolo2coco(
            src_dir=args.src_dir, dst_dir=args.dst_dir, names=args.names,
            use_link=args.use_link, split=args.split, class_start_index=args.class_start_index, 
            img_idx=args.img_idx, ann_idx=args.ann_idx
        )
    elif args.mode == "voc2labelme":
        voc2labelme(args.lbl_dir, args.dst_dir, args.img_dir)
    elif args.mode == "voc2yolo":
        voc2yolo(args.lbl_dir, args.dst_dir, args.names, args.img_dir)
    elif args.mode == "voc2coco":
        voc2coco(
            img_dir=args.img_dir, dst_dir=args.dst_dir, names=args.names, 
            lbl_dir=args.lbl_dir, img_idx=args.img_idx, ann_idx=args.ann_idx, 
            use_link=args.use_link, split=args.split, year=args.year, 
            class_start_index=args.class_start_index
        )
    elif args.mode == "coco2labelme":
        coco2labelme(args.lbl_dir, args.dst_dir)
    elif args.mode == "coco2voc":
        coco2voc(args.lbl_dir, args.dst_dir)
    elif args.mode == "coco2yolo":
        coco2yolo(args.lbl_dir, args.dst_dir)
    elif args.mode == "genNames":
        statitic_gen_names(args.lbl_dir, args.dst_dir, args.suffix)
    
    
    elif args.mode == "font":
        if args.download:
            font_download()
    elif args.mode == "cutImg":
        cut_op = CutImgFromLabel(args.src_dir, args.dst_dir, pattern=args.pattern, target_size=args.size)
        cut_op(img_dir_name=args.img_dir_name, lbl_dir_name=args.lbl_dir_name)
    
    elif args.mode == "yoloLabelExclude":
        include_classes, data_yaml = args.include_classes, args.data_yaml
        yle = YoloLabelExclude(include_classes=include_classes, data_yaml=data_yaml)
        yle(dst_dir=args.dst_dir, cp_img=args.cp_img)
    else:
        print("Invalid subcommand")