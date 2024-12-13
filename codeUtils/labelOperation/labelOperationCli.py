#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   labelOperationCli.py
@Time    :   2024/12/10 15:55:01
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   标签转换工具的命令行接口
'''

from codeUtils.__base__ import args
from codeUtils.labelOperation.labelme2other import labelme2yolo
from codeUtils.labelOperation.yoloLabelExclude import YoloLabelExclude
from codeUtils.labelOperation.voc2other import voc2yolo, voc_gen_names



def elfin():
    print("Welcome to elfin's label operation tool!")
    if args.mode == "labelme2yolo":
        src_dir, dst_dir, classes = args.src_dir, args.dst_dir, args.classes
        labelme2yolo(src_dir, dst_dir, classes)
    elif args.mode == "yoloLabelExclude":
        include_classes, data_yaml = args.include_classes, args.data_yaml
        yle = YoloLabelExclude(include_classes=include_classes, data_yaml=data_yaml)
        yle(save_dir=args.save_dir, cp_img=args.cp_img)
    elif args.mode == "voc2yolo":
        src_dir, dst_dir, names2id, img_valid = args.src_dir, args.dst_dir, args.names2id, args.img_valid
        voc2yolo(src_dir, dst_dir, names2id, img_valid)
    elif args.mode == "vocGenNames2Id":
        src_dir, dst_file = args.src_dir, args.dst_file
        voc_gen_names(src_dir, dst_file)
    else:
        print("Invalid subcommand")


    
