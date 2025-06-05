#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   yoloLabelExclude.py
@Time    :   2024/12/12 10:47:40
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   生成yolo格式的标签文件--排除不在include_classes中的类别, 并重置类别序号。
'''

import shutil
import yaml
from copy import deepcopy
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from codeUtils.labelOperation.readLabel import read_yolo
from codeUtils.labelOperation.saveLabel import save_yolo_label

class YoloLabelExclude(object):
    """将YOLO格式的标签文件中不在include_classes中的类别剔除, 并重置类别序号

    1. 重置后的类别序号从0开始, 依次为include_classes中的类别索引序号
    2. 剔除不在include_classes中的类别, 删除对应的标签行
    3. 默认在labels同级文件夹创建新文件存放新标签, 也可以指定保存地址
    4. 指定存放地址是指定yaml文件的path路径, 数据集子集会根据结构重建
    5. cp_img参数控制是否复制图片到新文件夹, 默认不复制, cp_img=True时, 新标签才会保存到dst_dir
    6. dst_dir存在时, 新的yaml文件会根据dst_dir的地址和原yaml文件名生成, 并保存到dst_dir
    7. dst_dir存在时, classes.txt文件会在dst_dir中生成, 否则在原数据集下生成(不会替换原文件)

    :param list include_classes: 选中留下的标签
    :param str data_yaml: 原始数据的yaml文件路径

    NOTE: 标签转换使用__call__方法, 传入参数为新的path地址, 是否复制图片

    Example:
        >>> from codeUtils.labelOperation.yoloLabelExclude import YoloLabelExclude
        >>> include_classes = [2,4,8]
        >>> data_yaml = 'path/to/old.yaml'
        >>> yolo_label_exclude = YoloLabelExclude(include_classes, data_yaml)
        >>> yolo_label_exclude('path/to/save_total_datasets', cp_img=True)
    
    Cli:
        >>> elfin yoloLabelExclude  2 4 8  path/to/old.yaml --dst_dir path/to/save_total_datasets --cp_img
    """

    def __init__(self, include_classes: list[int], data_yaml: str):
        self.include_classes = include_classes
        self.data_yaml = data_yaml
        self.old_id_to_new = {v: i for i, v in enumerate(include_classes)}
        self.data = self.load_yaml()
        self.new_names = {i: self.data['names'][v] for i, v in enumerate(include_classes)}
        self.data_root = Path(self.data['path']).absolute()
        self.errors = []
        self.exclude_title = "_".join([str(i) for i in include_classes])

    def load_yaml(self):
        data = yaml.load(open(self.data_yaml, 'r'), Loader=yaml.FullLoader)
        return data
    
    def save_classes(self, save_dir: str = None):
        """保存新的classes.txt文件

        :param str save_dir: 保存地址, defaults to None
        """
        if save_dir is None:
            save_dir = self.data_root
        else:
            save_dir = Path(save_dir)
        with open(save_dir / f'classes_{self.exclude_title}.txt', 'w') as f:
            for i in range(len(self.include_classes)):
                f.write(f'{self.new_names[i]}\n')
    
    def save_yaml(self, save_dir: str = None):
        """保存新标签文件
        
        :param str save_dir: 新标签保存地址
        """
        file_stem = Path(self.data_yaml).stem
        if save_dir is None:
            new_yaml_file = self.data_root / f'{file_stem}_{self.exclude_title}.yaml'
        else:
            new_yaml_file = Path(save_dir) / f'{file_stem}_{self.exclude_title}.yaml'
        new_yaml_data = deepcopy(self.data)
        new_yaml_data['names'] = self.new_names
        if save_dir:
            new_yaml_data['path'] = str(save_dir)
        
        # 保存新yaml文件
        with open(new_yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(new_yaml_data, f, allow_unicode=True)
    
    def label_exclude(
            self, lbl_path: str, img_save_dir: str = None, 
            lbl_save_dir: str = None, img_path: str = None, 
            cp_img: bool = False
        ):
        """剔除不在include_classes中的类别, 并重置类别序号

        :param str lbl_path: 原标签路径
        :param str img_save_dir: 图片保存的地址, defaults to None
        :param str lbl_save_dir: 标签保存的地址, defaults to None
        :param str img_path: 原图片的地址, defaults to None
        :param bool cp_img: 是否复制图片, defaults to False
        """
        
        lbl_list = read_yolo(lbl_path)
        if lbl_list is None:
            self.errors.append(lbl_path)
            return None
        new_lbl_list = []
        for lbl in lbl_list:
            if lbl[0] not in self.include_classes:
                continue
            new_lbl_list.append([self.old_id_to_new[lbl[0]], *lbl[1:]])
        # 保存新标签
        save_yolo_label(lbl_save_dir / Path(lbl_path).name, new_lbl_list)
        if cp_img:
            shutil.copy(img_path, img_save_dir / Path(img_path).name)

    def exclude_classes_by_subset(
            self, img_save_dir: str = None, lbl_save_dir: str = None, 
            img_dir: str = None, lbl_dir: str = None, cp_img: bool = False
        ):
        """根据子集的图片和标签文件, 剔除不在include_classes中的类别, 并重置类别序号

        :param str img_save_dir: 图片保存的地址, defaults to None
        :param str lbl_save_dir: 标签保存的地址, defaults to None
        :param str img_dir: 原子集图片的地址, defaults to None
        :param str lbl_dir: 原子集标签的地址, defaults to None
        :param bool cp_img: 是否复制图片, defaults to False
        """
        results = []
        with ThreadPoolExecutor() as executor:
            for img_file in img_dir.iterdir():
                lbl_file = lbl_dir / f'{img_file.stem}.txt'
                if not lbl_file.exists() and not cp_img:
                    continue
                elif not lbl_file.exists():
                    continue

                if lbl_file.stem.startswith('.') or lbl_file.stem == 'classes':
                    continue

                # 读取lbl_file并转换
                cycle_res = executor.submit(
                    self.label_exclude, lbl_file, img_save_dir, lbl_save_dir, img_file, cp_img
                )
                results.append(cycle_res)
        # 等待所有线程完成
        for res in results:
            res.result()

    def __call__(self, dst_dir: str = None, cp_img: bool = False):
        """标签转换接口

        :param str dst_dir: 新标签保存地址, defaults to None
        :param bool cp_img: 是否移动图片到save_dir, defaults to False
        """
        if cp_img and dst_dir is None:
            raise ValueError("save_dir must be set when cp_img is True.")
        if dst_dir is  None or Path(dst_dir).absolute() == self.data_root:
            cp_img = False
        if dst_dir is not None:
            dst_dir = Path(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
        
        all_subsets = []
        for mode in ['train', 'val', 'test']:
            subsets = self.data.get(mode, [])
            if subsets:
                all_subsets += subsets
        all_subsets = list(set(all_subsets))
        logger.info(f"All subsets: {all_subsets}")

        for subset in all_subsets:
            # 初始化每个子集的配置
            lbl_dir = self.data_root / str(subset) / 'labels'
            img_dir = self.data_root / str(subset) / 'images'
            if cp_img:
                img_save_dir = Path(dst_dir) / str(subset) / 'images'
                lbl_save_dir = Path(dst_dir) / str(subset) / 'labels'
                img_save_dir.mkdir(parents=True, exist_ok=True)
                lbl_save_dir.mkdir(parents=True, exist_ok=True)
            else:
                img_save_dir = None
                lbl_save_dir = lbl_dir.parent / f'labels_{self.exclude_title}'
                lbl_save_dir.mkdir(parents=True, exist_ok=True)
            self.exclude_classes_by_subset(
                img_save_dir=img_save_dir, lbl_save_dir=lbl_save_dir, 
                img_dir=img_dir, lbl_dir=lbl_dir, cp_img=cp_img
            )
            logger.info(f"Subset {subset} done.")

        # 生成新的yaml文件
        self.save_yaml(dst_dir)
        # 生成新的classes.txt文件
        self.save_classes(dst_dir)
