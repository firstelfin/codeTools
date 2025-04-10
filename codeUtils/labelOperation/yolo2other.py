#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   yolo2other.py
@Time    :   2024/12/12 22:13:25
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import time
import shutil
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from codeUtils.labelOperation.readLabel import read_txt, read_yolo
from codeUtils.labelOperation.saveLabel import save_json


# def yolo_to_voc(src_dir: str, dst_dir: str = None, classes: str = None) -> None:
#     """Convert YOLO format to VOC format.

#     :param str src_dir: yolo format file directory.
#     :param str dst_dir: voc format file directory.
#     :param str classes: yolo classes.txt file path.
#     """

#     if classes is None:
#         raise ValueError('classes is None, please provide classes.txt file path.')
#     img_dir = Path(src_dir) / 'images'
#     txt_dir = Path(src_dir) / 'labels'
#     if not Path(img_dir).exists():
#         raise ValueError(f'image directory {img_dir} not exists.')
#     if not Path(txt_dir).exists():
#         raise ValueError(f'label directory {txt_dir} not exists.')

#     if classes is None:
#         clssess = Path(src_dir) / 'classes.txt'
        
#     # read classes
#     with open(classes, 'r+', encoding='utf-8') as f:
#         classes = f.readlines
#         classes = [c.strip().replace('\n', '') for c in classes]
    
#     if dst_dir is None:
#         dst_dir = Path(src_dir).parent /  f'{src_dir.stem}_voc'
#     dst_dir = Path(dst_dir)
#     dst_dir.mkdir(exist_ok=True, parents=True)

    

#     # convert yolo to voc
#     with ThreadPoolExecutor() as executor:
#         for file in :
#             dst_file = dst_dir / (file.stem + '.xml')
#             img_file = file.parent / (file.stem + '.jpg')


class YOLO2COCO(object):
    """Convert YOLO format to COCO format.
    
    Args:
        src_dir (str): yolo format file directory. sub_dir should be 'images' and 'labels'.
        dst_dir (str): coco format file directory.
        classes (str): yolo classes.txt file path.
        data (str): yolo data.yaml file path. load classes, train, test and val from data.yaml.
        use_link (bool): whether to use symbolic link to save images.
        split (str): data split, 'train', 'val', 'test'.
        year (str): year of dataset.
        class_start_index (int): start index of class id. default is 0. lt 2.
    """

    def __init__(
            self, src_dir: str = None, dst_dir: str = None, classes: str = None, 
            data: str = None, use_link: bool = False, split: str = 'train', 
            year: str = None, class_start_index: int = 0
        ):
        assert class_start_index < 2, 'class_start_index should be lt 2.'
        if data is None:
            assert src_dir is not None and classes is not None, 'src_dir and classes are None, \
                please provide src_dir and classes, when data is None.'
        if dst_dir is None and data is None:
            raise ValueError('dst_dir and data are None, please provide dst_dir or data.')
        if classes is None and data is None:
            raise ValueError('classes is None, please provide classes.txt file path.')
        self.src_dir = Path(src_dir) if src_dir is not None else None
        self.dst_dir = Path(dst_dir)
        self.classes = classes
        self.data = data
        self.use_link = use_link
        self.split = split
        self.year = year if year is not None else time.strftime("%Y", time.localtime())
        self.use_yaml = data is not None
        self.coco_images = dict()
        self.coco_annotations = dict()
        self.class_start_index = int(class_start_index)

    def load_yolo_yaml(self):
        import yaml
        with open(self.data, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.names = data['names']
        self.data_split = [
            data[split] for split in ['train', 'val', 'test'] if split in data
        ]
        self.datasets_path = Path(data["path"])
        self.splits = [split for split in ['train', 'val', 'test'] if split in data]
        self.coco_images = {split: [] for split in self.splits}
        self.coco_annotations = {split: [] for split in self.splits}
    
    def load_classes(self):
        names = read_txt(self.classes)
        self.names = {i: name for i, name in enumerate(names)}
        self.coco_images = {split: [] for split in self.splits}
        self.coco_annotations = {self.split: []}

    def load_items(self):
        if self.use_yaml:
            self.load_yolo_yaml()
            for i, split in enumerate(self.splits):
                for sub_dir in self.data_split[i]:
                    for img_file in (self.datasets_path / sub_dir / 'images').iterdir():
                        lbl_file = self.datasets_path / sub_dir / 'labels' / (img_file.stem + '.txt')
                        yield img_file, lbl_file, split
        else:
            self.load_classes()
            for img_file in (self.src_dir / 'images').iterdir():
                lbl_file = self.src_dir / 'labels' / (img_file.stem + '.txt')
                yield img_file, lbl_file, self.split

    def coco_prepare(self, img_file: Path, lbl_file: Path, split: str, img_id: int):
        """coco数据集整备

        :param Path img_file: 图片路径
        :param Path lbl_file: 标签路径
        :param str split: 数据集划分(不包含年份)
        :param int img_id: 图片id
        :raises ValueError: 标签格式错误
        """

        # 给COCO添加图片信息
        src_img = cv.imread(str(img_file))
        img_h, img_w = src_img.shape[:2]
        img_info = {
            'id': img_id,
            'file_name': str(self.dst_dir / f"{split}{self.year}" / img_file.name),
            'height': img_h,
            'width': img_w,
        }
        self.coco_images[split].append(img_info)

        # 给COCO添加标注信息
        file_labels = None
        for _ in range(3):
            file_labels = read_yolo(lbl_file)
            if file_labels is not None:
                break
            
        if file_labels is None:
            file_labels = []
        
        for label in file_labels:
            cls_id = label[0]
            if len(label[1:]) == 4:
                x, y, w, h = label[1:]
                x_min, y_min = int((x - w / 2) * img_w), int((y - h / 2) * img_h)
                box_w, box_h = int(w * img_w), int(h * img_h)
            else:
                raise ValueError(f'label {label} format error. segmentation label is not supported.')
            ann_info = {
                "id": 0,
                "image_id": img_id,
                "category_id": cls_id+self.class_start_index,
                "bbox": [x_min, y_min, box_w, box_h],
                "iscrowd": 0,
                "area": box_w * box_h,
                "segmentation": [],
            }
            self.coco_annotations[split].append(ann_info)
        
        # 保存图片
        img_link = self.dst_dir / f"{split}{self.year}" / img_file.name
        img_link.parent.mkdir(exist_ok=True, parents=True)
        if self.use_link:
            if not img_link.exists():
                img_link.symlink_to(img_file)
        else:
            shutil.copy(img_file, img_link)
        return len(file_labels)

    def anno_id_modify(self, start_index: int = 0):
        for split in self.coco_annotations:
            for ann in self.coco_annotations[split]:
                ann['id'] = start_index
                start_index += 1
    
    def save_coco_json(self, anno_file: Path, split: str):
        coco_info = {
            "description": f"COCO {self.year} Dataset",
            "version": "1.0",
            "year": self.year,
            "contributor": "firstelfin",
            "date_created": time.strftime("%Y/%m/%d", time.localtime()),
        }
        coco_images = self.coco_images[split]
        coco_annotations = self.coco_annotations[split]
        coco_dict = {
            "info": coco_info,
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [
                {
                    "id": i+self.class_start_index, 
                    "name": name, 
                    "supercategory": name
                } for i, name in self.names.items()
            ],
        }
        save_json(anno_file, coco_dict)
        logger.info(f"save {split} coco json file {anno_file} success.")

    def __call__(self, image_index: int = 0, anno_index: int = 0, *args, **kwds):
        
        res = []
        with ThreadPoolExecutor() as executor:
            for img_file, lbl_file, split in self.load_items():
                convert_res = executor.submit(self.coco_prepare, img_file, lbl_file, split, image_index)
                image_index += 1
                res.append(convert_res)
        
            for convert_res in  tqdm(as_completed(res), total=len(res), desc='yolo2coco'):
                convert_res.result()

        # 保存COCO格式数据集
        self.anno_id_modify(start_index=anno_index)
        anno_dir = self.dst_dir / 'annotations'
        anno_dir.mkdir(exist_ok=True, parents=True)

        for split in self.coco_annotations:
            anno_file = anno_dir / f"{split}{self.year}.json"
            self.save_coco_json(anno_file=anno_file, split=split)
            

def yolo2coco(
        src_dir: str = None, dst_dir: str = None, classes: str = None, 
        data: str = None, use_link: bool = False, split: str = 'train', 
        year: str = None, class_start_index: int = 0, image_index: int = 0, anno_index: int = 0
    ):
    """Convert YOLO format to COCO format.

    :param str src_dir: yolo format file directory. sub_dir should be 'images' and 'labels'.
    :param str dst_dir: coco format file directory.
    :param str classes: yolo classes.txt file path.
    :param str data: yolo data.yaml file path. load classes, train, test and val from data.yaml.
    :param bool use_link: whether to use symbolic link to save images.
    :param str split: data split, 'train', 'val', 'test'.
    :param str year: year of dataset.
    :param int class_start_index: start index of class id. default is 0. lt 2.
    :param int image_index: start index of image id. default is 0.
    :param int anno_index: start index of annotation id. default is 0.
    """
    YOLO2COCO(
        src_dir=src_dir, dst_dir=dst_dir, classes=classes, data=data, 
        use_link=use_link, split=split, year=year, class_start_index=class_start_index
    )(image_index=image_index, anno_index=anno_index)
