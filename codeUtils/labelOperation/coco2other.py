#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   coco2other.py
@Time    :   2024/12/12 22:07:25
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

from loguru import logger
from pathlib import Path
from codeUtils.labelOperation.converter import COCOToAll
from codeUtils.labelOperation.saveLabel import save_voc_label


def coco_show():
    coco_dict = {
        "info": {
            "description": "Example COCO dataset",
            "url": "https://github.com/cocodataset/cocoapi",
            "version": "1.0",
            "year": 2014,
            "contributor": "COCO Consortium",
            "date_created": "2019/05/01"
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            }
        ],
        "images": [
            {
                "license": 1,
                "file_name": "000000397133.jpg",
                "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
                "height": 427,
                "width": 640,
                "date_captured": "2013-11-14 17:02:52",
                "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
                "id": 397133
            },
            {
                "license": 1,
                "file_name": "000000037777.jpg",
                "coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg",
                "height": 427,
                "width": 640,
                "date_captured": "2013-11-14 17:02:52",
                "flickr_url": "http://farm9.staticflickr.com/8041/8024364248_4e5a7e36c3_z.jpg",
                "id": 37777
            }
        ],
        "annotations": [
            {
                "segmentation": [
                    [
                        192.81,
                        247.09,
                        192.81,
                        230.51,
                        176.23,
                        223.93,
                        176.23,
                        207.35,
                        192.81,
                        200.77,
                        192.81,
                        247.09
                    ]
                ],
                "area": 1035.749,
                "iscrowd": 0,
                "image_id": 397133,
                "bbox": [
                    176.23,
                    200.77,
                    16.58,
                    6.57
                ],
                "category_id": 18,
                "id": 42986
            },
            {
                "segmentation": [
                    [
                        325.12,
                        247.09,
                        325.12,
                        230.51,
                        308.54,
                        223.93,
                        308.54,
                        207.35,
                        325.12,
                        200.77,
                        325.12,
                        247.09
                    ]
                ],
                "area": 1035.749,
                "iscrowd": 0,
                "image_id": 397133,
                "bbox": [
                    308.54,
                    200.77,
                    16.58,
                    6.57
                ],
                "category_id": 18,
                "id": 42987
            }
        ],
        "categories": [
            {
                "supercategory": "person",
                "id": 18,
                "name": "person"
            }
        ]
    }
    print(coco_dict)
    return coco_dict


class COCO2Labelme(COCOToAll):

    def __init__(self, img_dir: str, lbl_file: str, dst_dir: str):
        super().__init__(img_dir, lbl_file, dst_dir)
    
    def save_label(self, img_path: str, labelme_dict: dict, **kwargs):
        super().save_label(img_path, labelme_dict)


class COCO2VOC(COCOToAll):

    def __init__(self, img_dir: str, lbl_file: str, dst_dir: str):
        super().__init__(img_dir, lbl_file, dst_dir)
    
    def save_label(self, img_path: str, labelme_dict: dict, extra_keys: list = []):
        xml_file = self.dst_dir / f"{Path(img_path).stem}.xml"
        voc_dict = {
            'folder': self.dst_dir.name,
            'filename': Path(img_path).name,
            'path': Path(img_path).name,
            'source': {"database": "Unknown"},
            'segmented': 0,  # TODO: check if it's 0 or 1
            'size': {
                'width': int(labelme_dict['imageWidth']),
                'height': int(labelme_dict['imageHeight']),
                'depth': 3
            },
            'object': [
                {
                    'name': obj["label"],
                    'pose': "Unspecified",
                    'truncated': 0,
                    'difficult': 0,
                    'bndbox': {
                        'xmin': int(obj["points"][0][0]),
                        'ymin': int(obj["points"][0][1]),
                        'xmax': int(obj["points"][1][0]),
                        'ymax': int(obj["points"][1][1]),
                    },
                    **{key: obj.find(key).text for key in extra_keys}
                } for obj in labelme_dict['shapes']
            ]
        }
        for obj in labelme_dict['shapes']:
            if obj['shape_type'] == "rectangle":
                voc_dict['object'].append({
                    'name': obj["label"],
                    'pose': "Unspecified",
                    'truncated': 0,
                    'difficult': 0,
                    'bndbox': {
                        'xmin': int(obj["points"][0][0]),
                        'ymin': int(obj["points"][0][1]),
                        'xmax': int(obj["points"][1][0]),
                        'ymax': int(obj["points"][1][1]),
                    },
                    **{key: obj.find(key).text for key in extra_keys}
                })
            elif obj['shape_type'] == "polygon":
                logger.warning(f"Unsupported shape type: {obj['shape_type']}")
                # voc_dict['object'].append({
                #     'name': obj["label"],
                #     'pose': "Unspecified",
                #     'truncated': 0,
                #     'difficult': 0,
                #     'polygon': [  # Note: polygon is a list of points, 自定义数据格式，暂时未支持
                #         {'x': int(point[0]), 'y': int(point[1])} for point in obj["points"]
                #     ],
                #     **{key: obj.find(key).text for key in extra_keys}
                # })
            else:
                logger.warning(f"Unsupported shape type: {obj['shape_type']}")
        save_voc_label(xml_file, voc_dict)


def coco2labelme(img_dir: str, lbl_file: str, dst_dir: str):
    converter = COCO2Labelme(img_dir, lbl_file, dst_dir)
    converter()


def coco2voc(img_dir: str, lbl_file: str, dst_dir: str, extra_keys: list = []):
    converter = COCO2VOC(img_dir, lbl_file, dst_dir)
    converter(extra_keys=extra_keys)

