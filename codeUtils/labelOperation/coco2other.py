#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   coco2other.py
@Time    :   2024/12/12 22:07:25
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''


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
