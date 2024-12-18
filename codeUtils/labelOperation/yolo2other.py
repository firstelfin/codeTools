#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   yolo2other.py
@Time    :   2024/12/12 22:13:25
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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

