#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   splitImg.py
@Time    :   2024/10/31 09:58:47
@Author  :   firstElfin 
@Version :   0.1.6
@Desc    :   图片裁切工具
'''

import math

# Code from https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/dataset.py

def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: set[tuple], width: int, height: int, image_size=640):
    """寻找最接近原始图片宽高比的宽高比
    选择更接近的宽高比; 宽高比的差距相同, 选择面积更小的宽高比, 即使用尽可能多(也不能太多)的patch去裁剪. 
    ratio[0], ratio[1]足够大, 才能使`0.5 * image_size * image_size * ratio[0] * ratio[1]`小于原始图片面积。

    :param float aspect_ratio: 图片原始的宽高比
    :param set[tuple] target_ratios: 根据需求预设的宽高比集合
    :param int width: 原始图片的宽度
    :param int height: 原始图片的高度
    :param int image_size: 期望裁剪的图片大小, defaults to 640
    :return tuple: 最接近原始图片宽高比的宽高比
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:  # 选择更接近的宽高比
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:  # 宽高比的差距相同, 选择面积更小的宽高比，即使用尽可能多的patch去裁剪
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def internvl_split(image, min_num=1, max_num=6, image_size=640, use_thumbnail=False):
    """internvl模型裁剪图片func, 裁剪数量尽可能接近max_num, 且宽高比尽可能接近原图

    :param pillow.Image.Image image: pillow图片, 待裁剪的原始图片
    :param int min_num: 最小裁剪数量, defaults to 1
    :param int max_num: 最大裁剪数量, defaults to 6
    :param int image_size: 裁剪后图片的期望大小, defaults to 640
    :param bool use_thumbnail: 是否原图缩减, defaults to False
    :return list: 裁剪后的图片列表
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 选择满足入参善良需求的最接近原图比例的裁剪图比例, 记录宽高的裁剪次数
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def cut_num_compute(image_size=640, src_img_size=1280, overlap_ratio=0.25):
    """裁剪图片的数量计算, 只根据单个变长计算这个边需要裁切多少个子图

    :param int image_size: 裁切的目标图片大小, defaults to 640
    :param int src_img_size: 原始图片大小, defaults to 1280
    :param float overlap_ratio: 子图之间的重叠比例, defaults to 0.25
    :return int: 裁切产生的子图数量
    """
    cut_trail_num = (src_img_size - overlap_ratio * image_size) // (image_size - overlap_ratio * image_size)
    cut_trail_num_modify = (src_img_size - overlap_ratio * image_size) / (image_size - overlap_ratio * image_size) % 1
    if cut_trail_num_modify > 0 and cut_trail_num_modify >= overlap_ratio:
        cut_trail_add = 1
    else:
        cut_trail_add = 0
    cut_trail_num += cut_trail_add
    return int(cut_trail_num)


def overlap_split(image, image_size=640, overlap_ratio=0.25, use_thumbnail=False):
    """重叠裁切图片func, 裁切后的图片尽大小尽可能接近image_size, 重叠比例接近overlap_ratio

    :param _type_ image: 原始图片
    :param int image_size: 期望裁切图片的大小, defaults to 640
    :param float overlap_ratio: 图片裁切的重叠比例, defaults to 0.25
    :param bool use_thumbnail: 是否使用整张图的缩略图, defaults to False
    :return tuple: 裁切后的图片列表, 裁切后的bbox列表
    """
    orig_width, orig_height = image.size

    cut_width_num = cut_num_compute(image_size=image_size, src_img_size=orig_width, overlap_ratio=overlap_ratio)
    cut_height_num = cut_num_compute(image_size=image_size, src_img_size=orig_height, overlap_ratio=overlap_ratio)
    stride_ratio = 1 - overlap_ratio
    cut_width_size = math.ceil(orig_width  / (cut_width_num * stride_ratio + overlap_ratio))
    cut_height_size = math.ceil(orig_height / (cut_height_num * stride_ratio + overlap_ratio))
    cut_stride_width = math.ceil(cut_width_size * stride_ratio)
    cut_stride_height = math.ceil(cut_height_size * stride_ratio)
    overlap_width = math.ceil(cut_width_size * overlap_ratio)
    overlap_height = math.ceil(cut_height_size * overlap_ratio)

    processed_images = []
    processed_bbox = []
    for i in range(cut_width_num):
        for j in range(cut_height_num):
            box = (
                int(i * cut_stride_width),
                int(j * cut_stride_height),
                min(int((i + 1) * cut_stride_width + overlap_width), orig_width),
                min(int((j + 1) * cut_stride_height + overlap_height), orig_height)
            )
            split_img = image.crop(box)
            split_img = split_img.resize((image_size, image_size))
            processed_images.append(split_img)
            processed_bbox.append(box)
            if box[-2] > orig_width - 2 and box[-1] > orig_height - 2:
                print("True")

    if use_thumbnail and len(processed_images) != 1:
        # 保持缩略图的宽高比
        scale_ratio = min(image_size / orig_width, image_size / orig_height)
        new_width = int(orig_width * scale_ratio)
        new_height = int(orig_height * scale_ratio)
        thumbnail_img = image.resize((new_width, new_height))
        add_thumbnail_img = Image.new('RGB', (image_size, image_size), (114, 114, 114))
        add_thumbnail_img.paste(thumbnail_img, ((image_size - new_width) // 2, (image_size - new_height) // 2))
        processed_images.append(add_thumbnail_img)  # 最后一个图片是缩略图

        processed_bbox.append((0, 0, orig_width, orig_height))  # 最后一个bbox是原图的bbox
    
    return processed_images, processed_bbox


if __name__ == '__main__':
    from PIL import Image
    src_img = Image.open('...')
    a, b = overlap_split(src_img, image_size=640, overlap_ratio=0.25, use_thumbnail=False)
    pass
