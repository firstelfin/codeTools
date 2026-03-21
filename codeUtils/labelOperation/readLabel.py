#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   readLabel.py
@Time    :   2024/12/09 15:34:02
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import yaml
import json
from typing import Literal
from pathlib import Path
from natsort import natsorted
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def parser_json(json_file: str | Path):
    """读取json文件, 返回一个字典"""
    for _ in range(3):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if data is not None:
                return data
        except:
            continue
    return None


def read_json(json_file: str | Path):
    return parser_json(json_file)


def read_yolo(label_file: str | Path):
    """读取yolo格式的标签文件, 返回一个列表"""
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
    except:
        return None
    
    labels = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        label = line.split()
        box = map(float, label[1:])
        cls_id = int(label[0])
        labels.append([cls_id, *box])
    return labels


def read_voc(label_file: str | Path, extra_keys: list = []) -> dict | None:
    """读取voc格式的标签文件, 返回一个json对象, 指定extra_keys可以读取标注实例中额外的键值对

    :param str | Path label_file: voc格式的标签文件路径
    :param list extra_keys: 额外需要添加到object元素字典中的键列表, 格式为["key1", "key2",...]
    :return dict: voc格式的标签文件内容, 组织格式是超文本标签组织格式的映射
    """
    
    # 读取xml文件
    try:
        with open(label_file, 'r') as f:
            xml_str = f.read()
    except:
        return None

    # 解析xml文件
    soup = BeautifulSoup(xml_str, 'xml')

    def safe_text(tag, default=""):
        return tag.text.strip() if tag else default
    
    def safe_find(tag, *keys):
        """安全地进行多级 find() TODO: 形参是否会改变实参"""
        for key in keys:
            if tag is None:
                return None
            tag = tag.find(key)
        return tag

    voc_dict = {
        'folder': safe_text(soup.find('folder')),
        'filename': safe_text(soup.find('filename')),
        'path': safe_text(soup.find('path')),
        'source': {"database": safe_text(safe_find(soup, 'source', 'database'), "Unknown")},
        'segmented': int(safe_text(soup.find('segmented'), "0")),
        'size': {
            'width': int(safe_text(safe_find(soup, "size", "width"), "0")),
            'height': int(safe_text(safe_find(soup, "size", "height"), "0")),
            'depth': int(safe_text(safe_find(soup, "size", "depth"), "3")),
        },
        'object': [
            {
                'name': safe_text(safe_find(obj, 'name')),
                'pose': safe_text(safe_find(obj, 'pose'), default="Unspecified"),
                'truncated': int(safe_text(safe_find(obj, 'truncated'), default="0")),
                'difficult': int(safe_text(safe_find(obj, 'difficult'), default="0")),
                'bndbox': {
                    'xmin': int(float(safe_text(safe_find(obj, 'bndbox', 'xmin'), default="0"))),
                    'ymin': int(float(safe_text(safe_find(obj, 'bndbox', 'ymin'), default="0"))),
                    'xmax': int(float(safe_text(safe_find(obj, 'bndbox', 'xmax'), default="0"))),
                    'ymax': int(float(safe_text(safe_find(obj, 'bndbox', 'ymax'), default="0")))
                },
                **{key: safe_text(safe_find(obj, key)) for key in extra_keys}
            } for obj in soup.find_all('object')
        ]
    }
    
    return voc_dict


def read_txt(txt_file: str):
    res = []
    with open(txt_file, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().replace("\n", "")
        if not line:
            continue
        res.append(line)
    return res


def read_yaml(file_path: Path | str) -> dict:
    """
    安全地读取 YAML 文件
    """
    path = Path(file_path)

    # 检查文件是否存在
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    # 使用 utf-8 编码打开
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 处理空文件情况 (safe_load 读取空文件会返回 None)
    return data if data is not None else {}


def get_lbl_names(lbl_file: str, suffix: Literal[".json", ".xml"] = ".json") -> dict:
    """获取标签文件包含的类别统计信息

    :param str lbl_file: 标签文件路径
    :param Literal[".json", ".xml"] suffix: 文件后缀, defaults to ".json"
    :return dict: 类别统计信息字典
    """
    lbl_dict = read_voc(lbl_file) if suffix == ".xml" else read_json(lbl_file)
    names = dict()
    if lbl_dict is None:
        return names
    shapes = lbl_dict.get("object", list()) if suffix == ".xml" else lbl_dict.get("shapes", list())
    key_name = "name" if suffix == ".xml" else "label"
    for obj in shapes:
        names[obj[key_name]] = names.get(obj[key_name], 0) + 1
    return names


def statitic_gen_names(lbl_dir: list[str|Path], dst_dir: str|Path|None = None, suffix: Literal[".json", ".xml"] = ".json") -> list[str]:
    """生成classes.txt文件

    :param list[str|Path] src_dir: xml标注目录路径, 文件夹下平铺所有XML文件
    :param str|Path|None dst_dir: classes.txt文件保存路径, defaults to None
    :param Literal[".json", ".xml"] suffix: 标签文件后缀, defaults to ".json"
    :return list: names列表
    """

    all_names = dict()
    all_objects_list = []
    all_files = []
    for src_dir in lbl_dir:
        all_files.extend(list(Path(src_dir).rglob(f"*{suffix}")))
    print(f"共有{len(all_files)}个{suffix}文件")
    with ThreadPoolExecutor() as executor:
        for lbl_file in all_files:
            all_objects_list.append(executor.submit(get_lbl_names, str(lbl_file), suffix))
    
        for names in tqdm(as_completed(all_objects_list), total=len(all_objects_list), desc="GetAllNames"):
            name_dict = names.result()
            for name, count in name_dict.items():
                all_names[name] = all_names.get(name, 0) + count

    if dst_dir is None:
        dst_file = Path(lbl_dir[0]) / "classesElfin.txt"
        dst_dir = dst_file.parent
    else:
        dst_file = Path(dst_dir) / "classesElfin.txt"
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_statistic_file = dst_dir / "classesStatitic.json"

    # 使用自然顺序排序类别
    classes = natsorted(all_names.keys())
    print(f"共有{len(classes)}个类别")
        
    with open(dst_file, "w+", encoding="utf-8") as f:
        for name in classes:
            f.write(name + "\n")
    with open(dst_statistic_file, "w+", encoding="utf-8") as f:
        json.dump(all_names, f, indent=4, ensure_ascii=False)
    
    return classes
