#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   cache_file.py
@Time    :   2024/09/18 11:53:04
@Author  :   firstElfin 
@Version :   0.0.6
@Desc    :   支持缓存文件到本地: image, dict
'''

import os
import shutil
import time
import json
import msgpack
import numpy as np
import cv2 as cv
from loguru import logger
from contextlib import contextmanager
from numpy import ndarray
from PIL import Image
from pathlib import Path, PosixPath
from concurrent.futures import ThreadPoolExecutor
from . import CPU_KERNEL_NUM


class CacheFile:
    """缓存文件类, 支持msgpack, json, jpeg, webp, png格式缓存. 支持软链接. 增删操作均将相关文件加锁, 防止被其他进程读取.

    :param str | PosixPath file_path: 缓存文件路径
    :param bool symlink: 是否创建软链接, defaults to True

    Example:
        ```python
        from pathlib import Path
        from codeUtils.tools.cache_file import CacheFile

        cache = CacheFile('runs', max_workers=6)
        suffix_trans = {".jpg": "jpeg", ".jpeg": "jpeg", ".json": "msgpack"}

        def batch_load_test(data_list):
            all_data = cache.batch_cache_load(data_list)
            return all_data

        def batch_save_test(data_list, dataset):
            inject_list = [
                {"stem": file.stem, "sud_dir": ["inject_data"], "mode": file.suffix} for file in data_list
            ]
            all_data = []
            for i, data in enumerate(dataset):
                all_data.append({"data": data, "inject_obj": inject_list[i], "mode": suffix_trans[inject_list[i]["mode"]]})
            all_data = cache.batch_cache_add(all_data)
            return all_data


        if __name__ == '__main__':
            import time
            all_files = [Path('xxx.jpeg'), Path('xxx.json')]
            s1 = time.perf_counter()
            all_files_data = batch_load_test(all_files)
            status = batch_save_test(all_files, all_files_data)
            s2 = time.perf_counter()
            print(f"load time: {s2-s1:.4f}s")
            cache.batch_cache_delete(list(Path("xxx/runs/links/inject_data").iterdir()))
            print(status)

        ```
    """

    class_name = "CacheFile"

    def __init__(
            self, file_path: str|Path, symlink: bool=True, 
            max_workers: int=4, usage_mode: str="or", 
            gb_threshold: float=0.2, ratio_threshold:float=0.05
        ):
        """初始化缓存文件类

        :param str | PosixPath file_path: 缓存文件路径文件夹
        :param bool symlink: 是否创建软链接, defaults to True
        :param int max_workers: 线程池最大线程数, defaults to 6
        :param str usage_mode: 磁盘使用模式, defaults to "or"
        :param float gb_threshold: 剩余空间阈值(单位GB), defaults to 0.2
        :param float ratio_threshold: 剩余空间比例阈值, defaults to 0.05
        """
        self.file_path = Path(file_path)
        self.symlink = symlink
        self.gb = 2**30  # 1GB
        self.cpu_cores = max(min(int(CPU_KERNEL_NUM * 0.8), max_workers), 6)
        self.usage_mode = usage_mode.lower()
        assert self.usage_mode in ["or", "and"], "Unsupported usage_mode"
        self.free_gb_threshold = gb_threshold  # 200MB
        self.free_ratio_threshold = ratio_threshold  # 5%
        self.file_path.mkdir(parents=True, exist_ok=True)
    
    def check_disk_usage(self):
        """检查磁盘空间是否足够"""
        total, _, free = shutil.disk_usage(self.file_path)
        free_gb = free / self.gb
        free_ratio = free / total
        ratio_status = free_ratio < self.free_ratio_threshold
        gb_status = free_gb < self.free_gb_threshold

        if self.usage_mode == "and" and (ratio_status and gb_status):
            return False
        elif self.usage_mode == "or" and (ratio_status or gb_status):
            return False
        else:
            return True

    @staticmethod
    @contextmanager
    def lock(link_paths: list[Path], timeout: int=10):
        """上下文管理器，确保在删除软链接时不被读取。"""

        lock_links = [Path(link_path).with_suffix('.lock') for link_path in link_paths]
        should_cleanup = [False] * len(lock_links)
        try:
            # 创建锁文件，表示正在使用链接
            for i,lock_file in enumerate(lock_links):
                lock_file.touch(exist_ok=False)
                should_cleanup[i] = True
            yield  # 在这个上下文中执行
        except FileExistsError:
            logger.warning('CacheFile: The file has been locked, please try again later.')
            raise
        finally:
            # 删除锁文件
            for j, lock_file in enumerate(lock_links):
                if lock_file.exists() and should_cleanup[j]:
                    lock_file.unlink()

    def cache_save(self, data: str|bytes|dict|list|ndarray, file_path: str|Path, mode: str="msgpack"):
        """保存Python字典(列表)到缓存文件, 支持msgpack和json格式. msgpack数据保存比json数据保存快快快太多了.

        :param str | bytes | dict | list data: Python数据
        :param str | PosixPath file_path: 缓存文件路径
        :param str mode: 缓存文件格式, defaults to "msgpack"
        :raises ValueError: Unsupported mode
        """
        mode = mode.lower()
        file_path = Path(file_path)
        lock_files = [file_path, file_path.resolve()] if file_path.is_symlink() else [file_path]
        try:
            with self.lock(lock_files):
                if mode == "msgpack":
                    packed_data = msgpack.packb(data)
                    if not isinstance(packed_data, bytes):
                        raise TypeError(f"msgpack.packb() should return bytes, got {type(packed_data)}")
                    with open(file_path, "wb") as f:
                        f.write(packed_data)
                elif mode == "json":
                    with open(file_path, "w+", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                elif mode == "webp":
                    # 当前webp格式cv在保存数据上的支持没有pillow好
                    if not isinstance(data, ndarray):
                        raise TypeError(f"data should be numpy.ndarray, got {type(data)}")
                    image_data = Image.fromarray(data)
                    image_data.save(file_path, format=mode)
                elif mode in ["jpeg", "png"]:
                    if not isinstance(data, ndarray):
                        raise TypeError(f"data should be numpy.ndarray, got {type(data)}")
                    cv.imwrite(str(file_path), data)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
        except FileExistsError:
            return False
        return True
    
    def cache_load(self, file_path: str|Path, mode: str="msgpack"):
        """加载缓存文件为Python字典(列表), msgpack数据加载比json数据加载快30%.
        加载缓存文件时, 会自动对target文件进行锁定, 防止被其他进程读取。

        :param str | PosixPath file_path: 缓存文件路径
        :param str mode: 缓存文件格式, defaults to "msgpack"
        :raises ValueError: Unsupported mode
        :return dict | list: 缓存内容
        """
        file_path = Path(file_path)
        mode = mode.lower()
        lock_files = [file_path, file_path.resolve()] if file_path.is_symlink() else [file_path]
        try:
            with self.lock(lock_files):
                if mode == "msgpack":
                    with open(file_path, "rb") as f:
                        packed_data = f.read()
                        data_dict = msgpack.unpackb(packed_data)
                elif mode == "json":
                    with open(file_path, "r+") as f:
                        data_dict = json.load(f)
                elif mode in ["jpeg", "webp", "png", "jpg"]:
                    data_dict = cv.imread(str(file_path))
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
        except FileExistsError:
            return None
        return data_dict

    def inject_obj_parser(self, inject_obj: dict, suffix="cache", soft_link=True):
        """根据inject_obj生成缓存文件路径

        :param dict inject_obj: 需要缓存的对象路径封装
        :param str suffix: 缓存文件后缀, defaults to "cache"
        :param bool soft_link: 是否创建软链接, defaults to True
        :return PosixPath: 缓存文件路径
        """
        sub_dir = inject_obj.get("sud_dir", [])
        assert isinstance(sub_dir, list), "sub_dir must be a list"
        stem = inject_obj.get("stem", "test")
        assert isinstance(stem, str), "stem must be a string"
        inject_save_root = self.file_path / f"cache/{'/'.join(sub_dir)}"
        if not inject_save_root.exists():
            inject_save_root.mkdir(parents=True, exist_ok=True)
        
        # 创建软连接
        if soft_link:
            inject_link_root = self.file_path / f"links/{'/'.join(sub_dir)}"
            if not inject_link_root.exists():
                inject_link_root.mkdir(parents=True, exist_ok=True)
            return inject_save_root / f"{stem}.{suffix}", inject_link_root / f"{stem}.{suffix}"
        return inject_save_root / f"{stem}.{suffix}", None

    def cache_add(self, data: str|bytes|dict|list|ndarray, inject_obj: dict, mode: str="msgpack") -> bool:
        """缓存数据到本地, 并创建软链接(若self.symlink为True).

        :param str | bytes | dict | list | ndarray data: 需要缓存的数据
        :param dict inject_obj: 需要缓存的对象路径封装
        :param str mode: 缓存文件格式, defaults to "msgpack"
        :return bool: 是否成功缓存
        """
        # 判断缓存空间是否足够
        if not self.check_disk_usage():
            return False
        # 缓存文件保存
        save_path, link_path = self.inject_obj_parser(inject_obj, suffix=mode.lower(), soft_link=self.symlink)
        # 文件已经存在, 删除原文件极其软链接
        if self.symlink and link_path is not None:
            self.cache_delete(link_path)
        else:
            self.cache_delete(save_path)
        self.cache_save(data, save_path, mode=mode)
        # 缓存文件软连接创建
        # self.file_path/links/.../xxx.cache -> self.file_path/cache/.../xxx.cache
        if link_path is not None and self.symlink:
            with self.lock([link_path]):
                link_path.symlink_to(save_path.absolute())
        return True
    
    def cache_delete(self, link_path: str|Path):
        """删除缓存文件软链接, link_path如果是软链接路径, target文件也会被删除, lock是自适应.

        :param str | PosixPath link_path: 缓存文件软链接路径 或 缓存文件路径
        :return bool: 是否成功删除
        """
        link_path = Path(link_path)
        if not link_path.exists():
            return True
        
        is_link = link_path.is_symlink()
        lock_files = [link_path, link_path.resolve()] if is_link else [link_path]
        
        try:
            with self.lock(lock_files):
                link_path.unlink()
                if is_link:
                    lock_files[-1].unlink()
            return True
        except FileExistsError:
            return False
        except:
            if link_path.exists():
                status = False
            elif is_link and lock_files[-1].exists():
                status = False
            else:
                status = True
            return status

    def batch_cache_add(self, data_list: list[dict]):
        """批量缓存数据到本地, 并创建软链接(若self.symlink为True).

        :param list[dict] data_list: 需要缓存的数据列表, item是每个文件打包好的参数对象

        Example:
            ```python
            >>> data_list = [
                    {
                        "data": np.random.rand(2448, 3264, 3),
                        "inject_obj": {
                            "sud_dir": ["test"],
                            "stem": "000"
                        },
                        "mode": "jpeg"
                    },
                    {
                        "data": test_dict,
                        "inject_obj": {
                            "sud_dir": ["test"],
                            "stem": "001"
                        },
                        "mode": "json"
                    }
                ]
            >>> cache_file = CacheFile("test.cache")
            >>> cache_file.batch_cache(data_list)
            ```
        """
        
        res = []
        with ThreadPoolExecutor(max_workers=self.cpu_cores) as executor:
            for data_dict in data_list:
                req = executor.submit(self.cache_add, data_dict["data"], data_dict["inject_obj"], data_dict["mode"])
                res.append(req)
        result = [req.result() for req in res]
        return result
    
    def batch_cache_delete(self, data_list: list[str|PosixPath]):
        """批量删除缓存文件软链接, link_path如果是软链接路径, target文件也会被删除.

        :param list[str | PosixPath] data_list: 缓存文件软链接路径 或 缓存文件路径列表
        :return list[bool]: 是否成功删除列表
        """

        res = []
        with ThreadPoolExecutor(max_workers=self.cpu_cores) as executor:
            for link_path in data_list:
                req = executor.submit(self.cache_delete, link_path)
                res.append(req)
        result = [req.result() for req in res]
        return result

    def batch_cache_load(self, data_list: list[dict|str|PosixPath]):

        res = []
        with ThreadPoolExecutor(max_workers=self.cpu_cores) as executor:
            for data_dict in data_list:
                # 缓存文件路径解析
                if isinstance(data_dict, str) or isinstance(data_dict, PosixPath):
                    cache_file_path = Path(data_dict)
                    mode = cache_file_path.suffix[1:].lower()
                else:
                    parser_path = self.inject_obj_parser(data_dict["inject_obj"], suffix=data_dict["mode"].lower(), soft_link=self.symlink)
                    cache_file_path = parser_path[int(self.symlink)]
                    mode = data_dict["mode"].lower()

                assert cache_file_path is not None, "cache_file_path is None"
                req = executor.submit(self.cache_load, cache_file_path, mode)
                res.append(req)
        result = [req.result() for req in res]
        return result
