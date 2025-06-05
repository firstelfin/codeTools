#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   errorDecorator.py
@Time    :   2024/08/18 17:52:59
@Author  :   firstElfin 
@Version :   1.0
@Desc    :   None
'''

import traceback
from loguru import logger
from ..tools import is_async_function


ERROR_NAME2CODE = {
    "DownloadError": 10,    # 下载类错误
    "ImgError": 20,         # 图片类错误
    "MatchError": 30,       # 尺寸类错误
    "TorchError":40,        # torch类错误
    "GroupValueError": 15,  # 传参错误
    "JsonError": 16,        # json解析错误
    "OtherError": 99,       # 未定义错误类型
}


class ErrorCheck(object):
    r"""错误管理类, 执行算子的错误管理, 遇到错误会给`res_obj`注入错误的顺序调用栈到`error_msg`, 注入错误编码到`error_code`.

    Example::

        ```
        class MatchError(Exception): ...

        class ResItem(BaseModel):
            resCode: int = 0
            resMsg:  str = ""


        @ErrorCheck(res_obj="res_item")
        class Elfin:

            class_name = "Elfin"

            def __call__(self, num, **kwargs):
                if num < 10:
                    raise MatchError(f"Num: {num} should be greater than 10.")
                else:
                    raise KeyError(f"Num: {num} should be less than 10.")
        
        if __name__ == "__main__":
            elfin = Elfin()
            res = ResItem()
            elfin(20, res_item=res)
            print(res)
        
        ```

    """

    class_name = "ErrorCheck"
    type2code = ERROR_NAME2CODE

    def __init__(self, res_obj="res_item", error_code="resCode", error_msg="resMsg") -> None:
        self.error_code = error_code
        self.error_msg  = error_msg
        self.res_obj    = res_obj
    
    def __call__(self, cls) -> type:

        # 定义__call__
        is_class = isinstance(cls, type)
        cls_call = cls.__call__ if is_class else cls

        async_status = is_async_function(cls_call)
        res_item_error_msg = f"ResItemAttrError: {cls.__name__} does not have parameter {self.res_obj}; "\
            f"Please use the format {self.res_obj}={self.res_obj}_param to pass parameters within the __call__ method."

        def call_hook(_self, *args, **kwargs):
            res_item = kwargs.get(self.res_obj, None)
            assert res_item is not None, res_item_error_msg

            try:
                res = cls_call(_self, *args, **kwargs)
                return res
            except Exception as e:
                res_item = self.inject(res_item, e)
                return res_item
        
        async def async_call_hook(_self, *args, **kwargs):
            res_item = kwargs.get(self.res_obj, None)
            assert res_item is not None, res_item_error_msg

            try:
                res = await cls_call(_self, *args, **kwargs)
                return res
            except Exception as e:
                res_item = self.inject(res_item, e)
                return res_item
        
        if is_class:
            cls.__call__ = call_hook if not async_status else async_call_hook
        else:
            cls = call_hook if not async_status else async_call_hook
        return cls

    def inject(self, res_item, e):
        error_code, error_msg = self.get_error_code_msg(e)
        if isinstance(res_item, dict):
            res_item.update({f"{self.error_code}": error_code, f"{self.error_msg}": error_msg})
            return res_item
        setattr(res_item, self.error_code, error_code)
        setattr(res_item, self.error_msg, error_msg)
        return res_item
    
    def inject_type_to_code(self, trans: dict):
        self.type2code = self.type2code.update(trans)
    
    def get_error_code_msg(self, e):
        error_str = str(e)
        traceback_list = traceback.extract_tb(e.__traceback__)[1:]
        traceback_list.reverse()
        traceback_str = f" --> ".join([str(tl) for tl in traceback_list])
        try:
            suffix_desc = f" --> errorColNum:[{traceback_list[-1].colno+1}->{traceback_list[-1].end_colno+1}]"
        except:
            suffix_desc = ""
        error_str += " tracebackStr:" + traceback_str + suffix_desc
        error_class = error_str.split(":")[0]
        error_code = self.type2code.get(error_class, 99)
        logger.info(f"errorCode:{error_code} -> errorMsg:{error_str}")
        return error_code, error_str
