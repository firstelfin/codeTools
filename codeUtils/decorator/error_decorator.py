#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   error_decorator.py
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
        cls_call = cls.__call__
        async_status = is_async_function(cls_call)

        def call_hook(_self, *args, **kwargs):
            try:
                assert kwargs.get(self.res_obj, False), \
                    f"ResItemAttrError: {_self.class_name} does not have parameter res_item."
                res = cls_call(_self, *args, **kwargs)
                return res
            except Exception as e:
                # res_item = kwargs.get("res_item")
                res_item = self.inject(kwargs.get(self.res_obj), e)
                return res_item
        
        async def async_call_hook(_self, *args, **kwargs):
            try:
                assert kwargs.get(self.res_obj, False), \
                    f"ResItemAttrError: {_self.class_name} does not have parameter res_item."
                res = await cls_call(_self, *args, **kwargs)
                return res
            except Exception as e:
                res_item = self.inject(kwargs.get(self.res_obj), e)
                return res_item
        
        cls.__call__ = call_hook if not async_status else async_call_hook

        return cls

    def inject(self, res_item, e):
        error_code, error_msg = self.get_error_code_msg(e)
        if isinstance(res_item, dict):
            res_item.update({"self.error_code": error_code, "self.error_msg": error_msg})
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
        suffix_desc = f" --> errorColNum:[{traceback_list[-1].colno}->{traceback_list[-1].end_colno}]"
        error_str += " tracebackStr:" + traceback_str + suffix_desc
        error_class = error_str.split(":")[0]
        error_code = self.type2code.get(error_class, 99)
        logger.info(f"errorCode:{error_code} -> errorMsg:{error_str}")
        return error_code, error_str
