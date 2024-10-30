# Operator Definition

## 1. 概述

标准化算子是一种特殊模块整合方式。算子不仅包含常规的算法功能实现，还包含了初始化超参数、参数检查、必传参数查询、返回参数查询等功能。结合注册器，可以实现算子的自动加载，降低动态加载的难度，提升算子的复用性。

## 2. operator_def模块

operator_def模块提供了标准化算子的基类定义和IOS算子实现，并将两者注入OPERATORS_REGISTRY注册器。建议所有子项目、算子定义都继承自operator_def模块的基类，并注入OPERATORS_REGISTRY注册器。

### 2.1 算子初始化

```python
def __init__(self):
    super().__init__()
    self.threshold = FloatParam(0.5)
```

算子初始化时，需要调用父类的初始化方法，并定义参数。参数类型可以是IntParam、FloatParam、StrParam、ListParam等。

上面提到两个关键点是必须要严格准守的：
1. 超参数不要初始化传参，必须使用类属性默认值定义，定义类型必须是ConfigBase对象，否则工程会直接报错。
2. 父类初始化方法，必须显示调用，父类有不可或缺的属性注入，否则容易导致报错。

### 2.2 算子执行方法

算子执行统一使用`__call__`方法，形参有两个：`params` 和 `res_item`，分别表示算子参数和结果。内部可以实现具体的算法逻辑，要求低功耗，高性能。

一个不成熟的案例：

```python
def __call__(self, params: dict, res_item: dict = {}) -> dict:
    res_item = super().__call__(params, res_item)
    box1, box2 = params["bboxes"]
    double = params.get("double", False)
    ios = ios_box(box1, box2, mode="xyxy", double=double)
    res_item["ios"] = ios if double else [ios]  # 格式对齐
    return res_item
```

### 2.3 算子必传参数和注入参数

IOS算子案例：

```python
def used_keys(self) -> list:
    return ["bboxes"]

def inject_keys(self) -> list:
    return ["ios"]
```

### 2.4 参数注释和文档字符串

注意事项如下：
1. 形参类型注释：使用PEP484格式，如`params: dict`，`res_item: dict = {}`, 避免使用typing库；
2. 算子注释：使用Google/sphinx风格的文档字符串，描述算子功能、输入参数、输出参数、返回值等；
3. 尽可能给出Example示例代码，方便其他开发者快速理解和使用;
4. 方法、函数内部代码尽量精简，提高复用性和可读性。

## 3. 算子与任务调度器实用案例

```python
from codeUtils.scheduler import PARAM_REGISTRY, ConfigBase
from codeUtils.scheduler import IntParam, StrParam
from codeUtils.scheduler import OPERATORS_REGISTRY, BaseOperator
from codeUtils.scheduler import TaskManager


@PARAM_REGISTRY
class MyParam(ConfigBase):

    def __init__(self, name, age):
        self.name = StrParam(name)
        self.age = IntParam(age)

    def check(self):
        self.name.check()
        self.age.check()
    
    def get(self):
        return self.name.get(), self.age.get()
    
    def set(self, values):
        self.name.set(values[0])
        self.age.set(values[1])


@OPERATORS_REGISTRY
class MyOperator(BaseOperator):

    def __init__(self):
        super().__init__()
        self.threshold = MyParam("elfin", 18)

    def __call__(self, params: dict, res_item: dict = {}) -> dict:
        res_item = super().__call__(params, res_item)
        print(f"MyOperator: {self.threshold.get()}")
        return res_item
    
    def used_keys(self) -> list:
        return []
    
    def inject_keys(self) -> list:
        return []


if __name__ == "__main__":
    tasks = {"my_op": "MyOperator"}
    taskManage = TaskManager(tasks=tasks, operators=OPERATORS_REGISTRY)
    taskManage.save_json_cfg("task_op.json")
    taskManage.load_json_cfg("task_op.json")
    for task_obj in taskManage.generate_graph():
        task_obj({})
```
