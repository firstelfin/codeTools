# codeTools

代码工具箱，记录关键代码防止重复开发, Github地址: [codeUtils](https://github.com/firstelfin/codeTools)

## Install

```shell
pip install --index-url https://pypi.org/simple elfinCodeUtils
```

## update code

```shell
python3 -m pip install --upgrade build
python -m build -sw -nx
twine upload dist/*
```

> build 库是一个编译库，参考PYPI官网：https://pypi.org/project/build/

## taskManage

- [X] 完成taskManage模块的编写: 参考[文档](./docs/taskManage.md)

## Shell Cli

- [X] 完成labelme2yolo的开发：使用 `elfin labelme2yolo -h`查看帮助;
- [ ] 完成labelme2coco的开发：使用 `elfin labelme2coco -h`查看帮助;
- [ ] 完成labelme2voc的开发：使用 `elfin labelme2voc -h`查看帮助;
- [ ] 完成yolo2labelme的开发：使用 `elfin yolo2labelme -h`查看帮助;
- [ ] 完成yolo2coco的开发：使用 `elfin yolo2coco -h`查看帮助;
- [ ] 完成yolo2voc的开发：使用 `elfin yolo2voc -h`查看帮助;
- [ ] 完成coco2yolo的开发：使用 `elfin coco2yolo -h`查看帮助;
- [ ] 完成coco2labelme的开发：使用 `elfin coco2labelme -h`查看帮助;
- [ ] 完成coco2voc的开发：使用 `elfin coco2voc -h`查看帮助;
- [ ] 完成voc2yolo的开发：使用 `elfin voc2yolo -h`查看帮助;
- [ ] 完成voc2labelme的开发：使用 `elfin voc2labelme -h`查看帮助;
- [ ] 完成voc2coco的开发：使用 `elfin voc2coco -h`查看帮助;
- [X] 完成yoloLabelExclude的开发：使用 `elfin yoloLabelExclude -h`查看帮助;
