# 命令行工具使用指南

## voc2yoloClasses：从voc标注生成YOLO的classes.txt文件

```shell
elfin voc2yoloClasses test/test/elfin/yoloLabelTest/voc --dst_file test/test/elfin/yoloLabelTest/classes.txt
```

参数说明：

1. src_dir: voc标注文件所在文件夹
2. dst_file: classes.txt文件存放路径, 默认放到src_dir文件夹下

## voc2yolo：从voc标注生成yolo的标注文件

```shell
elfin voc2yolo\
    test/test/elfin/yoloLabelTest/voc\
    test/test/elfin/yoloLabelTest/voc_yolo\
    test/test/elfin/yoloLabelTest/classes.txt
```

参数说明：

1. src_dir：voc标注文件所在文件夹
2. dst_dir：yolo格式的生成标注文件存放路径
3. classes：classes.txt文件路径

## labelme2yolo: 从labelme标注文件转yolo标注

```shell
elfin labelme2yolo labelme_jsons labelme_jsons labelme_jsons/classes.txt
```

参数说明：

1. src_dir：labelme标注文件所在文件夹
2. dst_dir：yolo格式的生成标注文件存放路径
3. classes：classes.txt文件路径

## yoloLabelExclude: 排出部分yolo标注并连续化剩余标签

```shell
elfin yoloLabelExclude 1 2 test/test/test.yaml --dst_dir test2 --cp_img
```

参数说明：

1. include_classes：要保留的数据类别编码, 整型
2. data_yaml：yolo的data yaml文件
3. --dst_dir：新数据集的地址, 默认是data_yaml指定的数据集根目录
4. --cp_img：是否复制图片

此命令会同步生成新的yaml文件。注意 `cp_img`为false时，标签仍然存放在原始标签同级目录下，名称会增加 `include_class`后缀，但是配置文件会在dst_dir下存放。
