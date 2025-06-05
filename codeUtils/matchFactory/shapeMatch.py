# #!/usr/bin/env python3
# # encoding: utf-8
# # @author: firstelfin
# # @time: 2024/08/19 11:50:42

# import os
# import sys
# import json
# import warnings
# from pathlib import Path
# from copy import deepcopy
# warnings.filterwarnings('ignore')
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # project root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# import numpy as np
# import cv2 as cv
# import shape_based_matching_py


# class ShapeBasedMatch(object):
#     """结构相似度查询, 这里对执行耗时没有限制

#     :param object: _description_
#     :type object: _type_
#     :return: _description_
#     :rtype: _type_
#     """

#     class_name = "ShapeBasedMatch"

#     def __init__(self) -> None:
#         pass

#     @staticmethod
#     def img_padding(img: np.ndarray, mask: np.ndarray|None, padding: int=100) -> np.ndarray:
#         """同步padding图片及其对应的mask, 一般用于旋转放缩预处理

#         :param np.ndarray img: 待处理的图片
#         :param np.ndarray|None mask: 图片关注区域mask
#         :param int padding: 每个边padding的尺寸, defaults to 100
#         :return np.ndarray: padding后的图像
#         """
#         imgh, imgw, imgc = img.shape
#         padded_templ = np.zeros(shape=(imgh+2*padding, imgw+2*padding, imgc), dtype=np.uint8)
#         padded_templ[padding:padding+imgh, padding:padding+imgw, :] = img[...]
#         if mask is None: padded_mask = None
#         else:
#             padded_mask  = np.zeros(shape=(imgh+2*padding, imgw+2*padding), dtype=np.uint8)
#             padded_mask[padding:padding+imgh, padding:padding+imgw] = mask[...]
#         return padded_templ, padded_mask

#     @staticmethod
#     def mask_gen(img: np.ndarray, contours: list=[]) -> np.ndarray:
#         """模板的mask生成函数

#         :param np.ndarray img: 源图片
#         :param list contours: 多边形候选区域列表, defaults to []
#         :return np.ndarray: roi的区域
#         """
#         if not contours:
#             mask = np.ones(shape=img.shape[:2], dtype=np.uint8)
#             mask *= 255
#         else:
#             mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)
#             cv.drawContours(mask, contours, -1, (255, 255, 255), cv.FILLED)
#         return mask

#     def inject_template(
#             self, templ: np.ndarray, contour: np.ndarray, 
#             num_features: int = 128, padding: list = [100, 250], 
#             pyramid_levels: list = [4, 8], angle_range: list = [-40, 40], 
#             use_rot: bool = True, scale_range: list = [1]
#         ):
#         detector = shape_based_matching_py.Detector(num_features, pyramid_levels)
#         templ_mask = self.mask_gen(templ, contours=[contour])
#         padded_templ, padded_mask = self.img_padding(templ, templ_mask, padding[0])

#         shapes = shape_based_matching_py.shapeInfo_producer(padded_templ, padded_mask)
#         shapes.angle_range = angle_range
#         shapes.angle_step = 1
#         # shapes.scale_range = [0.8, 1.2]
#         shapes.scale_range = scale_range
#         shapes.produce_infos()
        
#         class_id = "test"
#         is_first = True
#         first_id = 0
#         first_angle = 0

#         for info in shapes.infos:
#             templ_id = 0
#             if is_first:
#                 templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info))
#                 first_id = templ_id
#                 first_angle = info.angle
#                 if use_rot: is_first = False
#                 if templ_id == -1: return 1
#             else:
#                 templ_id = detector.addTemplate_rotate(
#                     class_id, 
#                     first_id,
#                     info.angle-first_angle,
#                     shape_based_matching_py.CV_Point2f(padded_templ.shape[1]/2.0, padded_templ.shape[0]/2.0)
#                 )
#         return detector

#     def match(self, detector, search_img, padding):
#         """匹配函数

#         :param detector: 识别器
#         :type detector: _type_
#         :param search_img: 搜索图
#         :type search_img: np.ndarray
#         :param padding: 用于旋转时的padding
#         :type padding: int
#         :return: 匹配结果和搜索图
#         :rtype: tuple
#         """
#         ids = []
#         ids.append('test')
#         padded_img, _ = self.img_padding(search_img, None, padding)

#         stride = 16
#         img_rows = int(padded_img.shape[0] / stride) * stride
#         img_cols = int(padded_img.shape[1] / stride) * stride
#         img = padded_img[0:img_rows, 0:img_cols, :]
#         matches = detector.match(img, 0, ids)
#         return matches, img

#     def shape_base_match(
#             self,
#             search_img: np.ndarray, 
#             templ: np.ndarray, 
#             contours: list[np.ndarray] = [],
#             num_features: int = 128, 
#             padding: list = [100, 250],
#             pyramid_levels: list = [4, 8], 
#             angle_range: list = [-40, 40],
#             use_rot: bool = True,
#             show=True
#         ) -> float:
#         """基于边缘梯度方向的模板匹配算法

#         :param np.ndarray search_img: 搜索图
#         :param np.ndarray templ: 模板图
#         :param list[np.ndarray] contours: 关注区域多边形列表, 注意多边形的坐标要锚定templ, defaults to []
#         :param int num_features: 需要配置的特征数量, defaults to 128
#         :param list padding: 分别表示模板和搜索图的padding大小, defaults to [100, 250]
#         :param list pyramid_levels: line memory的T, 即模板分块的边长, defaults to [4, 8]
#         :param list angle_range: 模板匹配旋转角度, defaults to [-40, 40]
#         :param bool use_rot: _description_, defaults to True
#         :param bool show: 是否展示模板关键点匹配图, defaults to True
#         :return float: 最佳匹配得分
#         """
#         import  time
#         start_time = time.perf_counter()
#         ssim = 0

#         detector = shape_based_matching_py.Detector(num_features, pyramid_levels)
#         templ_mask = self.mask_gen(templ, contours=contours)
#         padded_templ, padded_mask = self.img_padding(templ, templ_mask, padding[0])

#         shapes = shape_based_matching_py.shapeInfo_producer(padded_templ, padded_mask)
#         shapes.angle_range = angle_range
#         shapes.angle_step = 1
#         shapes.scale_range = [1]
#         shapes.produce_infos()

#         class_id = "test"
#         is_first = True
#         first_id = 0
#         first_angle = 0

#         for info in shapes.infos:
#             templ_id = 0
#             if is_first:
#                 templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info))
#                 first_id = templ_id
#                 first_angle = info.angle
#                 if use_rot: is_first = False
#                 if templ_id == -1: return 1
#             else:
#                 templ_id = detector.addTemplate_rotate(
#                     class_id, 
#                     first_id,
#                     info.angle-first_angle,
#                     shape_based_matching_py.CV_Point2f(padded_templ.shape[1]/2.0, padded_templ.shape[0]/2.0)
#                 )
        
#         middle_time = time.perf_counter()

#         # 开始搜索图的匹配计算
#         ids = []
#         ids.append('test')
#         padded_img, _ = self.img_padding(search_img, None, padding[1])

#         stride = 16
#         img_rows = int(padded_img.shape[0] / stride) * stride
#         img_cols = int(padded_img.shape[1] / stride) * stride
#         img = np.zeros((img_rows, img_cols, padded_img.shape[2]), np.uint8)
#         img[:, :, :] = padded_img[0:img_rows, 0:img_cols, :]
#         matches = detector.match(img, 0, ids)
#         top5 = 1
#         if top5 > len(matches): top5 = 1
#         if len(matches): ssim = matches[0].similarity
#         end_time = time.perf_counter()
#         print(f"execTime of Match: {end_time-middle_time}, execTime of addTempl: {middle_time - start_time}")
#         if show and len(matches):
#             for i in range(top5):
#                 match = matches[i]
#                 templ = detector.getTemplates("test", match.template_id)
                
#                 for feat in templ[0].features:
#                     img = cv.circle(img, (feat.x+match.x, feat.y+match.y), 3, (0, 0, 255), -1)

#                 # cv have no RotatedRect constructor?
#                 print('match.template_id: {}'.format(match.template_id))
#                 print('match.similarity: {}'.format(match.similarity))
#             cv.imshow("img", img)
#             # cv.imwrite("test-img.png", img)
#             cv.waitKey(0)

#         if len(matches) == 0:
#             print(matches)
#         return ssim
    
#     def shape_base_match2(
#             self,
#             search_img: np.ndarray, 
#             templ: np.ndarray, 
#             contours: list[np.ndarray] = [],
#             num_features: int = 128, 
#             padding: list = [100, 250],
#             pyramid_levels: list = [4, 8], 
#             angle_range: list = [-40, 40],
#             use_rot: bool = True,
#             show=True
#         ) -> float:
#         """基于边缘梯度方向的模板匹配算法

#         :param np.ndarray search_img: 搜索图
#         :param np.ndarray templ: 模板图
#         :param list[np.ndarray] contours: 关注区域多边形列表, 注意多边形的坐标要锚定templ, defaults to []
#         :param int num_features: 需要配置的特征数量, defaults to 128
#         :param list padding: 分别表示模板和搜索图的padding大小, defaults to [100, 250]
#         :param list pyramid_levels: line memory的T, 即模板分块的边长, defaults to [4, 8]
#         :param list angle_range: 模板匹配旋转角度, defaults to [-40, 40]
#         :param bool use_rot: _description_, defaults to True
#         :param bool show: 是否展示模板关键点匹配图, defaults to True
#         :return float: 最佳匹配得分
#         """
#         import  time
#         start_time = time.perf_counter()
#         detector = self.inject_template(
#             templ=templ, contour=contours[0], 
#             num_features=num_features, padding=padding, 
#             pyramid_levels=pyramid_levels, angle_range=[-20, 20], use_rot=False
#         )
        
#         middle_time = time.perf_counter()

#         # 开始搜索图的匹配计算
#         matches, img = self.match(detector=detector, search_img=search_img, padding=padding[1])
#         end_time = time.perf_counter()

#         top5 = 3
#         if top5 > len(matches): top5 = 1
#         if len(matches): ssim = matches[0].similarity
#         print(f"execTime of Match: {end_time-middle_time}, execTime of addTempl: {middle_time - start_time}")
#         if show and len(matches):
#             for i in range(top5):
#                 match = matches[i]
#                 templ = detector.getTemplates("test", match.template_id)
                
#                 for feat in templ[0].features:
#                     img = cv.circle(img, (feat.x+match.x, feat.y+match.y), 3, (0, 0, 255), -1)

#                 # cv have no RotatedRect constructor?
#                 print('match.template_id: {}'.format(match.template_id))
#                 print('match.similarity: {}'.format(match.similarity))
#             cv.imshow("img", img)
#             # cv.imwrite("test-img.png", img)
#             cv.waitKey(0)

#         if len(matches) == 0:
#             print(matches)
#         return ssim
    

# class ComponentStructuralSimilarity(ShapeBasedMatch):

#     class_name = "ComponentStructuralSimilarity"

#     def __init__(self, templ_dir: str) -> None:
#         super().__init__()
#         self.templ_dir = templ_dir
#         self.matcher = {}

#     def add_template(self):
#         """向对象添加所有的模板"""
#         for category in Path(self.templ_dir).iterdir():
#             self.matcher[category.stem] = []
#             for file in category.iterdir():
#                 if file.suffix == ".json" or file.stem.startswith("."):
#                     continue
#                 img = cv.imread(str(file))
#                 with open(file.with_suffix(".json"), "r+", encoding="utf-8") as f:
#                     contour_data = json.load(f)
#                 src_contour = np.array(contour_data["shapes"][0]["points"], dtype=np.int32).reshape(-1, 1, 2)
#                 bbox = cv.boundingRect(src_contour)
#                 x, y, w, h = bbox
#                 templ_img = deepcopy(img[y:y+h, x:x+w, :])    # template
#                 src_contour = src_contour - [x, y]       # template mask
#                 detector = self.inject_template(templ_img, src_contour, num_features=128, angle_range=[-10, 10], scale_range=[0.8, 1.2])
#                 self.matcher[category.stem].append(detector)
    
#     @staticmethod
#     def cosine_similarity(vec1, vec2):
#         # 转换为 NumPy 数组
#         vec1 = np.array(vec1)
#         vec2 = np.array(vec2)
        
#         # 计算点积
#         dot_product = np.dot(vec1, vec2)
        
#         # 计算模长
#         norm_vec1 = np.linalg.norm(vec1)
#         norm_vec2 = np.linalg.norm(vec2)
        
#         # 计算余弦相似度
#         similarity = dot_product / (norm_vec1 * norm_vec2)
        
#         return similarity
    
#     def similarity_generate(self, dete_img: np.ndarray, base_img: np.ndarray, category: str = "damper") -> list:
#         """相似度计算

#         :param dete_img: 巡视图
#         :type dete_img: np.ndarray
#         :param base_img: 底图
#         :type base_img: np.ndarray
#         :return: 所有的相似度度量指标
#         :rtype: list
#         """
#         dete_simi = [self.match(matcher, dete_img, 100)[0][0] for matcher in self.matcher[category]]
#         base_simi = [self.match(matcher, base_img, 100)[0][0] for matcher in self.matcher[category]]
#         dete_similarity = [simi.similarity for simi in dete_simi]
#         base_similarity = [simi.similarity for simi in base_simi]
#         # X, Y的位置信息
#         # dete_x = [simi.x for simi in dete_simi]
#         # base_x = [simi.x for simi in base_simi]
#         diff_simi = [abs(dete_similarity[i] - base_similarity[i]) for i in range(len(dete_simi))]
#         diff_simi.sort()
#         # TOP2 = diff_simi.sort()[:2]

#         # vec_simi = self.cosine_similarity(np.array(dete_similarity), np.array(base_similarity))
#         vec_simi = 100 - sum(diff_simi[:2]) / 2
#         return [vec_simi]
    
#     @staticmethod
#     def similarity_verification(simi_list: list, thresh: list = [80]) -> bool:
#         vec_simi = simi_list[0]
#         if vec_simi > thresh[0]:
#             return True
#         else:
#             return False
    
#     @staticmethod
#     def base_dete_simi(dete_img: np.ndarray, base_img: np.ndarray) -> float:
#         """底图和检测图的相似度计算

#         :param dete_img: 巡视图
#         :type dete_img: np.ndarray
#         :param base_img: 底图
#         :type base_img: np.ndarray
#         :return: 底图和检测图的相似度
#         :rtype: float
#         """
#         dete_simi = cv.matchTemplate(dete_img, base_img, cv.TM_CCOEFF_NORMED)
#         _, max_val, _, max_loc = cv.minMaxLoc(dete_simi)
#         return max_val
    
#     def inference(self, dete_img: np.ndarray, base_img: np.ndarray, category: str = "damper") -> bool:

#         simi_list = self.similarity_generate(dete_img, base_img, category=category)
#         base_detd_simi = self.base_dete_simi(dete_img, base_img)
#         print(simi_list, base_detd_simi)
#         if self.similarity_verification(simi_list, thresh=[80, base_detd_simi / 1.0]):
#             return True
        
#         else:
#             return False


# def read_img_and_json(img_path: str, json_path: str, target_labels: list = []):
#     tuple_img = cv.imread(img_path)
#     dete_img = tuple_img[:, tuple_img.shape[1]//2:, :]
#     base_img = tuple_img[:, :tuple_img.shape[1]//2, :]
#     with open(json_path, "r+", encoding="utf-8") as f:
#         json_data = json.load(f)
#     target_obj = [obj for obj in json_data["shapes"] if obj["label"] in target_labels]
#     imgh, imgw = dete_img.shape[:2]

#     for obj in target_obj:
#         [x1, y1], [x2, y2] = obj["points"]
#         delta = 15
#         x1 = int(max(0, x1 - delta))
#         y1 = int(max(0, y1 - delta))
#         x2 = int(min(imgw, x2 + delta))
#         y2 = int(min(imgh, y2 + delta))
#         dete_instance_img = dete_img[y1:y2, x1:x2, :]
#         base_instance_img = base_img[y1:y2, x1:x2, :]
#         yield dete_instance_img, base_instance_img, obj["group_id"]

