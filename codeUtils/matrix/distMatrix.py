#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   distMatrix.py
@Time    :   2025/05/13 15:41:01
@Author  :   firstElfin 
@Version :   0.1.11.15
@Desc    :   分布式matrix的python实现
'''


import json
import warnings
import numpy as np
from pathlib import Path

from codeUtils.tools import checks


class SimpleClass:
    """
    A simple base class for creating objects with string representations of their attributes.

    This class provides a foundation for creating objects that can be easily printed or represented as strings,
    showing all their non-callable attributes. It's useful for debugging and introspection of object states.

    Methods:
        __str__: Returns a human-readable string representation of the object.
        __repr__: Returns a machine-readable string representation of the object.
        __getattr__: Provides a custom attribute access error message with helpful information.

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
        - Copy from https://github.com/ultralytics/ultralytics/
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names={}, on_plot=None):
    """
    Plot precision-recall curve.

    Args:
        px (np.ndarray): X values for the PR curve.
        py (np.ndarray): Y values for the PR curve.
        ap (np.ndarray): Average precision values.
        save_dir (Path, optional): Path to save the plot.
        names (dict, optional): Dictionary mapping class indices to class names.
        on_plot (callable, optional): Function to call after plot is saved.
    
    Note: Copy from https://github.com/ultralytics/ultralytics/
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names={}, xlabel="Confidence", ylabel="Metric", on_plot=None):
    """
    Plot metric-confidence curve.

    Args:
        px (np.ndarray): X values for the metric-confidence curve.
        py (np.ndarray): Y values for the metric-confidence curve.
        save_dir (Path, optional): Path to save the plot.
        names (dict, optional): Dictionary mapping class indices to class names.
        xlabel (str, optional): X-axis label.
        ylabel (str, optional): Y-axis label.
        on_plot (callable, optional): Function to call after plot is saved.
    
    Note: Copy from https://github.com/ultralytics/ultralytics/
    """
    import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.1)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    
    Note: Copy from https://github.com/ultralytics/ultralytics/
    """
    # Append sentinel values to beginning and end
    m_rec = np.concatenate(([0.0], recall, [1.0]))
    m_pre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        func = np.trapezoid if checks.check_version(np.__version__, ">=2.0") else np.trapz  # np.trapz deprecated
        ap = func(np.interp(x, m_rec, m_pre), x)  # integrate
    else:  # 'continuous'
        i = np.where(m_rec[1:] != m_rec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((m_rec[i + 1] - m_rec[i]) * m_pre[i + 1])  # area under curve

    return ap, m_pre, m_rec


def ap_per_class(
    tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names={}, eps=1e-16, prefix=""
):
    """
    Compute the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): shape=[pred_num, iou_num], Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered.
        save_dir (Path, optional): Directory to save the PR curves.
        names (dict, optional): Dict of class names to plot PR curves.
        eps (float, optional): A small value to avoid division by zero.
        prefix (str, optional): A prefix string for saving the plot files.

    Returns:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class.
        p (np.ndarray): Precision values at threshold given by max F1 metric for each class.
        r (np.ndarray): Recall values at threshold given by max F1 metric for each class.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class.
        r_curve (np.ndarray): Recall curves for each class.
        f1_curve (np.ndarray): F1-score curves for each class.
        x (np.ndarray): X-axis values for the curves.
        p_rec_values (np.ndarray): Precision values at mAP@0.5 for each class.
    
    Note: Copy from https://github.com/ultralytics/ultralytics/
    """
    # Sort by objectness
    i = np.argsort(-conf)  # order是指定按照那个字段进行排序, 这里使用-conf表示获取conf降序排列索引
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)  # 统计GT标签有哪些类及其对应数量
    nc = unique_classes.shape[0]  # number of classes

    # Create Precision-Recall curve and compute AP for each class
    x, p_rec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c  # 预测是c的索引
        n_l = nt[ci]       # 类别为c的GT数量
        n_p = i.sum()      # 正确预测为c的数量
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs, cumsum是累积求和, 常用于分布计算
        fpc = (1 - tp[i]).cumsum(0)  # 误报数量数组, shape=[pred_of_c_num, iou_num], i是预测为c的索引
        tpc = tp[i].cumsum(0)        # 正确预测数量数组, shape=[pred_of_c_num, iou_num]

        # Recall
        recall = tpc / (n_l + eps)   # 召回数组, shape=[pred_of_c_num, iou_num], 记录了不同匹配iou, 不同置信度下的召回率
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):  # for each iou threshold
            ap[ci, j], m_pre, m_rec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:  # j = 0 时, IOU=0.5
                p_rec_values.append(np.interp(x, m_rec, m_pre))  # precision at mAP@0.5

    p_rec_values = np.array(p_rec_values) if p_rec_values else np.zeros((1, 1000))  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)  # shape=(nc, 1000)
    names = [v for k, v in names.items() if k in unique_classes or v in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    cls_is_int = True if isinstance(unique_classes[0], int) else False
    if plot:
        plot_pr_curve(x, p_rec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index, 所有类的mF1中最大值的索引
    argmax_f1 = {cls_name: x[smooth(f1_curve[cls_i, :], 0.1).argmax()] for cls_i, cls_name in names.items()}
    with open(save_dir / f"{prefix}argmax_f1.json", "w+") as f:
        json.dump(argmax_f1, f, indent=4, ensure_ascii=False)
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * nt).round()  # true positives的数量
    fp = (tp / (p + eps) - tp).round()  # false positives
    if cls_is_int:
        return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, p_rec_values
    else:
        return tp, fp, p, r, f1, ap, unique_classes, p_curve, r_curve, f1_curve, x, p_rec_values


class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for Ultralytics YOLO models.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    
    Note: Copy from https://github.com/ultralytics/ultralytics/
    """

    def __init__(self) -> None:
        """Initialize a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """
        Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Return the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Return the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Return the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Return the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Return mean of results, mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """Return class-aware result, p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """Return mAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Return model fitness as a weighted combination of metrics."""
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.nan_to_num(np.array(self.mean_results())) * w).sum()

    def update(self, results):
        """
        Update the evaluation metrics with a new set of results.

        Args:
            results (tuple): A tuple containing evaluation metrics:
                - p (list): Precision for each class.
                - r (list): Recall for each class.
                - f1 (list): F1 score for each class.
                - all_ap (list): AP scores for all classes and all IoU thresholds.
                - ap_class_index (list): Index of class for each AP score.
                - p_curve (list): Precision curve for each class.
                - r_curve (list): Recall curve for each class.
                - f1_curve (list): F1 curve for each class.
                - px (list): X values for the curves.
                - prec_values (list): Precision values for each class.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self):
        """Return a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Return a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class DetMetrics(SimpleClass):
    """
    Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class.
        names (dict): A dictionary of class names.
        box (Metric): An instance of the Metric class for storing detection results.
        speed (dict): A dictionary for storing execution times of different parts of the detection process.
        task (str): The task type, set to 'detect'.
    
    Note: Copy from https://github.com/ultralytics/ultralytics/
    """

    def __init__(self, save_dir=Path("."), plot=False, names={}, **kwargs) -> None:
        """
        Initialize a DetMetrics instance with a save directory, plot flag, and class names.

        Args:
            save_dir (Path, optional): Directory to save plots.
            plot (bool, optional): Whether to plot precision-recall curves.
            names (dict, optional): Dictionary mapping class indices to names.
        """
        super().__init__()
        self.save_dir = save_dir
        self.plot = plot
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"

    def process(self, tp, conf, pred_cls, target_cls, on_plot=None):
        """
        Process predicted results for object detection and update metrics.

        Args:
            tp (np.ndarray): True positive array.
            conf (np.ndarray): Confidence array.
            pred_cls (np.ndarray): Predicted class indices array.
            target_cls (np.ndarray): Target class indices array.
            on_plot (callable, optional): Function to call after plots are generated.
        """
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=on_plot,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """Return a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Return mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Return the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Return the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Return dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Return a list of curves for accessing specific metrics curves."""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self):
        """Return dictionary of computed performance metrics and statistics."""
        return self.box.curves_results



