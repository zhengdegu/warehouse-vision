"""Metric computation utilities for the accuracy evaluation framework."""

from typing import Dict, List, Tuple


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个 bbox 的 IoU（像素坐标 [x1, y1, x2, y2]）。

    Args:
        box1: [x1, y1, x2, y2] 像素坐标
        box2: [x1, y1, x2, y2] 像素坐标

    Returns:
        IoU 值，范围 [0, 1]。无交集时返回 0.0。
    """
    # Intersection coordinates
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    # Intersection area (0 if no overlap)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    if inter == 0.0:
        return 0.0

    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0.0:
        return 0.0

    return inter / union


def compute_ap(
    precisions: List[float], recalls: List[float], method: str = "interp11"
) -> float:
    """计算 Average Precision。

    Args:
        precisions: 精确率列表（按置信度降序排列的累积值）
        recalls: 召回率列表（按置信度降序排列的累积值）
        method: "interp11" 为 11 点插值法，"all_points" 为全点插值法

    Returns:
        AP 值，范围 [0, 1]。
    """
    if not precisions or not recalls:
        return 0.0

    if method == "interp11":
        # 11-point interpolation at recall thresholds [0, 0.1, ..., 1.0]
        ap = 0.0
        for t in [i / 10.0 for i in range(11)]:
            # Max precision at recall >= t
            p_interp = 0.0
            for p, r in zip(precisions, recalls):
                if r >= t:
                    p_interp = max(p_interp, p)
            ap += p_interp
        ap /= 11.0
        return ap

    elif method == "all_points":
        # All-points interpolation (area under the P-R curve)
        # Prepend (recall=0, precision=1) and append (recall=1, precision=0)
        # to ensure the curve starts and ends correctly.
        mrec = [0.0] + list(recalls) + [1.0]
        mpre = [0.0] + list(precisions) + [0.0]

        # Make precision monotonically decreasing (right to left)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # Compute area under the curve
        ap = 0.0
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                ap += (mrec[i] - mrec[i - 1]) * mpre[i]
        return ap

    else:
        raise ValueError(f"Unknown AP method: {method}. Use 'interp11' or 'all_points'.")


def compute_map(ap_per_class: Dict[int, float]) -> float:
    """计算 mAP（排除无样本类别，即 AP 为 NaN 或 None 的类别）。

    Args:
        ap_per_class: {class_id: ap_value}。值为 None 表示无样本。

    Returns:
        mAP 值。无有效类别时返回 0.0。
    """
    valid_aps = [ap for ap in ap_per_class.values() if ap is not None]
    if not valid_aps:
        return 0.0
    return sum(valid_aps) / len(valid_aps)


def compute_confusion_matrix(
    matches: List[dict], class_names: List[str]
) -> list:
    """生成混淆矩阵，含背景类。

    行 = 真值类别，列 = 预测类别。最后一行/列为背景（FP/FN）。

    Args:
        matches: 匹配结果列表，每个 dict 包含:
            - gt_class (int or None): 真值类别索引，None 表示 FP（无对应 GT）
            - pred_class (int or None): 预测类别索引，None 表示 FN（漏检）
        class_names: 类别名称列表

    Returns:
        2D list，大小 (num_classes+1) x (num_classes+1)。
    """
    n = len(class_names)
    size = n + 1  # +1 for background
    matrix = [[0] * size for _ in range(size)]

    bg = n  # background index = last

    for m in matches:
        gt_cls = m.get("gt_class")
        pred_cls = m.get("pred_class")

        row = gt_cls if gt_cls is not None else bg
        col = pred_cls if pred_cls is not None else bg

        matrix[row][col] += 1

    return matrix


def compute_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """计算 Precision, Recall, F1-Score。

    Args:
        tp: True Positives
        fp: False Positives
        fn: False Negatives

    Returns:
        (precision, recall, f1)。当 tp=0 时返回 (0.0, 0.0, 0.0)。
    """
    if tp == 0:
        return (0.0, 0.0, 0.0)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    denom = precision + recall
    if denom == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / denom

    return (precision, recall, f1)


def compute_counting_error(expected: int, actual: int) -> Tuple[int, float]:
    """计算计数误差。

    Args:
        expected: 期望计数（非负整数）
        actual: 实际计数（非负整数）

    Returns:
        (abs_error, rel_error)。
        abs_error = |actual - expected|
        rel_error = abs_error / expected (expected > 0)
                  = 0.0 (expected == 0 and actual == 0)
                  = float('inf') (expected == 0 and actual > 0)
    """
    abs_error = abs(actual - expected)

    if expected > 0:
        rel_error = abs_error / expected
    elif actual == 0:
        rel_error = 0.0
    else:
        rel_error = float("inf")

    return (abs_error, rel_error)
