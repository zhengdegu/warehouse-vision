"""
几何与规则基础模块
提供多边形判定、线段交叉、越线判定等基础几何运算。
"""

from typing import List, Tuple

Point = Tuple[float, float]
Polygon = List[Point]


def bbox_center(bbox: list) -> Point:
    """计算 bbox 中心点"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def foot_point(bbox: list) -> Point:
    """计算 bbox 脚点（底边中点）"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    射线法判断点是否在多边形内
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def _cross(o: Point, a: Point, b: Point) -> float:
    """向量叉积 OA × OB"""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    """判断点 q 是否在线段 pr 上"""
    if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
        return True
    return False


def segment_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """
    判断线段 p1p2 与 p3p4 是否相交
    """
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if d1 == 0 and _on_segment(p3, p1, p4):
        return True
    if d2 == 0 and _on_segment(p3, p2, p4):
        return True
    if d3 == 0 and _on_segment(p1, p3, p2):
        return True
    if d4 == 0 and _on_segment(p1, p4, p2):
        return True

    return False


def line_crossing(prev_pos: Point, curr_pos: Point,
                  line_p1: Point, line_p2: Point) -> str:
    """
    判断从 prev_pos 到 curr_pos 的运动是否穿越了 line_p1-line_p2 线段。
    返回:
        "none"  - 未穿越
        "positive" - 正方向穿越（从线的左侧到右侧）
        "negative" - 反方向穿越（从线的右侧到左侧）
    """
    if not segment_intersect(prev_pos, curr_pos, line_p1, line_p2):
        return "none"

    # 用叉积判断方向
    cross_val = _cross(line_p1, line_p2, curr_pos)
    if cross_val > 0:
        return "positive"
    elif cross_val < 0:
        return "negative"
    return "none"
