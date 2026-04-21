"""
工具函数模块
包含深度图预处理、旋转矩阵构造等辅助功能
"""
import cv2
import numpy as np
from typing import Optional


def preprocess_depth(depth_map):
    """预处理深度图，填充无效值"""
    # 复制深度图
    depth_processed = depth_map.copy()

    # 统计无效点
    invalid_mask = (depth_processed == 0)
    n_invalid = np.sum(invalid_mask)

    if n_invalid > 0:
        print(f"⚠️  深度图中有 {n_invalid} 个无效点 (深度=0)")

        # 方法 1：使用最近邻有效深度填充
        valid_mask = (depth_processed > 0)

        if np.any(valid_mask):
            # 获取有效点的坐标
            valid_coords = np.argwhere(valid_mask)
            invalid_coords = np.argwhere(invalid_mask)

            # 使用 KDTree 找到最近的有效点
            from scipy.spatial import cKDTree
            tree = cKDTree(valid_coords)

            # 批量查找最近邻
            distances, indices = tree.query(invalid_coords, k=1)

            # 用最近有效点的深度填充
            for i, (y, x) in enumerate(invalid_coords):
                nearest_y, nearest_x = valid_coords[indices[i]]
                depth_processed[y, x] = depth_processed[nearest_y, nearest_x]

            print(f"✅ 使用最近邻填充了 {len(invalid_coords)} 个无效点")

        # 方法 2：可选的中值滤波去除噪声
        depth_processed = cv2.medianBlur(depth_processed.astype(np.float32), 3)

    return depth_processed

def apply_mask_shrink(shrink_factor,mask):
        """应用掩码收缩"""
        if shrink_factor <= 0:
            return mask.copy()

        # 确保掩码是二值化的
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # 创建腐蚀核
        kernel_size = shrink_factor * 2 + 1  # 确保核大小为奇数
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # 应用形态学腐蚀操作
        mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)

        # 转换为浮点格式（0-1范围）
        mask_shrunk = (mask_eroded > 0).astype(np.float32)

        # 计算收缩前后的像素数量
        original_pixels = np.sum(mask_binary > 0)
        shrunk_pixels = np.sum(mask_eroded > 0)
        shrinkage_percent = (1 - shrunk_pixels / original_pixels) * 100 if original_pixels > 0 else 0

        print(f"🔧 掩码收缩: {original_pixels} → {shrunk_pixels} 像素 (减少 {shrinkage_percent:.1f}%)")

        return mask_shrunk


def rotation_matrix_from_z_to_v(z_target):
    """
    构造旋转矩阵 R，使得 R @ [0,0,1] = z_target（单位向量）
    """
    z_target = z_target / np.linalg.norm(z_target)
    z0 = np.array([0, 0, 1.0])

    if np.allclose(z_target, z0):
        return np.eye(3)
    if np.allclose(z_target, -z0):
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]])  # 绕 x 轴旋转 180°

    v = np.cross(z0, z_target)
    s = np.linalg.norm(v)
    c = np.dot(z0, z_target)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R


def interpolate_scan_points(scan_points, points_per_segment=5):
    """
    路径点插值加密
    
    返回:
      interpolated_points: [(x, y), ...]
      segment_info: [{'orig_idx': i, 'is_original': bool, 't': float}, ...]
        - orig_idx: 来自第 i 段（i: 0~N-2）
        - is_original: 是否为原始路径点（首/尾/中间原始点）
        - t: 归一化参数 [0,1]，用于判断位置
    """
    if len(scan_points) < 2:
        points = [tuple(p) for p in scan_points]
        info = [{'orig_idx': -1, 'is_original': True, 't': 0.0 if i == 0 else 1.0} for i, p in enumerate(scan_points)]
        return points, info

    interpolated = []
    segment_info = []

    for i in range(len(scan_points) - 1):
        p0 = np.array(scan_points[i], dtype=float)
        p1 = np.array(scan_points[i + 1], dtype=float)

        # 添加起点（仅第一次）
        if i == 0:
            interpolated.append(tuple(scan_points[i]))
            segment_info.append({'orig_idx': i, 'is_original': True, 't': 0.0})

        # 确定本段插值数
        n_insert = points_per_segment if i % 2 == 0 else 30  # 奇数段插 30

        # 插中间点
        ts = np.linspace(0, 1, n_insert + 2)[1:-1]  # 排除 0 和 1
        for t in ts:
            pt = (1 - t) * p0 + t * p1
            interpolated.append((int(round(pt[0])), int(round(pt[1]))))
            segment_info.append({'orig_idx': i, 'is_original': False, 't': float(t)})

        # 添加终点（仅最后一段）
        if i == len(scan_points) - 2:
            interpolated.append(tuple(scan_points[-1]))
            segment_info.append({'orig_idx': i, 'is_original': True, 't': 1.0})

    return interpolated, segment_info
