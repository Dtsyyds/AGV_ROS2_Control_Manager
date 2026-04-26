"""
路径生成模块
包含多种路径生成算法：自适应轮廓、旋转矩形网格、平滑轮廓跟随
"""
import cv2
import numpy as np
from typing import List, Tuple


class PathGenerator:
    """路径生成器类"""
    
    def __init__(self, scan_mode='Short', spacing=10):
        """
        初始化路径生成器
        
        Args:
            scan_mode: 扫描模式 ('Long'或'Short')
            spacing: 切割线间距（像素）
        """
        self.scan_mode = scan_mode
        self.spacing = spacing
    
    def generate_from_mask(self, mask):
        """
        从掩码生成路径（主入口函数）
        
        Args:
            mask: 二值掩码
            
        Returns:
            contour: 轮廓
            scan_points: 路径点列表
        """
        # 调整掩码大小以匹配原图
        img_height, img_width = mask.shape[:2]
        mask_binary = (mask > 0.5).astype(np.uint8)

        # 获取掩膜轮廓
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("⚠️ 未找到轮廓")
            return None, []

        # 取最大轮廓
        contour = max(contours, key=cv2.contourArea)

        # 生成自适应轮廓路径
        scan_points = self.generate_contour_adaptive_path(contour, mask_binary)
        
        return contour, scan_points
    
    def generate_contour_adaptive_path(self, contour, mask_binary):
        """
        生成自适应轮廓的栅格路径

        原理：
        1. 获取掩膜的最小外接矩形，确定主要方向
        2. 沿长轴方向以一定间距生成切割线
        3. 计算切割线与掩膜轮廓的交点
        4. 连接交点形成之字形路径
        """
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        center, (w, h), angle_deg = rect

        # 确定长轴（扫描方向）和短轴（步进方向）
        if self.scan_mode == "Long":
            if w >= h:
                scan_len, step_len = w, h
                scan_angle = angle_deg
            else:
                scan_len, step_len = h, w
                scan_angle = angle_deg + 90
        elif self.scan_mode == "Short":
            if w <= h:
                scan_len, step_len = w, h
                scan_angle = angle_deg
            else:
                scan_len, step_len = h, w
                scan_angle = angle_deg + 90

        # 将轮廓点转换为 numpy 数组
        contour_points = contour.reshape(-1, 2).astype(np.float32)

        # 创建旋转矩阵（将坐标旋转到矩形坐标系）
        angle_rad = np.deg2rad(scan_angle)
        R = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)]
        ])
        R_inv = np.linalg.inv(R)  # 逆矩阵

        # 将轮廓点转换到矩形坐标系
        contour_centered = contour_points - center
        contour_rotated = (R @ contour_centered.T).T

        # 在矩形坐标系中生成切割线
        v_min = contour_rotated[:, 1].min()
        v_max = contour_rotated[:, 1].max()

        # 生成切割线位置
        v_vals = np.arange(v_min, v_max + self.spacing, self.spacing)
        scan_points = []

        for i, v in enumerate(v_vals):
            # 获取当前切割线与轮廓的交点
            intersections = self.find_contour_intersections(contour_rotated, v)

            if len(intersections) < 2:
                continue

            # 按扫描方向（x 坐标）排序
            intersections = sorted(intersections, key=lambda p: p[0])

            # 之字形路径：奇数行反向
            if i % 2 == 1:
                intersections = intersections[::-1]

            # 将交点转换回原图像坐标系
            for point_rot in intersections:
                # 变换回原坐标系
                point_rot_np = np.array(point_rot, dtype=np.float32).reshape(1, 2)
                point_original = (R_inv @ point_rot_np.T).T + center
                x, y = int(round(point_original[0, 0])), int(round(point_original[0, 1]))

                # 确保点在掩膜内
                if (0 <= y < mask_binary.shape[0] and
                        0 <= x < mask_binary.shape[1] and
                        mask_binary[y, x] > 0):
                    scan_points.append((x, y))

        return scan_points
    
    def find_contour_intersections(self, contour_points, v, epsilon=0.5):
        """
        找到轮廓与水平线 y=v 的交点

        Args:
            contour_points: 轮廓点（在旋转后的坐标系中）
            v: 水平线 y 坐标
            epsilon: 容差值

        Returns:
            交点列表 [(x1, y1), (x2, y2), ...]
        """
        intersections = []
        n = len(contour_points)

        for i in range(n):
            p1 = contour_points[i]
            p2 = contour_points[(i + 1) % n]

            # 检查线段是否与水平线相交
            if (p1[1] - v) * (p2[1] - v) <= 0:
                # 确保不是垂直线（避免除以 0）
                if abs(p2[1] - p1[1]) > 1e-6:
                    # 线性插值求交点
                    t = (v - p1[1]) / (p2[1] - p1[1])
                    x_interp = p1[0] + t * (p2[0] - p1[0])
                    intersections.append((x_interp, v))
                elif abs(p1[1] - v) < epsilon:
                    # 线段几乎水平，取中点
                    intersections.append(((p1[0] + p2[0]) / 2, v))

        # 去重（由于轮廓闭合，可能产生重复点）
        unique_intersections = []
        for pt in intersections:
            is_duplicate = False
            for existing in unique_intersections:
                if abs(pt[0] - existing[0]) < epsilon and abs(pt[1] - existing[1]) < epsilon:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(pt)

        return unique_intersections
    
    def generate_rotated_rect_path(self, contour, mask_binary, 
                                   scan_spacing=20, step_spacing=30):
        """
        使用旋转矩形网格生成路径

        原理：
        1. 创建旋转矩形网格
        2. 计算网格与掩膜轮廓的交点
        3. 只保留在掩膜内的路径点
        """
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        center, (w, h), angle_deg = rect

        # 确定网格参数
        if w >= h:
            scan_angle = angle_deg
        else:
            scan_angle = angle_deg + 90

        angle_rad = np.deg2rad(scan_angle)

        # 创建网格点
        scan_points = []

        # 沿步进方向生成线
        num_steps = int(max(w, h) / step_spacing) + 2
        for step_idx in range(num_steps):
            # 计算当前线的偏移
            offset = (step_idx - num_steps / 2) * step_spacing

            # 计算线的起点和终点（在矩形坐标系中）
            length = min(w, h) * 1.5  # 线长度略大于矩形

            # 计算线的两个端点（在旋转后的坐标系中）
            dx1 = -length / 2 * np.cos(angle_rad)
            dy1 = -length / 2 * np.sin(angle_rad)
            dx2 = length / 2 * np.cos(angle_rad)
            dy2 = length / 2 * np.sin(angle_rad)

            # 加上步进方向的偏移
            dx_offset = offset * np.sin(angle_rad)
            dy_offset = -offset * np.cos(angle_rad)

            # 计算实际端点
            start_point = np.array([
                center[0] + dx1 + dx_offset,
                center[1] + dy1 + dy_offset
            ])
            end_point = np.array([
                center[0] + dx2 + dx_offset,
                center[1] + dy2 + dy_offset
            ])

            # 生成线上的点
            num_points_on_line = int(length / scan_spacing)
            for i in range(num_points_on_line + 1):
                t = i / num_points_on_line if num_points_on_line > 0 else 0
                # 之字形：奇数行反向
                if step_idx % 2 == 1:
                    t = 1 - t

                # 计算点坐标
                x = start_point[0] * (1 - t) + end_point[0] * t
                y = start_point[1] * (1 - t) + end_point[1] * t

                x_int, y_int = int(round(x)), int(round(y))

                # 检查点是否在掩膜内
                if (0 <= y_int < mask_binary.shape[0] and
                        0 <= x_int < mask_binary.shape[1] and
                        mask_binary[y_int, x_int] > 0):
                    scan_points.append((x_int, y_int))

        return scan_points
    
    def generate_smooth_contour_path(self, contour, mask_binary, point_spacing=10):
        """
        生成平滑的轮廓跟随路径

        原理：
        1. 对轮廓进行多边形逼近
        2. 在逼近的多边形上均匀采样点
        3. 形成连续的扫描路径
        """
        # 对轮廓进行多边形逼近
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 将近似多边形转换为点列表
        approx_points = approx.reshape(-1, 2)

        # 计算总长度
        total_length = 0
        lengths = []
        for i in range(len(approx_points)):
            p1 = approx_points[i]
            p2 = approx_points[(i + 1) % len(approx_points)]
            segment_length = np.linalg.norm(p2 - p1)
            lengths.append(segment_length)
            total_length += segment_length

        # 沿着多边形均匀采样点
        scan_points = []
        current_length = 0
        target_distance = 0

        while target_distance < total_length:
            # 找到当前目标距离所在的线段
            segment_start = 0
            for i, segment_length in enumerate(lengths):
                if current_length + segment_length > target_distance:
                    # 在这个线段上插值
                    t = (target_distance - current_length) / segment_length
                    p1 = approx_points[i]
                    p2 = approx_points[(i + 1) % len(approx_points)]
                    x = p1[0] * (1 - t) + p2[0] * t
                    y = p1[1] * (1 - t) + p2[1] * t

                    x_int, y_int = int(round(x)), int(round(y))

                    # 检查点是否在掩膜内
                    if (0 <= y_int < mask_binary.shape[0] and
                            0 <= x_int < mask_binary.shape[1] and
                            mask_binary[y_int, x_int] > 0):
                        scan_points.append((x_int, y_int))

                    break
                current_length += segment_length

            target_distance += point_spacing

        return scan_points
