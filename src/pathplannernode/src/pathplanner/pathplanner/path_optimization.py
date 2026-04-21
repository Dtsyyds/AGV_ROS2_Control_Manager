"""
路径优化模块
包含路径插值、注意力引导密度调整、路径平滑等功能
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
import os

class PathOptimizer:
    """路径优化器类"""
    
    def __init__(self):
        """初始化路径优化器"""   
        pass
    
    def interpolate_scan_points(self, scan_points, points_per_segment=5):
        """
        路径点插值加密
        
        Args:
            scan_points: 原始路径点 [(x,y), ...]
            points_per_segment: 每段插值点数
            
        Returns:
            interpolated_points: [(x, y), ...]
            segment_info: [{'orig_idx': i, 'is_original': bool, 't': float}, ...]
        """
        if len(scan_points) < 2:
            points = [tuple(p) for p in scan_points]
            info = [{'orig_idx': -1, 'is_original': True, 't': 0.0 if i == 0 else 1.0} 
                    for i, p in enumerate(scan_points)]
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
    
    def attention_guided_interpolation(self, scan_points, attention_map, 
                                       min_density=3, max_density=15):
        """
        根据注意力热力图动态调整插值密度
        
        Args:
            scan_points: 原始 2D 路径点 [(x,y), ...]
            attention_map: (H, W) 注意力热力图 [0~1]
            min_density: 最小插值密度
            max_density: 最大插值密度
            
        Returns:
            interpolated: 加密后的路径点
            segment_info: 段信息列表
        """
        interpolated = []
        segment_info = []

        H, W = attention_map.shape[:2]

        for i in range(len(scan_points) - 1):
            p0 = np.array(scan_points[i], dtype=float)
            p1 = np.array(scan_points[i + 1], dtype=float)

            # 计算线段中点的注意力值
            mid_pt = ((p0 + p1) / 2).astype(int)
            if 0 <= mid_pt[1] < H and 0 <= mid_pt[0] < W:
                att_val = attention_map[mid_pt[1], mid_pt[0]]
            else:
                att_val = 0.5

            # 根据注意力值动态调整插值密度
            density = int(min_density + (max_density - min_density) * att_val)

            # 添加起点（仅第一次）
            if i == 0:
                interpolated.append(tuple(scan_points[i]))
                segment_info.append({
                    'orig_idx': i, 
                    'is_original': True, 
                    't': 0.0,
                    'attention': float(att_val)
                })

            # 插中间点
            ts = np.linspace(0, 1, density + 2)[1:-1]
            for t in ts:
                pt = (1 - t) * p0 + t * p1
                # 四舍五入到像素坐标
                pt_int = (int(round(pt[0])), int(round(pt[1])))
                interpolated.append(pt_int)
                segment_info.append({
                    'orig_idx': i,
                    'is_original': False,
                    't': float(t),
                    'attention': float(att_val)
                })

            # 添加终点 (仅在最后一段添加，避免重复)
            if i == len(scan_points) - 2:
                interpolated.append(tuple(scan_points[-1]))
                segment_info.append({
                    'orig_idx': i, 
                    'is_original': True, 
                    'attention': float(att_val)
                })

        print(f"✅ 注意力引导插值完成：{len(scan_points)} -> {len(interpolated)} 点 (密度动态调整)")
        return interpolated, segment_info
    
    def smooth_path_within_mask(self, scan_points, mask_binary, window_size=5):
        """
        使用移动平均平滑路径，并强制约束点在掩码内
        
        Args:
            scan_points: 路径点列表
            mask_binary: 二值掩码
            window_size: 滑动窗口大小
            
        Returns:
            smoothed_points: 平滑后的路径点
        """
        smoothed_points = []
        points_array = np.array(scan_points)

        for i in range(len(points_array)):
            # 确定滑动窗口范围
            start = max(0, i - window_size // 2)
            end = min(len(points_array), i + window_size // 2 + 1)
            window = points_array[start:end]

            # 计算窗口内点的平均值
            avg_point = np.mean(window, axis=0)
            x, y = int(round(avg_point[0])), int(round(avg_point[1]))

            # 确保点在掩码内
            if (0 <= y < mask_binary.shape[0] and 
                0 <= x < mask_binary.shape[1] and 
                mask_binary[y, x] > 0):
                smoothed_points.append((x, y))
            else:
                # 如果不在掩码内，保留原始点
                smoothed_points.append(scan_points[i])

        print(f"✅ 路径平滑完成，已约束在掩码范围内")
        return smoothed_points
    
    def map_2d_to_3d(self, scan_points, mask_resized, get_point_func, 
                     segment_info=None,attention_ture=False):
        """
        将 2D 路径映射到 3D 空间
        
        Args:
            scan_points: 2D 路径点
            mask_resized: 调整后的掩码
            get_point_func: 获取 3D 点和法向量的函数
            segment_info: 段信息列表
            
        Returns:
            scan_points_3d: 3D 路径点数组
            scan_normals: 法向量数组
            scan_Points: 组合数据（点 + 法向量）
            scan_orig_indices: 段索引数组
        """
        scan_points_3d = []
        scan_normals = []
        scan_Points = []
        scan_orig_indices = []

        # 分组：按 orig_idx 把点分到各段
        from collections import defaultdict
        segments = defaultdict(list)

        for idx, (x, y) in enumerate(scan_points):
            info = segment_info[idx] if segment_info else {'orig_idx': 0, 'is_original': False, 't': 0.0}
            orig_idx = info['orig_idx']

            # 映射为 3D 点
            if 0 <= y < mask_resized.shape[0] and 0 <= x < mask_resized.shape[1]:
                if mask_resized[y, x] > 0.5:
                    pt3d, normal,pixel_to_index = get_point_func(x, y)
                    if pt3d is not None:
                        segments[orig_idx].append({
                            'pt2d': (x, y),
                            'pt3d': pt3d,
                            'normal': normal,
                            'info': info
                        })

        # 按 orig_idx 顺序合并
        for orig_idx in sorted(segments.keys()):
            group = segments[orig_idx]
            if not group:
                continue

            # 判断是否为奇数段
            is_odd_segment = (orig_idx % 2 == 1)
            
                
                
            if is_odd_segment and len(group) > 2:
                # 奇数段：保留首尾两个点
                selected = [group[0], group[-1]]
            else:
                # 偶数段：全保留
                selected = group
            if attention_ture:
                selected = group

            for item in selected:
                scan_points_3d.append(item['pt3d'])
                scan_normals.append(item['normal'])
                scan_Points.append(item['pt3d'].tolist() + item['normal'].tolist())
                scan_orig_indices.append(orig_idx)

        scan_points_3d = np.array(scan_points_3d) if scan_points_3d else np.array([])
        print(f"✅ 映射得到 {len(scan_points_3d)} 个有效 3D 路径点")
        print(f"段分布：{sorted(set(scan_orig_indices))}")

        return scan_points_3d, scan_normals, scan_Points, scan_orig_indices,pixel_to_index
