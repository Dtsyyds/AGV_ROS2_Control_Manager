"""
点云处理模块
包含点云生成、法向量计算、点查询等功能
"""
import numpy as np
from typing import Tuple, Optional


class PointCloudProcessor:
    """点云处理器类"""
    
    def __init__(self, fx, fy, cx, cy):
        """
        初始化点云处理器
        
        Args:
            fx, fy: 相机焦距（像素）
            cx, cy: 相机主点坐标（像素）
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.pixel_to_index = None
        self.pointcloud = None
        self.normals = None
    
    def mask_depth_to_color_pointcloud(self, mask, depth_map, color_image, 
                                       depth_scale=1.0, mask_threshold=0.5):
        """将掩码区域转换为带颜色的 3D 点云"""
        if mask.dtype == np.uint8:
            if mask.max() > 1:
                mask = mask.astype(np.float32) / 255.0
            else:
                mask = mask.astype(np.float32)
        else:
            mask = mask.astype(np.float32)

        mask_binary = (mask > mask_threshold)
        n_mask_px = mask_binary.sum()
        print(f"🔹 掩码前景像素数：{n_mask_px}")

        if n_mask_px == 0:
            print("❌ 掩码全黑！请检查 SAM 分割结果。")
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        v, u = np.where(mask_binary)
        depths_raw = depth_map[v, u].astype(np.float32)
        depths = depths_raw * depth_scale

        # 提取对应像素的颜色
        colors_bgr = color_image[v, u]  # OpenCV 使用 BGR 格式

        n_valid = np.count_nonzero(depths > 0)
        print(f"---🔹 掩码区域内深度 >0 的点数：{n_valid} / {len(depths)}---")

        if n_valid == 0:
            sample_vals = depths_raw[:10]
            print(f"⚠️ 掩码区域深度值（前 10 个）: {sample_vals}")
            print("❗ 提示：深度全为 0？检查物体是否反光/透明/太远/被遮挡。")
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        valid_mask = depths > 0
        u_valid, v_valid, Z = u[valid_mask], v[valid_mask], depths[valid_mask]
        colors_bgr_valid = colors_bgr[valid_mask]

        # ✅ 修正：正确构建像素→点云索引映射
        H, W = mask.shape[:2]
        self.pixel_to_index = -np.ones((H, W), dtype=np.int32)  # 初始化全 -1

        # 为每个有效像素位置赋予索引
        idx = 0
        for v in range(H):
            for u in range(W):
                if mask_binary[v, u] and depth_map[v, u] > 0:
                    self.pixel_to_index[v, u] = idx
                    idx += 1
                else:
                    self.pixel_to_index[v, u] = -1

        # 投影到 3D 空间
        X = (u_valid - self.cx) * Z / self.fx
        Y = (v_valid - self.cy) * Z / self.fy
        points_3d = np.stack([X, Y, Z], axis=1)

        # 将颜色从 BGR 转换为 RGB，并归一化到 [0,1]
        colors_rgb = colors_bgr_valid[:, ::-1] / 255.0  # BGR 转 RGB，并归一化

        self.pointcloud = points_3d.astype(np.float32)
        print(f"---✅ 成功生成彩色点云：{points_3d.shape[0]} 个点---")

        return points_3d.astype(np.float32), colors_rgb.astype(np.float32)
    
    def calculate_normals(self, pointcloud=None, colors=None, 
                          radius=20, max_nn=50, orientation=[0.0, 0.0, -1.0]):
        """计算点云的法向量"""
        import open3d as o3d
        
        if pointcloud is None:
            pointcloud = self.pointcloud
            
        if pointcloud is None or len(pointcloud) == 0:
            print("❌ 点云为空，无法计算法向量")
            return

        print("---4正在计算点云法向量...---")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # 估计法向量
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )

        # 对齐法向量方向（朝向指定方向）
        pcd.orient_normals_to_align_with_direction(orientation)

        # 获取法向量
        normals = np.asarray(pcd.normals)  # shape (N, 3)
        self.normals = normals
        print(f"   法向量形状：{normals.shape}")
        print("   范围：x[{:.3f}, {:.3f}] y[{:.3f}, {:.3f}] z[{:.3f}, {:.3f}]".format(
            normals[:, 0].min(), normals[:, 0].max(),
            normals[:, 1].min(), normals[:, 1].max(),
            normals[:, 2].min(), normals[:, 2].max()
        ))
        
        print("---✅ 法向量计算完成---")
      
        return normals
    
    def get_point_and_normal(self, u, v, search_radius=2):
        """
        改进版本：如果直接坐标无对应点，搜索附近有效点
        
        Args:
            u, v: 像素坐标（列，行）
            search_radius: 搜索半径（像素）
            
        Returns:
            point_3d: (X,Y,Z) 或 None
            normal: (Nx,Ny,Nz) 或 None
        """
        if self.pixel_to_index is None or self.pointcloud is None:
            return None, None

        H, W = self.pixel_to_index.shape
        if not (0 <= v < H and 0 <= u < W):
            return None, None

        # 直接查找
        idx = self.pixel_to_index[v, u]

        # 如果直接查找失败，搜索附近区域
        if idx == -1 and search_radius > 0:
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    ny, nx = v + dy, u + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        temp_idx = self.pixel_to_index[ny, nx]
                        if temp_idx != -1:
                            idx = temp_idx
                            break
                if idx != -1:
                    break

        if idx == -1:
            return None, None

        if idx >= len(self.pointcloud):
            return None, None

        point_3d = self.pointcloud[idx]

        # 获取法向量
        if self.normals is not None and idx < len(self.normals):
            normal = self.normals[idx]
        else:
            normal = None

        return point_3d, normal,self.pixel_to_index
