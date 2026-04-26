"""
注意力热力图计算模块
基于点云几何特征（法向量变化、PCA 曲率、深度梯度）计算掩码区域的复杂度注意力
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional
import threading


def compute_internal_attention_from_masked(
        mask: np.ndarray,
        pointcloud: np.ndarray,
        normals: np.ndarray,
        pixel_to_index: np.ndarray,
        k_neighbors: int = 16,
        radius: Optional[float] = None,
        weight_normal_std: float = 0.4,
        weight_curvature: float = 0.4,
        weight_laplacian: float = 0.2,
        depth_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算掩码内部区域的几何复杂度注意力（非仅轮廓！）

    Returns:
        attention_map: (H, W) float32, [0,1]，掩码内有效，外为 0
        attention_values: (N,) 每个点云点的注意力值（便于后续查表）
    """
    H, W = mask.shape
    N = len(pointcloud)
    if N == 0:
        return np.zeros((H, W), dtype=np.float32), np.array([])

    assert pointcloud.shape == (N, 3)
    assert normals.shape == (N, 3)
    assert pixel_to_index.shape == (H, W)

    # === Step 1: 构建 k-d tree 加速邻域搜索 ===
    tree = cKDTree(pointcloud)

    # 邻域查询策略
    if radius is not None:
        # 半径搜索（更均匀）
        distances, indices = tree.query(pointcloud, k=k_neighbors + 1, distance_upper_bound=radius)
        # 注意：query 会返回自身（dist=0），需剔除
        neighbor_indices = []
        for i in range(N):
            valid = (distances[i] < radius) & (distances[i] > 0)  # 剔除自身
            neighbor_indices.append(indices[i][valid])
    else:
        # 固定 k 近邻（更稳定，推荐）
        distances, indices = tree.query(pointcloud, k=k_neighbors + 1)
        neighbor_indices = [idx[1:] for idx in indices]  # 跳过第一个（自身）

    # 初始化特征
    normal_std_score = np.zeros(N, dtype=np.float32)
    curvature_score = np.zeros(N, dtype=np.float32)
    laplacian_score = np.zeros(N, dtype=np.float32)

    # === Step 2: 遍历每个点，计算局部几何特征 ===
    for i in range(N):
        nbrs = neighbor_indices[i]
        if len(nbrs) < 3:
            continue  # 邻域太小，跳过

        # --- (a) 法向量标准差（角度空间）---
        nbr_normals = normals[nbrs]  # (K, 3)
        # 转为单位向量夹角（与中心法向）
        center_normal = normals[i]
        cos_angles = np.clip(np.dot(nbr_normals, center_normal), -1.0, 1.0)
        angles = np.arccos(cos_angles)  # (K,)
        normal_std_score[i] = np.std(angles)

        # --- (b) PCA 曲率特征 ---
        nbr_points = pointcloud[nbrs]  # (K, 3)
        centroid = np.mean(nbr_points, axis=0)
        cov = np.cov((nbr_points - centroid).T)  # 3x3
        try:
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.abs(eigvals)
            eigvals.sort()  # λ0 ≤ λ1 ≤ λ2
            # 线性度（棱边）、平面度（平面）、球状度（凸点）
            # 推荐：使用 λ0 / (λ0+λ1+λ2) —— 越小越可能是棱/角
            if eigvals.sum() > 1e-8:
                curvature_score[i] = eigvals[0] / (eigvals.sum() + 1e-8)
            else:
                curvature_score[i] = 0.0
        except np.linalg.LinAlgError:
            curvature_score[i] = 0.0

        # --- (c) 深度 Laplacian（若提供 depth_map）---
        if depth_map is not None:
            # 反向映射：点云点 → 像素坐标（需相机内参，此处简化用 pixel_to_index 逆映射）
            # 更稳健方式：在 pixel_to_index 中搜索该点索引对应的坐标
            idx_where = np.where(pixel_to_index.ravel() == i)[0]
            if len(idx_where) > 0:
                flat_idx = idx_where[0]
                v, u = divmod(flat_idx, W)
                # 检查邻域像素深度（3x3）
                u0, u1 = max(0, u - 1), min(W, u + 2)
                v0, v1 = max(0, v - 1), min(H, v + 2)
                patch = depth_map[v0:v1, u0:u1]
                if patch.size > 1 and patch[patch > 0].size > 1:
                    center_depth = depth_map[v, u]
                    avg_depth = np.mean(patch[patch > 0])
                    laplacian_score[i] = abs(center_depth - avg_depth)

    # === Step 3: 归一化各特征 ===
    def safe_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        return np.zeros_like(x)

    n_std_norm = safe_normalize(normal_std_score)
    curv_norm = safe_normalize(curvature_score)
    lap_norm = safe_normalize(laplacian_score)

    # === Step 4: 融合注意力 ===
    attention_values = (
            weight_normal_std * n_std_norm +
            weight_curvature * curv_norm +
            weight_laplacian * lap_norm
    )
    attention_values = safe_normalize(attention_values)

    # === Step 5: 映射回 (H, W) 图像空间 ===
    attention_map = np.zeros((H, W), dtype=np.float32)
    # 利用 pixel_to_index: 对每个像素，若 idx >=0，则 attention_map[v,u] = attention_values[idx]
    valid_mask = pixel_to_index >= 0
    idx_flat = pixel_to_index[valid_mask]
    attention_map[valid_mask] = attention_values[idx_flat]

    # 可选：轻微平滑（避免邻域跳跃）
    attention_map = cv2.GaussianBlur(attention_map, (7, 7), sigmaX=0.8)

    return attention_map, attention_values


def compute_internal_attention_from_masked_pc(
        mask: np.ndarray,#目标物体掩码
        pointcloud: np.ndarray,#目标物体点云
        normals: np.ndarray,#目标物体法向量
        pixel_to_index: np.ndarray,#3d点转二维图索引
        k_neighbors: int = 32,#邻域点数量（推荐 16-32，过大可能过平滑）
        radius: Optional[float] = 0.015,
        weight_normal_std: float = 0.4,
        weight_linearity: float = 0.6,
        min_neighbors: int = 6,
        depth_map: Optional[np.ndarray] = None,#深度图
) -> Tuple[np.ndarray, np.ndarray]:
    """
    优化版注意力计算（修复邻域搜索问题）
    
    Returns:
        attention_map: (H, W) float32, [0,1]
        attention_values: (N,) 每个点云点的注意力值
    """
    print(f"🔹 计算内部注意力: k={k_neighbors}, radius={radius}, weights=({weight_normal_std:.2f}, {weight_linearity:.2f}), min_neighbors={min_neighbors}")
    H, W = mask.shape
    N = len(pointcloud)
    if N == 0:
        return np.zeros((H, W), dtype=np.float32), np.array([])

    assert pointcloud.shape == (N, 3)
    assert normals.shape == (N, 3)
    assert pixel_to_index.shape == (H, W)

    tree = cKDTree(pointcloud)

    # === 修复：安全构建邻域 ===
    neighbor_indices = []
    if radius is not None:
        # 查询时允许略大半径，避免因浮点误差漏点
        distances, indices = tree.query(
            pointcloud,
            k=k_neighbors + 1,
            distance_upper_bound=radius * 1.2
        )
        for i in range(N):
            # ✅ 关键：剔除无效索引（== N）、自身（dist≈0）、超半径
            valid = (
                    (distances[i] < radius) &
                    (distances[i] > 1e-6) &
                    (indices[i] < N) &  # ← 防 IndexError 的核心！
                    (indices[i] != i)  # 冗余保险（dist>0 通常已排除自身）
            )
            nbrs = indices[i][valid]
            # 若仍不足，从所有有效邻居中取最近的 min_neighbors 个
            if len(nbrs) < min_neighbors:
                # 收集所有非自身、非无效的邻居
                all_valid = (indices[i] < N) & (indices[i] != i)
                candidates = indices[i][all_valid]
                nbrs = candidates[:min_neighbors]
            neighbor_indices.append(nbrs)
    else:
        distances, indices = tree.query(pointcloud, k=k_neighbors + 1)
        for i in range(N):
            nbrs = indices[i][1:]  # 跳过自身
            nbrs = nbrs[nbrs < N]  # 防越界
            if len(nbrs) < min_neighbors:
                nbrs = nbrs[:min_neighbors]
            neighbor_indices.append(nbrs)

    # === 特征计算 ===
    normal_std_score = np.zeros(N, dtype=np.float32)#法向标准差
    linearity_score = np.zeros(N, dtype=np.float32)#线性结构

    for i in range(N):
        nbrs = np.array(neighbor_indices[i])
        # ✅ 双重保险：再过滤一次越界（极罕见，但安全）
        nbrs = nbrs[nbrs < N]
        if len(nbrs) < min_neighbors:
            continue

        # --- 法向标准差 ---
        #计算中心点法向量与所有邻居法向量的夹角。
        nbr_normals = normals[nbrs]
        center_normal = normals[i]
        cos_angles = np.clip(np.dot(nbr_normals, center_normal), -1.0, 1.0)
        angles = np.arccos(cos_angles)
        # 简单离群剔除
        if len(angles) > 2:
            angle_std = np.std(angles)
            inlier_mask = np.abs(angles - np.mean(angles)) < 2.5 * (angle_std + 1e-4)
            if np.sum(inlier_mask) >= 3:
                normal_std_score[i] = np.std(angles[inlier_mask])
        # 否则保持 0

        # --- 线性结构 ---
        #对邻居点坐标去中心化后计算协方差矩阵，进行特征值分解。
        nbr_points = pointcloud[nbrs]
        centroid = np.mean(nbr_points, axis=0)
        cov = np.cov((nbr_points - centroid).T)
        try:
            eigvals = np.linalg.eigvalsh(cov)  # 比 eigh 更稳
            eigvals = np.abs(eigvals)
            eigvals.sort()
            lam0, lam1, lam2 = eigvals + 1e-9
            linearity = 1.0 - min(1.0, lam0 / (lam1 + 1e-9))
            planarity = lam0 / (lam0 + lam1 + 1e-9)
            linearity_score[i] = linearity * (1.0 - planarity)
        except:
            linearity_score[i] = 0.0

    # === 归一化 & 融合 ===
    #Min-Max 归一化：safe_normalize 函数将特征值缩放到 [0, 1]，并处理 NaN/Inf 异常值。
    def safe_normalize(x):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min() + 1e-8)
        return np.zeros_like(x)

    n_std_norm = safe_normalize(normal_std_score)
    linearity_norm = safe_normalize(linearity_score)

    attention_values = (
            weight_normal_std * n_std_norm +
            weight_linearity * linearity_norm
    )
    #非线性增强
    #这是一个 Gamma 校正类似的操作。大于 1 的指数会压低中间值，使高响应区域更突出（接近 1），低响应区域更暗（接近 0），增加对比度。
    attention_values = np.power(np.clip(attention_values, 0, 1), 1.8)
    attention_values = safe_normalize(attention_values)

    # === 映射回图像空间 ===
    attention_map = np.zeros((H, W), dtype=np.float32)
    valid_mask = pixel_to_index >= 0
    idx_flat = pixel_to_index[valid_mask]
    # ✅ 再加一层保险：只取有效索引
    valid_idx = idx_flat[idx_flat < N]
    mask_sub = valid_mask.copy()
    mask_sub_flat = mask_sub.ravel()
    mask_sub_flat[valid_mask.ravel()] = (idx_flat < N)
    attention_map.flat[mask_sub_flat] = attention_values[valid_idx]

    # 降噪后处理
    #图像后处理 (Post-processing)
    #中值滤波：cv2.medianBlur(..., 3)。去除孤立的噪点（Salt-and-pepper noise）。
    attention_map = cv2.medianBlur(attention_map, 3)
    #高斯模糊：cv2.GaussianBlur(..., (5, 5), sigmaX=0.6)。使热力图过渡更平滑，避免锯齿感，更符合视觉注意力的连续性。
    attention_map = cv2.GaussianBlur(attention_map, (5, 5), sigmaX=0.6)
    print(f"✅ 内部注意力计算完成，非零像素数: {(attention_map > 1e-5).sum()} / {H*W}")
    return attention_map, attention_values


def visualize_internal_attention(
        masked_image: np.ndarray,
        attention_map: np.ndarray,
        alpha: float = 0.6,
        colormap: int = cv2.COLORMAP_JET,
        highlight_percentile: float = 95.0,
        draw_contours: bool = False,
        mask_binary: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    可视化内部注意力热力图（无轮廓点模式）
    """
    H, W = attention_map.shape[:2]
    overlay = masked_image.copy()

    # 彩色热力图
    att_uint8 = (np.clip(attention_map, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(att_uint8, colormap)
    cv2.imshow("HeatAttentionMap", heat_color)
    if threading.current_thread() is threading.main_thread():
        cv2.waitKey(500)
    else:
        cv2.waitKey(1)
    cv2.destroyWindow("HeatAttentionMap")
    
    # 叠加（仅在 attention > 0 区域）
    nonzero_mask = attention_map > 1e-5
    if np.any(nonzero_mask):
        overlay[nonzero_mask] = cv2.addWeighted(
            overlay[nonzero_mask], 1 - alpha,
            heat_color[nonzero_mask], alpha,
            0
        ).astype(np.uint8)

    # 🔴 高亮高注意力区域：二值化 + 轮廓描边（比散点更直观）
    thresh = np.percentile(attention_map[nonzero_mask], highlight_percentile) if np.any(nonzero_mask) else 0.5
    high_mask = (attention_map >= thresh) & nonzero_mask

    if np.any(high_mask):
        # 提取高注意力区域轮廓并描红边
        contours, _ = cv2.findContours(high_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)  # 🔴 红色轮廓

    # 叠加原始掩码轮廓（辅助参考）
    if draw_contours and mask_binary is not None:
        contours, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=1)  # 🟢 绿色外轮廓

    # 添加文字
    cv2.putText(overlay, f'Internal Attention (>{highlight_percentile:.0f}th %ile)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(overlay, f'Internal Attention (>{highlight_percentile:.0f}th %ile)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Internal Attention", overlay)
    if threading.current_thread() is threading.main_thread():
        cv2.waitKey(500)
    else:
        cv2.waitKey(1)
    cv2.destroyWindow("Internal Attention")
    cv2.imwrite("result/internal_attention.jpg", overlay)

    return overlay

import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree
from typing import Tuple, Optional
class GeometricPhysicalAttention:
    """
    高级几何物理注意力建模模块
    设计者：物理Attention (Gemini Co-Researcher)
    """
    def __init__(self, k_neighbors: int = 20, search_radius: float = 0.02):
        self.k = k_neighbors # k近邻数量 太小会过于局部(很碎)，太大可能过平滑(很大)
        self.radius = search_radius

    def safe_normalize(self, data: np.ndarray) -> np.ndarray:
        """带异常值处理的归一化"""
        data = np.nan_to_num(data, nan=0.0)
        low, high = np.percentile(data, [1, 99]) # 鲁棒性归一化，剔除1%离群点
        data = np.clip(data, low, high)
        if (high - low) < 1e-7: return np.zeros_like(data)
        return (data - low) / (high - low + 1e-8)

    def compute_attention(
        self, 
        pointcloud: np.ndarray, 
        normals: np.ndarray, 
        pixel_to_index: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        核心算法：多尺度各向异性张量注意力
        """
        N = pointcloud.shape[0]
        H, W = img_shape
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        # pcd=pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # 1. 构建 KDTree
        tree = o3d.geometry.KDTreeFlann(pcd)
        
        anisotropy_scores = np.zeros(N)
        flux_scores = np.zeros(N)
        curvature_change = np.zeros(N)

        for i in range(N):
            # 搜索邻域
            [_, idx, _] = tree.search_knn_vector_3d(pcd.points[i], self.k)
            if len(idx) < 5: continue
            
            # 提取邻域数据
            nbr_pts = pointcloud[idx]
            nbr_normals = normals[idx]
            center_pt = pointcloud[i]
            center_normal = normals[i]

            # --- (A) 几何张量分析 (PCA) ---
            cov = np.cov((nbr_pts - nbr_pts.mean(axis=0)).T)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(np.abs(eigvals)) # l0 <= l1 <= l2
            
            # 各向异性（棱边感知识别）：l2远大于l1时，为强边缘
            anisotropy_scores[i] = (eigvals[2] - eigvals[1]) / (eigvals[2] + 1e-8)

            # --- (B) 物理通量分析 (Normal Flux) ---
            # 计算邻域法线相对于中心点的发散程度，识别凹凸突变
            relative_pos = nbr_pts - center_pt
            # 归一化位移矢量
            dist = np.linalg.norm(relative_pos, axis=1, keepdims=True) + 1e-8
            unit_rel_pos = relative_pos / dist
            # 计算投影
            flux = np.mean(np.abs(np.sum(nbr_normals * unit_rel_pos, axis=1)))
            flux_scores[i] = flux

        # 2. 特征融合
        # 赋予边缘(Anisotropy)和几何突变(Flux)不同的物理权重
        f1 = self.safe_normalize(anisotropy_scores)
        f2 = self.safe_normalize(flux_scores)
        
        # 创新融合：乘性融合（只有当又是边缘且又有法向突变时，注意力最高）
        combined_att = np.power(f1 * 0.0 + f2 * 1.0, 1.5)
        combined_att = self.safe_normalize(combined_att)

        # 3. 映射至图像域
        attention_map = np.zeros((H, W), dtype=np.float32)
        valid_pixel_mask = pixel_to_index >= 0
        indices = pixel_to_index[valid_pixel_mask]
        
        # 过滤可能越界的索引
        valid_indices = indices < N
        attention_map[valid_pixel_mask] = 0 # 初始化
        
        # 高效映射
        flat_map = attention_map.ravel()
        valid_flat_mask = valid_pixel_mask.ravel()
        flat_map[valid_flat_mask] = combined_att[indices]
        
        # 4. 视觉后处理：导向滤波（Guided Filter）思想的平滑
        # 这里简化使用中值+高斯
        attention_map = cv2.medianBlur(attention_map, 3)
        attention_map = cv2.GaussianBlur(attention_map, (5, 5), 0)

        return attention_map, combined_att



class AdvancedGeometricAttention:
    """
    高级几何物理注意力建模模块 (V2.0)
    改进：引入微分几何算子、矩阵化加速、各向异性流分析
    """
    def __init__(self, k_neighbors: int = 30, sigma_factor: float = 1.5):
        self.k = k_neighbors
        self.sigma_factor = sigma_factor

    def safe_normalize(self, data: np.ndarray) -> np.ndarray:
        # 使用分位数进行健壮性缩放
        data = np.nan_to_num(data)
        q_min, q_max = np.percentile(data, [2, 98])
        if q_max - q_min < 1e-7:
            return np.zeros_like(data)
        return np.clip((data - q_min) / (q_max - q_min + 1e-8), 0, 1)

    def compute_attention(
        self, 
        pointcloud: np.ndarray, 
        normals: np.ndarray, 
        pixel_to_index: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        N = pointcloud.shape[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # 1. 矩阵化特征提取：局部几何协方差 (利用Open3D内部加速)
        # 获取每个点的特征值分析
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        linearity = np.zeros(N)
        planarity = np.zeros(N)
        surface_variation = np.zeros(N) # 传统意义上的曲率近似

        # 我们保留循环但优化内部计算
        points_np = np.asarray(pcd.points)
        normals_np = np.asarray(pcd.normals)
        
        # 预计算：为了演示创新，引入法向变化率
        # 
        for i in range(0, N, 1): # 可跨步采样加速或全采样
            [_, idx, _] = kdtree.search_knn_vector_3d(pcd.points[i], self.k)
            if len(idx) < 5: continue
            
            # 局部坐标中心化
            neighbors = points_np[idx]
            local_cov = np.cov(neighbors.T)
            eigenvalues, _ = np.linalg.eigh(local_cov)
            l = np.sort(np.abs(eigenvalues)) # l0 <= l1 <= l2
            
            # --- 创新点：多维特征描述符 ---
            sum_l = np.sum(l) + 1e-8
            linearity[i] = (l[2] - l[1]) / (l[2] + 1e-8)   # 线特征 (边缘)
            planarity[i] = (l[1] - l[0]) / (l[2] + 1e-8)   # 面特征
            surface_variation[i] = l[0] / sum_l            # 曲率变化 (突变)

        # 2. 物理通量改进：各向异性法向散度
        # 我们定义注意力 w = exp(- |n_i \cdot n_j|) 的变体
        # 这里使用简化版的全局/局部法向偏差
        flux_attention = self.safe_normalize(surface_variation)

        # 3. 融合策略：物理启发式加权
        # 创新：Edge-aware Attention. 
        # 只有在线性度高(Edge)且曲率变化大(Curvature Change)的地方权重最大
        combined_score = (linearity * 0.6 + surface_variation * 0.4) 
        combined_score = self.safe_normalize(combined_score)

        # 4. 投影与导向平滑
        H, W = img_shape
        attention_map = np.zeros((H, W), dtype=np.float32)
        
        # 扁平化映射
        flat_pixel_idx = pixel_to_index.flatten()
        valid_mask = flat_pixel_idx >= 0
        
        # 填充
        temp_map = np.zeros(H * W, dtype=np.float32)
        temp_map[valid_mask] = combined_score[flat_pixel_idx[valid_mask]]
        attention_map = temp_map.reshape(H, W)

        # 5. 改进：形态学形态学增强 + 边缘保持滤波
        # 移除孤立噪点并增强线状结构
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        attention_map = cv2.morphologyEx(attention_map, cv2.MORPH_CLOSE, kernel)
        
        # 使用快速导向滤波(Guided Filter)的替代方案：双边滤波
        attention_map = (attention_map * 255).astype(np.uint8)
        attention_map = cv2.bilateralFilter(attention_map, d=5, sigmaColor=75, sigmaSpace=75)
        
        return attention_map.astype(np.float32) / 255.0, combined_score
    

class AnisotropicAttentionSystem:
    """
    自适应各向异性注意力系统 (AAT-v1)
    功能：自动切换 边缘(Linearity) 与 突变(Variation) 的权重
    """
    def __init__(self, k_neighbors: int = 25):
        self.k = k_neighbors

    def _compute_local_tensor(self, neighbors: np.ndarray):
        """计算局部结构张量与特征值"""
        cov = np.cov(neighbors.T)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1] # 降序: l2 >= l1 >= l0
        return vals[idx], vecs[:, idx]

    def compute_aat_attention(
        self, 
        pcd_np: np.ndarray, 
        pixel_to_index: np.ndarray, 
        img_shape: Tuple[int, int]
    ) -> np.ndarray:
        N = pcd_np.shape[0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        tree = o3d.geometry.KDTreeFlann(pcd)

        # 结果容器
        linearity = np.zeros(N)
        variation = np.zeros(N)
        adaptive_weight = np.zeros(N)

        for i in range(N):
            [_, idx, _] = tree.search_knn_vector_3d(pcd.points[i], self.k)
            if len(idx) < 5: continue
            
            l, _ = self._compute_local_tensor(pcd_np[idx])
            l2, l1, l0 = l[0], l[1], l[2]
            sum_l = np.sum(l) + 1e-8

            # 1. 提取几何描述符
            linearity[i] = (l2 - l1) / (l2 + 1e-8)   # 对应边缘
            variation[i] = l0 / sum_l                # 对应突变 (曲率)

            # 2. 自适应权重计算 (Soft-Switch)
            # 使用指数函数增强两者的差异竞争
            w1 = np.exp(linearity[i] * 5.0)
            w2 = np.exp(variation[i] * 10.0) # 突变通常量纲小，系数给大一点
            adaptive_weight[i] = w1 / (w1 + w2 + 1e-8)

        # 3. 融合：根据自适应权重动态混合
        # 当 weight 接近 1，偏向边缘；接近 0，偏向突变
        combined_att = adaptive_weight * linearity + (1 - adaptive_weight) * variation
        # combined_att=variation
        
        # 归一化
        combined_att = self._robust_norm(combined_att)

        # 4. 映射到图像域
        return self._map_to_image(combined_att, pixel_to_index, img_shape)

    def _robust_norm(self, data: np.ndarray) -> np.ndarray:
        low, high = np.percentile(data, [1, 99])
        return np.clip((data - low) / (high - low + 1e-8), 0, 1)

    def _map_to_image(self, score, p2i, shape):
        h, w = shape
        img = np.zeros(h * w, dtype=np.float32)
        mask = p2i.flatten() >= 0
        img[mask] = score[p2i.flatten()[mask]]
        img = img.reshape(h, w)
        
        # 最后的结构保持：双边滤波
        img_u8 = (img * 255).astype(np.uint8)
        refined = cv2.bilateralFilter(img_u8, 9, 75, 75)
        return refined.astype(np.float32) / 255.0


#3.19物理attention熵的改进 EGA) 算法
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree
from typing import Tuple

class EntropyGuidedAttentionSystem:
    """
    基于信息熵引导的向量化自适应几何注意力系统 (AAT-v2)
    特点：全向量化计算、无经验超参数、引入平面惩罚与香农熵门控
    """
    def __init__(self, k_neighbors: int = 25):
        self.k = k_neighbors

    def compute_aat_attention(
        self, 
        pcd_np: np.ndarray, 
        pixel_to_index: np.ndarray, 
        img_shape: Tuple[int, int]
    ) -> np.ndarray:
        
        N = pcd_np.shape[0]
        
        # 1. 向量化 KNN 查询 (极其高效)
        tree = cKDTree(pcd_np)
        # workers=-1 表示使用所有 CPU 核心进行并行查询
        _, indices = tree.query(pcd_np, k=self.k, workers=-1) 
        
        # 2. 向量化协方差矩阵计算
        # 获取所有邻居点 shape: (N, k, 3)
        neighbors = pcd_np[indices] 
        # 计算质心 shape: (N, 1, 3)
        centroids = np.mean(neighbors, axis=1, keepdims=True)
        # 去中心化 shape: (N, k, 3)
        centered = neighbors - centroids
        
        # 使用 einsum 极速计算批量协方差矩阵 shape: (N, 3, 3)
        cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (self.k - 1)
        
        # 3. 向量化求解特征值 (默认升序排列: l0, l1, l2)
        vals = np.linalg.eigvalsh(cov_matrices)
        vals = np.clip(vals, 1e-12, None) # 防止数值下溢
        
        l0 = vals[:, 0]
        l1 = vals[:, 1]
        l2 = vals[:, 2]
        sum_l = l0 + l1 + l2
        
        # 4. 提取无量纲几何特征 (Demantké 等人的理论框架)
        linearity = (l2 - l1) / l2
        planarity = (l1 - l0) / l2
        variation = l0 / sum_l
        
        # 5. 信息熵门控计算 (Shannon Entropy)
        e0, e1, e2 = l0/sum_l, l1/sum_l, l2/sum_l
        # 局部信息熵
        entropy = - (e0 * np.log(e0) + e1 * np.log(e1) + e2 * np.log(e2))
        # 最大熵理论值为 ln(3) 约 1.0986，进行归一化
        entropy_norm = np.clip(entropy / np.log(3.0), 0.0, 1.0)
        
        # 6. 核心物理 Attention 建模
        # (1 - planarity) 作为显著性掩码，压制平坦区域
        # entropy_norm 作为软开关：高熵趋向variation(角/突变)，低熵趋向linearity(边缘)
        saliency_mask = 1.0 - planarity
        dynamic_fusion = (1.0 - entropy_norm) * linearity + entropy_norm * variation
        dynamic_fusion=variation
        
        combined_att = saliency_mask * dynamic_fusion
        combined_att=dynamic_fusion
        
        # 鲁棒归一化
        combined_att = self._robust_norm(combined_att)
        
        # 7. 映射至图像域
        return self._map_to_image(combined_att, pixel_to_index, img_shape)

    def _robust_norm(self, data: np.ndarray) -> np.ndarray:
        """剔除极端异常值后的归一化"""
        low, high = np.percentile(data, [1, 99.5])
        return np.clip((data - low) / (high - low + 1e-8), 0, 1)

    # =========================================================================
    # 以下方法请直接添加到你的 EntropyGuidedAttentionSystem 类中
    # =========================================================================

    def _get_pca_and_shift(self, pcd_np: np.ndarray):
        """
        [内部核心基建] 统一计算特征值、定向法向与质心偏移
        """
        tree = cKDTree(pcd_np)
        _, indices = tree.query(pcd_np, k=self.k, workers=-1)
        
        neighbors = pcd_np[indices] # (N, k, 3)
        centroids = np.mean(neighbors, axis=1) # (N, 3)
        centered = neighbors - centroids[:, np.newaxis, :] # (N, k, 3)
        
        # 批量协方差与特征值分解
        cov_matrices = np.einsum('nki,nkj->nij', centered, centered) / (self.k - 1)
        vals, vecs = np.linalg.eigh(cov_matrices)
        vals = np.clip(vals, 1e-12, None)
        
        l0, l1, l2 = vals[:, 0], vals[:, 1], vals[:, 2]
        
        # 提取法向 (最小特征值对应的特征向量)
        normals = vecs[:, :, 0] # (N, 3)
        
        # 视点定向 (假设深度图相机在原点 [0,0,0])
        # 确保法向指向相机: N 向量与视线向量 (-P) 的点积应大于 0
        view_dirs = -pcd_np
        dots = np.einsum('ni,ni->n', normals, view_dirs)
        normals[dots < 0] *= -1.0
        
        # 计算质心在法向上的投影位移 h
        # shift_vec = 质心 - 当前点
        shift_vecs = centroids - pcd_np
        h = np.einsum('ni,ni->n', shift_vecs, normals)
        
        return l0, l1, l2, h

    def get_bending_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
        """
        1. 提取绝对弯曲度 (Bending / Surface Variation)
        物理意义：不区分凹凸，只要不是纯平面就高亮。
        """
        l0, l1, l2, _ = self._get_pca_and_shift(pcd_np)
        bending = l0 / (l0 + l1 + l2)
        
        bending = self._robust_norm(bending)
        return self._map_to_image(bending, pixel_to_index, img_shape)

    def get_convex_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
        """
        2. 提取凸起结构 (Convex / Dome)
        物理意义：如球体外表面、凸起的按键、把手。质心偏移 h < 0。
        """
        l0, l1, l2, h = self._get_pca_and_shift(pcd_np)
        bending = l0 / (l0 + l1 + l2)
        
        # 筛选 h < 0 的区域 (凸)，并与弯曲度结合
        convex_score = np.where(h < 0, np.abs(h) * bending, 0.0)
        
        convex_score = self._robust_norm(convex_score)
        return self._map_to_image(convex_score, pixel_to_index, img_shape)

    def get_concave_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
        """
        3. 提取凹陷结构 (Concave / Cup)
        物理意义：如杯子内部、孔洞、深坑。质心偏移 h > 0。
        """
        l0, l1, l2, h = self._get_pca_and_shift(pcd_np)
        bending = l0 / (l0 + l1 + l2)
        
        # 筛选 h > 0 的区域 (凹)，并与弯曲度结合
        concave_score = np.where(h > 0, h * bending, 0.0)
        
        concave_score = self._robust_norm(concave_score)
        return self._map_to_image(concave_score, pixel_to_index, img_shape)

    def get_saddle_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
        """
        4. 提取马鞍面 (Saddle)
        物理意义：两个主曲率方向符号相反，正负抵消导致质心几乎在切平面上 (h 接近 0)，但弯曲度很大。
        """
        l0, l1, l2, h = self._get_pca_and_shift(pcd_np)
        bending = l0 / (l0 + l1 + l2)
        
        # 归一化 h 以便计算高斯衰减
        h_norm = h / (np.std(h) + 1e-8)
        
        # 当 h 接近 0 时，exp(-h^2) 接近 1；结合高弯曲度即为马鞍面
        saddle_score = bending * np.exp(- (h_norm ** 2) * 5.0) 
        
        saddle_score = self._robust_norm(saddle_score)
        return self._map_to_image(saddle_score, pixel_to_index, img_shape)

    def get_valley_ridge_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
        """
        5. 提取谷底与山脊 (Valley & Ridge)
        物理意义：具有强烈的单向延伸性 (高 Linearity) 的凹凸结构。如折痕、台阶边缘。
        """
        l0, l1, l2, h = self._get_pca_and_shift(pcd_np)
        
        # 线性度 (衡量延伸性)
        linearity = (l2 - l1) / (l2 + 1e-8)
        bending = l0 / (l0 + l1 + l2)
        
        # 既要像线一样延伸，又要发生弯曲
        valley_ridge_score = linearity * bending * np.abs(h)
        
        valley_ridge_score = self._robust_norm(valley_ridge_score)
        return self._map_to_image(valley_ridge_score, pixel_to_index, img_shape)
    def _map_to_image(self, score: np.ndarray, p2i: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """点云特征重投影到 2D 像素域并进行结构化滤波"""
        h, w = shape
        img = np.zeros(h * w, dtype=np.float32)
        
        # 有效映射索引
        valid_mask = p2i.flatten() >= 0
        valid_indices = p2i.flatten()[valid_mask]
        
        img[valid_mask] = score[valid_indices]
        img = img.reshape(h, w)
        
        # 双边滤波：保留边缘的同时平滑噪声
        img_u8 =(img * 255).astype(np.float32).astype(np.uint8)
        refined = cv2.bilateralFilter(img_u8, d=9, sigmaColor=75, sigmaSpace=75)
        
        return refined.astype(np.float32) / 255.0



class HighOrderGeometricExtractor:
    """
    高阶物理几何特征提取器
    包含：DoN (法向差分), Shape Index (形状指数), Normal Roughness (法向粗糙度)
    """
    def __init__(self, pcd_np: np.ndarray):
        self.pcd_np = pcd_np
        self.N = pcd_np.shape[0]
        self.tree = cKDTree(pcd_np)
        
        # 将 numpy 数组转换为 Open3D 格式以利用其优化的法向计算
        self.pcd_o3d = o3d.geometry.PointCloud()
        self.pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)

    def compute_don(self, radius_small: float, radius_large: float) -> np.ndarray:
        """提取多尺度法向差分 (Difference of Normals)"""
        # 计算小尺度法向
        self.pcd_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius_small)
        )
        self.pcd_o3d.orient_normals_consistent_tangent_plane(k=15)
        normals_small = np.asarray(self.pcd_o3d.normals).copy()

        # 计算大尺度法向
        self.pcd_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamRadius(radius=radius_large)
        )
        self.pcd_o3d.orient_normals_consistent_tangent_plane(k=15)
        normals_large = np.asarray(self.pcd_o3d.normals).copy()

        # DoN 向量计算与模长提取
        don_vectors = (normals_small - normals_large) / 2.0
        don_magnitude = np.linalg.norm(don_vectors, axis=1)
        
        # 归一化到 [0, 1] 方便后续 Attention 融合
        return np.clip(don_magnitude / (np.max(don_magnitude) + 1e-8), 0, 1)

    def compute_normal_roughness(self, k_neighbors: int = 30) -> np.ndarray:
        """提取局部法向粗糙度"""
        # 确保基础法向已计算
        if not self.pcd_o3d.has_normals():
            self.pcd_o3d.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
            )
            self.pcd_o3d.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
            
        normals = np.asarray(self.pcd_o3d.normals)
        _, indices = self.tree.query(self.pcd_np, k=k_neighbors, workers=-1)
        
        # 提取邻居法向: shape (N, K, 3)
        neighbor_normals = normals[indices]
        # 中心法向扩维: shape (N, 1, 3)
        center_normals = normals[:, np.newaxis, :]
        
        # 批量点积计算夹角余弦: shape (N, K)
        # 使用 einsum 极速计算点积
        dot_products = np.abs(np.einsum('nki,nmi->nk', neighbor_normals, center_normals))
        
        # 粗糙度 = 1 - 平均余弦值
        roughness = 1.0 - np.mean(dot_products, axis=1)
        return roughness / (np.max(roughness) + 1e-8)

    def compute_shape_index(self, k_neighbors: int = 20) -> np.ndarray:
        """
        [核心创新点] 提取 Shape Index (形状指数)
        基于局部切平面投影与二次曲面近似拟合求主曲率
        """
        # 1. 找邻居
        _, indices = self.tree.query(self.pcd_np, k=k_neighbors, workers=-1)
        neighbors = self.pcd_np[indices] # (N, K, 3)
        
        # 2. 局部去中心化
        centroids = np.mean(neighbors, axis=1, keepdims=True)
        centered = neighbors - centroids # (N, K, 3)
        
        # 3. 计算局部协方差获取法向 (即最小特征值对应的特征向量)
        cov = np.einsum('nki,nkj->nij', centered, centered)
        vals, vecs = np.linalg.eigh(cov)
        
        # z轴为法向 (局部坐标系), x, y 为切平面基向量
        local_z = vecs[:, :, 0] # (N, 3) 最小特征向量
        local_x = vecs[:, :, 1] # (N, 3)
        local_y = vecs[:, :, 2] # (N, 3)
        
        # 4. 将邻居点投影到局部二维切平面 (u, v) 以及高度 w
        u = np.einsum('nki,ni->nk', centered, local_x) # (N, K)
        v = np.einsum('nki,ni->nk', centered, local_y) # (N, K)
        w = np.einsum('nki,ni->nk', centered, local_z) # (N, K)
        
        # 5. 二次曲面拟合 w = 0.5 * A * u^2 + B * u * v + 0.5 * C * v^2
        # 构建设计矩阵 M: shape (N, K, 3)
        M = np.stack([0.5 * u**2, u * v, 0.5 * v**2], axis=-1)
        
        # 使用伪逆批量求解最小二乘 M * X = w  => X = (M^T M)^-1 M^T w
        Mt = np.transpose(M, (0, 2, 1)) # (N, 3, K)
        MtM = np.matmul(Mt, M) # (N, 3, 3)
        
        # 增加极小的正则化项防止奇异矩阵 (共面情况)
        reg = np.eye(3).reshape(1, 3, 3) * 1e-6
        MtM_inv = np.linalg.inv(MtM + reg) 
        
        Mt_w = np.matmul(Mt, w[..., np.newaxis]) # (N, 3, 1)
        
        # 得到拟合系数 X: [A, B, C]
        X = np.matmul(MtM_inv, Mt_w).squeeze(-1) # (N, 3)
        
        A = X[:, 0]
        B = X[:, 1]
        C = X[:, 2]
        
        # 6. 计算海森矩阵 (Hessian) 的特征值作为主曲率 kappa_1, kappa_2
        # Hessian = [[A, B], [B, C]]
        trace = A + C
        det = A * C - B**2
        
        # 求解二次方程 lambda^2 - trace*lambda + det = 0
        discriminant = np.sqrt(np.clip(trace**2 - 4 * det, 0, None))
        kappa_1 = (trace + discriminant) / 2.0
        kappa_2 = (trace - discriminant) / 2.0
        
        # 7. 计算 Shape Index (处理分母为0的情况)
        diff = kappa_1 - kappa_2
        sum_k = kappa_1 + kappa_2
        
        # 避免纯平面的除零错误
        valid_mask = diff > 1e-5 
        si = np.zeros(self.N)
        si[valid_mask] = (2.0 / np.pi) * np.arctan(sum_k[valid_mask] / diff[valid_mask])
        
        return si # 返回值在 [-1, 1] 之间
    def _map_to_image(self, score: np.ndarray, p2i: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """点云特征重投影到 2D 像素域并进行结构化滤波"""
        h, w = shape
        img = np.zeros(h * w, dtype=np.float32)
        
        # 有效映射索引
        valid_mask = p2i.flatten() >= 0
        valid_indices = p2i.flatten()[valid_mask]
        
        img[valid_mask] = score[valid_indices]
        img = img.reshape(h, w)
        
        # 双边滤波：保留边缘的同时平滑噪声
        img_u8 =(img * 255).astype(np.float32).astype(np.uint8)
        refined = cv2.bilateralFilter(img_u8, d=9, sigmaColor=75, sigmaSpace=75)
        
        return refined.astype(np.float32) / 255.0
    def get_don_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
        """获取 DoN 特征图 (H, W)，直接可用于你的画图程序"""
        # 1. 提取 (N,) 的点云特征
      
        don_feat = self.compute_don(radius_small=0.02, radius_large=0.1)
        
        # 2. 调用你自己的映射函数转为 (H, W) 的图像
        return self._map_to_image(don_feat, pixel_to_index, img_shape)

    def get_roughness_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
        """获取粗糙度特征图 (H, W)"""
    
        rough_feat = self.compute_normal_roughness(k_neighbors=100)
        
        return self._map_to_image(rough_feat, pixel_to_index, img_shape)

    def get_shape_index_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
        """获取 Shape Index 特征图 (H, W)"""
     
        si_feat = self.compute_shape_index(k_neighbors=20)
        
        # 【关键修正】将 [-1, 1] 线性映射到 [0, 1]，防止你的 _map_to_image 报错
        si_feat_normalized = (si_feat + 1.0) / 2.0 
        
        return self._map_to_image(si_feat_normalized, pixel_to_index, img_shape)





class FeatureVisualizer:
    """
    点云物理特征的 2D 伪彩色渲染器
    """
    def __init__(self, pixel_to_index: np.ndarray, img_shape: Tuple[int, int]):
        self.p2i = pixel_to_index
        self.shape = img_shape
        self.h, self.w = img_shape
        
        # 预计算有效映射掩码
        self.valid_mask = self.p2i.flatten() >= 0
        self.valid_indices = self.p2i.flatten()[self.valid_mask]

    def render_pseudocolor(
        self, 
        feature_array: np.ndarray, 
        feature_name: str,
        value_range: Tuple[float, float],
        cmap: int = cv2.COLORMAP_JET,
        save_to_disk: bool = True
    ) -> np.ndarray:
        """
        核心渲染逻辑
        :param feature_array: shape 为 (N,) 的一维特征数组
        :param feature_name: 特征名称 (用于保存文件名)
        :param value_range: 特征的理论或经验物理值域 (min, max)
        """
        # 1. 初始化灰度画布
        img_gray = np.zeros(self.h * self.w, dtype=np.float32)
        
        # 2. 将点云特征填入图像对应的像素位置
        img_gray[self.valid_mask] = feature_array[self.valid_indices]
        img_gray = img_gray.reshape(self.h, self.w)
        
        # 3. 基于物理值域进行线性归一化 [min, max] -> [0, 255]
        min_val, max_val = value_range
        img_clipped = np.clip(img_gray, min_val, max_val)
        img_normalized = ((img_clipped - min_val) / (max_val - min_val + 1e-8) * 255.0).astype(np.uint8)
        
        # 4. 应用 OpenCV 伪彩色映射 (默认使用 JET 色图: 蓝-绿-黄-红)
        color_img = cv2.applyColorMap(img_normalized, cmap)
        
        # 5. [关键] 掩码过滤：把没有数据的背景设为纯黑
        # 因为 applyColorMap 会把 0 值映射为深蓝色，我们需要手动擦除背景
        bg_mask = (self.p2i < 0)
        color_img[bg_mask] = [0, 0, 0]
        
        # 6. 保存与输出
        if save_to_disk:
            filename = f"feature_vis_{feature_name}.png"
            cv2.imwrite(filename, color_img)
            print(f"[{feature_name}] 伪彩色图已成功保存至: {filename}")
            
        return color_img






# class HighOrderGeometricExtractor:
#     """
#     高阶物理几何特征提取器 (优化版：基于预计算法向)
#     专为包含 pcd_np, normals_np, p2i, shape 的输入流设计
#     """
#     def __init__(
#         self, 
#         pcd_np: np.ndarray, 
#         normals_np: np.ndarray, 
#         pixel_to_index: np.ndarray, 
#         img_shape: Tuple[int, int]
#     ):
#         self.pcd_np = pcd_np
#         self.normals_np = normals_np
#         self.p2i = pixel_to_index
#         self.shape = img_shape
#         self.N = pcd_np.shape[0]
        
#         # 构建 KD-Tree，用于所有基于邻域的拓扑计算
#         self.tree = cKDTree(self.pcd_np)

#     def compute_don(self, k_large: int = 50) -> np.ndarray:
#         """
#         [创新点] 极速多尺度法向差分 (Difference of Normals)
#         原理：将传入的 normal 视作原法向，通过对大邻域(k_large)法向求均值得到平滑法向。
#         """
#         # 1. 查询大尺度邻居
#         _, indices = self.tree.query(self.pcd_np, k=k_large, workers=-1)
        
#         # 2. 提取邻居法向: shape (N, k_large, 3)
#         neighbor_normals = self.normals_np[indices]
        
#         # 3. 向量化求均值并重新归一化，得到大尺度低频法向
#         large_scale_normals = np.mean(neighbor_normals, axis=1)
#         norms = np.linalg.norm(large_scale_normals, axis=1, keepdims=True) + 1e-8
#         large_scale_normals = large_scale_normals / norms
        
#         # 4. 计算 DoN 向量与模长
#         don_vectors = (self.normals_np - large_scale_normals) / 2.0
#         don_magnitude = np.linalg.norm(don_vectors, axis=1)
        
#         # 鲁棒归一化 [0, 1]
#         p99 = np.percentile(don_magnitude, 99.5)
#         return np.clip(don_magnitude / (p99 + 1e-8), 0, 1)

#     def compute_roughness(self, k_neighbors: int = 20) -> np.ndarray:
#         """
#         提取局部法向粗糙度 (Local Normal Roughness)
#         原理：中心点法向与邻居法向夹角余弦的离散度。
#         """
#         _, indices = self.tree.query(self.pcd_np, k=k_neighbors, workers=-1)
        
#         neighbor_normals = self.normals_np[indices]      # (N, K, 3)
#         center_normals = self.normals_np[:, np.newaxis, :] # (N, 1, 3)
        
#         # Einsum 计算批量点积: (N, K)
#         dot_products = np.abs(np.einsum('nki,nmi->nk', neighbor_normals, center_normals))
        
#         # 粗糙度 = 1 - 邻域法向一致性
#         roughness = 1.0 - np.mean(dot_products, axis=1)
        
#         p99 = np.percentile(roughness, 99.5)
#         return np.clip(roughness / (p99 + 1e-8), 0, 1)

#     def compute_shape_index(self, k_neighbors: int = 20) -> np.ndarray:
#         """
#         [核心优化] 基于先验法向的定向切平面 Shape Index 计算
#         彻底解决传统 PCA 切平面法向不一致导致的曲率符号翻转问题。
#         """
#         _, indices = self.tree.query(self.pcd_np, k=k_neighbors, workers=-1)
#         neighbors = self.pcd_np[indices] # (N, K, 3)
        
#         # 1. 局部去中心化
#         centroids = np.mean(neighbors, axis=1, keepdims=True)
#         centered = neighbors - centroids # (N, K, 3)
        
#         # 2. 构建极其稳定的局部正交标架 (Local Frame)
#         local_z = self.normals_np # (N, 3) 直接使用传入的高质量法向
        
#         # 构造参考向量 (避免与 Z 轴共线)
#         ref_vec = np.zeros_like(local_z)
#         ref_vec[:, 0] = 1.0
#         # 如果 Z 轴太靠近 X 轴，则用 Y 轴作为参考
#         mask = np.abs(local_z[:, 0]) > 0.9
#         ref_vec[mask] = [0.0, 1.0, 0.0]
        
#         # 向量化叉乘生成 X, Y 轴
#         local_x = np.cross(ref_vec, local_z)
#         local_x /= (np.linalg.norm(local_x, axis=1, keepdims=True) + 1e-8)
#         local_y = np.cross(local_z, local_x)
        
#         # 3. 投影到切平面 (u, v) 以及高度 w
#         u = np.einsum('nki,ni->nk', centered, local_x) # (N, K)
#         v = np.einsum('nki,ni->nk', centered, local_y) # (N, K)
#         w = np.einsum('nki,ni->nk', centered, local_z) # (N, K)
        
#         # 4. 二次曲面最小二乘拟合 w = 0.5*A*u^2 + B*uv + 0.5*C*v^2
#         M = np.stack([0.5 * u**2, u * v, 0.5 * v**2], axis=-1) # (N, K, 3)
#         Mt = np.transpose(M, (0, 2, 1)) # (N, 3, K)
#         MtM = np.matmul(Mt, M)          # (N, 3, 3)
        
#         # 正则化防奇异
#         reg = np.eye(3).reshape(1, 3, 3) * 1e-6
#         MtM_inv = np.linalg.inv(MtM + reg)
#         Mt_w = np.matmul(Mt, w[..., np.newaxis]) # (N, 3, 1)
        
#         # 求解系数
#         X = np.matmul(MtM_inv, Mt_w).squeeze(-1) # (N, 3)
#         A, B, C = X[:, 0], X[:, 1], X[:, 2]
        
#         # 5. 求主曲率
#         trace = A + C
#         det = A * C - B**2
#         discriminant = np.sqrt(np.clip(trace**2 - 4 * det, 0, None))
        
#         kappa_1 = (trace + discriminant) / 2.0
#         kappa_2 = (trace - discriminant) / 2.0
        
#         # 6. 计算 Shape Index
#         diff = kappa_1 - kappa_2
#         sum_k = kappa_1 + kappa_2
        
#         valid_mask = np.abs(diff) > 1e-6
#         si = np.zeros(self.N)
#         si[valid_mask] = (2.0 / np.pi) * np.arctan(sum_k[valid_mask] / diff[valid_mask])
        
#         return si
#     def _map_to_image(self, score: np.ndarray, p2i: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
#         """点云特征重投影到 2D 像素域并进行结构化滤波"""
#         h, w = shape
#         img = np.zeros(h * w, dtype=np.float32)
        
#         # 有效映射索引
#         valid_mask = p2i.flatten() >= 0
#         valid_indices = p2i.flatten()[valid_mask]
        
#         img[valid_mask] = score[valid_indices]
#         img = img.reshape(h, w)
        
#         # 双边滤波：保留边缘的同时平滑噪声
#         img_u8 =(img * 255).astype(np.float32).astype(np.uint8)
#         refined = cv2.bilateralFilter(img_u8, d=9, sigmaColor=75, sigmaSpace=75)
        
#         return refined.astype(np.float32) / 255.0
#     def get_don_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
#         """获取 DoN 特征图 (H, W)，直接可用于你的画图程序"""
#         # 1. 提取 (N,) 的点云特征
       
#         don_feat =self.compute_don()
        
#         # 2. 调用你自己的映射函数转为 (H, W) 的图像
#         return self._map_to_image(don_feat, pixel_to_index, img_shape)

#     def get_roughness_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
#         """获取粗糙度特征图 (H, W)"""
       
#         rough_feat = self.compute_roughness(k_neighbors=30)
        
#         return self._map_to_image(rough_feat, pixel_to_index, img_shape)

#     def get_shape_index_image(self, pcd_np, pixel_to_index, img_shape) -> np.ndarray:
#         """获取 Shape Index 特征图 (H, W)"""
     
#         si_feat = self.compute_shape_index(k_neighbors=20)
        
#         # 【关键修正】将 [-1, 1] 线性映射到 [0, 1]，防止你的 _map_to_image 报错
#         si_feat_normalized = (si_feat + 1.0) / 2.0 
        
#         return self._map_to_image(si_feat_normalized, pixel_to_index, img_shape)






def main(pcd_np,normals_np,p2i,img_shape):

    print(f"输入点云数量: {pcd_np.shape[0]}")
    
    # 2. 实例化我们设计的提取器
    extractor = HighOrderGeometricExtractor(pcd_np, normals_np, p2i, img_shape)
    visualizer = FeatureVisualizer(p2i, img_shape)
    
    # 3. 计算并渲染 DoN (多尺度法向差分)
    # 应该在半球与平面的交界处(阶跃边缘)看到高亮的红圈
    print("正在计算 DoN...")
    don = extractor.compute_don(k_large=50)
    visualizer.render(don, "DoN", v_range=(0.0, 1.0))
    
    # 4. 计算并渲染粗糙度
    # 应该看到平面的高斯噪声呈现散点状的微弱高亮
    print("正在计算 Roughness...")
    roughness = extractor.compute_roughness(k_neighbors=20)
    visualizer.render(roughness, "Roughness", v_range=(0.0, 1.0))
    
    # 5. 计算并渲染 Shape Index (最硬核的拓扑特征)
    # 应该看到：平面呈现绿色(0左右)，半球凸起呈现红色(1.0, Dome)
    print("正在计算 Shape Index...")
    shape_index = extractor.compute_shape_index(k_neighbors=25)
    visualizer.render(shape_index, "ShapeIndex", v_range=(-1.0, 1.0))
    
    print("所有特征提取与渲染测试完成！")






class PhysicsAttentionV3:
    """
    高级几何物理注意力建模 V3.0
    优化点：内棱边增强、平滑曲率抑制、形态学特征提取
    """
    def __init__(self, k_neighbors: int = 30, power_factor: float = 2.5):
        self.k = k_neighbors
        self.power_factor = power_factor # 抑制平缓曲率的指数因子

    def _safe_normalize(self, data: np.ndarray) -> np.ndarray:
        """健壮性归一化，过滤1%的极值"""
        data = np.nan_to_num(data)
        if np.max(data) == np.min(data):
            return np.zeros_like(data)
        low, high = np.percentile(data, [1, 99])
        data = np.clip(data, low, high)
        return (data - low) / (high - low + 1e-8)

    def compute_attention(
        self, 
        pointcloud: np.ndarray, 
        normals: np.ndarray, 
        pixel_to_index: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        N = pointcloud.shape[0]
        H, W = img_shape
        
        # 构建 Open3D 点云对象加速检索
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        tree = o3d.geometry.KDTreeFlann(pcd)

        # 特征容器
        linearity_scores = np.zeros(N)  # 捕获外轮廓
        crease_scores = np.zeros(N)     # 捕获内棱边/物理折痕

        # 1. 局部几何特征扫描
        for i in range(N):
            [_, idx, _] = tree.search_knn_vector_3d(pcd.points[i], self.k)
            if len(idx) < 5: continue
            
            # --- [A] 结构张量分析 (PCA) ---
            local_pts = pointcloud[idx]
            cov = np.cov(local_pts.T)
            eigenvals, _ = np.linalg.eigh(cov)
            l = np.sort(np.abs(eigenvals)) # l0 <= l1 <= l2
            
            # Linearity: 边缘线特征
            linearity_scores[i] = (l[2] - l[1]) / (l[2] + 1e-8)

            # --- [B] 法向一致性分析 (Normal Jump) ---
            center_n = normals[i]
            local_n = normals[idx]
            # 计算局部法线与中心法线的余弦相似度
            cos_sims = np.dot(local_n, center_n)
            # 使用标准差衡量法向跳变的剧烈程度
            # 对于平面和拱形，std很小；对于折痕，std会激增
            crease_scores[i] = np.std(cos_sims)

        # 2. 特征非线性强化
        # 针对“拱形误报”，通过高幂次运算抑制低分干扰
        f_edge = self._safe_normalize(linearity_scores)
        f_crease = self._safe_normalize(np.power(crease_scores, self.power_factor))

        # 3. 融合策略：自适应竞争
        # 外轮廓与内棱边通过 Max 逻辑融合，确保两类特征都能被保留
        combined_att = np.maximum(f_edge, f_crease)

        # 4. 投影至图像域
        attention_map = np.zeros((H, W), dtype=np.float32)
        flat_p2i = pixel_to_index.flatten()
        valid_mask = flat_p2i >= 0
        
        # 填充图像
        temp_img = np.zeros(H * W, dtype=np.float32)
        temp_img[valid_mask] = combined_att[flat_p2i[valid_mask]]
        attention_map = temp_img.reshape(H, W)

        # 5. 图像域形态学优化
        # 使用 Top-Hat 变换：原图 - 开运算结果
        # 它可以移除掉背景中大面积的缓慢变化（如拱形的渐变），只保留细小的显著几何结构
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        attention_map = cv2.morphologyEx(attention_map, cv2.MORPH_TOPHAT, kernel)
        
        # 最后的平滑处理
        attention_map = self._safe_normalize(attention_map)
        attention_map = cv2.GaussianBlur(attention_map, (3, 3), 0)

        return attention_map, combined_att


#待优化
class RobustPhysicsAttentionV4:
    """
    针对噪声点云优化的物理注意力算法 (V4.0)
    改进：局部加权平滑、法向一致性增强、形态学背景抑制
    """
    def __init__(self, k_neighbors: int = 40, sigma_dist: float = 0.01, power_factor: float = 3.5):
        self.k = k_neighbors
        self.sigma_dist = sigma_dist # 高斯平滑参数
        self.power_factor = power_factor

    def _safe_normalize(self, data: np.ndarray) -> np.ndarray:
        data = np.nan_to_num(data)
        low, high = np.percentile(data, [5, 95]) # 提升剔除比例，应对噪点
        if high - low < 1e-7: return np.zeros_like(data)
        return np.clip((data - low) / (high - low + 1e-8), 0, 1)

    def compute_attention(
        self, 
        pointcloud: np.ndarray, 
        normals: np.ndarray, 
        pixel_to_index: np.ndarray,
        img_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        N = pointcloud.shape[0]
        H, W = img_shape
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        tree = o3d.geometry.KDTreeFlann(pcd)

        linearity = np.zeros(N)
        crease_score = np.zeros(N)

        # 1. 鲁棒特征提取
        for i in range(N):
            [_, idx, _] = tree.search_knn_vector_3d(pcd.points[i], self.k)
            if len(idx) < 10: continue
            
            # --- [A] 局部加权 PCA (抗噪边缘) ---
            neighbors = pointcloud[idx]
            diff = neighbors - pointcloud[i]
            dists_sq = np.sum(diff**2, axis=1)
            # 高斯权重：离中心点越近权重越高，抑制边缘噪声
            weights = np.exp(-dists_sq / (2 * self.sigma_dist**2))
            
            # 加权协方差矩阵
            weighted_diff = diff * np.sqrt(weights[:, np.newaxis])
            cov = np.dot(weighted_diff.T, weighted_diff) / (np.sum(weights) + 1e-8)
            
            vals, _ = np.linalg.eigh(cov)
            l = np.sort(np.abs(vals))
            # 改进的 Linearity (各向异性增强)
            linearity[i] = (l[2] - l[1]) / (l[2] + 1e-8)

            # --- [B] 法向投影一致性 (抗噪折痕) ---
            # 不直接用std，而是计算邻域法线在主轴上的投影分布
            local_n = normals[idx]
            mean_n = np.mean(local_n, axis=0)
            # 计算局部法线偏离平均法线的程度
            n_deviation = 1.0 - np.abs(np.dot(local_n, mean_n))
            # 使用中值抑制孤立噪点带来的干扰
            crease_score[i] = np.median(n_deviation)

        # 2. 非线性抑制与融合
        f_edge = self._safe_normalize(linearity)
        # 强力抑制拱形表面的微小法向偏移
        f_crease = self._safe_normalize(np.power(crease_score, self.power_factor))
        
        # 融合：优先保留强边缘和强折痕
        combined = np.maximum(f_edge, f_crease)

        # 3. 投影至图像域
        attention_map = np.zeros((H, W), dtype=np.float32)
        flat_p2i = pixel_to_index.flatten()
        valid = flat_p2i >= 0
        
        temp_img = np.zeros(H * W, dtype=np.float32)
        temp_img[valid] = combined[flat_p2i[valid]]
        attention_map = temp_img.reshape(H, W)

        # 4. 视觉后处理：抗噪双边滤波 + 闭运算
        # 闭运算连接断裂的边缘，Top-Hat 移除平面背景噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 移除大面积缓慢变化的背景（如拱形的渐变）
        attention_map = cv2.morphologyEx(attention_map, cv2.MORPH_TOPHAT, kernel)
        
        # 使用较大的双边滤波核抑制深度相机特有的“空洞”和“毛刺”
        attention_map = (self._safe_normalize(attention_map) * 255).astype(np.uint8)
        refined_map = cv2.bilateralFilter(attention_map, d=7, sigmaColor=50, sigmaSpace=50)

        return refined_map.astype(np.float32) / 255.0, combined
