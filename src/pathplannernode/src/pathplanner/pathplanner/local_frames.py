import cv2
import numpy as np
from ultralytics import SAM
import torch
import open3d as o3d

class Local_frames:
    def __init__(self):
        self.alpha_min=0.0
        self.alpha_max = 0.5
        self.curvature_k = 2.0
        self.use_adaptive_alpha = True
        self.alpha=0.3 #1.0全局，0.0局部

        self.scan_points_3d=None
        self.local_frames = None
        self.scan_normals=None
    def compute_local_frames(self):
        """
        为每个路径点计算局部坐标系：
          - z: 法向量
          - x: 扫查方向，保持统一方向（所有段都指向同一方向）
          - y: z × x （右手系）

        奇数段（反向扫描）保持与偶数段相同的x轴方向
        """
        if self.scan_points_3d is None or len(self.scan_points_3d) == 0:
            print("⚠️ 路径点为空，无法计算局部坐标系")
            self.local_frames = []
            return

        # 获取原始段信息
        N = len(self.scan_points_3d)
        points = np.array(self.scan_points_3d, dtype=np.float32)

        # 归一化法向量
        normals = np.array(self.scan_normals, dtype=np.float32)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=(norms != 0))

        # 第一步：计算原始的切线方向（基于路径点序列）
        raw_tangents = np.zeros_like(points)
        for i in range(N):
            if i < N - 1:
                raw_tangents[i] = points[i + 1] - points[i]
            else:
                if N > 1:
                    raw_tangents[i] = points[i] - points[i - 1]
                else:
                    raw_tangents[i] = np.array([1.0, 0.0, 0.0])

        # 归一化原始切线
        raw_tangent_norms = np.linalg.norm(raw_tangents, axis=1, keepdims=True)
        raw_tangents = np.divide(raw_tangents, raw_tangent_norms,
                                 out=np.zeros_like(raw_tangents),
                                 where=(raw_tangent_norms != 0))

        # 第二步：确定参考方向（使用第一个偶数段的切线方向）
        # 找到第一个偶数段的点
        reference_tangent = None
        if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
            for i in range(N):
                if self.scan_orig_indices[i] % 2 == 0:  # 偶数段
                    reference_tangent = raw_tangents[i]
                    break

        # 如果没有段信息，使用第一个点的切线方向
        if reference_tangent is None:
            reference_tangent = raw_tangents[0]

        # 确保参考方向不为零
        if np.linalg.norm(reference_tangent) < 1e-6:
            reference_tangent = np.array([1.0, 0.0, 0.0])

        print(f"参考切线方向: {reference_tangent}")

        # 第三步：计算每个点的最终x轴方向
        # 所有点都使用参考方向（保持一致性），然后投影到切平面
        uniform_tangents = np.tile(reference_tangent, (N, 1))

        # 投影到法向量正交平面（确保x ⊥ z）
        dot_xz = np.sum(uniform_tangents * normals, axis=1, keepdims=True)
        x_axes = uniform_tangents - dot_xz * normals

        # 归一化x轴
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)

        # 处理平行情况（x轴与法向量平行）
        fallback_mask = (x_norms.squeeze() < 1e-6)
        if np.any(fallback_mask):
            print(f"⚠️ {np.sum(fallback_mask)} 个点的参考方向与法向量平行")
            fallback_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            for i in np.where(fallback_mask)[0]:
                z = normals[i]

                # 尝试找到一个与法向量不正交的向量
                candidates = [np.array([1.0, 0.0, 0.0]),
                              np.array([0.0, 1.0, 0.0]),
                              np.array([0.0, 0.0, 1.0])]

                best_candidate = None
                best_dot = 0

                for cand in candidates:
                    proj = cand - np.dot(cand, z) * z
                    proj_norm = np.linalg.norm(proj)
                    if proj_norm > 1e-6:
                        # 检查与参考方向的一致性
                        if np.dot(proj / proj_norm, reference_tangent) > best_dot:
                            best_dot = np.dot(proj / proj_norm, reference_tangent)
                            best_candidate = proj / proj_norm

                if best_candidate is not None:
                    x_axes[i] = best_candidate
                else:
                    # 最后手段：使用交叉积
                    x_axes[i] = np.cross(z, [0, 0, 1])
                    x_norm_temp = np.linalg.norm(x_axes[i])
                    if x_norm_temp > 1e-6:
                        x_axes[i] = x_axes[i] / x_norm_temp
                    else:
                        x_axes[i] = np.cross(z, [1, 0, 0]) / np.linalg.norm(np.cross(z, [1, 0, 0]))

        # 重新归一化所有x轴
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)
        x_axes = np.divide(x_axes, x_norms, out=np.zeros_like(x_axes), where=(x_norms != 0))

        # 第四步：计算y轴 = z × x （右手系）
        y_axes = np.cross(normals, x_axes)
        y_norms = np.linalg.norm(y_axes, axis=1, keepdims=True)
        y_axes = np.divide(y_axes, y_norms, out=np.zeros_like(y_axes), where=(y_norms != 0))

        # 存储结果
        self.local_frames = []
        for i in range(N):
            frame = {
                'origin': points[i].tolist(),
                'x_axis': x_axes[i].tolist(),
                'y_axis': y_axes[i].tolist(),
                'z_axis': normals[i].tolist()
            }

            # 记录原始段信息（如果可用）
            if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
                frame['orig_idx'] = int(self.scan_orig_indices[i])
                frame['is_odd_segment'] = bool(self.scan_orig_indices[i] % 2 == 1)

            self.local_frames.append(frame)

        # 验证方向一致性
        if len(self.local_frames) > 0:
            # 计算第一个偶数段和第一个奇数段的x轴点积
            even_x = None
            odd_x = None

            for frame in self.local_frames:
                if 'orig_idx' in frame:
                    if even_x is None and frame['orig_idx'] % 2 == 0:
                        even_x = np.array(frame['x_axis'])
                    if odd_x is None and frame['orig_idx'] % 2 == 1:
                        odd_x = np.array(frame['x_axis'])

                if even_x is not None and odd_x is not None:
                    dot_product = np.dot(even_x, odd_x)
                    print(f"偶数段与奇数段x轴点积: {dot_product:.4f}")
                    if dot_product < 0.9:  # 应该接近1
                        print("⚠️ 警告: 奇数段与偶数段的x轴方向不一致")
                    break

        print(f"✅ 已计算 {N} 个路径点的局部坐标系（所有段保持相同x轴方向）")
        return self.local_frames

    def compute_local_frames_methd2(self):
        """
        奇数段保持与第一个偶数段相同的方向，偶数段交替反向
        规则：
          - 段0（第一个偶数段）：原始方向
          - 段1、3、5...（奇数段）：与段0同向
          - 段2（偶数段）：反向
          - 段4（偶数段）：原始方向
          - 段6（偶数段）：反向
          - 以此类推
        """
        if self.scan_points_3d is None or len(self.scan_points_3d) == 0:
            print("⚠️ 路径点为空，无法计算局部坐标系")
            self.local_frames = []
            return

        N = len(self.scan_points_3d)
        points = np.array(self.scan_points_3d, dtype=np.float32)

        # 归一化法向量
        normals = np.array(self.scan_normals, dtype=np.float32)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=(norms != 0))

        # 步骤1：计算每个点的原始切线方向
        raw_tangents = np.zeros_like(points)
        for i in range(N):
            if i < N - 1:
                raw_tangents[i] = points[i + 1] - points[i]
            elif i > 0:
                raw_tangents[i] = points[i] - points[i - 1]
            else:
                raw_tangents[i] = np.array([1.0, 0.0, 0.0])

        # 归一化原始切线
        raw_tangent_norms = np.linalg.norm(raw_tangents, axis=1, keepdims=True)
        raw_tangents = np.divide(raw_tangents, raw_tangent_norms,
                                 out=np.zeros_like(raw_tangents),
                                 where=(raw_tangent_norms != 0))

        # 步骤2：计算段0的平均原始切线方向（作为奇数段的参考方向）
        first_even_avg_direction = None
        first_even_points_count = 0

        if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
            # 找到段0的所有点
            segment0_indices = []
            for i in range(N):
                if self.scan_orig_indices[i] == 0:  # 段0
                    segment0_indices.append(i)

            if segment0_indices:
                # 计算段0的平均切线方向
                sum_direction = np.zeros(3)
                for idx in segment0_indices:
                    sum_direction += raw_tangents[idx]

                avg_direction = sum_direction / len(segment0_indices)
                norm = np.linalg.norm(avg_direction)
                if norm > 1e-6:
                    first_even_avg_direction = avg_direction / norm
                    first_even_points_count = len(segment0_indices)
                    print(f"段0有 {first_even_points_count} 个点，平均方向: {first_even_avg_direction}")
                else:
                    first_even_avg_direction = np.array([1.0, 0.0, 0.0])
                    print("⚠️ 段0的平均方向为零向量，使用默认方向")
            else:
                print("⚠️ 未找到段0的点，使用默认方向")
                first_even_avg_direction = np.array([1.0, 0.0, 0.0])
        else:
            print("⚠️ 没有段信息，使用默认方向")
            first_even_avg_direction = np.array([1.0, 0.0, 0.0])

        # 步骤3：确定每个段的方向因子
        segment_factors = {}

        if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
            unique_segments = sorted(set(self.scan_orig_indices))

            for seg_idx in unique_segments:
                if seg_idx == 0:  # 第一个偶数段
                    segment_factors[seg_idx] = 1  # 原始方向
                elif seg_idx % 2 == 0:  # 其他偶数段
                    # 偶数段交替反向：段2反向，段4原始方向，段6反向...
                    factor = -1 if (seg_idx // 2) % 2 == 1 else 1
                    segment_factors[seg_idx] = factor
                else:  # 奇数段
                    # 所有奇数段都使用段0的方向（因子为1）
                    segment_factors[seg_idx] = 1

            print("段方向因子:", segment_factors)
        else:
            print("⚠️ 没有段信息，所有点使用相同方向")
            segment_factors = {0: 1}  # 默认

        # 步骤4：应用方向因子
        adjusted_tangents = np.zeros_like(raw_tangents)

        for i in range(N):
            if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
                seg_idx = self.scan_orig_indices[i]
                factor = segment_factors.get(seg_idx, 1)

                if seg_idx % 2 == 1:  # 奇数段
                    # 奇数段使用段0的平均方向，而不是自己的原始方向
                    adjusted_tangents[i] = first_even_avg_direction * factor
                else:  # 偶数段
                    # 偶数段使用自己的原始方向，乘以方向因子
                    adjusted_tangents[i] = raw_tangents[i] * factor
            else:
                # 没有段信息，使用原始方向
                adjusted_tangents[i] = raw_tangents[i]

        # 步骤5：投影到法向量切平面，得到x轴
        dot_xz = np.sum(adjusted_tangents * normals, axis=1, keepdims=True)
        x_axes = adjusted_tangents - dot_xz * normals

        # 归一化x轴
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)

        # 处理与法向量平行的情况
        fallback_mask = (x_norms.squeeze() < 1e-6)
        if np.any(fallback_mask):
            print(f"⚠️ {np.sum(fallback_mask)} 个点的切线与法向量平行，使用备选方向")

            for i in np.where(fallback_mask)[0]:
                z = normals[i]

                # 确定期望的方向
                expected_direction = adjusted_tangents[i]

                # 尝试多个方向
                candidates = [np.array([1.0, 0.0, 0.0]),
                              np.array([0.0, 1.0, 0.0]),
                              np.array([0.0, 0.0, 1.0])]

                best_candidate = None
                best_dot = -1

                for cand in candidates:
                    # 投影到切平面
                    proj = cand - np.dot(cand, z) * z
                    proj_norm = np.linalg.norm(proj)
                    if proj_norm > 1e-6:
                        proj_normalized = proj / proj_norm
                        # 计算与期望方向的点积
                        dot_val = np.dot(proj_normalized, expected_direction)
                        if dot_val > best_dot:
                            best_dot = dot_val
                            best_candidate = proj_normalized

                if best_candidate is not None:
                    x_axes[i] = best_candidate
                else:
                    # 最后手段：使用交叉积
                    x_axes[i] = np.cross(z, [0, 0, 1])
                    x_norm_temp = np.linalg.norm(x_axes[i])
                    if x_norm_temp > 1e-6:
                        x_axes[i] = x_axes[i] / x_norm_temp
                    else:
                        x_axes[i] = np.array([1.0, 0.0, 0.0])

        # 重新归一化所有x轴
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)
        x_axes = np.divide(x_axes, x_norms, out=np.zeros_like(x_axes), where=(x_norms != 0))

        # 步骤6：计算y轴 = z × x （右手系）
        y_axes = np.cross(normals, x_axes)
        y_norms = np.linalg.norm(y_axes, axis=1, keepdims=True)
        y_axes = np.divide(y_axes, y_norms, out=np.zeros_like(y_axes), where=(y_norms != 0))

        # 存储结果
        self.local_frames = []
        for i in range(N):
            frame = {
                'origin': points[i].tolist(),
                'x_axis': x_axes[i].tolist(),
                'y_axis': y_axes[i].tolist(),
                'z_axis': normals[i].tolist()
            }

            if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
                seg_idx = self.scan_orig_indices[i]
                frame['orig_idx'] = int(seg_idx)
                frame['segment_type'] = 'odd' if seg_idx % 2 == 1 else 'even'
                frame['direction_factor'] = segment_factors.get(seg_idx, 1)
                frame['uses_first_even_dir'] = (seg_idx % 2 == 1)  # 标记是否使用了段0的方向

            self.local_frames.append(frame)

        # 验证方向一致性
        if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
            # 计算每个段的平均x轴方向
            segment_x_directions = {}

            for i, frame in enumerate(self.local_frames):
                seg_idx = frame['orig_idx']
                x_axis = np.array(frame['x_axis'])

                if seg_idx not in segment_x_directions:
                    segment_x_directions[seg_idx] = []

                segment_x_directions[seg_idx].append(x_axis)

            # 计算每个段的平均方向
            print("\n段方向统计:")
            for seg_idx in sorted(segment_x_directions.keys()):
                seg_x_axes = np.array(segment_x_directions[seg_idx])
                avg_x = np.mean(seg_x_axes, axis=0)
                avg_x_norm = avg_x / np.linalg.norm(avg_x)

                seg_type = "奇数" if seg_idx % 2 == 1 else "偶数"
                uses_ref = "（使用段0方向）" if seg_idx % 2 == 1 else ""
                print(
                    f"  段{seg_idx}({seg_type}){uses_ref}: 平均x轴方向 = [{avg_x_norm[0]:.3f}, {avg_x_norm[1]:.3f}, {avg_x_norm[2]:.3f}]")

            # 检查奇数段与段0的方向一致性
            if 0 in segment_x_directions:
                # 计算段0的平均方向
                seg0_x_axes = np.array(segment_x_directions[0])
                seg0_avg_x = np.mean(seg0_x_axes, axis=0)
                seg0_avg_x_norm = seg0_avg_x / np.linalg.norm(seg0_avg_x)

                # 检查每个奇数段与段0的方向一致性
                for seg_idx in sorted(segment_x_directions.keys()):
                    if seg_idx % 2 == 1:  # 奇数段
                        seg_x_axes = np.array(segment_x_directions[seg_idx])
                        seg_avg_x = np.mean(seg_x_axes, axis=0)
                        seg_avg_x_norm = seg_avg_x / np.linalg.norm(seg_avg_x)

                        dot_product = np.dot(seg0_avg_x_norm, seg_avg_x_norm)
                        print(f"    段{seg_idx}与段0的x轴方向点积: {dot_product:.3f}")

                        if dot_product < 0.9:  # 应该接近1
                            print(f"    ⚠️ 警告: 段{seg_idx}与段0的方向不一致")

            # 检查偶数段交替反向
            for seg_idx in sorted(segment_x_directions.keys()):
                if seg_idx % 2 == 0 and seg_idx >= 2:
                    prev_even_seg = seg_idx - 2
                    if prev_even_seg in segment_x_directions:
                        # 计算当前段平均方向
                        seg_x_axes = np.array(segment_x_directions[seg_idx])
                        seg_avg_x = np.mean(seg_x_axes, axis=0)
                        seg_avg_x_norm = seg_avg_x / np.linalg.norm(seg_avg_x)

                        # 计算前一个偶数段平均方向
                        prev_seg_x_axes = np.array(segment_x_directions[prev_even_seg])
                        prev_avg_x = np.mean(prev_seg_x_axes, axis=0)
                        prev_avg_x_norm = prev_avg_x / np.linalg.norm(prev_avg_x)

                        dot_product = np.dot(seg_avg_x_norm, prev_avg_x_norm)
                        expected_dot = -1.0 if (seg_idx // 2) % 2 == 1 else 1.0
                        print(f"    段{seg_idx}与段{prev_even_seg}的x轴方向点积: {dot_product:.3f} (期望接近{expected_dot})")

                        if abs(dot_product - expected_dot) > 0.3:  # 允许一定误差
                            print(f"    ⚠️ 警告: 段{seg_idx}与段{prev_even_seg}方向关系不符合预期")

        print(f"\n✅ 已计算 {N} 个路径点的局部坐标系（奇数段使用段0方向，偶数段交替反向）")
        return self.local_frames

    #alpha=1的情况
    def compute_local_frames_p(self):
        """
        为每个路径点计算局部坐标系（改进版）：
          - z: 法向量（归一化）
          - x: 投影后的局部切线方向，经全局参考方向符号对齐（贴近路径走向，且所有段同向）
          - y: z × x （右手系）

        ✅ 改进点：
          - 以局部路径切线为主，投影到切平面得自然x方向
          - 通过符号对齐（flip 若 dot < 0）保证奇偶段x同向
          - fallback 更鲁棒（优先用参考方向而非固定[1,0,0]）
        """
        if self.scan_points_3d is None or len(self.scan_points_3d) == 0:
            print("⚠️ 路径点为空，无法计算局部坐标系")
            self.local_frames = []
            return

        N = len(self.scan_points_3d)
        points = np.array(self.scan_points_3d, dtype=np.float32)
        normals = np.array(self.scan_normals, dtype=np.float32)

        # === Step 1: 归一化法向量 ===
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=(norms != 0))

        # === Step 2: 计算原始切线（路径走向）并归一化 ===
        raw_tangents = np.zeros_like(points)
        if N == 1:
            raw_tangents[0] = np.array([1.0, 0.0, 0.0])
        else:
            # 前向差分（首点用前两点，末点用后两点，中间用中心差分）
            raw_tangents[0] = points[1] - points[0]
            raw_tangents[-1] = points[-1] - points[-2]
            for i in range(1, N - 1):
                raw_tangents[i] = points[i + 1] - points[i - 1]
            # 归一化
            tan_norms = np.linalg.norm(raw_tangents, axis=1, keepdims=True)
            raw_tangents = np.divide(raw_tangents, tan_norms,
                                     out=np.zeros_like(raw_tangents),
                                     where=(tan_norms != 0))

        # === Step 3: 投影切线到法向量正交平面 → 局部自然x候选 ===
        dot_tz = np.sum(raw_tangents * normals, axis=1, keepdims=True)
        local_x_candidates = raw_tangents - dot_tz * normals  # x ⊥ z

        # 归一化候选x
        x_cand_norms = np.linalg.norm(local_x_candidates, axis=1, keepdims=True)
        local_x_candidates = np.divide(local_x_candidates, x_cand_norms,
                                       out=np.zeros_like(local_x_candidates),
                                       where=(x_cand_norms > 1e-8))

        # === Step 4: 确定全局参考方向（用于符号对齐）===
        reference_dir = None
        if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
            # 优先使用第一个偶数段的 *投影后* 切线方向（更合理）
            for i, orig_idx in enumerate(self.scan_orig_indices):
                if orig_idx % 2 == 0:  # 偶数段
                    if np.linalg.norm(local_x_candidates[i]) > 1e-6:
                        reference_dir = local_x_candidates[i].copy()
                        break
        # 若无段信息或偶数段全退化，用第一个非退化点
        if reference_dir is None:
            for i in range(N):
                if np.linalg.norm(local_x_candidates[i]) > 1e-6:
                    reference_dir = local_x_candidates[i].copy()
                    break

        # 最终兜底
        if reference_dir is None or np.linalg.norm(reference_dir) < 1e-6:
            reference_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        print(f"参考方向（用于符号对齐）: {reference_dir}")

        # === Step 5: 符号对齐 —— 保证所有x轴与参考方向夹角 ≤ 90° ===
        dot_with_ref = np.sum(local_x_candidates * reference_dir, axis=1)  # shape: (N,)
        flip_mask = dot_with_ref < 0  # 需翻转的点索引
        x_axes = local_x_candidates.copy()
        x_axes[flip_mask] *= -1.0

        # === Step 6: 处理退化情况（投影后为零向量）===
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)
        degenerate_mask = (x_norms.squeeze() < 1e-6)
        if np.any(degenerate_mask):
            deg_count = np.sum(degenerate_mask)
            print(f"⚠️ {deg_count} 个点的局部切线 ∥ 法向量，需 fallback")
            for i in np.where(degenerate_mask)[0]:
                z = normals[i]
                # 尝试用参考方向投影
                proj = reference_dir - np.dot(reference_dir, z) * z
                proj_norm = np.linalg.norm(proj)
                if proj_norm > 1e-6:
                    x_axes[i] = proj / proj_norm
                else:
                    # 参考方向也平行 → 用标准基叉积
                    cand = np.cross(z, np.array([0.0, 0.0, 1.0]))
                    if np.linalg.norm(cand) < 1e-6:
                        cand = np.cross(z, np.array([1.0, 0.0, 0.0]))
                    x_axes[i] = cand / np.linalg.norm(cand)

        # 最终归一化（fallback后可能未归一）
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)
        x_axes = np.divide(x_axes, x_norms, out=np.zeros_like(x_axes), where=(x_norms > 1e-8))

        # === Step 7: 计算 y = z × x（右手系）===
        y_axes = np.cross(normals, x_axes)
        y_norms = np.linalg.norm(y_axes, axis=1, keepdims=True)
        y_axes = np.divide(y_axes, y_norms, out=np.zeros_like(y_axes), where=(y_norms > 1e-8))

        # === Step 8: 存储结果 ===
        self.local_frames = []
        for i in range(N):
            frame = {
                'origin': points[i].tolist(),
                'x_axis': x_axes[i].tolist(),
                'y_axis': y_axes[i].tolist(),
                'z_axis': normals[i].tolist()
            }
            if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
                orig_idx = int(self.scan_orig_indices[i])
                frame['orig_idx'] = orig_idx
                frame['is_odd_segment'] = bool(orig_idx % 2 == 1)
            self.local_frames.append(frame)

        # === Step 9: 验证方向一致性 ===
        if self.local_frames:
            even_x = odd_x = None
            for frame in self.local_frames:
                if 'orig_idx' in frame:
                    x_vec = np.array(frame['x_axis'])
                    if even_x is None and frame['orig_idx'] % 2 == 0:
                        even_x = x_vec
                    if odd_x is None and frame['orig_idx'] % 2 == 1:
                        odd_x = x_vec
                if even_x is not None and odd_x is not None:
                    dot_prod = np.dot(even_x, odd_x)
                    print(f"偶数段与奇数段x轴点积: {dot_prod:.4f}（应≈1.0）")
                    if dot_prod < 0.95:
                        print("⚠️ 警告: 奇偶段x方向不一致（可能段索引错误或路径突变）")
                    break

        print(f"✅ 已计算 {N} 个路径点的局部坐标系（x贴近路径走向，全局方向一致）")
        return self.local_frames

    # 加权处理若希望 x 更贴近路径走向（而非强统一），可改用：
    # x = normalize( (raw_tangent - (raw_tangent⋅z)z) + α * reference_projection )
    # 用加权混合局部切线与全局方向。
    def compute_local_frames_jiaquan(self, alpha=0.0):
        """
        为每个路径点计算局部坐标系（加权版）：
          - z: 法向量（归一化）
          - x: (1-α) * 局部投影切线 + α * 全局参考投影方向，再归一化 & 符号对齐
          - y: z × x （右手系）

        Args:
            alpha (float): 权重 ∈ [0, 1]
                - 0.0: 完全信任局部路径走向（等价于上一版）
                - 1.0: 完全强制统一方向（类似你最原始版本）
                - 0.2~0.5: 推荐值，平衡自然性与一致性

        """
        if self.scan_points_3d is None or len(self.scan_points_3d) == 0:
            print("⚠️ 路径点为空，无法计算局部坐标系")
            self.local_frames = []
            return

        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"

        N = len(self.scan_points_3d)
        points = np.array(self.scan_points_3d, dtype=np.float32)
        normals = np.array(self.scan_normals, dtype=np.float32)

        # === Step 1: 归一化法向量 ===
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=(norms != 0))

        # === Step 2: 计算并投影局部切线 ===
        raw_tangents = np.zeros_like(points)
        if N == 1:
            raw_tangents[0] = np.array([1.0, 0.0, 0.0])
        else:
            raw_tangents[0] = points[1] - points[0]
            raw_tangents[-1] = points[-1] - points[-2]
            for i in range(1, N - 1):
                raw_tangents[i] = points[i + 1] - points[i - 1]
            tan_norms = np.linalg.norm(raw_tangents, axis=1, keepdims=True)
            raw_tangents = np.divide(raw_tangents, tan_norms,
                                     out=np.zeros_like(raw_tangents),
                                     where=(tan_norms != 0))

        # 投影到切平面 → 局部自然方向
        dot_tz = np.sum(raw_tangents * normals, axis=1, keepdims=True)
        local_x_dir = raw_tangents - dot_tz * normals
        local_x_norms = np.linalg.norm(local_x_dir, axis=1, keepdims=True)
        local_x_dir = np.divide(local_x_dir, local_x_norms,
                                out=np.zeros_like(local_x_dir),
                                where=(local_x_norms > 1e-8))

        # === Step 3: 构建全局参考投影方向 ===
        # 参考方向：第一个偶数段（或首个有效点）的投影切线
        reference_tangent = None
        if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
            for i, idx in enumerate(self.scan_orig_indices):
                if idx % 2 == 0 and np.linalg.norm(local_x_dir[i]) > 1e-6:
                    reference_tangent = local_x_dir[i].copy()
                    break
        if reference_tangent is None:
            for i in range(N):
                if np.linalg.norm(local_x_dir[i]) > 1e-6:
                    reference_tangent = local_x_dir[i].copy()
                    break
        if reference_tangent is None or np.linalg.norm(reference_tangent) < 1e-6:
            reference_tangent = np.array([1.0, 0.0, 0.0])

        # 将参考方向投影到每个点的切平面（使其 ⊥ z_i）
        dot_rz = np.sum(reference_tangent * normals, axis=1, keepdims=True)
        global_x_dir = reference_tangent - dot_rz * normals  # shape: (N, 3)
        global_x_norms = np.linalg.norm(global_x_dir, axis=1, keepdims=True)
        global_x_dir = np.divide(global_x_dir, global_x_norms,
                                 out=np.zeros_like(global_x_dir),
                                 where=(global_x_norms > 1e-8))

        # === Step 4: 【加权混合】 x = (1-α) * local + α * global ===
        mixed_x = (1.0 - alpha) * local_x_dir + alpha * global_x_dir

        # === Step 5: 符号对齐（防止因加权导致整体翻转）===
        # 用参考方向检查整体朝向
        mixed_x_norms = np.linalg.norm(mixed_x, axis=1, keepdims=True)
        mixed_x = np.divide(mixed_x, mixed_x_norms,
                            out=np.zeros_like(mixed_x),
                            where=(mixed_x_norms > 1e-8))

        dot_with_ref = np.sum(mixed_x * reference_tangent, axis=1)
        flip_mask = dot_with_ref < 0
        x_axes = mixed_x.copy()
        x_axes[flip_mask] *= -1.0

        # === Step 6: 处理退化（norm ≈ 0）===
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)
        degenerate_mask = (x_norms.squeeze() < 1e-6)
        if np.any(degenerate_mask):
            print(f"⚠️ {np.sum(degenerate_mask)} 个点需 fallback（加权后退化）")
            for i in np.where(degenerate_mask)[0]:
                z = normals[i]
                # 优先尝试 local_x_dir（若有效）
                if np.linalg.norm(local_x_dir[i]) > 1e-6:
                    x_axes[i] = local_x_dir[i] if np.dot(local_x_dir[i], reference_tangent) >= 0 else -local_x_dir[i]
                else:
                    # fallback 到参考方向投影
                    proj = reference_tangent - np.dot(reference_tangent, z) * z
                    if np.linalg.norm(proj) > 1e-6:
                        x_axes[i] = proj / np.linalg.norm(proj)
                        if np.dot(x_axes[i], reference_tangent) < 0:
                            x_axes[i] *= -1
                    else:
                        cand = np.cross(z, [0, 0, 1])
                        if np.linalg.norm(cand) < 1e-6:
                            cand = np.cross(z, [1, 0, 0])
                        x_axes[i] = cand / np.linalg.norm(cand)

        # 最终归一化
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)
        x_axes = np.divide(x_axes, x_norms, out=np.zeros_like(x_axes), where=(x_norms > 1e-8))

        # === Step 7: y = z × x ===
        y_axes = np.cross(normals, x_axes)
        y_norms = np.linalg.norm(y_axes, axis=1, keepdims=True)
        y_axes = np.divide(y_axes, y_norms, out=np.zeros_like(y_axes), where=(y_norms > 1e-8))

        # === 存储 & 验证（同前，略）===
        self.local_frames = []
        for i in range(N):
            frame = {
                'origin': points[i].tolist(),
                'x_axis': x_axes[i].tolist(),
                'y_axis': y_axes[i].tolist(),
                'z_axis': normals[i].tolist()
            }
            if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
                orig_idx = int(self.scan_orig_indices[i])
                frame['orig_idx'] = orig_idx
                frame['is_odd_segment'] = bool(orig_idx % 2 == 1)
            self.local_frames.append(frame)

        # 验证（同前）
        if self.local_frames:
            even_x = odd_x = None
            for frame in self.local_frames:
                if 'orig_idx' in frame:
                    x_vec = np.array(frame['x_axis'])
                    if even_x is None and frame['orig_idx'] % 2 == 0:
                        even_x = x_vec
                    if odd_x is None and frame['orig_idx'] % 2 == 1:
                        odd_x = x_vec
                if even_x is not None and odd_x is not None:
                    dot_prod = np.dot(even_x, odd_x)
                    print(f"alpha={alpha:.2f} → 偶/奇段x点积: {dot_prod:.4f}")
                    if dot_prod < 0.95:
                        print("⚠️ 方向一致性不足（可增大 alpha）")
                    break

        print(f"✅ 已计算 {N} 个路径点的局部坐标系（alpha={alpha}，加权混合）")
        return self.local_frames
    #alpha_i = alpha_min + (alpha_max - alpha_min) * sigmoid(k * curvature_i)
    #curvature_i：点 i 的离散曲率估计（用三点夹角或曲率半径倒数）
    # k：灵敏度系数（控制过渡陡峭度）
    # alpha_min/alpha_max：α 的上下界（如 0.0 ~ 0.5）
    # 📌 为什么用 sigmoid？
    #
    # 自然平滑过渡
    # 避免曲率噪声导致 α 震荡
    # 可解释性强（阈值附近柔性切换）

    # 区域
    # 曲率
    # α
    # 实际值
    # 行为
    # 直线段
    # ≈0
    # α ≈ alpha_min = 0.0
    # 完全跟随路径走向
    # 缓弯
    # 1.0
    # α ≈ 0.12
    # 轻微向参考方向靠拢
    # 急弯 / 拐角
    # > 2.0
    # α → alpha_max = 0.5
    # 显著抑制抖动，避免
    # x
    # 翻转
    # 段切换处（奇→偶）
    # 可能突变
    # α
    # 升高
    # 保证方向平滑过渡
    # alpha_min = 0.0,  # 直段完全信任局部
    # alpha_max = 0.5,  # 弯道最多混合50%全局方向
    # curvature_k = 2.0,  # 灵敏度（可调：值大 → 更敏感）
    # use_adaptive_alpha = True  # 默认开启
    def compute_local_frames_adpative(self, alpha_min=0.0, alpha_max=0.5, curvature_k=2.0, use_adaptive_alpha=True):
        """
        自适应 α 局部坐标系计算：
          - 若 use_adaptive_alpha=True：α_i ∝ 局部曲率
          - 否则：使用固定 alpha = (alpha_min + alpha_max)/2

        Args:
            alpha_min (float): 曲率→0 时的 α 下限（直段）
            alpha_max (float): 曲率→∞ 时的 α 上限（急弯）
            curvature_k (float): sigmoid 灵敏度（越大，过渡越陡）
            use_adaptive_alpha (bool): 是否启用自适应
        """
        if self.scan_points_3d is None or len(self.scan_points_3d) == 0:
            print("⚠️ 路径点为空，无法计算局部坐标系")
            self.local_frames = []
            return

        N = len(self.scan_points_3d)
        points = np.array(self.scan_points_3d, dtype=np.float32)
        normals = np.array(self.scan_normals, dtype=np.float32)

        # === Step 1: 归一化法向量 ===
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=(norms != 0))

        # === Step 2: 计算局部切线（中心差分）===
        raw_tangents = np.zeros_like(points)
        if N == 1:
            raw_tangents[0] = np.array([1.0, 0.0, 0.0])
        else:
            raw_tangents[0] = points[1] - points[0]
            raw_tangents[-1] = points[-1] - points[-2]
            for i in range(1, N - 1):
                raw_tangents[i] = points[i + 1] - points[i - 1]
            tan_norms = np.linalg.norm(raw_tangents, axis=1, keepdims=True)
            raw_tangents = np.divide(raw_tangents, tan_norms,
                                     out=np.zeros_like(raw_tangents),
                                     where=(tan_norms != 0))

        # === Step 3: 投影切线到切平面 → 局部自然方向 ===
        dot_tz = np.sum(raw_tangents * normals, axis=1, keepdims=True)
        local_x_dir = raw_tangents - dot_tz * normals
        local_x_norms = np.linalg.norm(local_x_dir, axis=1, keepdims=True)
        local_x_dir = np.divide(local_x_dir, local_x_norms,
                                out=np.zeros_like(local_x_dir),
                                where=(local_x_norms > 1e-8))

        # === Step 4: 计算局部曲率（基于三点夹角）===
        curvatures = np.zeros(N)
        if N >= 3:
            # 用相邻三点计算转向角（越接近 π，越直；越小，越弯）
            for i in range(1, N - 1):
                v1 = points[i] - points[i - 1]
                v2 = points[i + 1] - points[i]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 1e-8 and norm2 > 1e-8:
                    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta = np.arccos(cos_theta)  # ∈ [0, π]
                    # 曲率 ≈ 转向角 / 弧长（简化：用平均边长归一）
                    avg_len = (norm1 + norm2) / 2.0
                    if avg_len > 1e-6:
                        curvatures[i] = theta / avg_len
                    else:
                        curvatures[i] = 0.0
                else:
                    curvatures[i] = 0.0
            # 首尾点：复制邻点
            curvatures[0] = curvatures[1]
            curvatures[-1] = curvatures[-2]
        else:
            curvatures[:] = 0.0  # 退化情况

        # 可视化调试（可选）
        # print(f"曲率统计: min={curvatures.min():.3f}, max={curvatures.max():.3f}, mean={curvatures.mean():.3f}")

        # === Step 5: 构建全局参考方向 & 其投影 ===
        reference_tangent = None
        if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
            for i, idx in enumerate(self.scan_orig_indices):
                if idx % 2 == 0 and np.linalg.norm(local_x_dir[i]) > 1e-6:
                    reference_tangent = local_x_dir[i].copy()
                    break
        if reference_tangent is None:
            for i in range(N):
                if np.linalg.norm(local_x_dir[i]) > 1e-6:
                    reference_tangent = local_x_dir[i].copy()
                    break
        if reference_tangent is None or np.linalg.norm(reference_tangent) < 1e-6:
            reference_tangent = np.array([1.0, 0.0, 0.0])

        # 投影参考方向到各点切平面
        dot_rz = np.sum(reference_tangent * normals, axis=1, keepdims=True)
        global_x_dir = reference_tangent - dot_rz * normals
        global_x_norms = np.linalg.norm(global_x_dir, axis=1, keepdims=True)
        global_x_dir = np.divide(global_x_dir, global_x_norms,
                                 out=np.zeros_like(global_x_dir),
                                 where=(global_x_norms > 1e-8))

        # === Step 6: 【自适应 α】计算每个点的 alpha_i ===
        if use_adaptive_alpha:
            # sigmoid 映射：curvature → [0,1]
            # 调整曲率尺度（经验：多数场景 curvature ∈ [0, 5] 即可触发高 α）
            scaled_curv = curvatures * curvature_k
            sigmoid = 1.0 / (1.0 + np.exp(-scaled_curv + 2.0))  # +2.0: 阈值偏移（在 curvature≈1.0 时 α 达中值）
            alphas = alpha_min + (alpha_max - alpha_min) * sigmoid
            # 可选：限制极值
            alphas = np.clip(alphas, alpha_min, alpha_max)
            effective_alpha = "adaptive"
        else:
            alphas = np.full(N, 0.5 * (alpha_min + alpha_max))
            effective_alpha = f"fixed={alphas[0]:.2f}"

        # === Step 7: 加权混合：x = (1-α_i) * local + α_i * global ===
        mixed_x = (1.0 - alphas[:, None]) * local_x_dir + alphas[:, None] * global_x_dir

        # 归一化
        mixed_x_norms = np.linalg.norm(mixed_x, axis=1, keepdims=True)
        mixed_x = np.divide(mixed_x, mixed_x_norms,
                            out=np.zeros_like(mixed_x),
                            where=(mixed_x_norms > 1e-8))

        # === Step 8: 符号对齐（防整体翻转）===
        dot_with_ref = np.sum(mixed_x * reference_tangent, axis=1)
        flip_mask = dot_with_ref < 0
        x_axes = mixed_x.copy()
        x_axes[flip_mask] *= -1.0

        # === Step 9: 退化处理（同前）===
        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)
        degenerate_mask = (x_norms.squeeze() < 1e-6)
        if np.any(degenerate_mask):
            print(f"⚠️ {np.sum(degenerate_mask)} 个点需 fallback")
            for i in np.where(degenerate_mask)[0]:
                z = normals[i]
                if np.linalg.norm(local_x_dir[i]) > 1e-6:
                    x_axes[i] = local_x_dir[i] if np.dot(local_x_dir[i], reference_tangent) >= 0 else -local_x_dir[i]
                else:
                    proj = reference_tangent - np.dot(reference_tangent, z) * z
                    if np.linalg.norm(proj) > 1e-6:
                        x_axes[i] = proj / np.linalg.norm(proj)
                        if np.dot(x_axes[i], reference_tangent) < 0:
                            x_axes[i] *= -1
                    else:
                        cand = np.cross(z, [0, 0, 1])
                        if np.linalg.norm(cand) < 1e-6:
                            cand = np.cross(z, [1, 0, 0])
                        x_axes[i] = cand / np.linalg.norm(cand)

        x_norms = np.linalg.norm(x_axes, axis=1, keepdims=True)
        x_axes = np.divide(x_axes, x_norms, out=np.zeros_like(x_axes), where=(x_norms > 1e-8))

        # === Step 10: y = z × x ===
        y_axes = np.cross(normals, x_axes)
        y_norms = np.linalg.norm(y_axes, axis=1, keepdims=True)
        y_axes = np.divide(y_axes, y_norms, out=np.zeros_like(y_axes), where=(y_norms > 1e-8))

        # === 存储结果 + 记录 alpha（便于调试）===
        self.local_frames = []
        for i in range(N):
            frame = {
                'origin': points[i].tolist(),
                'x_axis': x_axes[i].tolist(),
                'y_axis': y_axes[i].tolist(),
                'z_axis': normals[i].tolist(),
                'alpha_used': float(alphas[i]) if use_adaptive_alpha else float(alphas[0])
            }
            if hasattr(self, 'scan_orig_indices') and self.scan_orig_indices is not None:
                orig_idx = int(self.scan_orig_indices[i])
                frame['orig_idx'] = orig_idx
                frame['is_odd_segment'] = bool(orig_idx % 2 == 1)
            self.local_frames.append(frame)

        # === 验证（增强：报告 α 范围）===
        if self.local_frames:
            even_x = odd_x = None
            for frame in self.local_frames:
                if 'orig_idx' in frame:
                    x_vec = np.array(frame['x_axis'])
                    if even_x is None and frame['orig_idx'] % 2 == 0:
                        even_x = x_vec
                    if odd_x is None and frame['orig_idx'] % 2 == 1:
                        odd_x = x_vec
                if even_x is not None and odd_x is not None:
                    dot_prod = np.dot(even_x, odd_x)
                    alpha_range = f"[{alphas.min():.2f}, {alphas.max():.2f}]" if use_adaptive_alpha else f"{alphas[0]:.2f}"
                    print(f"α={effective_alpha} ({alpha_range}) → 偶/奇x点积: {dot_prod:.4f}")
                    if dot_prod < 0.95:
                        print("⚠️ 方向一致性不足（可增大 alpha_max）")
                    break

        print(f"✅ 已计算 {N} 个路径点的局部坐标系（α自适应：直段低α，弯道高α）")
        return self.local_frames