import numpy as np
import cv2

class AttentionPathOptimizer:
    """
    基于物理 Attention 引导的路径优化器 v2.0
    修复了长线段中间高分区域被漏检的问题，引入全线段像素级扫描验证。
    """
    def __init__(self, drift_radius=2, densify_threshold=0.6, extra_points=8):
        """
        Args:
            drift_radius (int): 引力漂移的搜索半径（不要太大，建议1或2，否则容易拉扯变形）。
            densify_threshold (float): 触发加密的阈值。建议调低到 0.5~0.7 左右，以便捕获黄色区域。
            extra_points (int): 遇到高 Attention 区域时，额外插入的点数。
        """
        self.drift_radius = drift_radius
        self.densify_threshold = densify_threshold
        self.extra_points = extra_points

    def _get_max_attention_on_line(self, p0, p1, attention_map):
        """
        [核心新增] 检查 p0 到 p1 这条线段所经过的所有像素，找出最高的 Attention 分数。
        这能保证即使 p0 和 p1 都在低分区，只要中间穿过了高分区，也能被精准捕获！
        """
        x0, y0 = p0
        x1, y1 = p1
        
        # 计算线段的像素长度
        length = int(np.hypot(x1 - x0, y1 - y0))
        if length <= 1:
            return attention_map[y0, x0]
            
        # 沿着线段生成一系列采样点坐标
        x_vals = np.linspace(x0, x1, length).astype(int)
        y_vals = np.linspace(y0, y1, length).astype(int)
        
        # 防止越界
        h, w = attention_map.shape
        x_vals = np.clip(x_vals, 0, w - 1)
        y_vals = np.clip(y_vals, 0, h - 1)
        
        # 批量获取这条线上所有像素的 Attention 分数
        line_scores = attention_map[y_vals, x_vals]
        
        # 返回这条线上的最大分数
        return np.max(line_scores)

    def optimize(self, scan_points, segment_info, attention_map):
        if not scan_points or attention_map is None:
            return scan_points, segment_info

        h, w = attention_map.shape
        
        # ==========================================
        # 第一阶段：引力漂移 (Gravitational Drift)
        # ==========================================
        drifted_points = []
        for x, y in scan_points:
            x0 = max(0, x - self.drift_radius)
            x1 = min(w - 1, x + self.drift_radius)
            y0 = max(0, y - self.drift_radius)
            y1 = min(h - 1, y + self.drift_radius)
            
            patch = attention_map[y0:y1+1, x0:x1+1]
            if patch.size == 0:
                drifted_points.append((x, y))
                continue
                
            max_idx = np.unravel_index(np.argmax(patch), patch.shape)
            max_val = patch[max_idx]
            current_val = attention_map[y, x]
            
            if max_val > current_val + 1e-3:
                new_x = x0 + max_idx[1]
                new_y = y0 + max_idx[0]
                drifted_points.append((new_x, new_y))
            else:
                drifted_points.append((x, y))

        # ==========================================
        # 第二阶段：全线段自适应加密
        # ==========================================
        final_points = []
        final_info = []
        
        for i in range(len(drifted_points) - 1):
            p0 = drifted_points[i]
            p1 = drifted_points[i+1]
            info0 = segment_info[i]
            info1 = segment_info[i+1]
            
            final_points.append(p0)
            final_info.append(info0)
            
            # 【关键修改】不再只看两头，而是看整条线段最高分！
            line_max_att = self._get_max_attention_on_line(p0, p1, attention_map)
            
            # 只要这条线碰到了高分区，就给它塞满点
            if line_max_att >= self.densify_threshold:
                for j in range(1, self.extra_points + 1):
                    t_sub = j / (self.extra_points + 1.0)
                    new_x = int(round(p0[0] * (1 - t_sub) + p1[0] * t_sub))
                    new_y = int(round(p0[1] * (1 - t_sub) + p1[1] * t_sub))
                    final_points.append((new_x, new_y))
                    
                    new_t = info0['t'] * (1 - t_sub) + info1['t'] * t_sub
                    final_info.append({
                        'orig_idx': info0['orig_idx'],
                        'is_original': False,
                        't': float(new_t)
                    })
                    
        final_points.append(drifted_points[-1])
        final_info.append(segment_info[-1])
        
        return final_points, final_info