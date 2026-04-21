"""
SamPathPlanning - ROS2 适配版本
基于 SAM2 分割和注意力机制的超声扫描路径规划系统
支持 numpy 数组输入和 ROS2 集成

模块结构:
├── attention_path_ros2.py     # 本文件：ROS2适配的主流程
├── attention_computation.py   # 注意力热力图计算
├── pointcloud_processing.py   # 点云生成与处理
├── path_generation.py         # 路径生成算法
├── path_optimization.py       # 路径优化与插值
├── local_coordinate.py        # 局部坐标系计算
├── visualization.py           # 可视化功能
└── utils.py                   # 工具函数
"""
import os
import sys
import cv2
import numpy as np
import threading
from ultralytics import SAM
import torch

# 导入自定义模块（使用相对导入）
from .attention_computation import (
    compute_internal_attention_from_masked_pc,
    visualize_internal_attention,
    GeometricPhysicalAttention,
    AdvancedGeometricAttention,
    AnisotropicAttentionSystem,
    PhysicsAttentionV3
)
from . import attention_computation
from .pointcloud_processing import PointCloudProcessor
from .path_generation import PathGenerator
from .path_optimization import PathOptimizer
from .local_coordinate import LocalCoordinateCalculator
from .visualization import PathVisualizer
from .utils import preprocess_depth, apply_mask_shrink
from . import AttentionPathOptimizer

# 默认参数
default_shrink_factor = 12  # 掩码收缩因子（像素数）
default_InterPoins = 5  # 两两 插值点数
default_scan = 'Long'  # Long Short 长为扫查
default_spacing = 10  # 步进间隔 10


def extract_masked_roi(original_image, mask, background_color=(0, 0, 0)):
    """
    根据掩码提取感兴趣区域 (ROI)，将非掩码区域设置为指定背景色（默认为黑色）。
    
    参数:
        original_image (np.ndarray): 输入的原始彩色图像 (BGR 格式)。
        mask (np.ndarray): 掩码图像。
            - 可以是布尔数组 (True/False)
            - 可以是浮点数数组 (0.0-1.0)
            - 可以是 uint8 数组 (0-255)
        background_color (tuple): 背景颜色，默认为黑色 (0, 0, 0)。格式为 (B, G, R)。
    
    返回:
        np.ndarray: 处理后的图像，掩码区域保留原色，其他区域为背景色。
    """
    # 1. 创建图像的副本，避免修改原图
    if len(original_image.shape) == 2:
        # 如果输入是灰度图，先转为 BGR 以便统一处理
        result = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        result = original_image.copy()
    
    # 2. 标准化掩码为 uint8 格式 (0 或 255)
    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    elif mask.dtype == np.float32 or mask.dtype == np.float64:
        # 假设浮点数范围是 0.0-1.0
        if mask.max() <= 1.0:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            # 如果已经是 0-255 的浮点数
            mask_uint8 = np.clip(mask, 0, 255).astype(np.uint8)
    else:
        # 已经是 uint8，确保二值化 (防止有中间值)
        _, mask_uint8 = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3. 确保掩码尺寸与图像一致
    h, w = result.shape[:2]
    if mask_uint8.shape != (h, w):
        print(f"⚠️ 掩码尺寸 {mask_uint8.shape} 与图像尺寸 {(h, w)} 不匹配，正在调整...")
        mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)

    # 4. 创建背景色画布
    if len(result.shape) == 3:
        mask_3ch = cv2.merge([mask_uint8, mask_uint8, mask_uint8])
        
        # 计算反向掩码 (背景区域)
        inv_mask_3ch = cv2.bitwise_not(mask_3ch)
        
        # 创建纯背景色图像
        background = np.full_like(result, background_color)
        
        # 融合：结果 = (原图 & 掩码) + (背景 & 反掩码)
        foreground_part = cv2.bitwise_and(result, result, mask=mask_uint8)
        background_part = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask_uint8))
        
        final_result = cv2.add(foreground_part, background_part)
        
    else:
        # 如果是单通道意外情况
        final_result = result
        final_result[mask_uint8 == 0] = background_color[0]

    return final_result


class InteractiveSegmentationROS2:
    """ROS2 适配的交互式分割与路径规划主类"""
    
    def __init__(self, depth_path=None, depth_array=None, fx=498.3686770748583, 
                 fy=501.9355502582987, cx=314.3019441792476, cy=225.6695918834769,
                 model_path=None):
        """
        初始化交互式分割系统 - ROS2 适配版本
        
        Args:
            depth_path: 深度图文件路径（可选，与 depth_array 二选一）
            depth_array: 深度图 numpy 数组（可选，与 depth_path 二选一）
            fx, fy, cx, cy: 相机内参
            model_path: SAM2 模型路径
        """
        # 初始化模型
        self.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        if model_path is None:
            model_path = os.path.join(self.dir, 'sam2model', 'sam2.1_l.pt')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            # 尝试其他可能的路径
            alt_paths = [
                os.path.join(self.dir, 'Sam2Model', 'sam2.1_l.pt'),
                '/home/dts/agv_git/src/pathplannernode/src/pathplanner/sam2model/sam2.1_l.pt',
                './sam2model/sam2.1_l.pt',
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"找到模型文件: {model_path}")
                    break
            else:
                raise FileNotFoundError(f"❌ 找不到SAM2模型文件: {model_path}\n"
                                        f"请确保 sam2.1_l.pt 存在于以下位置之一:\n"
                                        f"  - {os.path.join(self.dir, 'sam2model', 'sam2.1_l.pt')}\n"
                                        f"  - {os.path.join(self.dir, 'Sam2Model', 'sam2.1_l.pt')}")
        
        print(f"正在加载SAM2模型: {model_path}")
        self.model = SAM(model_path)
        
        # 状态变量
        self.click_point = None
        self.current_mask = None
        self.original_image = None
        self.shrink_factor = default_shrink_factor
        
        # 相机内参
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # 点云相关
        self.pointcloud = None
        self.pointcloud_colors = None
        self.normals = None
        self.pixel_to_index = None
        self.scan_points_3d = None
        self.scan_Points = None
        self.scan_normals = None
        self.scan_points = None
        self.PathPoint = None
        self.local_frames = None
        self.scan_orig_indices = None
        
        # 注意力相关
        self.maskroi = None
        self.use_attention_density = True
        self.use_path_smoothing = False
        self.attention_map = None
        
        # 加载深度图（支持文件路径或 numpy 数组）
        if depth_array is not None:
            print("---1使用传入的深度图数组---")
            self.depth_map = preprocess_depth(depth_array)
        elif depth_path is not None:
            print(f"---1正在加载深度图：{depth_path}---")
            self.depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            self.depth_map = preprocess_depth(self.depth_map)
        else:
            self.depth_map = None
            print("---1深度图未初始化，将在处理时设置---")
        
        if self.depth_map is not None:
            if len(self.depth_map.shape) == 3:
                print("⚠️ 深度图是多通道，正在提取第 1 通道...")
                self.depth_map = self.depth_map[:, :, 0]
            
            print(f"---✅ 深度图加载成功：{self.depth_map.shape}, dtype={self.depth_map.dtype}, "
                  f"min={self.depth_map.min()}, max={self.depth_map.max()}---")
        
        print(f"---2🔧 掩码收缩设置：{self.shrink_factor} 像素---")
        
        # 初始化各模块处理器
        self.pc_processor = PointCloudProcessor(self.fx, self.fy, self.cx, self.cy)
        self.path_generator = PathGenerator(scan_mode=default_scan, spacing=default_spacing)
        self.path_optimizer = PathOptimizer()
        self.coord_calculator = LocalCoordinateCalculator(method='weighted')
        self.visualizer = PathVisualizer()
        self.attention_optimizer = None
    
    def set_depth(self, depth_array):
        """动态设置深度图（用于 ROS2 实时数据）"""
        self.depth_map = preprocess_depth(depth_array)
        if len(self.depth_map.shape) == 3:
            self.depth_map = self.depth_map[:, :, 0]
        print(f"---✅ 深度图已更新：{self.depth_map.shape}---")
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = [x, y]
            print(f"点击坐标：({x}, {y})")
            self.perform_segmentation(enable_visualization=True)
    
    def perform_segmentation(self, enable_visualization=True):
        """执行分割"""
        if self.click_point and self.original_image is not None:
            print("正在分割...")
            try:
                results = self.model(
                    self.original_image,
                    points=[self.click_point],
                    labels=[1],
                    device="cpu" if not torch.cuda.is_available() else "cuda",
                    retina_masks=True,
                    conf=0.3,
                    verbose=False
                )

                if len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'masks') and result.masks is not None:
                        masks = result.masks.data.cpu().numpy()
                        if len(masks) > 0:
                            self.current_mask = masks[0]
                            print("分割完成，正在显示结果...", len(results))
                            self.visualize_result(self.current_mask, enable_visualization)
                    else:
                        print("未检测到掩码")
                else:
                    print("未检测到分割结果")
            except Exception as e:
                print(f"分割过程中出现错误：{e}")
                import traceback
                traceback.print_exc()
    
    def process_pipeline(self, color_image=None, color_path=None, click_point=None, 
                         auto_center=False, enable_visualization=True):
        """
        完整处理流程 - ROS2 适配版本
        
        Args:
            color_image: 彩色图像 numpy 数组（可选，与 color_path 二选一）
            color_path: 彩色图像文件路径（可选，与 color_image 二选一）
            click_point: 指定点击点坐标 [x, y]，为None时使用交互模式或自动中心
            auto_center: 是否自动使用图像中心作为点击点
            enable_visualization: 是否启用可视化（cv2.imshow）
            
        Returns:
            scan_points: 2D 路径点
            local_frames: 局部坐标系
            scan_points_3d: 3D 路径点
        """
        # 1.加载彩色图（支持 numpy 数组或文件路径）
        if color_image is not None:
            self.original_image = color_image.copy()
            print("---使用传入的彩色图像数组---")
        elif color_path is not None:
            self.original_image = cv2.imread(color_path)
            if self.original_image is None:
                raise FileNotFoundError(f"无法加载彩色图：{color_path}")
            print(f"---从文件加载彩色图：{color_path}---")
        else:
            raise ValueError("必须提供 color_image 或 color_path 之一")
        
        # 检查深度图是否已设置
        if self.depth_map is None:
            raise ValueError("深度图未设置，请先调用 set_depth() 或在初始化时提供")
        
        # 判断使用哪种模式
        if click_point is not None:
            # 模式1: 指定点击点（ROS2模式）
            self.click_point = click_point
            print(f"使用指定点击点: ({click_point[0]}, {click_point[1]})")
            self.perform_segmentation(enable_visualization)
        elif auto_center:
            # 模式2: 自动使用图像中心
            h, w = self.original_image.shape[:2]
            self.click_point = [w // 2, h // 2]
            print(f"自动使用图像中心作为点击点: ({self.click_point[0]}, {self.click_point[1]})")
            self.perform_segmentation(enable_visualization)
        else:
            # 模式3: 交互模式（弹窗等待点击）
            if enable_visualization:
                cv2.imshow('Click to Segment', self.original_image)
                cv2.setMouseCallback('Click to Segment', self.mouse_callback)
                print("请在图像上点击选择分割区域...")
                cv2.waitKey(0)
                cv2.destroyWindow('Click to Segment')
            else:
                print("❌ 未指定点击点且未启用可视化，无法继续")
                return None, None, None
        
        if self.current_mask is None:
            print("❌ 未进行分割，无法继续")
            return None, None, None
        
        roi_img = extract_masked_roi(self.original_image, self.current_mask, background_color=(0, 0, 0))
        if enable_visualization:
            cv2.imshow("roi", roi_img)
            if threading.current_thread() is threading.main_thread():
                cv2.waitKey(500)
            else:
                cv2.waitKey(1)
            cv2.destroyWindow("roi")
            cv2.imwrite("roi.png", roi_img)
        
        # 2.生成彩色点云
        print(f"---3正在生成点云---")
        mask_resized = cv2.resize(self.current_mask.astype(np.float32),
                                  (self.original_image.shape[1], self.original_image.shape[0]))
        self.pointcloud, self.pointcloud_colors = self.pc_processor.mask_depth_to_color_pointcloud(
            mask_resized, self.depth_map, self.original_image
        )
        
        # 检查点云是否成功生成
        if self.pointcloud is None or len(self.pointcloud) == 0:
            print("❌ 点云生成失败，无法继续")
            return None, None, None
        
        # 3.计算点云法向量
        self.calculate_normals()
        
        # 4.生成rgb路径
        print(f"---5正在生成路径点---")
        print(f"   路径生成模式：{self.path_generator.scan_mode}, 步进间隔：{self.path_generator.spacing} 像素")
        contour, self.scan_points = self.path_generator.generate_from_mask(mask_resized)
        print(f"---✅ rgb路径点完成。初始路径点数: {len(self.scan_points)}---")
        
        # 检查路径点是否生成
        if not self.scan_points or len(self.scan_points) == 0:
            print("❌ 路径点生成失败，无法继续")
            return None, None, None
        
        # 5. 可视化rgb路径
        if enable_visualization:
            print(f"---5正在可视化路径点---")
            self.visualizer.visualize_contour_path(
                self.original_image, contour, self.scan_points,
                save=True
            )
            print(f"---✅ 路径点可视化完成---")

        # 6.插值
        print(f"---6正在插值路径点,两两点插值间隔: {default_InterPoins}---")
        self.scan_points, segment_info = self.path_optimizer.interpolate_scan_points(
            self.scan_points, points_per_segment=default_InterPoins
        )
        print(f"---✅插值完成，插值后路径点数: {len(self.scan_points)}---")

        # 7.将路径点映射回3D空间
        print(f"---7正在将路径点映射回3D空间---")
        self.scan_points_3d, self.scan_normals, self.scan_Points, \
            self.scan_orig_indices, self.pixel_to_index = self.path_optimizer.map_2d_to_3d(
                self.scan_points, mask_resized, self.pc_processor.get_point_and_normal, 
                segment_info=segment_info
            )
        print(f"---✅映射完成，映射后路径点数: {len(self.scan_points_3d)}---")
        
        # 8. 计算局部坐标系
        print(f"---8正在计算局部坐标系---")
        self.compute_local_frames()
        print(f"---✅ 局部坐标系计算完成，局部坐标系数量: {len(self.local_frames)}---")
        
        # 9. 可视化点云
        if enable_visualization:
            print(f"---9正在可视化点云---")
            self.visualizer.visualize_color_pointcloud(
                self.pointcloud, 
                colors=self.pointcloud_colors,
                normals=self.normals,
                scan_points_3d=self.scan_points_3d,
                scan_normals=self.scan_normals,
                local_frames=self.local_frames
            )
            print(f"---✅ 点云可视化完成---")
        
        # 保存点云（无论是否可视化）
        self.save_pointcloud()

        # # 计算注意力热力图
        # attention_switch = "EGA"  # Geometric, Anisotropic, Advanced, EGA
        # if attention_switch == "Advanced":
        #     researcher2 = AdvancedGeometricAttention(k_neighbors=45)
        #     self.attention_map, att_vals = researcher2.compute_attention(
        #         self.pointcloud, self.normals, self.pixel_to_index, (480, 640)
        #     )
        # elif attention_switch == "Geometric":
        #     researcher = GeometricPhysicalAttention(k_neighbors=45)
        #     self.attention_map, att_vals = researcher.compute_attention(
        #         self.pointcloud, self.normals, self.pixel_to_index, (480, 640)
        #     )
        # elif attention_switch == "Anisotropic":
        #     researcher3 = AnisotropicAttentionSystem(k_neighbors=45)
        #     self.attention_map = researcher3.compute_aat_attention(
        #         self.pointcloud, self.pixel_to_index, (480, 640)
        #     )
        # elif attention_switch == "EGA":
        #     researcher3 = attention_computation.EntropyGuidedAttentionSystem(200)
        #     self.attention_map = researcher3.compute_aat_attention(
        #         self.pointcloud, self.pixel_to_index, (480, 640)
        #     )
        # else:
        #     researcher4 = PhysicsAttentionV3(45)
        #     self.attention_map, att_vals = researcher4.compute_attention(
        #         self.pointcloud, self.normals, self.pixel_to_index, (480, 640)
        #     )
        
        # # Attention 优化
        # self.attention_optimizer = AttentionPathOptimizer.AttentionPathOptimizer(
        #     drift_radius=2, densify_threshold=0.6, extra_points=4
        # )
        # self.scan_points, segment_info = self.attention_optimizer.optimize(
        #     self.scan_points, segment_info, self.attention_map
        # )
        # print(f"---✅ Attention 优化完成，最终自适应路径点数: {len(self.scan_points)}---")

        # # 重新映射到3D空间
        # print(f"---7正在将优化后的路径点映射回3D空间---")
        # self.scan_points_3d, self.scan_normals, self.scan_Points, \
        #     self.scan_orig_indices, self.pixel_to_index = self.path_optimizer.map_2d_to_3d(
        #         self.scan_points, mask_resized, self.pc_processor.get_point_and_normal, 
        #         segment_info=segment_info, attention_ture=True
        #     )
        # print(f"---✅映射完成，映射后路径点数: {len(self.scan_points_3d)}---")
        
        # # 重新计算局部坐标系
        # print(f"---8正在计算局部坐标系---")
        # self.compute_local_frames()
        # print(f"---✅ 局部坐标系计算完成，局部坐标系数量: {len(self.local_frames)}---")

        # if self.use_attention_density:
        #     # 可视化注意力
        #     mask_binary = (mask_resized > 0.5).astype(np.uint8)
        #     masked_image = self.original_image.copy()
        #     masked_image[mask_binary == 0] = [0, 0, 0]
        #     self.maskroi = masked_image
            
        #     if enable_visualization:
        #         vis_img = visualize_internal_attention(
        #             masked_image=masked_image,
        #             attention_map=self.attention_map,
        #             alpha=0.5,
        #             highlight_percentile=95,
        #             draw_contours=True,
        #             mask_binary=mask_binary
        #         )
        # else:
        #     # 即使不计算注意力，也生成 maskroi 用于后续显示
        #     mask_binary = (mask_resized > 0.5).astype(np.uint8)
        #     self.maskroi = self.original_image.copy()
        #     self.maskroi[mask_binary == 0] = [0, 0, 0]
        
        # # 显示点云
        # if enable_visualization:
        #     print(f"---9正在可视化点云---")
        #     self.visualizer.visualize_color_pointcloud(
        #         self.pointcloud, 
        #         colors=self.pointcloud_colors,
        #         normals=self.normals,
        #         scan_points_3d=self.scan_points_3d,
        #         scan_normals=self.scan_normals,
        #         local_frames=self.local_frames
        #     )
        #     print(f"---✅ 点云可视化完成---")
        
        return self.scan_points, self.local_frames, self.scan_points_3d
    
    def calculate_normals(self):
        """计算法向量（委托给 PointCloudProcessor）"""
        self.normals = self.pc_processor.calculate_normals(
            self.pointcloud, 
            self.pointcloud_colors
        )
    
    def compute_local_frames(self):
        """计算局部坐标系（委托给 LocalCoordinateCalculator）"""
        if self.scan_points_3d is None or len(self.scan_points_3d) == 0:
            print("❌ 没有3D路径点，无法计算局部坐标系")
            return
        
        # 调用 LocalCoordinateCalculator 的 compute 方法
        # 返回的是字典列表格式，与 visualization.py 兼容
        self.local_frames = self.coord_calculator.compute(
            self.scan_points_3d,
            self.scan_normals,
            self.scan_orig_indices
        )
    def local_frames_to_scan_points(self):
        """将局部坐标系转换为输出格式"""
        return self.coord_calculator.frames_to_scan_points(self.local_frames)
    
    def save_pointcloud(self):
        """保存点云到文件"""
        try:
            import open3d as o3d
            import os
            
            if self.pointcloud is None or len(self.pointcloud) == 0:
                print("⚠️ 点云为空，无法保存")
                return
            
            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.pointcloud)
            
            if self.pointcloud_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(self.pointcloud_colors)
            
            if self.normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(self.normals)
            
            # 保存路径：使用绝对路径 /home/zyj/Code/pathplannernode/result/output.ply
            result_dir = '/home/zyj/Code/pathplannernode/result'
            os.makedirs(result_dir, exist_ok=True)
            
            save_path = os.path.join(result_dir, "output.ply")
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"💾 点云已保存: {save_path} ({len(self.pointcloud)} 个点)")
            
        except Exception as e:
            print(f"❌ 保存点云失败: {e}")
    
    def visualize_result(self, mask, enable_visualization=True):
        """可视化分割结果（委托给 PathVisualizer）"""
        self.current_mask = apply_mask_shrink(self.shrink_factor, mask)
        self.visualizer.visualize_segmentation_result(
            self.original_image,
            self.current_mask,
            self.click_point,
            save=True,
            show=enable_visualization
        )


def main(input_path, output_path):
    """
    主函数
    
    Args:
        input_path: 输入路径（包含 depth.png 和 color.png）
        output_path: 输出路径
    """
    input_path_str = str(input_path)
    depth_path = os.path.join(input_path_str, "depth.png")
    color_path = os.path.join(input_path_str, "color.png")
    
    # 创建交互式分割实例
    interactive_seg = InteractiveSegmentationROS2(depth_path=depth_path)
    
    # 运行完整流程
    interactive_seg.process_pipeline(color_path=color_path)
    
    return interactive_seg.scan_points_3d


if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) < 2:
        print("用法：python attention_path_ros2.py <input_folder>")
        print("输入文件夹应包含 depth.png 和 color.png")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = "./result"
    os.makedirs(output_folder, exist_ok=True)

    print(f"可选参数配置：掩码收缩器{default_shrink_factor},路径点数{default_InterPoins}，扫描方式{default_scan}，步进间隔{default_spacing}")
    
    result = main(input_folder, output_folder)
    print(f"✅ 处理完成，生成 {len(result)} 个路径点")
