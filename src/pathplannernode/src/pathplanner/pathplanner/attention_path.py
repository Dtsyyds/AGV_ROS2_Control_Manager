"""
SamPathPlanning - 主入口文件
基于 SAM2 分割和注意力机制的超声扫描路径规划系统

模块结构:
├── attention_path.py          # 本文件：主流程和交互逻辑
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

    # 4. 创建背景色画布 (可选优化：直接操作像素更高效，但此方法逻辑清晰)
    # 方法 A: 直接将非掩码区域设为背景色 (推荐，速度快)
    # 构建一个三通道掩码以便广播
    if len(result.shape) == 3:
        mask_3ch = cv2.merge([mask_uint8, mask_uint8, mask_uint8])
        
        # 计算反向掩码 (背景区域)
        inv_mask_3ch = cv2.bitwise_not(mask_3ch)
        
        # 创建纯背景色图像
        background = np.full_like(result, background_color)
        
        # 融合：结果 = (原图 & 掩码) + (背景 & 反掩码)
        # 注意：bitwise_and 对于 uint8 有效
        foreground_part = cv2.bitwise_and(result, result, mask=mask_uint8)
        background_part = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask_uint8))
        
        final_result = cv2.add(foreground_part, background_part)
        
    else:
        # 如果是单通道意外情况
        final_result = result
        final_result[mask_uint8 == 0] = background_color[0]

    return final_result


class InteractiveSegmentation:
    """交互式分割与路径规划主类"""
    
    def __init__(self, depth_path):
        """
        初始化交互式分割系统
        
        Args:
            depth_path: 深度图路径
        """
        # 初始化模型
        self.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        model_path = os.path.join(self.dir, 'sam2model', 'sam2.1_l.pt')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            # 尝试其他可能的路径
            alt_paths = [
                os.path.join(self.dir, 'Sam2Model', 'sam2.1_l.pt'),
                '/home/zyj/Code/pathplannernode/src/pathplanner/sam2model/sam2.1_l.pt',
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
        self.click_point = None #
        self.current_mask = None  #掩码
        self.original_image = None
        self.shrink_factor = default_shrink_factor
        
        # 相机内参
        self.fx = 498.3686770748583
        self.fy = 501.9355502582987
        self.cx = 314.3019441792476
        self.cy = 225.6695918834769
        
        # 点云相关
        self.pointcloud = None # 3D 点云
        self.pointcloud_colors = None # 点云颜色
        self.normals = None# 法线
        self.pixel_to_index = None# 像素索引
        self.scan_points_3d = None# 3D 路径点
        self.scan_Points = None# 3D 扫描点
        self.scan_normals = None#3d路径的点的法向量
        self.scan_points = None# 2d rgb路径点
        self.PathPoint = None
        self.local_frames = None# 3d路径点的局部坐标系
        self.scan_orig_indices=None
        
        # 注意力相关
        self.maskroi = None
        self.use_attention_density = True  # 是否计算注意力热力图
        self.use_path_smoothing = False
        self.attention_map = None
        
        # 加载深度图
        print(f"---1正在加载深度图：{depth_path}---")
        self.depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        self.depth_map = preprocess_depth(self.depth_map)
        
        if self.depth_map is None:
            raise FileNotFoundError(f"❌ 无法加载深度图：{depth_path}")
        
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
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = [x, y]
            print(f"点击坐标：({x}, {y})")
            self.perform_segmentation()
    
    def perform_segmentation(self):
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
                            self.visualize_result(self.current_mask)
                    else:
                        print("未检测到掩码")
                else:
                    print("未检测到分割结果")
            except Exception as e:
                print(f"分割过程中出现错误：{e}")
                import traceback
                traceback.print_exc()
    
    def process_pipeline(self, color_path, click_point=None, auto_center=False):
        """
        完整处理流程
        
        Args:
            color_path: 彩色图像路径
            click_point: 指定点击点坐标 [x, y]，为None时使用交互模式或自动中心
            auto_center: 是否自动使用图像中心作为点击点
            
        Returns:
            scan_points: 2D 路径点
            local_frames: 局部坐标系
        """
        # 1.加载彩色图
        self.original_image = cv2.imread(color_path)
        if self.original_image is None:
            raise FileNotFoundError(f"无法加载彩色图：{color_path}")
        
        # 判断使用哪种模式
        if click_point is not None:
            # 模式1: 指定点击点（ROS2模式）
            self.click_point = click_point
            print(f"使用指定点击点: ({click_point[0]}, {click_point[1]})")
            self.perform_segmentation()
        elif auto_center:
            # 模式2: 自动使用图像中心
            h, w = self.original_image.shape[:2]
            self.click_point = [w // 2, h // 2]
            print(f"自动使用图像中心作为点击点: ({self.click_point[0]}, {self.click_point[1]})")
            self.perform_segmentation()
        else:
            # 模式3: 交互模式（弹窗等待点击）
            cv2.imshow('Click to Segment', self.original_image)
            cv2.setMouseCallback('Click to Segment', self.mouse_callback)
            print("请在图像上点击选择分割区域...")
            cv2.waitKey(0)
        
        if self.current_mask is None:
            print("❌ 未进行分割，无法继续")
            return None, None
        
        roi_img = extract_masked_roi(self.original_image, self.current_mask, background_color=(0, 0, 0))
        cv2.imshow("roi", roi_img)
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
            return None, None
        
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
            return None, None
        
        #5. 可视化rgb路径
        print(f"---5正在可视化路径点---")
        self.visualizer.visualize_contour_path(
            self.original_image, contour, self.scan_points,
            save=True
        )
        print(f"---✅ 路径点可视化完成---")

        #6.插值
        print(f"---6正在插值路径点,两两点插值间隔: {default_InterPoins}---")
        self.scan_points,segment_info = self.path_optimizer.interpolate_scan_points(self.scan_points, points_per_segment=default_InterPoins)
        print(f"---✅插值完成，插值后路径点数: {len(self.scan_points)}---")

        # 7.将路径点映射回3D空间
        print(f"---7正在将路径点映射回3D空间---")
        self.scan_points_3d,self.scan_normals,self.scan_Points,self.scan_orig_indices,self.pixel_to_index=self.path_optimizer.map_2d_to_3d(self.scan_points, mask_resized, self.pc_processor.get_point_and_normal, segment_info=segment_info)
        print("---✅映射完成，映射后路径点数: {len(self.scan_points_3d)}---")
        
        #8. 计算局部坐标系
        print(f"---8正在计算局部坐标系---")
        self.compute_local_frames()
        print(f"---✅ 局部坐标系计算完成，局部坐标系数量: {len(self.local_frames)}---")
        

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

        
         
           # 计算注意力热力图（可选）
        #
        attention_switch="EGA"  #Geometric  #  Anisotropic  Advanced  EGA
        if attention_switch=="Advanced":
            researcher2 =AdvancedGeometricAttention(k_neighbors=45)
            self.attention_map, att_vals = researcher2.compute_attention(self.pointcloud, self.normals, self.pixel_to_index, (480, 640))
        elif attention_switch=="Geometric":
            researcher = GeometricPhysicalAttention(k_neighbors=45)
            self.attention_map, att_vals = researcher.compute_attention(self.pointcloud, self.normals, self.pixel_to_index, (480, 640))
        elif attention_switch=="Anisotropic":
            researcher3 = AnisotropicAttentionSystem(k_neighbors=45)
            self.attention_map = researcher3.compute_aat_attention(self.pointcloud, self.pixel_to_index, (480, 640))
        elif attention_switch=="EGA":
            researcher3 = attention_computation.EntropyGuidedAttentionSystem(200) #500 45 200
            self.attention_map = researcher3.compute_aat_attention(self.pointcloud, self.pixel_to_index, (480, 640))
        else:
            researcher4=PhysicsAttentionV3(45)
            self.attention_map, att_vals = researcher4.compute_attention(self.pointcloud, self.normals, self.pixel_to_index, (480, 640))
        #0.6 Attention 优化
        self.attention_optimizer=AttentionPathOptimizer.AttentionPathOptimizer(drift_radius=2, densify_threshold=0.6, extra_points=4)
        self.scan_points, segment_info = self.attention_optimizer.optimize(self.scan_points, segment_info, self.attention_map)
        print(f"---✅ Attention 优化完成，最终自适应路径点数: {len(self.scan_points)}---")



         # 7.将路径点映射回3D空间
        print(f"---7正在将路径点映射回3D空间---")
        self.scan_points_3d,self.scan_normals,self.scan_Points,self.scan_orig_indices,self.pixel_to_index=self.path_optimizer.map_2d_to_3d(self.scan_points, mask_resized, self.pc_processor.get_point_and_normal, segment_info=segment_info,attention_ture=True)
        print("---✅映射完成，映射后路径点数: {len(self.scan_points_3d)}---")
        
        #8. 计算局部坐标系
        print(f"---8正在计算局部坐标系---")
        self.compute_local_frames()
        print(f"---✅ 局部坐标系计算完成，局部坐标系数量: {len(self.local_frames)}---")

        # aa=attention_computation.HighOrderGeometricExtractor(self.pointcloud)
        # aa.compute_don(0.01,0.02)
        # aa.compute_normal_roughness(30)
        # aa.compute_shape_index(20)
        # aa=attention_computation.HighOrderGeometricExtractor(self.pointcloud)
        # # get_roughness_image
        # self.attention_map= aa.get_roughness_image(self.pointcloud, self.pixel_to_index, (480, 640))
        
        # rea=attention_computation.RobustPhysicsAttentionV4(k_neighbors=40)
        # self.attention_map, att_vals = rea.compute_attention(self.pointcloud, self.normals, self.pixel_to_index, (480, 640))

        if self.use_attention_density:
        #     self.attention_map, _ = compute_internal_attention_from_masked_pc(
        #         mask=mask_resized,
        #         pointcloud=self.pointcloud,
        #         normals=self.normals,
        #         pixel_to_index=self.pixel_to_index,
        #         k_neighbors=32,
        #         radius=2,
        #         weight_normal_std=0.0,
        #         weight_linearity=1.0,
        #         min_neighbors=6,
        #         depth_map=self.depth_map,
        #     )
            
            # 可视化注意力
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            masked_image = self.original_image.copy()
            masked_image[mask_binary == 0] = [0, 0, 0]
            self.maskroi = masked_image
            
            vis_img = visualize_internal_attention(
                masked_image=masked_image,
                attention_map=self.attention_map,
                alpha=0.5,
                highlight_percentile=95,
                draw_contours=True,
                mask_binary=mask_binary
            )
        else:
            # 即使不计算注意力，也生成 maskroi 用于后续显示
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            self.maskroi = self.original_image.copy()
            self.maskroi[mask_binary == 0] = [0, 0, 0]
         
         # 显示点云
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

        
        
        
        # 转换为输出格式
        scan_points_full = self.local_frames_to_scan_points()
        
        return self.scan_points, self.local_frames
    
    def calculate_normals(self):
        """计算法向量（委托给 PointCloudProcessor）"""
        self.normals = self.pc_processor.calculate_normals(
            self.pointcloud, 
            self.pointcloud_colors
        )
    
    def compute_local_frames(self):
        """计算局部坐标系（委托给 LocalCoordinateCalculator）"""
        if self.scan_points_3d is None or len(self.scan_points_3d) == 0:
            print("⚠️ 3D 路径点为空，无法计算局部坐标系")
            self.local_frames = []
            return
        
        self.local_frames = self.coord_calculator.compute(
            self.scan_points_3d,
            self.scan_normals,
            self.scan_orig_indices if hasattr(self, 'scan_orig_indices') else None,
            alpha=1.0
        )
    
    def local_frames_to_scan_points(self):
        """将局部坐标系转换为输出格式"""
        return self.coord_calculator.frames_to_scan_points(self.local_frames)
    
    def visualize_result(self, mask):
        """可视化分割结果（委托给 PathVisualizer）"""
        self.current_mask=apply_mask_shrink(self.shrink_factor,mask)
        self.visualizer.visualize_segmentation_result(
            self.original_image, 
            self.current_mask, 
            self.click_point,
            save=True
        )


def GetEdge(maskroi, mask, target_color=(0, 255, 0)):
    """获取边缘（委托给 PathVisualizer）"""
    # 检查 maskroi 是否为 None
    if maskroi is None:
        print("⚠️ maskroi 为 None，无法绘制边缘")
        return None
    
    visualizer = PathVisualizer()
    return visualizer.draw_edge_on_mask(maskroi, mask, target_color)




def main(input_path, output_path):
    """
    主函数
    
    Args:
        input_path: 输入路径（包含 depth.png 和 color.png）
        output_path: 输出路径
    """
    #capture_2_depth capture_2_color gong
    input_path_str = str(input_path)
    depth_path = os.path.join(input_path_str, "depth.png")
    color_path = os.path.join(input_path_str, "color.png")
    
    # 创建交互式分割实例
    interactive_seg = InteractiveSegmentation(depth_path)
    
    # 运行完整流程
    interactive_seg.process_pipeline(color_path)
    
    # 获取轮廓
    GetEdge(interactive_seg.maskroi, interactive_seg.current_mask)
    
    return interactive_seg.local_frames_to_scan_points()


if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) < 2:
        print("用法：python attention_path.py <input_folder>")
        print("输入文件夹应包含 depth.png 和 color.png")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = "./result"
    os.makedirs(output_folder, exist_ok=True)

# default_shrink_factor = 10  # 掩码收缩因子（像素数）
# default_InterPoins = 5  # 两两 插值点数
# default_scan = 'Long'  # Long Short 长为扫查
# default_spacing = 10  # 步进间隔 10
    print(f"可选参数配置：掩码收缩器{default_shrink_factor},路径点数{default_InterPoins}，扫描方式{default_scan}，步进间隔{default_spacing}")
    
    result = main(input_folder, output_folder)
    print(f"✅ 处理完成，生成 {len(result)} 个路径点")
