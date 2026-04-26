"""
可视化模块
包含点云、路径、注意力热力图、分割结果的可视化功能
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
import os
import threading

class PathVisualizer:
    """路径可视化工具类"""
    
    def __init__(self):
        """初始化可视化工具"""
        # 获取项目根目录（pathplannernode）
        current_file = os.path.abspath(__file__)
        # pathplanner/pathplanner/visualization.py -> 向上3层到项目根目录
        self.dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        pass
    
    def visualize_contour_path(self, image, contour, scan_points, 
                                save=False, show=True):
        """
        可视化轮廓路径
        
        Args:
            image: 原始图像
            contour: 轮廓
            scan_points: 路径点列表
            save_path: 保存路径
            show: 是否显示
            
        Returns:
            image_with_path: 绘制后的图像
        """
        image_with_path = image.copy()
        
        # 绘制轮廓
        cv2.drawContours(image_with_path, [contour], 0, (0, 255, 255), 2)

        # 绘制路径点
        if scan_points:
            # 绘制所有路径点
            for pt in scan_points:
                cv2.circle(image_with_path, pt, 2, (255, 0, 0), -1)
            
            # 重新绘制起点（红色）和终点（蓝色），用更大半径覆盖
            if len(scan_points) >= 1:
                start_pt = scan_points[0]
                end_pt = scan_points[-1]

                # 起点：红色，半径 5（更醒目）
                cv2.circle(image_with_path, start_pt, 5, (0, 0, 255), -1)  # Red (BGR)
                cv2.circle(image_with_path, start_pt, 5, (0, 0, 0), 1)  # 黑边增强对比

                # 终点：蓝色，半径 5
                cv2.circle(image_with_path, end_pt, 5, (255, 0, 0), -1)  # Blue
                cv2.circle(image_with_path, end_pt, 5, (0, 0, 0), 1)  # 黑边

            # 绘制路径连线（显示扫描方向）
            for i in range(len(scan_points) - 1):
                pt1 = scan_points[i]
                pt2 = scan_points[i + 1]
                cv2.line(image_with_path, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)

            print(f"---✅ 轮廓路径点数：{len(scan_points)}---")
        else:
            print("⚠️ 未生成路径点")

        # 保存图像
        if save:
            cv2.imwrite(os.path.join(self.dir,'result/image_with_path.png'), image_with_path)
        
        # 显示结果
        if show:
            cv2.imshow("Contour-based Scan Path", image_with_path)
            if threading.current_thread() is threading.main_thread():
                cv2.waitKey(500)
            else:
                cv2.waitKey(1)
            cv2.destroyWindow("Contour-based Scan Path")
        
        return image_with_path
    
    def visualize_segmentation_result(self, original_image, mask, click_point=None,
                                       save=False, show=True):
        """
        可视化分割结果
        
        Args:
            original_image: 原始图像
            mask: 分割掩码
            click_point: 点击位置
            save_path: 保存路径
            show: 是否显示
        """
        overlay = original_image.copy()
        
        # 调整掩码大小以匹配原图
        img_height, img_width = original_image.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.float32), (img_width, img_height))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # 在点击点位置画圆
        if click_point:
            cv2.circle(overlay, tuple(click_point), 5, (0, 0, 255), -1)

        # 叠加半透明绿色掩码
        green_overlay = np.zeros_like(overlay)
        green_overlay[mask_binary > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.7, green_overlay, 0.3, 0)

         # 显示纯 Mask
        mask_display = (mask_binary * 255).astype(np.uint8)
        mask_colored = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)

        # 保存图像
        if save:
            cv2.imwrite(os.path.join(self.dir,'result/overlay.png'), overlay)
            cv2.imwrite(os.path.join(self.dir,'result/mask.png'), mask_colored)
        
        # 显示结果
        if show:
            cv2.imshow('Segmentation Result', overlay)
            cv2.imshow("Mask", mask_colored)
            if threading.current_thread() is threading.main_thread():
                cv2.waitKey(500)
            else:
                cv2.waitKey(1)
            cv2.destroyWindow('Segmentation Result')
            cv2.destroyWindow("Mask")
        
        return overlay
    

    def visualize_color_pointcloud(self, pointcloud, colors=None, normals=None,scan_points_3d=None,scan_normals=None,local_frames=None,
                                   background_color=[1.0, 0.8, 1.0],
                                   point_size=2.0, show_normal=False):

        import open3d as o3d
        import os
        from . import utils
        """可视化彩色点云"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 使用已经计算好的法向量
        if normals is not None:
            normals_vis = normals * 1  # 仅用于可视化，放大效果
            # pcd.normals = o3d.utility.Vector3dVector(normals_vis)
            print("✅ 已加载预计算的法向量")
        else:
            print("⚠️ 法向量未计算，使用默认法向量")

        # 可选：体素滤波（去除噪声）
        voxel_size = 1  # 设置体素大小
        pcd_downsampled = pcd.voxel_down_sample(voxel_size)
        print(f"📊 点云统计: {len(pcd.points)} → {len(pcd_downsampled.points)} 个点（下采样后）")
        # 保存点云 ✅
        o3d.io.write_point_cloud(os.path.join(self.dir,'result',"output.ply"), pcd)  # 推荐：.ply（保留颜色/法向量/结构最全）
        geometries = [pcd_downsampled]  # 使用下采样后的点云

        if len(scan_points_3d) > 0:
            # 1. 创建路径点云（红点）
            path_pcd = o3d.geometry.PointCloud()
            path_pcd.points = o3d.utility.Vector3dVector(scan_points_3d)
            path_pcd.normals = o3d.utility.Vector3dVector(scan_normals)  # ✅ 关键：赋法向量
            path_pcd.colors = o3d.utility.Vector3dVector(
                np.tile([1.0, 0.0, 0.0], (len(scan_points_3d), 1))  # 红色
            )
            geometries.append(path_pcd)

            if  local_frames:
                coord_axes = o3d.geometry.TriangleMesh()
                for frame in local_frames:
                    origin = np.array(frame['origin'])
                    x_axis = np.array(frame['x_axis']) * 5.0  # 缩放箭头长度（单位：mm，按需调整）
                    y_axis = np.array(frame['y_axis']) * 5.0
                    z_axis = np.array(frame['z_axis']) * 5.0

                    # 创建坐标轴箭头（用 LineSet 或 create_arrow 更精确，这里用小圆柱近似示意）
                    # 方案：为每个轴创建一个小箭头（三角形网格）
                    for axis_vec, color in zip([x_axis, y_axis, z_axis], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
                        # 创建从 origin 指向 origin+axis_vec 的箭头
                        arrow = o3d.geometry.TriangleMesh.create_arrow(
                            cylinder_radius=0.3,  # 箭杆半径
                            cone_radius=0.6,  # 箭头半径
                            cylinder_height=np.linalg.norm(axis_vec) * 0.8,
                            cone_height=np.linalg.norm(axis_vec) * 0.2,
                            resolution=4,
                            cylinder_split=1,
                            cone_split=1
                        )
                        # 旋转 + 平移
                        # 计算旋转：从 [0,0,1] → axis_vec 方向

                        R = utils.rotation_matrix_from_z_to_v(axis_vec)
                        arrow.rotate(R, center=[0, 0, 0])
                        arrow.translate(origin)

                        arrow.paint_uniform_color(color)
                        coord_axes += arrow
                geometries.append(coord_axes)
            # >>> 新增：首尾点用球体高亮 <<<
            if len(scan_points_3d) >= 1:
                start_point = scan_points_3d[0]
                end_point = scan_points_3d[-1]

                # 起点：红色大球
                start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2)  # 半径可调
                start_sphere.translate(start_point)
                start_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
                geometries.append(start_sphere)

                # 终点：蓝色大球
                end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2)
                end_sphere.translate(end_point)
                end_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
                geometries.append(end_sphere)

                print(f"🎯 起点/终点已高亮：红色球（起点）→ 蓝色球（终点）")

            # 2. 创建连线（LineSet）
            if len(scan_points_3d) > 1:
                lines = [[i, i + 1] for i in range(len(scan_points_3d) - 1)]
                colors = [[1.0, 0.0, 0.0] for _ in lines]  # 橙色线
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(scan_points_3d)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                geometries.append(line_set)

            print(f"✅ 已添加 {len(scan_points_3d)} 个路径点及 {len(lines) if len(scan_points_3d) > 1 else 0} 条连线")

        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="彩色点云", width=1024, height=768)

        # 添加所有几何体
        for geom in geometries:
            vis.add_geometry(geom)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.array([1.0,0.8,1.0])  # 深灰色背景
        opt.point_size = 2.0  # 点的大小
        opt.show_coordinate_frame = False  # 显示坐标系
        opt.point_show_normal = True

        # 设置视角
        ctr = vis.get_view_control()

        # 计算点云中心
        if len(pointcloud) > 0:
            center = np.mean(pointcloud, axis=0)
            print(f"🎯 点云中心: {center}")

            # 自动设置合适的视角
            max_bound = np.max(pointcloud, axis=0)
            min_bound = np.min(pointcloud, axis=0)
            size = np.max(max_bound - min_bound)

            # 设置视角位置
            ctr.set_lookat(center)
            ctr.set_front([0, 0, -1])  # 调整视角方向
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(size)

        # 添加说明文本
        print("\n📌 点云可视化控制:")
        print("  鼠标左键: 旋转视角")
        print("  鼠标右键: 平移视角")
        print("  鼠标滚轮: 缩放")
        print("  R: 重置视角")
        print("  C: 切换背景颜色")
        print("  P: 保存当前视角截图")
        print("  Q/ESC: 退出")

        # 运行可视化
        vis.run()
        vis.destroy_window()


    def draw_edge_on_mask(self, maskroi, current_mask, target_color=(0, 255, 0)):
        """
        在掩码 ROI 上绘制边缘
        
        Args:
            maskroi: 掩码 ROI 图像
            current_mask: 当前掩码
            target_color: 目标颜色（BGR 格式）
            
        Returns:
            result: 绘制后的图像
        """
        # 检查输入是否为 None
        if maskroi is None:
            print("⚠️ maskroi 为 None，无法绘制边缘")
            return None
        
        if current_mask is None:
            print("⚠️ current_mask 为 None，无法绘制边缘")
            return maskroi.copy()
        
        result = maskroi.copy()
        # 关键：仅将掩码区域设为该颜色
        result[current_mask > 0] = target_color
        return result
