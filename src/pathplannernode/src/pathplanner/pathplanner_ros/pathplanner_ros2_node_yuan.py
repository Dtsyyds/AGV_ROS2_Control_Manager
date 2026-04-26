#!/usr/bin/env python3
"""
ROS2 路径规划节点 - 完整版
支持交互模式（cv2弹窗）和话题触发模式
发布 3D 路径点 (PointCloud2)
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped, PoseArray, Pose, Point,TransformStamped, PoseStamped
from tf2_geometry_msgs import do_transform_pose_stamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading
import time

# 导入路径规划模块
from pathplanner.attention_path_ros2 import InteractiveSegmentationROS2
from pathplanner.utils import preprocess_depth
from tf2_ros import Buffer, TransformListener

class PathPlannerROS2Node(Node):
    """
    ROS2 路径规划节点
    
    功能:
    - 订阅彩色图像 (/camera/color)
    - 订阅深度图像 (/camera/depth)
    - 订阅点击点 (/click_point) - 像素坐标
    - 发布 3D 路径点 (/path_planner/3d_path)
    - 支持交互模式: 弹窗选择点击点
    """
    
    def __init__(self):
        super().__init__('pathplanner_ros2_node')
        
        # 声明参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('color_topic', 'color/image_raw'),
                ('depth_topic', 'depth/image_raw'),
                ('click_point_topic', '/click_point'),
                ('output_3d_path_topic', '/path_planner/path_3d'),
                ('output_cartesian_path_topic', '/path_planner/cartesian_path'),
                ('scan_mode', 'Long'),
                ('spacing', 10),
                ('shrink_factor', 12),
                ('fx', 498.3686770748583),
                ('fy', 501.9355502582987),
                ('cx', 314.3019441792476),
                ('cy', 225.6695918834769),
                ('model_path', ''),
                ('enable_visualization', True),
                ('interactive_mode', False),  # 是否启用交互模式
            ]
        )
        
        # 获取参数
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.click_point_topic = self.get_parameter('click_point_topic').value
        self.output_3d_path_topic = self.get_parameter('output_3d_path_topic').value
        self.output_cartesian_path_topic = self.get_parameter('output_cartesian_path_topic').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.interactive_mode = self.get_parameter('interactive_mode').value
        # QoS 配置
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # 订阅话题
        self.color_sub = self.create_subscription(
            Image, self.color_topic, self.color_callback, qos
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos
        )
        self.click_sub = self.create_subscription(
            PointStamped, self.click_point_topic, self.click_callback, qos
        )
        
        # 发布话题
        self.path_3d_pub = self.create_publisher(PointCloud2, self.output_3d_path_topic, 10)
        self.cartesian_path_pub = self.create_publisher(PoseArray, self.output_cartesian_path_topic, 10)
        
        # 工具
        self.bridge = CvBridge()
        
        # 数据缓存
        self.latest_color = None
        self.latest_depth = None
        self.processing = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 缓存最新的路径点用于定时发布
        self.latest_path_6d = None
        self.path_publish_timer = None
        
        # 路径规划器（延迟初始化）
        self.planner = None
        
        # 线程锁
        self.lock = threading.Lock()
        
        self.get_logger().info('PathPlannerROS2Node 初始化完成')
        self.get_logger().info(f'订阅彩色图: {self.color_topic}')
        self.get_logger().info(f'订阅深度图: {self.depth_topic}')
        self.get_logger().info(f'订阅点击点: {self.click_point_topic}')
        self.get_logger().info(f'交互模式: {self.interactive_mode}')
        
        # 如果启用交互模式，启动交互线程
        if self.interactive_mode:
            self.interactive_thread = threading.Thread(target=self.interactive_loop)
            self.interactive_thread.daemon = True
            self.interactive_thread.start()
            self.get_logger().info('交互模式已启动，等待图像...')
    def color_callback(self, msg):
        """彩色图像回调"""
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().debug(f'收到彩色图像: {self.latest_color.shape}')
        except Exception as e:
            self.get_logger().error(f'处理彩色图像失败: {e}')
    
    def depth_callback(self, msg):
        """深度图像回调"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.get_logger().debug(f'收到深度图像: {self.latest_depth.shape}')
        except Exception as e:
            self.get_logger().error(f'处理深度图像失败: {e}')
    
    def click_callback(self, msg):
        """点击点回调 - 接收像素坐标，直接触发处理"""
        try:
            # PointStamped 中的点是归一化坐标还是像素坐标？
            # 根据需求，我们假设是像素坐标
            x = int(msg.point.x)
            y = int(msg.point.y)
            
            self.get_logger().info(f'收到点击点: ({x}, {y})')
            
            # 检查图像数据是否就绪
            if self.latest_color is None or self.latest_depth is None:
                self.get_logger().warn('图像数据尚未就绪，请等待...')
                return
            
            # 检查是否正在处理
            with self.lock:
                if self.processing:
                    self.get_logger().warn('正在处理中，请稍后再试...')
                    return
                self.processing = True
                click_point = [x, y]
            
            # 在单独线程中处理，避免阻塞 ROS2 主循环
            thread = threading.Thread(
                target=self.process_pipeline,
                args=(click_point,)
            )
            thread.start()
            
        except Exception as e:
            self.get_logger().error(f'处理点击点失败: {e}')
    
    def interactive_loop(self):
        """交互模式循环 - 在单独线程中运行"""
        while rclpy.ok():
            # 等待图像数据就绪
            if self.latest_color is None or self.latest_depth is None:
                time.sleep(0.1)
                continue
            
            self.get_logger().info('图像已就绪，准备交互...')
            
            # 复制图像数据
            with self.lock:
                if self.processing:
                    time.sleep(0.1)
                    continue
                self.processing = True
                color = self.latest_color.copy()
                depth = self.latest_depth.copy()
            
            # 显示图像并等待点击
            try:
                cv2.imshow('Click to Segment (Press q to quit)', color)
                self.get_logger().info('请在图像上点击选择分割区域...')
                
                click_point = [None]
                
                def mouse_callback(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        click_point[0] = [x, y]
                        self.get_logger().info(f'交互点击: ({x}, {y})')
                
                cv2.setMouseCallback('Click to Segment (Press q to quit)', mouse_callback)
                
                # 等待点击或按键
                while click_point[0] is None:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                
                cv2.destroyWindow('Click to Segment (Press q to quit)')
                
                if click_point[0] is not None:
                    # 执行路径规划
                    self.process_pipeline(click_point[0], color, depth)
                else:
                    with self.lock:
                        self.processing = False
                    
            except Exception as e:
                self.get_logger().error(f'交互模式错误: {e}')
                with self.lock:
                    self.processing = False
    
    def process_pipeline(self, click_point=None, color=None, depth=None):
        """
        执行路径规划流程
        
        Args:
            click_point: 点击点 [x, y]，None 表示交互模式
            color: 彩色图像（可选，默认使用 latest_color）
            depth: 深度图像（可选，默认使用 latest_depth）
        """
        try:
            # 获取图像数据
            if color is None:
                color = self.latest_color
            if depth is None:
                depth = self.latest_depth
            
            if color is None or depth is None:
                self.get_logger().error('图像数据缺失')
                return
            
            self.get_logger().info('开始路径规划...')
            
            # 获取参数
            fx = self.get_parameter('fx').value
            fy = self.get_parameter('fy').value
            cx = self.get_parameter('cx').value
            cy = self.get_parameter('cy').value
            scan_mode = self.get_parameter('scan_mode').value
            spacing = self.get_parameter('spacing').value
            shrink_factor = self.get_parameter('shrink_factor').value
            model_path = self.get_parameter('model_path').value
            
            # 初始化规划器（如果未初始化）
            if self.planner is None:
                kwargs = {
                    'depth_array': depth,
                    'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
                }
                if model_path:
                    kwargs['model_path'] = model_path
                
                self.planner = InteractiveSegmentationROS2(**kwargs)
                self.get_logger().info('路径规划器初始化完成')
            else:
                # 更新深度图
                self.planner.set_depth(depth)
            
            # 设置参数
            self.planner.shrink_factor = shrink_factor
            self.planner.path_generator.scan_mode = scan_mode
            self.planner.path_generator.spacing = spacing
            
            # 执行路径规划
            scan_points, local_frames, scan_points_3d = self.planner.process_pipeline(
                color_image=color,
                click_point=click_point,
                auto_center=(click_point is None and not self.interactive_mode),
                enable_visualization=self.enable_visualization
            )
            # 获取6维格式的路径点 [x, y, z, rx, ry, rz]
            scan_points_6d = self.planner.local_frames_to_scan_points()
            self.get_logger().info(f'路径: {scan_points_6d[0]} ')
            
            if scan_points_3d is not None and len(scan_points_3d) > 0:
                # 保存路径点用于定时发布
                self.latest_path_6d = scan_points_6d
                
                # 立即发布一次
                self.publish_3d_path(scan_points_6d)  # 直接传入6维格式数据
                self.publish_cartesian_path(scan_points_6d)  # 发布笛卡尔路径
                
                # 创建定时器，每1秒重复发布路径（如果还没有创建）
                if self.path_publish_timer is None:
                    self.path_publish_timer = self.create_timer(1.0, self.timer_publish_path)
                    self.get_logger().info('已创建路径定时发布器 (1秒间隔)')
                
                self.get_logger().info(f'路径规划完成: {len(scan_points_3d)} 个3D点')
                self.get_logger().info(f'路径规划完成: {scan_points_3d[0]} ')
            else:
                self.get_logger().warn('路径规划失败，未生成路径点')
                
        except Exception as e:
            self.get_logger().error(f'路径规划错误: {e}')
            import traceback
            traceback.print_exc()
        finally:
            with self.lock:
                self.processing = False
    
    def timer_publish_path(self):
        """定时器回调：定时发布缓存的路径
        
        用于持续发布路径，确保机械臂控制节点能接收到
        """
        if self.latest_path_6d is not None and len(self.latest_path_6d) > 0:
            self.get_logger().debug(f'定时发布路径: {len(self.latest_path_6d)} 个点')
            self.publish_3d_path(self.latest_path_6d)
            self.publish_cartesian_path(self.latest_path_6d)
    
    def publish_3d_path(self, points_6d):
        """发布 3D 路径点 (PointCloud2)
        
        格式: [x, y, z, rx, ry, rz] - 位置和欧拉角
        """
        try:
            from sensor_msgs.msg import PointField
            import struct
            
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'camera_color_frame'
            
                   # 尝试获取 TF 最新变换时间戳
            try:
                # 等待 TF 连接建立（若首次调用时缓冲区为空）
                if not self.tf_buffer.can_transform('arm2_cartesionbase_link', 'camera_color_frame', rclpy.time.Time()):
                    self.get_logger().info('等待 TF 变换 arm2_cartesionbase_link -> camera_color_frame ...')
                    if not self.tf_buffer.wait_for_transform(
                        'arm2_cartesionbase_link', 'camera_color_frame',
                        self.get_clock().now().to_msg(),
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    ):
                        raise Exception("等待超时")

                transform = self.tf_buffer.lookup_transform(
                    'arm2_cartesionbase_link',
                    'camera_color_frame',
                    self.get_clock().now().to_msg(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                header.stamp = transform.header.stamp
                self.get_logger().debug('成功同步 TF 时间戳')
            except Exception as e:
                # 降级：使用全零时间戳
                self.get_logger().warn(f'TF 时间戳获取失败，使用零时间戳: {e}')
                header.stamp =  self.get_clock().now().to_msg()

            # 构建 PointCloud2 消息 - 3维数据 (只包含位置 xyz)
            cloud_msg = PointCloud2()
            cloud_msg.header = header
            cloud_msg.height = 1
            cloud_msg.width = len(points_6d)
            cloud_msg.is_dense = True
            cloud_msg.point_step = 24  # 6 floats * 4 bytes
            cloud_msg.row_step = cloud_msg.point_step * len(points_6d)

            # 定义字段 - 只包含位置
            cloud_msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rx', offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name='ry', offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name='rz', offset=20, datatype=PointField.FLOAT32, count=1),
            ]

            # 打包数据（将位置从 mm 转换为 m，只取 xyz）
            buffer = []
            for p in points_6d:
                # p = [x, y, z, rx, ry, rz]
                # 转换：x, y, z 从 mm 转为 m；rx, ry, rz 保持弧度
                p_m = [p[0] / 1000.0, p[1] / 1000.0, p[2] / 1000.0, p[3], p[4], p[5]]
                buffer.append(struct.pack('ffffff', *p_m))
            cloud_msg.data = b''.join(buffer)
          
            self.path_3d_pub.publish(cloud_msg)
    
            self.get_logger().info(f'发布 3D 路径: {len(points_6d)} 个点 (格式: 6维 [x,y,z,rx,ry,rz])')
            
        except Exception as e:
            self.get_logger().error(f'发布 3D 路径失败: {e}')
    def publish_cartesian_path(self, points_6d):
        """发布笛卡尔路径 (PoseArray)
        
        格式: [x, y, z, rx, ry, rz] - 位置(米)和欧拉角(弧度)
        转换为 PoseArray，姿态使用四元数表示
        将点从相机坐标系转换到 arm2_cartesionbase_link 坐标系
        并补偿工具坐标系偏移，使 arm2_toolscan_link 到达目标点
        """
        try:
            # 获取 TF 变换: camera -> base
            try:
                transform_camera_to_base = self.tf_buffer.lookup_transform(
                    'arm2_cartesionbase_link',
                    'camera_color_frame',
                    self.get_clock().now().to_msg(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                self.get_logger().info('成功获取 TF 变换: camera_color_frame -> arm2_cartesionbase_link')
            except Exception as e:
                self.get_logger().error(f'获取 TF 变换失败: {e}')
                return
            
            # 获取 TF 变换: toolscan -> tool (工具到末端的变换)
            try:
                transform_toolscan_to_tool = self.tf_buffer.lookup_transform(
                    'arm2_tool_link',
                    'arm2_toolscan_link',
                    self.get_clock().now().to_msg(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                self.get_logger().info('成功获取 TF 变换: arm2_toolscan_link -> arm2_tool_link')
            except Exception as e:
                self.get_logger().error(f'获取工具 TF 变换失败: {e}')
                return

            
           

            pose_array = PoseArray()
            pose_array.header.stamp = transform_camera_to_base.header.stamp
            pose_array.header.frame_id = 'arm2_cartesionbase_link'
           
            for p in points_6d:
                # p = [x, y, z, rx, ry, rz]
                # 位置从 mm 转换为 m
                x_m = p[0] / 1000.0
                y_m = p[1] / 1000.0
                z_m = p[2] / 1000.0
                rx = p[3]  # 弧度
                ry = p[4]  # 弧度
                rz = p[5]  # 弧度
                
                # 创建 PoseStamped 在相机坐标系下
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = 'camera_color_frame'
                pose_stamped.header.stamp = transform_camera_to_base.header.stamp 
                pose_stamped.pose.position.x = float(x_m)
                pose_stamped.pose.position.y = float(y_m)
                pose_stamped.pose.position.z = float(z_m)
                
                # 欧拉角转四元数 (ZYX 顺序: rz -> ry -> rx)
                quaternion = self.euler_to_quaternion(rx, ry, rz)
                pose_stamped.pose.orientation.x = quaternion[0]
                pose_stamped.pose.orientation.y = quaternion[1]
                pose_stamped.pose.orientation.z = quaternion[2]
                pose_stamped.pose.orientation.w = quaternion[3]
                
                # 第1步: 变换到 arm2_cartesionbase_link 坐标系
                # 此时表示 toolscan 应该在基座坐标系下的目标位姿
                pose_in_base = do_transform_pose_stamped(pose_stamped, transform_camera_to_base)
                
                # # 第2步: 将目标点从 toolscan 坐标系转换到 tool_link 坐标系
                # # 这样当末端(tool_link)到达转换后的点时，工具(toolscan)正好到达原始目标点
                # pose_for_tool_link = do_transform_pose_stamped(pose_in_base, transform_toolscan_to_tool)
                
                pose_array.poses.append(pose_in_base.pose)
            
            self.cartesian_path_pub.publish(pose_array)
            self.get_logger().info(f'发布笛卡尔路径: {len(pose_array.poses)} 个点 (PoseArray) 在 arm2_cartesionbase_link 坐标系下 (工具坐标系补偿已应用)')
            
        except Exception as e:
            self.get_logger().error(f'发布笛卡尔路径失败: {e}')

    def euler_to_quaternion(self, rx, ry, rz):
        """将欧拉角 (rx, ry, rz) 转换为四元数 [x, y, z, w]
        
        旋转顺序: ZYX (rz -> ry -> rx)
        """
        # 计算半角
        cx = np.cos(rx * 0.5)
        sx = np.sin(rx * 0.5)
        cy = np.cos(ry * 0.5)
        sy = np.sin(ry * 0.5)
        cz = np.cos(rz * 0.5)
        sz = np.sin(rz * 0.5)
        
        # ZYX 顺序的四元数
        w = cx * cy * cz + sx * sy * sz
        x = sx * cy * cz - cx * sy * sz
        y = cx * sy * cz + sx * cy * sz
        z = cx * cy * sz - sx * sy * cz
        
        return [x, y, z, w]
    
    def rotation_matrix_to_quaternion(self, R):
        """将 3x3 旋转矩阵转换为四元数 [x, y, z, w]"""
        # 使用标准转换公式
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return [x, y, z, w]


def main(args=None):
    rclpy.init(args=args)
    
    node = PathPlannerROS2Node()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
