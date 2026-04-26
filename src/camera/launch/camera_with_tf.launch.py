"""
相机节点启动文件（带 TF 发布）

发布：
- 彩色/深度图像
- CameraInfo（包含内参）
- 相机静态 TF（arm1_tool_link -> camera_color_frame/depth_frame）

使用方法：
1. 使用默认 TF：
   ros2 launch camera camera_with_tf.launch.py

2. 使用自定义 TF 配置文件：
   ros2 launch camera camera_with_tf.launch.py tf_config:=/path/to/custom_tf.yaml

3. 通过参数覆盖：
   ros2 launch camera camera_with_tf.launch.py \
     tf_translation:="[0.1, 0.0, 0.05]" \
     tf_rotation:="[0.0, 0.0, 0.0, 1.0]"
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 获取包路径
    pkg_camera = get_package_share_directory('camera')
    
    # 默认配置文件路径
    default_tf_config = os.path.join(pkg_camera, 'config', 'camera_tf.yaml')
    
    # 声明启动参数
    tf_config_arg = DeclareLaunchArgument(
        'tf_config',
        default_value=default_tf_config,
        description='相机 TF 配置文件路径'
    )
    
    # 相机节点
    camera_node = Node(
        package='camera',
        executable='camera_node',
        name='astraS_camera',
        output='screen',
        parameters=[LaunchConfiguration('tf_config')]
    )
    
    return LaunchDescription([
        tf_config_arg,
        camera_node
    ])
