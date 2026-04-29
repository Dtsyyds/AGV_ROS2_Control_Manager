import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """
    ROS2路径规划节点启动文件
    支持两种触发模式:
    1. topic模式: 通过发布点击点话题触发分割
    2. auto模式: 自动使用图像中心作为分割点
    """
    
    # 获取参数文件路径（从包的 share 目录）
    params_file = os.path.join(
        get_package_share_directory('pathplanner'),
        'config',
        'params.yaml'
    )
    
    # 声明启动参数
    scan_mode_arg = DeclareLaunchArgument(
        'scan_mode',
        default_value='Long',
        description='扫描模式: Long, Short'
    )
    
    spacing_arg = DeclareLaunchArgument(
        'spacing',
        default_value='10',
        description='步进间隔（像素）'
    )
    
    shrink_factor_arg = DeclareLaunchArgument(
        'shrink_factor',
        default_value='12',
        description='掩码收缩因子（像素）'
    )
    
    interpolation_points_arg = DeclareLaunchArgument(
        'interpolation_points',
        default_value='5',
        description='两点之间的插值点数'
    )
    
    attention_method_arg = DeclareLaunchArgument(
        'attention_method',
        default_value='EGA',
        description='注意力计算方法: Advanced, Geometric, Anisotropic, EGA'
    )
    
    k_neighbors_arg = DeclareLaunchArgument(
        'k_neighbors',
        default_value='45',
        description='邻居点数量'
    )
    
    result_dir_arg = DeclareLaunchArgument(
        'result_dir',
        default_value='./result',
        description='结果保存目录'
    )
    
    trigger_mode_arg = DeclareLaunchArgument(
        'trigger_mode',
        default_value='topic',
        description='触发模式: topic(等待点击点话题), auto(自动中心)'
    )
    
    auto_center_arg = DeclareLaunchArgument(
        'auto_center',
        default_value='false',
        description='是否自动使用图像中心'
    )
    
    click_point_topic_arg = DeclareLaunchArgument(
        'click_point_topic',
        default_value='input/click_point',
        description='点击点话题名称'
    )
    
    # 创建ROS2路径规划节点
    pathplanner_ros2_node = Node(
        package='pathplanner',
        executable='pathplanner_ros2_node',
        name='pathplanner_ros2_node',
        output='screen',
        parameters=[
            params_file,  # 加载YAML参数文件
            {
                # 这些参数会覆盖YAML文件中的值
                'color_topic': '/color/image_raw',
                'depth_topic': '/depth/image_raw',
                'output_3d_path_topic': '/path_planner/path_3d',
                'output_cartesian_path_topic': '/path_planner/cartesian_path',
            }
        ],
        remappings=[
            # 可以在这里重映射话题名称
            # ('color/image_raw', '/camera/color/image_raw'),
            # ('depth/image_raw', '/camera/depth/image_raw'),
            # ('input/camera_info', '/camera/color/camera_info'),
        ]
    )
    
    return LaunchDescription([
        scan_mode_arg,
        spacing_arg,
        shrink_factor_arg,
        interpolation_points_arg,
        attention_method_arg,
        k_neighbors_arg,
        result_dir_arg,
        trigger_mode_arg,
        auto_center_arg,
        click_point_topic_arg,
        pathplanner_ros2_node
    ])
