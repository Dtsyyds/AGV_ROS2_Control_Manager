from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
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
    
    # 创建路径规划节点
    pathplanner_node = Node(
        package='pathplanner',
        executable='pathplanner_node',
        name='pathplanner_node',
        output='screen',
        parameters=[
            {
                'scan_mode': LaunchConfiguration('scan_mode'),
                'spacing': LaunchConfiguration('spacing'),
                'shrink_factor': LaunchConfiguration('shrink_factor'),
                'interpolation_points': LaunchConfiguration('interpolation_points'),
                'attention_method': LaunchConfiguration('attention_method'),
                'k_neighbors': LaunchConfiguration('k_neighbors'),
                'result_dir': LaunchConfiguration('result_dir')
            }
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
        pathplanner_node
    ])
