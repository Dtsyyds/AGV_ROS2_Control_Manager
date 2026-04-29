from setuptools import find_packages, setup

package_name = 'pathplanner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/ros2_pathplanner_launch.py',
        ]),
        ('share/' + package_name + '/config', [
            '../config/params.yaml',
        ]),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python', 'open3d', 'scipy', 'rclpy', 'sensor_msgs', 'cv_bridge'],
    zip_safe=True,
    maintainer='zyj',
    maintainer_email='zhouyujiang73@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [  
            'pathplanner_ros2_node = pathplanner_ros.pathplanner_ros2_node:main',
            'test_click_publisher = pathplanner_ros.test_click_publisher:main',
        ],
    },
)
