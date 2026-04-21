#!/usr/bin/env python3
"""
测试脚本：发布点击点来触发路径规划
用法: ros2 run pathplanner test_click_publisher --ros-args -p x:=320 -p y:=240
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import sys


class TestClickPublisher(Node):
    def __init__(self):
        super().__init__('test_click_publisher')
        
        # 声明参数
        self.declare_parameter('x', 320)
        self.declare_parameter('y', 240)
        self.declare_parameter('topic', '/click_point')
        
        x = self.get_parameter('x').value
        y = self.get_parameter('y').value
        topic = self.get_parameter('topic').value
        
        # 创建发布者
        self.pub = self.create_publisher(PointStamped, topic, 10)
        
        # 创建定时器
        # self.timer = self.create_timer(1.0, self.publish_click)
        
        self.click_point = PointStamped()
        self.click_point.point.x = float(x)
        self.click_point.point.y = float(y)
        self.click_point.point.z = 0.0
        
        self.count = 0
        
        # 延迟发布，确保订阅者有足够时间连接
        self.create_timer(0.5, self.publish_once)
    
    def publish_once(self):
        """只发布一次消息"""
        if self.count >= 1:
            return
        
        # 设置时间戳和坐标系
        self.click_point.header.stamp = self.get_clock().now().to_msg()
        self.click_point.header.frame_id = 'camera_color_optical_frame'
        
        self.pub.publish(self.click_point)
        self.get_logger().info(f'发布点击点: ({self.click_point.point.x}, {self.click_point.point.y}) 到 /click_point')
        self.get_logger().info('发布完成')
        self.count += 1
    
    def publish_click(self):
        self.click_point.header.stamp = self.get_clock().now().to_msg()
        self.click_point.header.frame_id = 'camera_color_optical_frame'
        self.pub.publish(self.click_point)
        self.get_logger().info(f'发布点击点 ({self.click_point.point.x}, {self.click_point.point.y})')
        self.count += 1
        if self.count >= 5:
            self.get_logger().info('发布完成')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TestClickPublisher()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
