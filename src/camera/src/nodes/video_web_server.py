import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from flask import Flask, Response, stream_with_context
import threading
import time

class VideoStreamNode(Node):
    def __init__(self):
        super().__init__('video_stream_node')
        
        # 1. 线程锁：保护共享的 latest_frame
        self.lock = threading.Lock()
        
        self.subscription = self.create_subscription(
            Image,
            '/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_stamp = None

    def image_callback(self, msg):
        # 2. 转换图像
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 3. 加锁更新，确保写入时不会被读取
            with self.lock:
                self.latest_frame = cv_image
                self.latest_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        except Exception as e:
            self.get_logger().error(f"转换图像失败: {e}")

    def generate_frames(self):
        """生成 MJPEG 流的生成器"""
        last_stamp = None
        while rclpy.ok():
            # 4. 加锁读取，确保读取完整性
            with self.lock:
                frame = self.latest_frame
                stamp = self.latest_stamp
            
            if frame is not None and stamp is not None:
                if stamp == last_stamp:
                    time.sleep(0.01)
                    continue
                # 降低 JPEG 质量 (50-80) 以提高传输速度，减少 CPU 占用
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ret:
                    last_stamp = stamp
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.03)
            else:
                # 没有图像时稍微休眠，防止 CPU 空转
                time.sleep(0.1)

def main():
    rclpy.init()
    node = VideoStreamNode()
    
    # 启动 Flask 服务器
    app = Flask(__name__)

    @app.route('/video_feed')
    def video_feed():
        return Response(stream_with_context(node.generate_frames()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        return "<h1>ROS2 相机服务运行中</h1><p>请访问 <a href='/video_feed'>/video_feed</a> 查看视频流</p>"

    # 5. 设置 daemon=True，这样主程序退出时 Flask 线程也会自动退出
    flask_thread = threading.Thread(
        target=app.run,
        kwargs={'host': '192.168.1.103', 'port': 5000, 'threaded': True, 'use_reloader': False},
        daemon=True
    )
    flask_thread.start()

    node.get_logger().info("节点已启动，正在监听 /video_feed ...")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
