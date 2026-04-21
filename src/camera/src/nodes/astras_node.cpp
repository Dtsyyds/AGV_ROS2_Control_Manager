#include "camera/nodes/astras_node.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "camera/astras/astras_camera.hpp"

namespace camera {

/**
 * @brief AstraS 相机 ROS2 节点实现。
 *
 * 节点启动时：
 * - 创建 image_transport::CameraPublisher（彩色/深度）
 * - 打开并启动相机，将帧回调绑定到 publishFrame()
 */
AstraSNode::AstraSNode(const rclcpp::NodeOptions& options) : rclcpp::Node("astraS_camera", options) {
    depth_pub_ = image_transport::create_camera_publisher(this, "depth/image_raw");
    color_pub_ = image_transport::create_camera_publisher(this, "color/image_raw");     

    camera_ = std::make_unique<AstraSCamera>(true, true, false);
    camera_->setFrameCallback([this](const CameraFrameView& frame) {
        this->publishFrame(frame);
    });

    if (!camera_->open()) {
        auto* astra_camera = dynamic_cast<AstraSCamera*>(camera_.get());
        if (astra_camera != nullptr) {
            RCLCPP_ERROR(this->get_logger(), "相机打开失败: %s", astra_camera->lastError().c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "相机打开失败");
        }
        return;
    }

    if (!camera_->start()) {
        auto* astra_camera = dynamic_cast<AstraSCamera*>(camera_.get());
        if (astra_camera != nullptr) {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败: %s", astra_camera->lastError().c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "相机启动失败");
        }
        return;
    }
}

/**
 * @brief 析构时关闭相机（停止采集并释放资源）。
 */
AstraSNode::~AstraSNode() {
    if (camera_) {
        camera_->close();
    }
}

/**
 * @brief 将相机帧转换为 ROS2 Image，并连同 CameraInfo 一起发布。
 *
 * 当前仅发布 Color 与 Depth；IR 帧将被忽略。
 */
void AstraSNode::publishFrame(const CameraFrameView& frame) {
    image_transport::CameraPublisher* pub = nullptr;
    std::string ros_encoding;
    int cv_type = CV_8UC3;

    if (frame.type == CameraStreamType::Color) {
        pub = &color_pub_;
        ros_encoding = "rgb8";
        cv_type = CV_8UC3;
    } else if (frame.type == CameraStreamType::Depth) {
        pub = &depth_pub_;
        ros_encoding = "16UC1";
        cv_type = CV_16UC1;
    } else {
        return;
    }

    cv::Mat mat(frame.height, frame.width, cv_type, const_cast<void*>(frame.data));

    std_msgs::msg::Header header;
    header.stamp = this->get_clock()->now();
    header.frame_id = frame.frame_id;

    auto msg = cv_bridge::CvImage(header, ros_encoding, mat).toImageMsg();

    sensor_msgs::msg::CameraInfo info;
    info.header = header;
    info.width = frame.width;
    info.height = frame.height;

    pub->publish(*msg, info);
}

}
