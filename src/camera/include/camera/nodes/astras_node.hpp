#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>

#include "camera/interface/camera_interface.hpp"

namespace camera {

/**
 * @brief AstraS 相机 ROS2 节点。
 *
 * 发布话题：
 * - depth/image_raw + CameraInfo
 * - color/image_raw + CameraInfo
 */
class AstraSNode final : public rclcpp::Node {
public:
    /**
     * @brief 构造函数。
     * @param options ROS2 节点选项
     */
    explicit AstraSNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~AstraSNode() override;

private:
    /**
     * @brief 将相机帧转换并发布为 ROS Image/CameraInfo。
     */
    void publishFrame(const CameraFrameView& frame);

    std::unique_ptr<CameraInterface> camera_;
    image_transport::CameraPublisher depth_pub_;
    image_transport::CameraPublisher color_pub_;
};

}
