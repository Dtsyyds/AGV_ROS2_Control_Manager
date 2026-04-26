#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>

#include "camera/data/camera_types.hpp"
#include "camera/interface/camera_interface.hpp"

namespace camera {

/**
 * @brief AstraS 相机 ROS2 节点。
 *
 * 发布话题：
 * - depth/image_raw + CameraInfo
 * - color/image_raw + CameraInfo
 *
 * 发布 TF：
 * - arm1_tool_link -> camera_color_frame
 * - arm1_tool_link -> camera_depth_frame
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

    /**
     * @brief 发布相机静态 TF 变换。
     */
    void publishStaticTf();

    std::unique_ptr<CameraInterface> camera_;
    image_transport::CameraPublisher depth_pub_;
    image_transport::CameraPublisher color_pub_;

    CameraIntrinsics color_intrinsics_;
    CameraIntrinsics depth_intrinsics_;

    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;

    // TF 参数
    std::string parent_frame_id_;
    std::string color_frame_id_;
    std::string depth_frame_id_;
};

}
