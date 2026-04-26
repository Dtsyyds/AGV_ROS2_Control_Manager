#include "camera/nodes/astras_node.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include "camera/astras/astras_camera.hpp"

namespace camera {

/**
 * @brief AstraS 相机 ROS2 节点实现。
 *
 * 节点启动时：
 * - 创建 image_transport::CameraPublisher（彩色/深度）
 * - 打开并启动相机，将帧回调绑定到 publishFrame()
 * - 从相机获取内参并缓存
 * - 发布相机静态 TF
 */
AstraSNode::AstraSNode(const rclcpp::NodeOptions& options) : rclcpp::Node("astraS_camera", options) {
    // 声明参数
    this->declare_parameter<std::string>("parent_frame_id", "arm2_tool_link");
    this->declare_parameter<std::string>("color_frame_id", "camera_color_frame");
    this->declare_parameter<std::string>("depth_frame_id", "camera_depth_frame");

    // 获取参数
    this->get_parameter("parent_frame_id", parent_frame_id_);
    this->get_parameter("color_frame_id", color_frame_id_);
    this->get_parameter("depth_frame_id", depth_frame_id_);

    // 创建 TF 静态广播器
    tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    // 发布静态 TF
    publishStaticTf();

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

    // 获取并缓存相机内参
    auto* astra_camera = dynamic_cast<AstraSCamera*>(camera_.get());
    if (astra_camera != nullptr) {
        auto color_intrinsics = astra_camera->getColorIntrinsics();
        if (color_intrinsics.has_value()) {
            color_intrinsics_ = color_intrinsics.value();
            RCLCPP_INFO(this->get_logger(), "彩色相机内参已获取: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
                        color_intrinsics_.fx, color_intrinsics_.fy,
                        color_intrinsics_.cx, color_intrinsics_.cy);
        } else {
            RCLCPP_WARN(this->get_logger(), "无法获取彩色相机内参");
        }

        auto depth_intrinsics = astra_camera->getDepthIntrinsics();
        if (depth_intrinsics.has_value()) {
            depth_intrinsics_ = depth_intrinsics.value();
            RCLCPP_INFO(this->get_logger(), "深度相机内参已获取: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
                        depth_intrinsics_.fx, depth_intrinsics_.fy,
                        depth_intrinsics_.cx, depth_intrinsics_.cy);
        } else {
            RCLCPP_WARN(this->get_logger(), "无法获取深度相机内参");
        }
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
 * CameraInfo 使用从相机获取的内参填充 K 矩阵和畸变系数。
 */
void AstraSNode::publishFrame(const CameraFrameView& frame) {
    image_transport::CameraPublisher* pub = nullptr;
    std::string ros_encoding;
    int cv_type = CV_8UC3;
    const CameraIntrinsics* intrinsics = nullptr;

    if (frame.type == CameraStreamType::Color) {
        pub = &color_pub_;
        ros_encoding = "rgb8";
        cv_type = CV_8UC3;
        intrinsics = &color_intrinsics_;
    } else if (frame.type == CameraStreamType::Depth) {
        pub = &depth_pub_;
        ros_encoding = "16UC1";
        cv_type = CV_16UC1;
        intrinsics = &depth_intrinsics_;
    } else {
        return;
    }

    cv::Mat mat(frame.height, frame.width, cv_type, const_cast<void*>(frame.data));
    
    // 翻转图像 (0: 垂直翻转, 1: 水平翻转, -1: 垂直+水平翻转)
    cv::Mat flipped_mat;
    cv::flip(mat, flipped_mat, 1);  // 水平翻转，可根据需要改为 1 或 -1

    std_msgs::msg::Header header;
    header.stamp = this->get_clock()->now();
    header.frame_id = frame.frame_id;

    auto msg = cv_bridge::CvImage(header, ros_encoding, flipped_mat).toImageMsg();

    sensor_msgs::msg::CameraInfo info;
    info.header = header;
    info.width = frame.width;
    info.height = frame.height;

    // 填充相机内参矩阵 K [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    if (intrinsics != nullptr && intrinsics->isValid()) {
        info.k[0] = intrinsics->fx;
        info.k[1] = 0.0;
        info.k[2] = intrinsics->cx;
        info.k[3] = 0.0;
        info.k[4] = intrinsics->fy;
        info.k[5] = intrinsics->cy;
        info.k[6] = 0.0;
        info.k[7] = 0.0;
        info.k[8] = 1.0;

        // 填充畸变系数 D [k1, k2, p1, p2, k3]
        info.d.resize(5);
        for (size_t i = 0; i < 5; ++i) {
            info.d[i] = intrinsics->distortion_coeffs[i];
        }

        // 填充投影矩阵 P (与 K 相同，无畸变校正)
        info.p[0] = intrinsics->fx;
        info.p[1] = 0.0;
        info.p[2] = intrinsics->cx;
        info.p[3] = 0.0;
        info.p[4] = 0.0;
        info.p[5] = intrinsics->fy;
        info.p[6] = intrinsics->cy;
        info.p[7] = 0.0;
        info.p[8] = 0.0;
        info.p[9] = 0.0;
        info.p[10] = 1.0;
        info.p[11] = 0.0;

        // 畸变模型
        info.distortion_model = "plumb_bob";
        //  // 打印 CameraInfo 用于调试
        // RCLCPP_INFO(this->get_logger(), "%s CameraInfo: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f, d=[%.4f,%.4f,%.4f,%.4f,%.4f]",
        //             frame.type == CameraStreamType::Color ? "Color" : "Depth",
        //             info.k[0], info.k[4], info.k[2], info.k[5],
        //             info.d[0], info.d[1], info.d[2], info.d[3], info.d[4]);
    }

    pub->publish(*msg, info);
}

/**
 * @brief 发布相机静态 TF 变换。
 *
 * 从参数读取相机相对于机械臂末端的位姿：
 * - tf_translation: [x, y, z]（单位：米）
 * - tf_rotation: [qx, qy, qz, qw]（四元数）
 *
 * 发布两个 TF：
 * - parent_frame_id -> color_frame_id
 * - parent_frame_id -> depth_frame_id
 */
void AstraSNode::publishStaticTf() {
    // 声明位姿参数（默认值需要后续通过标定更新）
    this->declare_parameter<std::vector<double>>("tf_translation", {0.05, 0.0, 0.05});
    this->declare_parameter<std::vector<double>>("tf_rotation", {0.0, 0.0, 0.0, 1.0});

    // 获取位姿参数
    std::vector<double> translation;
    std::vector<double> rotation;
    this->get_parameter("tf_translation", translation);
    this->get_parameter("tf_rotation", rotation);

    // 验证参数长度
    if (translation.size() != 3) {
        RCLCPP_WARN(this->get_logger(), "tf_translation 参数长度应为 3，使用默认值");
        translation = {0.05, 0.0, 0.05};
    }
    if (rotation.size() != 4) {
        RCLCPP_WARN(this->get_logger(), "tf_rotation 参数长度应为 4，使用默认值");
        rotation = {0.0, 0.0, 0.0, 1.0};
    }

    RCLCPP_INFO(this->get_logger(), "发布相机 TF: %s -> %s, %s",
                parent_frame_id_.c_str(), color_frame_id_.c_str(), depth_frame_id_.c_str());
    RCLCPP_INFO(this->get_logger(), "TF 平移: [%.3f, %.3f, %.3f]",
                translation[0], translation[1], translation[2]);
    RCLCPP_INFO(this->get_logger(), "TF 旋转: [%.3f, %.3f, %.3f, %.3f]",
                rotation[0], rotation[1], rotation[2], rotation[3]);

    // 发布彩色相机 TF
    geometry_msgs::msg::TransformStamped color_tf;
    color_tf.header.stamp = this->get_clock()->now();
    color_tf.header.frame_id = parent_frame_id_;
    color_tf.child_frame_id = color_frame_id_;
    color_tf.transform.translation.x = translation[0];
    color_tf.transform.translation.y = translation[1];
    color_tf.transform.translation.z = translation[2];
    color_tf.transform.rotation.x = rotation[0];
    color_tf.transform.rotation.y = rotation[1];
    color_tf.transform.rotation.z = rotation[2];
    color_tf.transform.rotation.w = rotation[3];

    // 发布深度相机 TF（与彩色相机相同，因为已启用配准）
    geometry_msgs::msg::TransformStamped depth_tf;
    depth_tf.header.stamp = this->get_clock()->now();
    depth_tf.header.frame_id = parent_frame_id_;
    depth_tf.child_frame_id = depth_frame_id_;
    depth_tf.transform.translation.x = translation[0];
    depth_tf.transform.translation.y = translation[1];
    depth_tf.transform.translation.z = translation[2];
    depth_tf.transform.rotation.x = rotation[0];
    depth_tf.transform.rotation.y = rotation[1];
    depth_tf.transform.rotation.z = rotation[2];
    depth_tf.transform.rotation.w = rotation[3];

    tf_static_broadcaster_->sendTransform(color_tf);
    tf_static_broadcaster_->sendTransform(depth_tf);

    RCLCPP_INFO(this->get_logger(), "相机静态 TF 已发布");
}

}
