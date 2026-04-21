#pragma once

#include <hardware_interface/system_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/duration.hpp>

#include "agv_hardware/config.hpp"
#include "agv_hardware/robot_controller.hpp"
#include "agv_protocol/websocket_client.hpp"
#include "agv_protocol/websocket_server.hpp"
#include "agv_protocol/command_parser.hpp"
#include "agv_bridge/bridge_manager.hpp"

#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <rclcpp/rclcpp.hpp>

#include <array>
#include <memory>
#include <mutex>
#include <string>

namespace agv_hardware {

/**
 * @brief AGV 硬件接口主类
 * 
 * 实现 ROS2 hardware_interface::SystemInterface 接口，
 * 负责管理机器人控制器、WebSocket 通信、命令解析等模块
 */
class AgvHardwareInterface : public hardware_interface::SystemInterface {
public:
    /// @brief 生命周期配置回调
    hardware_interface::CallbackReturn
        on_configure(const rclcpp_lifecycle::State & previous_state) override;
    
    /// @brief 生命周期激活回调
    hardware_interface::CallbackReturn
        on_activate(const rclcpp_lifecycle::State & previous_state) override;
    
    /// @brief 生命周期停用回调
    hardware_interface::CallbackReturn
        on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

    /// @brief 硬件接口初始化回调
    hardware_interface::CallbackReturn
        on_init(const hardware_interface::HardwareComponentInterfaceParams & params) override;
    
    /// @brief 读取硬件状态
    hardware_interface::return_type
        read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
    
    /// @brief 写入硬件命令
    hardware_interface::return_type
        write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
    /**
     * @brief 发送关节位置数据到上位机
     * @param positions 6 个关节的角度数组
     */
    void reportJointPositions(const std::array<double, 6>& positions);
    
    /**
     * @brief 发送笛卡尔位置数据到上位机
     * @param positions 6 个笛卡尔位置数组
     */
    void reportPosturePositions(const std::array<double, 6>& positions);
    
    /**
     * @brief 处理接收到的命令
     * @param json_str JSON 格式的命令字符串
     */
    void handleCommand(const std::string& json_str);
    
    /**
     * @brief 初始化 RT 控制循环（实时模式）
     */
    void initRtControlLoop();

    // ==================== ROS 2 话题接口 ====================
    std::shared_ptr<rclcpp::Node> cmd_node_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_state_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cartesian_pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr click_point_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr cartesian_cmd_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr cartesian_path_sub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ==================== 配置与模块 ====================
    AgvConfig config_;                              ///< 配置参数
    std::unique_ptr<RobotController> robot_controller_;        ///< 机器人控制器
    std::unique_ptr<agv_protocol::WebSocketClientWrapper> ws_client_;        ///< WebSocket 客户端
    std::unique_ptr<agv_protocol::WebSocketServerWrapper> ws_server_;        ///< 本地 WebSocket 服务器
    agv_protocol::CommandParser command_parser_;                  ///< 命令解析器

    // ==================== 状态数据 ====================
    bool enable_status_report_ = false;  // 是否允许回传状态数据
    std::array<double, 6> joint_positions_{};       ///< 当前关节位置
    std::array<double, 6> posture_positions_{};     ///< 当前笛卡尔位置
    std::array<double, 6> joint_commands_{};        ///< 关节命令
    
    std::mutex io_mutex_;                           ///< IO 互斥锁

    // ==================== 异步 Executor ====================
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
    std::unique_ptr<std::thread> executor_thread_;
};

} // namespace agv_hardware