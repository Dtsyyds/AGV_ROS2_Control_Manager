#include "agv_hardware/agv_hardware.hpp"
#include "agv_hardware/self_check_manager.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>

namespace agv_hardware {

hardware_interface::CallbackReturn AgvHardwareInterface::on_init
    (const hardware_interface::HardwareInfo & info)
{
    if (hardware_interface::SystemInterface::on_init(info) !=
        hardware_interface::CallbackReturn::SUCCESS)
    {
        return hardware_interface::CallbackReturn::ERROR;
    }

    info_ = info;
    
    config_ = AgvConfig();
    robot_controller_ = std::make_unique<RobotController>(config_);
    
    ws_client_ = std::make_unique<agv_protocol::WebSocketClientWrapper>(
        config_.server_uri,
        config_.reconnect_interval_ms
    );
    agv_ = std::make_unique<AGVRobotMove>();
    agvIP = "192.168.1.109";
    agv_->init(agvIP);
    ws_server_ = std::make_unique<agv_protocol::WebSocketServerWrapper>(config_.local_server_port);
    
    // 统一指令处理器
    auto cmd_handler = [this](const std::string& msg) { handleCommand(msg); };
    ws_client_->setMessageHandler(cmd_handler);
    // ws_server_->setMessageHandler(cmd_handler);

    // ==================== 初始化 ROS 2 话题接口 ====================
    cmd_node_ = std::make_shared<rclcpp::Node>("agv_hardware_topic_node");

    // 初始化 TF
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(cmd_node_->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 1. 状态发布者
    joint_state_pub_ = cmd_node_->create_publisher<std_msgs::msg::Float64MultiArray>("arm_joint_states", 10);
    cartesian_pose_pub_ = cmd_node_->create_publisher<std_msgs::msg::Float64MultiArray>("arm_cartesian_pose", 10);
    click_point_pub_ = cmd_node_->create_publisher<geometry_msgs::msg::PointStamped>("/click_point", 10);

    // 2. 关节指令订阅者
    joint_cmd_sub_ = cmd_node_->create_subscription<std_msgs::msg::Float64MultiArray>(
        "cmd_arm_joint", 10,
        [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
            if (msg->data.size() == 6) {
                std::array<double, 6> target;
                std::copy(msg->data.begin(), msg->data.end(), target.begin());
                std::lock_guard<std::mutex> lock(io_mutex_);
                RCLCPP_INFO(cmd_node_->get_logger(), "收到关节指令，正在下发...");
                robot_controller_->sendJointCommand(target);
            }
        });

    // 3. 笛卡尔指令订阅者
    cartesian_cmd_sub_ = cmd_node_->create_subscription<std_msgs::msg::Float64MultiArray>(
        "cmd_arm_cartesian", 10,
        [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
            if (msg->data.size() == 6) {
                RCLCPP_INFO(cmd_node_->get_logger(), "收到笛卡尔指令，正在下发...");
                std::array<double, 6> p;
                std::copy(msg->data.begin(), msg->data.end(), p.begin());
                std::vector<std::array<double, 6>> points = {p};
                std::lock_guard<std::mutex> lock(io_mutex_);
                RCLCPP_INFO(cmd_node_->get_logger(), 
                    "内容: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]", 
                    p[0], p[1], p[2], p[3], p[4], p[5]);
                robot_controller_->sendCartesianCommand(points, "linear", 100.0);
            }
        });

    // 4. 轨迹路径订阅者 (PoseArray)
    // 轨迹路径订阅者 (PoseArray)
    cartesian_path_sub_ = cmd_node_->create_subscription<geometry_msgs::msg::PoseArray>(
        "/path_planner/cartesian_path", 10,
        [this](const geometry_msgs::msg::PoseArray::SharedPtr msg) {
            if (msg->poses.empty()) return;

            RCLCPP_INFO(cmd_node_->get_logger(), "收到路径规划轨迹，点数: %zu (frame: %s)", 
                msg->poses.size(), msg->header.frame_id.c_str());

            std::vector<std::array<double, 6>> cartesian_path;
            for (const auto& pose : msg->poses) {
                // 直接使用位置（米）和四元数，转换为欧拉角
                // tf2::Quaternion q(
                //     pose.orientation.x,
                //     pose.orientation.y,
                //     pose.orientation.z,
                //     pose.orientation.w);
                    tf2::Quaternion q(
                    0.13497,
                    0.75322,
                    -0.63979,
                    -0.071439);
                double rx, ry, rz;
                tf2::Matrix3x3(q).getRPY(rx, ry, rz);

                std::array<double, 6> p = {
                    pose.position.x,
                    pose.position.y,
                    pose.position.z + 0.2,
                    rx, ry, rz
                };
                cartesian_path.push_back(p);
            }

            if (!cartesian_path.empty()) {
                std::lock_guard<std::mutex> lock(io_mutex_);
                RCLCPP_INFO(cmd_node_->get_logger(), "下发整条规划轨迹，共 %zu 个点", cartesian_path.size());
                for(int i=0; i<cartesian_path.size(); i++){
                    robot_controller_->sendCartesianCommand({cartesian_path[i]}, "linear", 100.0);
                    robot_controller_->waitForMotionComplete(5000);
                }
            }
        });
    
    
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn AgvHardwareInterface::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/)
{
    return robot_controller_->connect() ? 
        hardware_interface::CallbackReturn::SUCCESS : 
        hardware_interface::CallbackReturn::ERROR;
}

hardware_interface::CallbackReturn AgvHardwareInterface::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/)
{
    if (!robot_controller_->activate()) {
        return hardware_interface::CallbackReturn::ERROR;
    }
    
    joint_positions_ = robot_controller_->getJointPositions();
    ws_client_->start();
    ws_server_->start();

    // 启动独立线程处理 ROS 2 话题回调
    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(cmd_node_);
    executor_thread_ = std::make_unique<std::thread>([this]() {
        executor_->spin();
    });
    
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn AgvHardwareInterface::on_deactivate(
    const rclcpp_lifecycle::State &)
{
    // 停止 Executor 线程
    if (executor_) {
        executor_->cancel();
    }
    if (executor_thread_ && executor_thread_->joinable()) {
        executor_thread_->join();
    }

    ws_server_->stop();
    ws_client_->stop();
    {
        std::lock_guard<std::mutex> lock(io_mutex_);
        enable_status_report_ = false;
    }
    robot_controller_->deactivate();
    return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::return_type AgvHardwareInterface::read(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
    // 1. 更新硬件状态
    {
        std::lock_guard<std::mutex> lock(io_mutex_);
        joint_positions_ = robot_controller_->getJointPositions();
        posture_positions_ = robot_controller_->getPosturePositions();
    }
    
    // 3. 实时发布到 ROS 2 话题 (无论网络状态如何)
    if (joint_state_pub_) {
        auto j_msg = std_msgs::msg::Float64MultiArray();
        j_msg.data.assign(joint_positions_.begin(), joint_positions_.end());
        joint_state_pub_->publish(j_msg);
    }
    if (cartesian_pose_pub_) {
        auto p_msg = std_msgs::msg::Float64MultiArray();
        p_msg.data.assign(posture_positions_.begin(), posture_positions_.end());
        cartesian_pose_pub_->publish(p_msg);
    }

    // 4. WebSocket 状态上报逻辑
    if (!ws_client_->isConnected()) {
        enable_status_report_ = false;
    }

    if (enable_status_report_) {
        // std::cerr << "[WSClient] 开始上报：" << std::endl;
        static int report_skip = 0;            // 静态计数器，跨调用保持
        const int SKIP_COUNT = 50;             // 每10次read上报一次，可自行调整

        if (++report_skip >= SKIP_COUNT) {
            report_skip = 0;                   // 重置计数器

            try {
                const std::string robot_id = "mobile_dual_arm_robot";
                ws_client_->send(command_parser_.buildStatusMessage(joint_positions_, robot_id));
                ws_client_->send(command_parser_.buildPosturePositionsMessage(posture_positions_, robot_id));
            } catch (const std::exception& e) {
                std::cerr << "[AgvHardware] 状态上报异常: " << e.what() << std::endl;
            }
        }
    }

    set_state("arm2_joint1/position", joint_positions_[0]);
    set_state("arm2_joint2/position", joint_positions_[1]);
    set_state("arm2_joint3/position", joint_positions_[2]);
    set_state("arm2_joint4/position", joint_positions_[3]);
    set_state("arm2_joint5/position", joint_positions_[4]);
    set_state("arm2_joint6/position", joint_positions_[5]);
    
    return hardware_interface::return_type::OK;
}

hardware_interface::return_type AgvHardwareInterface::write(
    const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
    // 实时控制模式
    if (config_.rt_mode) {
        std::lock_guard<std::mutex> lock(io_mutex_);
        robot_controller_->sendJointCommand(joint_commands_);
    }
    return hardware_interface::return_type::OK;
}

void AgvHardwareInterface::handleCommand(const std::string& json_str)
{
    auto cmd_opt = command_parser_.parse(json_str);
    if (!cmd_opt) return;
    
    const auto& cmd = *cmd_opt;
    std::cout << "[AgvHardware] 收到 " << cmd.header.msg_type << " 指令 (ID: " << cmd.header.robot_id << ")" << std::endl;

    // 1. 处理检查/自检指令
    if (cmd.header.msg_type == "check") {
        std::cout << "[AgvHardware] 开始执行系统自检..." << std::endl;
        std::vector<std::pair<std::string, bool>> devices;
        
        // 调用自检管理器进行真实检测
        for (const auto& dev : cmd.check_devices) {
            bool is_ok = SelfCheckManager::doCheck(dev, robot_controller_.get());
            devices.push_back({dev, is_ok});
            std::cout << "  - 设备 [" << dev << "]: " << (is_ok ? "正常" : "异常") << std::endl;
        }
        
        ws_client_->send(command_parser_.buildCheckResponse(devices));
        
        std::lock_guard<std::mutex> lock(io_mutex_);
        enable_status_report_ = true;
        return;
    }

    // 2. 处理急停
    if (cmd.emergency_stop) {
        std::cout << "[AgvHardware] 触发急停" << std::endl;
        robot_controller_->deactivate();
        ws_client_->send(command_parser_.buildCommandResponse("system", "emergency_stop", true));
        return;
    }

    // 3. 处理机械臂控制
    for (const auto& arm_cmd : cmd.arm_commands) {
        if (arm_cmd.arm_id != "left_arm") {
            std::cout << "[AgvHardware] 暂不支持 " << arm_cmd.arm_id << "，跳过" << std::endl;
            continue;
        }

        bool success = true;
        if (arm_cmd.is_joint_command) {
            if (config_.rt_mode) {
                std::lock_guard<std::mutex> lock(io_mutex_);
                joint_commands_ = arm_cmd.parameters;
            } else {
                robot_controller_->sendJointCommand(arm_cmd.parameters);
            }
        } else {
            if (config_.rt_mode) {
                std::cerr << "[AgvHardware] 实时模式不支持笛卡尔控制" << std::endl;
                success = false;
            } else {
                std::vector<std::array<double, 6>> points = {arm_cmd.parameters};
                std::string move_type = (arm_cmd.command == "cartesian_move") ? "linear" : "joint";
                robot_controller_->sendCartesianCommand(points, move_type, arm_cmd.speed * 10.0);
            }
        }

        // 等待非实时运动完成并反馈
        // if (success && cmd.header.msg_type == "command" && !config_.rt_mode) {
        //     robot_controller_->waitForMotionComplete();
        // }

        ws_client_->send(command_parser_.buildCommandResponse(arm_cmd.arm_id, arm_cmd.command, success));
    }

    // 4. 处理相机控制 (如点击点发布)
    if (cmd.has_camera_command) {
        if (cmd.camera_cmd.command == "image_pos") {
            auto msg = geometry_msgs::msg::PointStamped();
            msg.header.stamp = cmd_node_->get_clock()->now();
            msg.header.frame_id = "camera_color_optical_frame";
            msg.point.x = cmd.camera_cmd.x;
            msg.point.y = cmd.camera_cmd.y;
            msg.point.z = 0.0;
            
            click_point_pub_->publish(msg);
            RCLCPP_INFO(cmd_node_->get_logger(), "发布点击点: (%.2f, %.2f) 到 /click_point", msg.point.x, msg.point.y);
        }
    }

    // 5. 处理底盘控制
    if (cmd.has_chassis_command) {
        agv_bridge::BridgeManager::publishChassisCmd(
            cmd.chassis_velocity[0],
            cmd.chassis_velocity[1],
            cmd.chassis_velocity[2]
        );
        double vx = cmd.chassis_velocity[0];
        double vy = cmd.chassis_velocity[1];
        double wz = cmd.chassis_velocity[2];
        agv_->setVelocity(vx, vy, wz);
    }
}


} // namespace agv_hardware

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
    agv_hardware::AgvHardwareInterface,
    hardware_interface::SystemInterface)