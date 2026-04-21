#include "robot_move/nodes/universal_robot_driver.hpp"

#include <chrono>
#include <functional>
#include <thread>

#include "robot_move/factory/robot_move_factory.hpp"

/**
 * @class UniversalRobotDriver
 * @brief 通用机器人驱动节点类，负责机器人的控制和状态管理
 * 
 * 该类实现了一个ROS2节点，用于控制不同类型的机器人，包括AGV等。
 * 它提供了速度控制、扫描器配置、距离移动、角度旋转等功能，并定期发布机器人姿态信息。
 */

/**
 * @brief 构造函数
 * @param node_name 节点名称
 * 
 * 初始化机器人驱动节点，包括：
 * 1. 声明并获取机器人类型参数
 * 2. 根据机器人类型获取默认连接端口
 * 3. 创建机器人实例
 * 4. 初始化机器人硬件
 * 5. 创建各种订阅者和发布者
 * 6. 创建定时器用于定期发送控制命令和发布姿态信息
 */
UniversalRobotDriver::UniversalRobotDriver(const std::string& node_name) : Node(node_name)
{
    RCLCPP_INFO(this->get_logger(),"[%s]节点启动!", node_name.c_str());
    // 声明机器人类型参数，默认为"agv"
    this->declare_parameter<std::string>("robot_type", "agv");
    std::string type = this->get_parameter("robot_type").as_string();
    // 根据机器人类型获取默认连接端口
    std::string port = robot_move::defaultConnectionForType(type);
    RCLCPP_INFO(this->get_logger(), "正在初始化类型: [%s], 端口: [%s]", type.c_str(), port.c_str());

    // 创建机器人实例
    robot_ = robot_move::createRobotMove(type);
    if (!robot_)
    {
        RCLCPP_ERROR(this->get_logger(), "未知机器人类型: %s", type.c_str());
        return;
    }
    // 初始化机器人硬件
    if (!robot_->init(port)) 
    {
        RCLCPP_FATAL(this->get_logger(), "硬件初始化失败! Port: %s", port.c_str());
        robot_.reset();
        return;
    }
    RCLCPP_INFO(this->get_logger(), "硬件初始化成功!");
    // 记录最后命令时间
    last_cmd_time_ = this->now();

    // 创建速度控制订阅者
    robot_move_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10,
        std::bind(&UniversalRobotDriver::cmdVelCallback, this, std::placeholders::_1));

    // 创建扫描器配置订阅者
    scan_config_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "scan_config", 10,
        std::bind(&UniversalRobotDriver::scanconfigCallback, this, std::placeholders::_1));

    // 创建扫描器控制订阅者
    scannerctrl_sub_ = this->create_subscription<std_msgs::msg::Empty>(
        "scan_ctrl", 10,
        std::bind(&UniversalRobotDriver::scannerctrlCallback, this, std::placeholders::_1));

    // 创建距离移动订阅者
    robot_distance_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "robot_distance_move", 10,
        std::bind(&UniversalRobotDriver::stepMoveCallback, this, std::placeholders::_1));

    // 创建旋转移动订阅者
    robot_rotate_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "robot_rotate_move", 10,
        std::bind(&UniversalRobotDriver::stepRotateCallback, this, std::placeholders::_1));

    // 创建姿态发布者
    pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose2D>("agv_pose", 10);

    // 创建定时器，每20毫秒执行一次timerCallback
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(20),
        std::bind(&UniversalRobotDriver::timerCallback, this)
    );
}

/**
 * @brief 析构函数
 * 
 * 节点关闭时，停止机器人运动并关闭硬件连接
 */
UniversalRobotDriver::~UniversalRobotDriver()
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    if(robot_) 
    {
        RCLCPP_INFO(this->get_logger(), "节点关闭，停止机器人...");
        // 设置速度为0
        robot_->setVelocity(0.0, 0.0, 0.0);
        // 关闭机器人运动
        robot_->moveshutdown();
    }
}

/**
 * @brief 速度控制回调函数
 * @param msg 速度控制消息
 * 
 * 处理cmd_vel话题的消息，设置机器人的线速度和角速度
 */
void UniversalRobotDriver::cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(this->robot_mutex_);
    if (robot_) 
    {
        // 设置机器人速度：x方向线速度，y方向线速度，z方向角速度
        robot_->setVelocity(msg->linear.x, msg->linear.y, msg->angular.z);
        // 更新最后命令时间
        last_cmd_time_ = this->now();
    }
}

/**
 * @brief 扫描器配置回调函数
 * @param msg 扫描器配置消息
 * 
 * 处理scan_config话题的消息，配置扫描器参数
 */
void UniversalRobotDriver::scanconfigCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(this->robot_mutex_);
    if (robot_) 
    {
        // 设置扫描器配置参数
        robot_->setScannerConfig(msg->linear.x, msg->linear.y, msg->angular.z);
    }
}

/**
 * @brief 扫描器控制回调函数
 * @param msg 空消息
 * 
 * 处理scan_ctrl话题的消息，触发扫描器控制操作
 * 在新线程中执行扫描器控制，避免阻塞主线程
 */
void UniversalRobotDriver::scannerctrlCallback(const std_msgs::msg::Empty::SharedPtr msg)
{
    (void)msg; // 忽略消息参数
    if (!robot_) return;

    RCLCPP_INFO(this->get_logger(), "收到扫查器触发指令，开始执行 scannercontrol...");
    // 在新线程中执行扫描器控制
    std::thread([this]()
    {
        std::lock_guard<std::mutex> lock(this->robot_mutex_);
        if (this->robot_) 
        {
            this->robot_->scannercontrol();
        }
    }).detach();
}

/**
 * @brief 距离移动回调函数
 * @param msg 距离移动消息
 * 
 * 处理robot_distance_move话题的消息，执行指定距离的移动
 * 在新线程中执行移动操作，避免阻塞主线程
 */
void UniversalRobotDriver::stepMoveCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    if (!robot_) return;

    // 从消息中提取移动距离和速度参数
    double dist = msg->linear.z;  // 移动距离
    double vx = msg->linear.x;    // x方向速度
    double vy = msg->linear.y;    // y方向速度

    RCLCPP_INFO(this->get_logger(), "StepMove: dist=%.2f, vx=%.2f, vy=%.2f", dist, vx, vy);
    // 在新线程中执行距离移动
    std::thread([this, dist, vx, vy]()
    {
        std::lock_guard<std::mutex> lock(this->robot_mutex_);
        if (this->robot_) 
        { 
            this->robot_->movebydistance(dist, vx, vy);
        }
    }).detach();
}

/**
 * @brief 旋转回调函数
 * @param msg 旋转消息
 * 
 * 处理robot_rotate_move话题的消息，执行指定角度的旋转
 * 在新线程中执行旋转操作，避免阻塞主线程
 */
void UniversalRobotDriver::stepRotateCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    if (!robot_) return;

    // 从消息中提取旋转角度和角速度参数
    double angle = msg->angular.z;  // 旋转角度
    double vw = msg->angular.y;     // 角速度

    RCLCPP_INFO(this->get_logger(), "StepRotate: angle=%.2f, vw=%.2f", angle, vw);
    // 在新线程中执行角度旋转
    std::thread([this, angle, vw]()
    {
        std::lock_guard<std::mutex> lock(this->robot_mutex_);
        if (this->robot_) 
        {
            this->robot_->rotatebyangle(angle, vw);
        }
    }).detach();
}

/**
 * @brief 定时器回调函数
 * 
 * 定期执行以下操作：
 * 1. 发送控制命令到机器人
 * 2. 每5个周期发布一次机器人姿态信息
 */
void UniversalRobotDriver::timerCallback()
{
    std::lock_guard<std::mutex> lock(robot_mutex_);
    if (robot_) 
    {
        // 发送控制命令到机器人
        robot_->send_loop();
        // 每5个周期发布一次姿态信息
        static int loop_count = 0;
        if (++loop_count >= 5) 
        {
            loop_count = 0;
            double x = 0, y = 0, angle = 0;
            // 获取机器人姿态
            if (robot_->getpose(x, y, angle)) 
            {
                // 创建并发布姿态消息
                auto msg = geometry_msgs::msg::Pose2D();
                msg.x = x;
                msg.y = y;
                msg.theta = angle;
                pose_pub_->publish(msg);
            }
        }
    }
}

