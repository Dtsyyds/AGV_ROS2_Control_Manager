#ifndef UNIVERSAL_ROBOT_DRIVER_HPP
#define UNIVERSAL_ROBOT_DRIVER_HPP

#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>

#include "robot_move/interface/robot_move_base.hpp"

class UniversalRobotDriver : public rclcpp::Node
{
private:
    std::unique_ptr<RobotMove> robot_;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr robot_move_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr robot_distance_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr robot_rotate_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr scan_config_sub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr scannerctrl_sub_;

    rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pose_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::mutex robot_mutex_;
    rclcpp::Time last_cmd_time_;

    double target_linear_x_ = 0.0;
    double target_linear_y_ = 0.0;
    double target_angular_z_ = 0.0;

    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void scanconfigCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void scannerctrlCallback(const std_msgs::msg::Empty::SharedPtr msg);
    void stepMoveCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void stepRotateCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void timerCallback();

public:
    explicit UniversalRobotDriver(const std::string& node_name);
    ~UniversalRobotDriver() override;
};

#endif
