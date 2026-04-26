// ==========================================
// 指令
// 1、ros2 launch demo_robot_move move.launch.py type:=agv
// 2、ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.2, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
// ==========================================
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "robot_move/nodes/universal_robot_driver.hpp"

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UniversalRobotDriver>("robot_gotarget_node");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
