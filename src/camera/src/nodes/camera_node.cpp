#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "camera/nodes/astras_node.hpp"

/**
 * @brief camera 包入口：启动 AstraSNode 并进入 spin。
 */
int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<camera::AstraSNode>());
    rclcpp::shutdown();
    return 0;
}
