#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "agv_hardware/robot_controller.hpp"
#include "agv_bridge/bridge_manager.hpp"

namespace agv_hardware {

/**
 * @class SelfCheckManager
 * @brief 机器人全系统自检管理器
 */
class SelfCheckManager {
public:
    /**
     * @brief 执行特定设备的自检
     * @param device_name 设备名称 (agv, arms, radar, camera, eddy_current, etc.)
     * @param arm 机器人控制器指针
     * @return true=正常，false=异常
     */
    static bool doCheck(const std::string& device_name, RobotController* arm) {
        if (device_name == "agv" || device_name == "arm" || device_name == "arms") {
            return checkArm(arm);
        } else if (device_name == "radar") {
            return checkRadar();
        } else if (device_name == "camera") {
            return checkCamera();
        } else if (device_name == "chassis") {
            return checkChassis();
        } else if (device_name == "eddy_current") {
            return checkEddyCurrent();
        }
        
        // 未知设备默认返回失败
        return false;
    }

private:
    // 5. 涡流传感器自检逻辑 (占位实现)
    static bool checkEddyCurrent() {
        // 此处可根据实际需求添加话题检查或硬件通信检查
        // 默认返回 false，表示未检测到设备或未实现逻辑
        return false;
    }
    // 1. 机械臂自检逻辑
    static bool checkArm(RobotController* arm) {
        if (!arm) return false;
        
        // 检查 SDK 是否已连接
        if (!arm->isConnected()) return false;
        
        // 检查是否处于运动状态或其他错误（此处可扩展更深层的 SDK 调用）
        return true; 
    }

    // 2. 雷达自检逻辑：检查 ROS2 话题活跃度
    static bool checkRadar() {
        return agv_bridge::BridgeManager::isRadarOnline();
    }

    // 3. 相机自检逻辑：检查 ROS2 话题活跃度
    static bool checkCamera() {
        return agv_bridge::BridgeManager::isCameraOnline();
    }

    // 4. 底盘自检逻辑：电量或状态位检查
    static bool checkChassis() {
        double battery = agv_bridge::BridgeManager::getBatteryLevel();
        if (battery < 10.0) {
            std::cerr << "[SelfCheck] 警报：底盘电量过低 (" << battery << "%)" << std::endl;
            return false;
        }
        return true;
    }
};

} // namespace agv_hardware
