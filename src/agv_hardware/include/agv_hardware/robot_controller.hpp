#pragma once

#include "rokae/robot.h"
#include "rokae/data_types.h"
#include "agv_hardware/config.hpp"
#include <array>
#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>

namespace agv_hardware {

/**
 * @brief 机器人控制器封装类
 * 
 * 负责与珞石机器人控制器通信，执行运动控制命令，读取机器人状态
 */
class RobotController {
public:
    /**
     * @brief 构造函数
     * @param config 配置参数
     */
    explicit RobotController(const AgvConfig& config);
    
    /// @brief 析构函数
    ~RobotController();

    /**
     * @brief 连接到机器人控制器
     * @return true=连接成功，false=连接失败
     */
    bool connect();
    
    /**
     * @brief 断开与机器人控制器的连接
     */
    void disconnect();
    
    /**
     * @brief 检查是否已连接
     * @return true=已连接，false=未连接
     */
    bool isConnected() const { return is_connected_; }

    /**
     * @brief 激活机器人（上电、设置模式等）
     * @return true=激活成功，false=激活失败
     */
    bool activate();
    
    /**
     * @brief 停用机器人（停止运动、断电等）
     */
    void deactivate();

    /**
     * @brief 读取当前关节位置
     * @return 6 个关节的角度数组（弧度）
     */
    std::array<double, 6> getJointPositions();
    
    /**
     * @brief 读取当前笛卡尔位置
     * @return 6 个笛卡尔坐标系数据 （x, y, z, rx, ry, rz）
     */
    std::array<double, 6> getPosturePositions();

    /**
     * @brief 发送关节空间运动命令
     * @param joints 6 个关节的目标角度（弧度）
     * @return true=命令发送成功，false=发送失败
     */
    bool sendJointCommand(const std::array<double, 6>& joints);
    
    /**
     * @brief 发送笛卡尔空间运动命令
     * @param points 路径点数组，每个点包含 {x, y, z, rx, ry, rz}
     * @param move_type 运动类型："joint" 或 "linear"
     * @param speed 运动速度（mm/s）
     * @return true=命令发送成功，false=发送失败
     */
    bool sendCartesianCommand(const std::vector<std::array<double, 6>>& points,
                              const std::string& move_type, double speed);
    
    /**
     * @brief 等待运动完成
     * @param timeout_ms 超时时间（毫秒）
     * @return true=运动完成，false=超时或出错
     */
    bool waitForMotionComplete(int timeout_ms = 10000);
    
    /**
     * @brief 检查是否有运动正在执行
     * @return true=有运动正在执行，false=空闲
     */
    bool isMotionExecuting() const { return nrt_executing_; }
    
    /**
     * @brief 获取当前运动命令 ID
     */
    const std::string& getCurrentCommandId() const { return current_cmd_id_; }

private:
    /**
     * @brief 非实时模式发送关节命令
     * @param joints 6 个关节的目标角度（弧度）
     * @return true=命令发送成功，false=发送失败
     */
    bool sendNrtJointCommand(const std::array<double, 6>& joints);
    
    /**
     * @brief 非实时模式发送笛卡尔命令
     * @param points 路径点数组
     * @param move_type 运动类型
     * @param speed 运动速度
     * @return true=命令发送成功，false=发送失败
     */
    bool sendNrtCartesianCommand(const std::vector<std::array<double, 6>>& points,
                                  const std::string& move_type, double speed);

    AgvConfig config_;
    std::shared_ptr<rokae::xMateRobot> robot_;
    std::shared_ptr<rokae::RtMotionControlCobot<6>> rt_controller_;
    
    std::atomic<bool> is_connected_{false};
    std::atomic<bool> is_rt_mode_{false};
    std::atomic<bool> nrt_executing_{false};
    std::string current_cmd_id_;
    
    error_code ec_;
    std::mutex mutex_;
    
    // RT 模式回调初始化标志
    int callback_init_ = 0;
    
    // 存储当前关节命令（用于 RT 模式）
    std::array<double, 6> joint_commands_{};
    std::mutex command_mutex_;
};

} // namespace agv_hardware