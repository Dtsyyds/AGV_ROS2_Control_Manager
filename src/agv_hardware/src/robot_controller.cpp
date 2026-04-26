#include "agv_hardware/robot_controller.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace agv_hardware {

RobotController::RobotController(const AgvConfig& config)
    : config_(config)
    , robot_(std::make_shared<rokae::xMateRobot>())
{
}

RobotController::~RobotController()
{
    disconnect();
}

bool RobotController::connect()
{
    if (is_connected_) {
        return true;
    }
    
    try {
        robot_->connectToRobot(config_.robot_ip, config_.local_ip);
        is_connected_ = true;
        std::cout << "[RobotController] 已连接到机器人：" << config_.robot_ip << std::endl;
        return true;
    }
    catch (const rokae::NetworkException& e) {
        std::cerr << "[RobotController] 网络连接错误：" << e.what() << std::endl;
        return false;
    }
    catch (const rokae::ExecutionException& e) {
        std::cerr << "[RobotController] 执行错误：" << e.what() << std::endl;
        return false;
    }
}

void RobotController::disconnect()
{
    if (!is_connected_) {
        return;
    }
    
    deactivate();
    
    try {
        ec_.clear();
        robot_->disconnectFromRobot(ec_);
        is_connected_ = false;
        std::cout << "[RobotController] 已断开连接" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[RobotController] 断开连接异常：" << e.what() << std::endl;
    }
}

bool RobotController::activate()
{
    if (!is_connected_) {
        std::cerr << "[RobotController] 未连接，无法激活" << std::endl;
        return false;
    }
    
    try {
        ec_.clear();
        
        // 设置网络容差
        robot_->setRtNetworkTolerance(config_.rt_network_tolerance_ms, ec_);
        
        if (config_.rt_mode) {
            // === 实时模式初始化 ===
            std::cout << "[RobotController] 正在初始化实时模式..." << std::endl;
            
            robot_->setMotionControlMode(rokae::MotionControlMode::RtCommand, ec_);
            robot_->setOperateMode(rokae::OperateMode::automatic, ec_);
            robot_->setPowerState(true, ec_);
            
            // 启动状态接收
            robot_->startReceiveRobotState(std::chrono::milliseconds(1),
                {rokae::RtSupportedFields::jointPos_m});
            
            rt_controller_ = robot_->getRtMotionController().lock();
            if (!rt_controller_) {
                std::cerr << "[RobotController] 获取 RT 控制器失败" << std::endl;
                return false;
            }
            
            rt_controller_->startMove(rokae::RtControllerMode::jointPosition);
            is_rt_mode_ = true;
            
            std::cout << "[RobotController] 实时模式激活完成" << std::endl;
        } else {
            // === 非实时模式初始化 ===
            std::cout << "[RobotController] 正在初始化非实时模式..." << std::endl;
            
            robot_->setMotionControlMode(rokae::MotionControlMode::NrtCommand, ec_);
            robot_->setOperateMode(rokae::OperateMode::automatic, ec_);
            robot_->setPowerState(true, ec_);
            
            // 非实时模式需要设置默认速度
            robot_->setDefaultSpeed(config_.default_speed, ec_);
            robot_->adjustSpeedOnline(config_.speed_ratio, ec_);
            robot_->moveReset(ec_);
            
            is_rt_mode_ = false;
            
            std::cout << "[RobotController] 非实时模式激活完成" << std::endl;
        }
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[RobotController] 激活异常：" << e.what() << std::endl;
        return false;
    }
}

void RobotController::deactivate()
{
    try {
        if (is_rt_mode_ && rt_controller_) {
            rt_controller_->stopLoop();
            rt_controller_->stopMove();
            rt_controller_.reset();
        } else if (!is_rt_mode_) {
            ec_.clear();
            robot_->stop(ec_);
            robot_->moveReset(ec_);
        }
        
        ec_.clear();
        robot_->stopReceiveRobotState();
        robot_->setPowerState(false, ec_);
        
        is_rt_mode_ = false;
        nrt_executing_ = false;
        callback_init_ = 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[RobotController] 停用异常：" << e.what() << std::endl;
    }
}

std::array<double, 6> RobotController::getJointPositions()
{
    std::array<double, 6> positions{};
    
    if (!is_connected_) {
        return positions;
    }
    
    try {
        ec_.clear();
        auto joint_pos = robot_->jointPos(ec_);
        if (!ec_) {
            for (size_t i = 0; i < 6; ++i) {
                positions[i] = joint_pos[i];
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[RobotController] 读取关节位置异常：" << e.what() << std::endl;
    }
    
    return positions;
}

std::array<double, 6> RobotController::getPosturePositions()
{
    std::array<double, 6> positions{};
    
    if (!is_connected_) {
        return positions;
    }
    
    try {
        ec_.clear();
        auto joint_pos = robot_->posture(rokae::CoordinateType::endInRef, ec_);
        if (!ec_) {
            for (size_t i = 0; i < 6; ++i) {
                positions[i] = joint_pos[i];
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[RobotController] 读取关节位置异常：" << e.what() << std::endl;
    }
    
    return positions;
}

bool RobotController::sendJointCommand(const std::array<double, 6>& joints)
{
    if (config_.rt_mode) {
        // 实时模式：更新命令缓存，由 RT 循环执行
        std::lock_guard<std::mutex> lock(command_mutex_);
        joint_commands_ = joints;
        return true;
    } else {
        // 非实时模式：直接发送命令
        return sendNrtJointCommand(joints);
    }
}

bool RobotController::sendCartesianCommand(const std::vector<std::array<double, 6>>& points,
                                           const std::string& move_type, double speed)
{
    if (config_.rt_mode) {
        // 实时模式：暂不支持笛卡尔空间实时控制
        std::cerr << "[RobotController] 实时模式暂不支持笛卡尔空间控制" << std::endl;
        return false;
    } else {
        // 非实时模式：直接发送命令
        return sendNrtCartesianCommand(points, move_type, speed);
    }
}

bool RobotController::waitForMotionComplete(int timeout_ms)
{
    auto start = std::chrono::steady_clock::now();
    
    while (nrt_executing_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // 检查机器人实际状态
        try {
            ec_.clear();
            auto state = robot_->operationState(ec_);
            if (!ec_) {
                if (state == rokae::OperationState::idle) {
                    nrt_executing_ = false;
                    break;
                }
            }
        } catch (...) {
            // 忽略异常
        }
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > timeout_ms) {
            std::cerr << "[RobotController] 等待超时 " << timeout_ms << "ms" << std::endl;
            // 尝试停止运动
            ec_.clear();
            robot_->stop(ec_);
            nrt_executing_ = false;
            return false;
        }
    }
    
    return true;
}

bool RobotController::sendNrtJointCommand(const std::array<double, 6>& joints)
{
    try {
        // 如果当前正在执行，先停止并重置，允许新命令覆盖
        if (nrt_executing_) {
            std::cout << "[RobotController] 停止当前运动以执行新命令..." << std::endl;
            ec_.clear();
            robot_->moveReset(ec_);
            nrt_executing_ = false;
        }
        
        // 参数检查
        for (size_t i = 0; i < 6; ++i) {
            if (joints[i] < -3.142 || joints[i] > 3.142) {
                std::cerr << "[RobotController] 关节 " << i << " 角度 " << joints[i] 
                          << " 可能超出范围" << std::endl;
            }
        }
        
        rokae::JointPosition target;
        target.joints = std::vector<double>(joints.begin(), joints.end());
        
        rokae::MoveAbsJCommand cmd(target);
        
        std::string cmd_id;
        ec_.clear();
        
        robot_->moveAppend(cmd, cmd_id, ec_);
        if (ec_) {
            std::cerr << "[RobotController] moveAppend 失败：" << ec_.message() << std::endl;
            return false;
        }
        
        nrt_executing_ = true;
        current_cmd_id_ = cmd_id;
        
        ec_.clear();
        robot_->moveStart(ec_);
        if (ec_) {
            std::cerr << "[RobotController] moveStart 失败：" << ec_.message() << std::endl;
            nrt_executing_ = false;
            return false;
        }
        
        std::cout << "[RobotController] 发送关节命令成功：ID=" << cmd_id << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[RobotController] 发送关节命令异常：" << e.what() << std::endl;
        nrt_executing_ = false;
        return false;
    }
}

bool RobotController::sendNrtCartesianCommand(const std::vector<std::array<double, 6>>& points,
                                               const std::string& move_type, double speed)
{

    try {
        // 如果当前正在执行，先停止并重置，允许新命令覆盖
        if (nrt_executing_) {
            std::cout << "[RobotController] 停止当前笛卡尔运动以执行新命令..." << std::endl;
            ec_.clear();
            robot_->moveReset(ec_);
            nrt_executing_ = false;
        }
        
        ec_.clear();
        robot_->moveReset(ec_);  // 重置队列
        
        for (const auto& point : points) {
            if (move_type == "joint") {
                // 关节空间运动
                rokae::JointPosition target;
                target.joints = std::vector<double>(point.begin(), point.end());
                rokae::MoveAbsJCommand cmd(target);
                
                cmd.speed = speed > 0 ? speed : config_.default_speed;
                cmd.zone = config_.nrt_zone;
                
                ec_.clear();
                robot_->moveAppend(cmd, current_cmd_id_, ec_);
            } else {
                // 笛卡尔空间运动
                rokae::CartesianPosition target;
                target.trans[0] = point[0];  // X
                target.trans[1] = point[1];  // Y
                target.trans[2] = point[2];  // Z
                target.rpy[0] = point[3];    // Rx
                target.rpy[1] = point[4];    // Ry
                target.rpy[2] = point[5];    // Rz
                std::cerr << "[RobotController] 笛卡尔点：" << point[0] << ", " << point[1] << ", " << point[2] << ", "
                          << point[3] << ", " << point[4] << ", " << point[5] << std::endl;
                rokae::MoveLCommand cmd(target);
                cmd.speed = speed > 0 ? speed : 100.0;  // mm/s
                cmd.zone = 0.5;  // mm
                
                ec_.clear();
                robot_->moveAppend(cmd, current_cmd_id_, ec_);
            }
            
            if (ec_) {
                std::cerr << "[RobotController] 添加笛卡尔命令失败：" << ec_.message() << std::endl;
                return false;
            }
        }
        
        ec_.clear();
        robot_->moveStart(ec_);
        nrt_executing_ = true;

        std::cout << "[RobotController] 发送非实时笛卡尔命令：点数=" << points.size() 
                  << ", 类型=" << move_type << ", 速度=" << speed << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[RobotController] 发送非实时笛卡尔命令异常：" << e.what() << std::endl;
        return false;
    }
}

} // namespace agv_hardware