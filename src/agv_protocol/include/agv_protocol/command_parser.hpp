#pragma once

#include <jsoncpp/json/json.h>
#include <array>
#include <vector>
#include <optional>
#include <string>

namespace agv_protocol {

/**
 * @brief 消息头结构
 */
struct MessageHeader {
    /// @brief 机器人 ID: "mobile_dual_arm_robot", "vacuum_adsorption_robot" 等
    std::string robot_id = "mobile_dual_arm_robot";

    /// @brief 消息类型："check", "command", "query", "status_update"
    std::string msg_type = "command";
};

/**
 * @brief 机械臂控制命令
 */
struct ArmCommand {
    /// @brief 机械臂 ID: "left_arm" 或 "right_arm"
    std::string arm_id = "left_arm";

    /// @brief 命令类型："cartesian_move", "joint_move"
    std::string command = "joint_move";

    /// @brief 关节角度或笛卡尔位姿参数
    std::array<double, 6> parameters{};

    /// @brief 速度 (0-100 百分比)
    double speed = 50.0;

    /// @brief 是否为关节命令
    bool is_joint_command = true;
};

/**
 * @brief 相机控制命令（如图像点击位置）
 */
struct CameraCommand {
    std::string command;
    double x = 0.0;
    double y = 0.0;
    std::string scan_mode = "Long";
    int spacing = 0;
    int shrink_factor = 0;
    int default_InterPoints = 0;
};

/**
 * @brief 解析后的命令数据结构
 */
struct CommandData {
    /// @brief 消息头
    MessageHeader header;

    /// @brief 机械臂命令列表
    std::vector<ArmCommand> arm_commands;

    /// @brief 底盘速度命令 {vx, vy, wz}
    std::array<double, 3> chassis_velocity{};

    /// @brief 是否有底盘命令
    bool has_chassis_command = false;

    /// @brief 设备使能检查命令
    std::vector<std::string> check_devices;

    /// @brief 急停命令
    bool emergency_stop = false;

    /// @brief 是否有相机命令
    bool has_camera_command = false;
    
    /// @brief 相机命令数据
    CameraCommand camera_cmd;
};

/**
 * @brief JSON 命令解析器
 *
 * 负责解析从 WebSocket 接收到的 JSON 格式控制命令（新 ROS2 协议）
 */
class CommandParser {
public:
    /**
     * @brief 构建基础消息 JSON 对象
     * @param robot_id 设备ID
     * @param msg_type 消息类型
     * @return 返回包含 header 的基础消息 JSON 对象
     */
    Json::Value createBaseMessage(const std::string& robot_id, const std::string& msg_type);
    
    /**
     * @brief 构建新的命令数据
     * @param robot_id 设备ID
     * @param msg_type 命令类型
     * @param payload JSON 格式的 payload 数据
     * @return 返回构建的 CommandData 对象，失败返回 std::nullopt
     */
    std::string buildGenericMessage(const std::string& robot_id,
                                    const std::string& msg_type,
                                    const Json::Value& payload);

    /**
     * @brief 解析 JSON 字符串为命令数据
     * @param json_str JSON 格式的输入字符串
     * @return 解析成功返回 CommandData，失败返回 std::nullopt
     */
    std::optional<CommandData> parse(const std::string& json_str);

    /**
     * @brief 构建状态上报消息
     * @param joint_positions 关节位置数组
     * @param robot_id 机器人 ID
     * @return JSON 字符串
     */
    std::string buildStatusMessage(const std::array<double, 6>& joint_positions,
                                    const std::string& robot_id = "mobile_dual_arm_robot");

    /**
     * @brief 构建命令响应消息
     * @param arm_id 机械臂 ID
     * @param command 命令类型
     * @param success 是否成功
     * @return JSON 字符串
     */
    std::string buildCommandResponse(const std::string& arm_id,
                                      const std::string& command,
                                      bool success = true);

    /**
     * @brief 构建笛卡尔位姿状态上报消息
     * @param posture_positions 笛卡尔位姿参数数组
     * @param robot_id 机器人 ID
     * @return JSON 字符串
     */
    std::string buildPosturePositionsMessage(const std::array<double, 6>& posture_positions,
                                            const std::string& robot_id);

    /**
     * @brief 构建使能检查响应消息
     * @param devices 设备列表及其状态
     * @return JSON 字符串
     */
    std::string buildCheckResponse(const std::vector<std::pair<std::string, bool>>& devices);

private:
    /**
     * @brief 解析消息头
     * @param root JSON 根对象
     * @param cmd 输出命令数据
     * @return 解析是否成功
     */
    bool parseHeader(const Json::Value& root, CommandData& cmd);

    /**
     * @brief 解析 payload 部分
     * @param payload JSON payload 对象
     * @param cmd 输出命令数据
     * @return 解析是否成功
     */
    bool parsePayload(const Json::Value& payload, CommandData& cmd);

    /**
     * @brief 解析机械臂命令
     * @param arms JSON 数组
     * @param commands 输出命令列表
     * @return 解析是否成功
     */
    bool parseArmCommands(const Json::Value& arms, std::vector<ArmCommand>& commands);

    /**
     * @brief 解析底盘命令
     * @param chassis JSON 对象
     * @param cmd 输出命令数据
     * @return 解析是否成功
     */
    bool parseChassisCommand(const Json::Value& chassis, CommandData& cmd);

    /**
     * @brief 解析AGV命令（新格式）
     * @param agv JSON 对象
     * @param cmd 输出命令数据
     * @return 解析是否成功
     */
    bool parseAgvCommand(const Json::Value& agv, CommandData& cmd);

    /**
     * @brief 解析相机命令
     * @param camera JSON 对象
     * @param cmd 输出命令数据数据
     * @return 解析是否成功
     */
    bool parseCameraCommand(const Json::Value& camera, CommandData& cmd);

private:
    static double safeGetDouble(const Json::Value& node, const std::string& key, double default_val = 0.0);
    static std::string safeGetString(const Json::Value& node, const std::string& key, const std::string& default_val = "");

    template<size_t N>
    static void extractToVector(const Json::Value& node, const std::array<std::string, N>& keys, std::array<double, N>& out) {
        for (size_t i = 0; i < N; ++i) {
            out[i] = safeGetDouble(node, keys[i], 0.0);
        }
    }
};

} // namespace agv_protocol
